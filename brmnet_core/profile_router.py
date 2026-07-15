from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _profile_budget_tensor(
    profile_budgets: tuple[float, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if len(profile_budgets) < 2:
        raise ValueError("profile_budgets must contain at least two budget levels")
    values = torch.tensor(profile_budgets, device=device, dtype=dtype or torch.float32)
    if not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("profile_budgets must be strictly increasing")
    return values


class QualityBudgetRouter(nn.Module):
    """Map pre-encoder quality diagnostics to a deployable budget profile."""

    def __init__(
        self,
        profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        budgets = _profile_budget_tensor(profile_budgets)
        self.register_buffer("profile_budgets", budgets)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, len(profile_budgets)),
        )

    def forward(self, quality_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if quality_features.ndim != 2 or quality_features.shape[1] != 4:
            raise ValueError("quality_features must have shape (batch, 4)")
        logits = self.net(quality_features)
        probs = torch.softmax(logits, dim=1)
        budgets = self.profile_budgets.to(device=quality_features.device, dtype=quality_features.dtype)
        expected_budget = probs @ budgets.reshape(-1, 1)
        selected_profile = probs.argmax(dim=1)
        return {
            "profile_logits": logits,
            "profile_probs": probs,
            "expected_budget": expected_budget,
            "selected_profile": selected_profile,
            "selected_budget": budgets[selected_profile].reshape(-1, 1),
        }


def oracle_budget_profile_targets(
    quality_features: torch.Tensor,
    profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    clean_quality_threshold: float = 0.75,
    degraded_quality_threshold: float = 0.35,
    clean_uncertainty_threshold: float = 0.35,
    degraded_uncertainty_threshold: float = 0.70,
) -> dict[str, torch.Tensor]:
    """Create coarse profile labels for the first routing experiments."""

    if quality_features.ndim != 2 or quality_features.shape[1] != 4:
        raise ValueError("quality_features must have shape (batch, 4)")
    budgets = _profile_budget_tensor(
        profile_budgets,
        device=quality_features.device,
        dtype=quality_features.dtype,
    )
    q_main, q_aux, u_main, u_aux = quality_features.unbind(dim=1)
    min_quality = torch.minimum(q_main, q_aux)
    max_uncertainty = torch.maximum(u_main, u_aux)

    high_index = len(profile_budgets) - 1
    mid_index = len(profile_budgets) // 2
    low_index = 0
    target_indices = torch.full(
        (quality_features.shape[0],),
        mid_index,
        dtype=torch.long,
        device=quality_features.device,
    )
    clean = (min_quality >= clean_quality_threshold) & (max_uncertainty <= clean_uncertainty_threshold)
    degraded = (min_quality <= degraded_quality_threshold) | (max_uncertainty >= degraded_uncertainty_threshold)
    target_indices[clean] = high_index
    target_indices[degraded] = low_index
    return {
        "target_indices": target_indices,
        "target_budgets": budgets[target_indices].reshape(-1, 1),
    }


def budget_profile_routing_loss(
    profile_logits: torch.Tensor,
    expected_budget: torch.Tensor,
    targets: dict[str, torch.Tensor],
    budget_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    target_indices = targets["target_indices"].to(device=profile_logits.device, dtype=torch.long)
    target_budgets = targets["target_budgets"].to(
        device=expected_budget.device,
        dtype=expected_budget.dtype,
    )
    routing_cls = F.cross_entropy(profile_logits, target_indices)
    routing_budget = F.mse_loss(expected_budget, target_budgets.reshape_as(expected_budget))
    routing_total = routing_cls + float(budget_weight) * routing_budget
    return {
        "routing_total": routing_total,
        "routing_cls": routing_cls,
        "routing_budget": routing_budget,
    }


def evaluate_budget_profile_routing(
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    quality_by_mode: dict[str, list[float] | tuple[float, float, float, float]],
    profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    selected_budgets_by_mode: dict[str, float] | None = None,
    metric_key: str = "oa",
    resource_key: str = "expected_macs_ratio",
) -> dict[str, object]:
    budgets = tuple(float(budget) for budget in profile_budgets)
    normalized_profile_metrics = {float(budget): metrics for budget, metrics in profile_metrics.items()}
    missing_profiles = [budget for budget in budgets if budget not in normalized_profile_metrics]
    if missing_profiles:
        raise KeyError(f"Missing profile metrics for budgets: {missing_profiles}")

    per_mode: dict[str, dict[str, float]] = {}
    metric_values: list[float] = []
    selected_budget_values: list[float] = []
    resource_values: list[float] = []

    for mode, features in quality_by_mode.items():
        if selected_budgets_by_mode is None:
            feature_tensor = torch.as_tensor([features], dtype=torch.float32)
            targets = oracle_budget_profile_targets(feature_tensor, profile_budgets=budgets)
            selected_budget = float(targets["target_budgets"][0, 0].detach().cpu())
        else:
            selected_budget = float(selected_budgets_by_mode[mode])
        selected_budget = min(budgets, key=lambda budget: abs(budget - selected_budget))
        if selected_budget not in normalized_profile_metrics:
            raise KeyError(f"Missing selected budget profile: {selected_budget}")
        if mode not in normalized_profile_metrics[selected_budget]:
            raise KeyError(f"Missing mode {mode!r} for budget profile {selected_budget}")

        metrics = normalized_profile_metrics[selected_budget][mode]
        metric_value = float(metrics[metric_key])
        resource_value = float(metrics.get(resource_key, selected_budget))
        per_mode[mode] = {
            "selected_budget": selected_budget,
            metric_key: metric_value,
            resource_key: resource_value,
        }
        metric_values.append(metric_value)
        selected_budget_values.append(selected_budget)
        resource_values.append(resource_value)

    count = max(len(metric_values), 1)
    summary = {
        f"mean_{metric_key}": sum(metric_values) / count,
        "mean_selected_budget": sum(selected_budget_values) / count,
        f"mean_{resource_key}": sum(resource_values) / count,
        "modes": count,
    }
    return {"per_mode": per_mode, "summary": summary}
