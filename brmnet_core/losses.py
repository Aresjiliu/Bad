from __future__ import annotations

import torch
import torch.nn.functional as F

from .budget_gates import collect_budget_loss, collect_budget_stats
from .resources import resource_budget_loss


def modality_quality_loss(
    q_main: torch.Tensor,
    q_aux: torch.Tensor,
    target_main: torch.Tensor,
    target_aux: torch.Tensor,
) -> torch.Tensor:
    target_main = target_main.reshape_as(q_main).to(dtype=q_main.dtype, device=q_main.device)
    target_aux = target_aux.reshape_as(q_aux).to(dtype=q_aux.dtype, device=q_aux.device)
    return F.mse_loss(q_main, target_main) + F.mse_loss(q_aux, target_aux)


def brmnet_loss(
    model,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_budget: float = 1e-3,
    target_budget: float | None = None,
    patch_size: int | None = None,
    budget_metric: str = "macs",
    lambda_quality: float = 0.0,
    quality_targets: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    cls = F.cross_entropy(outputs["logits"], labels)
    resource_ratio = torch.zeros((), dtype=cls.dtype, device=labels.device)
    expected_params_ratio = torch.zeros((), dtype=cls.dtype, device=labels.device)
    expected_macs_ratio = torch.zeros((), dtype=cls.dtype, device=labels.device)
    if target_budget is None:
        budget = collect_budget_loss(model).to(device=labels.device)
        soft_retention = torch.zeros((), dtype=cls.dtype, device=labels.device)
        hard_retention = torch.zeros((), dtype=cls.dtype, device=labels.device)
        target = torch.zeros((), dtype=cls.dtype, device=labels.device)
    elif getattr(model, "gate_type", None) == "hard_concrete":
        if patch_size is None:
            raise ValueError("patch_size is required for hard_concrete resource budgets")
        budget, resource_stats = resource_budget_loss(
            model,
            target_budget=target_budget,
            patch_size=patch_size,
            metric=budget_metric,
        )
        budget = budget.to(device=labels.device)
        expected_params_ratio = resource_stats.params_ratio.to(device=labels.device)
        expected_macs_ratio = resource_stats.macs_ratio.to(device=labels.device)
        resource_ratio = (
            expected_params_ratio if budget_metric == "params" else expected_macs_ratio
        )
        soft_retention = torch.zeros((), dtype=cls.dtype, device=labels.device)
        hard_retention = torch.zeros((), dtype=cls.dtype, device=labels.device)
        target = budget.new_tensor(float(target_budget))
    else:
        if not 0.0 < target_budget <= 1.0:
            raise ValueError(f"target_budget must be in (0, 1], got {target_budget}")
        stats = collect_budget_stats(model)
        soft_retention = stats.soft_retention.to(device=labels.device)
        target = soft_retention.new_tensor(float(target_budget))
        hard_retention = soft_retention.new_tensor(stats.hard_retention)
        budget = torch.square(soft_retention - target)
    quality = torch.zeros((), dtype=cls.dtype, device=labels.device)
    if quality_targets is not None:
        quality = modality_quality_loss(outputs["q_main"], outputs["q_aux"], quality_targets[0], quality_targets[1])
    total = cls + lambda_budget * budget + lambda_quality * quality
    return {
        "total": total,
        "cls": cls,
        "budget": budget,
        "quality": quality,
        "soft_retention": soft_retention,
        "hard_retention": hard_retention,
        "target_budget": target,
        "resource_ratio": resource_ratio,
        "expected_params_ratio": expected_params_ratio,
        "expected_macs_ratio": expected_macs_ratio,
    }
