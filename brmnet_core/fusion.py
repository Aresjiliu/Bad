from __future__ import annotations

import torch
from torch import nn

from .budget_gates import BudgetGatedConv2d
from .gated_blocks import HardConcreteConvBlock


class ModalityQualityEstimator(nn.Module):
    """Lightweight reliability estimator for one modality feature map."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class ReliabilityGatedFusion(nn.Module):
    """Reliability-gated additive fusion for two modalities."""

    def __init__(self, temperature: float = 1.0, mode: str = "reliability") -> None:
        super().__init__()
        if mode not in {"reliability", "uniform"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        self.temperature = temperature
        self.mode = mode

    def forward(
        self,
        feature_main: torch.Tensor,
        feature_aux: torch.Tensor,
        quality_main: torch.Tensor,
        quality_aux: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_main.shape != feature_aux.shape:
            raise ValueError(f"Feature shapes must match: {feature_main.shape} vs {feature_aux.shape}")
        quality = torch.cat([quality_main, quality_aux], dim=1)
        if self.mode == "uniform":
            logits = torch.ones_like(quality)
        else:
            logits = quality / self.temperature
        if availability_mask is not None:
            availability_mask = availability_mask.to(device=logits.device, dtype=torch.bool)
            if availability_mask.shape != logits.shape:
                raise ValueError(f"Availability mask must have shape {tuple(logits.shape)}, got {tuple(availability_mask.shape)}")
            if bool((availability_mask.sum(dim=1) == 0).any()):
                raise ValueError("At least one modality must be available for each sample.")
            logits = logits.masked_fill(~availability_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = weights[:, 0:1, None, None] * feature_main + weights[:, 1:2, None, None] * feature_aux
        return fused, weights


class BudgetGatedFusionHead(nn.Module):
    """Budget-gated fusion classifier head."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        init_score: float = 0.85,
        gate_type: str = "legacy_sigmoid",
        initial_retention: float = 0.9,
        width: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        h1, h2 = width
        self.gate_type = gate_type
        if gate_type == "legacy_sigmoid":
            self.net = nn.Sequential(
                BudgetGatedConv2d(channels, h1, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
                nn.BatchNorm2d(h1),
                nn.ReLU(inplace=True),
                BudgetGatedConv2d(h1, h2, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
                nn.BatchNorm2d(h2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(h2, num_classes),
            )
        elif gate_type == "hard_concrete":
            self.net = nn.ModuleList(
                [
                    HardConcreteConvBlock(
                        channels, h1, kernel_size=3, padding=1, initial_retention=initial_retention
                    ),
                    HardConcreteConvBlock(
                        h1, h2, kernel_size=3, padding=1, initial_retention=initial_retention
                    ),
                ]
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(h2, num_classes)
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "legacy_sigmoid":
            return self.net(x)
        for block in self.net:
            x = block(x)
        return self.linear(self.flatten(self.pool(x)))
