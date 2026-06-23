from __future__ import annotations

import torch
from torch import nn

from .budget_gates import BudgetGatedConv2d


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

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        feature_main: torch.Tensor,
        feature_aux: torch.Tensor,
        quality_main: torch.Tensor,
        quality_aux: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_main.shape != feature_aux.shape:
            raise ValueError(f"Feature shapes must match: {feature_main.shape} vs {feature_aux.shape}")
        quality = torch.cat([quality_main, quality_aux], dim=1)
        weights = torch.softmax(quality / self.temperature, dim=1)
        fused = weights[:, 0:1, None, None] * feature_main + weights[:, 1:2, None, None] * feature_aux
        return fused, weights


class BudgetGatedFusionHead(nn.Module):
    """Budget-gated fusion classifier head."""

    def __init__(self, channels: int, num_classes: int, init_score: float = 0.85) -> None:
        super().__init__()
        self.net = nn.Sequential(
            BudgetGatedConv2d(channels, 128, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BudgetGatedConv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

