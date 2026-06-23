from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ModalityQualityEstimator(nn.Module):
    """Lightweight reliability estimator for one modality feature map.

    This module is intentionally standalone so it can be integrated into the
    existing experimental branches without changing their current behavior.

    Input shape:
        [batch, channels, height, width]
    Output shape:
        [batch, 1], with values in [0, 1]
    """

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
    """Two-modality reliability-gated feature fusion."""

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
            raise ValueError(
                "ReliabilityGatedFusion currently requires equal feature shapes: "
                f"{tuple(feature_main.shape)} vs {tuple(feature_aux.shape)}"
            )

        quality = torch.cat([quality_main, quality_aux], dim=1)
        weights = torch.softmax(quality / self.temperature, dim=1)
        fused = (
            weights[:, 0:1, None, None] * feature_main
            + weights[:, 1:2, None, None] * feature_aux
        )
        return fused, weights


def modality_quality_loss(
    quality_main: torch.Tensor,
    quality_aux: torch.Tensor,
    target_main: torch.Tensor,
    target_aux: torch.Tensor,
) -> torch.Tensor:
    """Regression loss for synthetic modality quality targets."""

    target_main = target_main.reshape_as(quality_main).to(
        dtype=quality_main.dtype,
        device=quality_main.device,
    )
    target_aux = target_aux.reshape_as(quality_aux).to(
        dtype=quality_aux.dtype,
        device=quality_aux.device,
    )
    return F.mse_loss(quality_main, target_main) + F.mse_loss(quality_aux, target_aux)

