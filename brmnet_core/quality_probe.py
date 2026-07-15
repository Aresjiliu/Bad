from __future__ import annotations

import torch
from torch import nn


class _ModalityQualityBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.quality_head = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        return self.quality_head(features), self.uncertainty_head(features)


class PreEncoderQualityProbe(nn.Module):
    """Estimate modality quality before the main encoders consume their budget."""

    def __init__(self, main_channels: int, aux_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        if min(main_channels, aux_channels, hidden_channels) <= 0:
            raise ValueError("Probe channel counts must be positive.")
        self.main_branch = _ModalityQualityBranch(main_channels, hidden_channels)
        self.aux_branch = _ModalityQualityBranch(aux_channels, hidden_channels)

    def forward(
        self,
        main_input: torch.Tensor,
        aux_input: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if main_input.shape[0] != aux_input.shape[0]:
            raise ValueError("Main and auxiliary inputs must have the same batch size.")
        q_main, u_main = self.main_branch(main_input)
        q_aux, u_aux = self.aux_branch(aux_input)
        if availability_mask is not None:
            availability_mask = availability_mask.to(device=main_input.device, dtype=torch.bool)
            expected_shape = (main_input.shape[0], 2)
            if tuple(availability_mask.shape) != expected_shape:
                raise ValueError(f"Availability mask must have shape {expected_shape}, got {tuple(availability_mask.shape)}")
            main_available = availability_mask[:, 0:1]
            aux_available = availability_mask[:, 1:2]
            q_main = torch.where(main_available, q_main, torch.zeros_like(q_main))
            q_aux = torch.where(aux_available, q_aux, torch.zeros_like(q_aux))
            u_main = torch.where(main_available, u_main, torch.ones_like(u_main))
            u_aux = torch.where(aux_available, u_aux, torch.ones_like(u_aux))
        return {
            "pre_q_main": q_main,
            "pre_q_aux": q_aux,
            "pre_u_main": u_main,
            "pre_u_aux": u_aux,
        }
