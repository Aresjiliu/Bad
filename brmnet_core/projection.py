from __future__ import annotations

import torch
from torch import nn


class FeatureProjection(nn.Module):
    """Map branch features into a common fusion width when needed."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if min(in_channels, out_channels) <= 0:
            raise ValueError("Projection channel counts must be positive.")
        self.net: nn.Module
        if in_channels == out_channels:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
