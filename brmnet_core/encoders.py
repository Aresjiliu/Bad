from __future__ import annotations

from torch import nn

from .budget_gates import BudgetGatedConv2d


class BudgetGatedEncoder(nn.Module):
    """Small multimodal branch encoder with budget-gated convolutions."""

    def __init__(self, in_channels: int, init_score: float = 0.85, width: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        c1, c2, c3 = width
        self.out_channels = c3
        self.net = nn.Sequential(
            BudgetGatedConv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            BudgetGatedConv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False, init_score=init_score),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            BudgetGatedConv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False, init_score=init_score),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

