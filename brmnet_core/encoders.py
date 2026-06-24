from __future__ import annotations

from torch import nn

from .budget_gates import BudgetGatedConv2d
from .gated_blocks import HardConcreteConvBlock
from .hard_concrete import HardConcreteGate


class BudgetGatedEncoder(nn.Module):
    """Small multimodal branch encoder with budget-gated convolutions."""

    def __init__(
        self,
        in_channels: int,
        init_score: float = 0.85,
        width: tuple[int, int, int] = (32, 64, 128),
        gate_type: str = "legacy_sigmoid",
        initial_retention: float = 0.9,
        output_gate: HardConcreteGate | None = None,
    ) -> None:
        super().__init__()
        c1, c2, c3 = width
        self.out_channels = c3
        self.gate_type = gate_type
        if gate_type == "legacy_sigmoid":
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
        elif gate_type == "hard_concrete":
            if output_gate is None:
                raise ValueError("hard_concrete encoders require a shared output_gate")
            self.net = nn.ModuleList(
                [
                    HardConcreteConvBlock(
                        in_channels, c1, kernel_size=3, padding=1, initial_retention=initial_retention
                    ),
                    HardConcreteConvBlock(
                        c1, c2, kernel_size=3, padding=1, initial_retention=initial_retention
                    ),
                    HardConcreteConvBlock(
                        c2,
                        c3,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        initial_retention=initial_retention,
                        gate=output_gate,
                    ),
                ]
            )
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

    def forward(self, x):
        if self.gate_type == "legacy_sigmoid":
            return self.net(x)
        for block in self.net:
            x = block(x)
        return x
