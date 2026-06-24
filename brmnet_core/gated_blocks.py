from __future__ import annotations

from torch import nn

from .hard_concrete import HardConcreteGate


class HardConcreteConvBlock(nn.Module):
    """Conv-BN-Gate-ReLU block for structured output-channel learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = False,
        initial_retention: float = 0.9,
        gate: HardConcreteGate | None = None,
    ) -> None:
        super().__init__()
        if gate is not None and gate.channels != out_channels:
            raise ValueError(
                f"Shared gate channels {gate.channels} do not match block output {out_channels}."
            )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.gate = gate or HardConcreteGate(out_channels, initial_retention=initial_retention)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.gate(self.bn(self.conv(inputs))))
