from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def sample_bernoulli(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.clamp(probs, min=0.0, max=1.0)
    return (torch.rand_like(probs) < probs).to(dtype=probs.dtype)


class BudgetGatedConv2d(nn.Conv2d):
    """Conv2d with learnable output-channel gates.

    This is a cleaned, device-safe extraction of the old `Conv2d_Prune`.
    It keeps the useful idea, but removes hard-coded `.cuda()` calls and exposes
    deterministic gate values for analysis/export.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        init_score: float = 0.85,
        init_temperature: float = 1.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gate_score = nn.Parameter(torch.full((out_channels,), float(init_score)))
        self.register_buffer("temperature", torch.full((out_channels,), float(init_temperature)))
        self.stochastic = stochastic

    def gate_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.temperature * self.gate_score)

    def hard_gate(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.gate_prob() >= threshold).to(dtype=self.weight.dtype)

    def set_temperature(self, value: torch.Tensor) -> None:
        self.temperature.copy_(value.to(device=self.temperature.device, dtype=self.temperature.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_prob()
        if self.training and self.stochastic:
            gate = gate * sample_bernoulli(gate)
        weight = self.weight * gate.view(-1, 1, 1, 1)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def iter_budget_gates(module: nn.Module) -> Iterable[BudgetGatedConv2d]:
    for child in module.modules():
        if isinstance(child, BudgetGatedConv2d):
            yield child


def collect_budget_loss(module: nn.Module, reduction: str = "mean") -> torch.Tensor:
    gates = [gate.gate_prob().sum() for gate in iter_budget_gates(module)]
    if not gates:
        return torch.tensor(0.0)
    value = torch.stack(gates)
    if reduction == "sum":
        return value.sum()
    if reduction == "mean":
        return value.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


@torch.no_grad()
def update_gate_temperature(module: nn.Module, gamma: float = 1.01, t_max: float = 2.0) -> None:
    for gate in iter_budget_gates(module):
        probs = gate.gate_prob()
        sampled = sample_bernoulli(probs)
        new_temperature = gate.temperature * torch.pow(torch.as_tensor(gamma, device=probs.device), sampled)
        gate.set_temperature(torch.clamp(new_temperature, max=t_max))

