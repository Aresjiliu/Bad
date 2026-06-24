from __future__ import annotations

import math

import torch
from torch import nn


class HardConcreteGate(nn.Module):
    """Learnable channel gate with differentiable L0-style activation probability."""

    def __init__(
        self,
        channels: int,
        initial_retention: float = 0.9,
        temperature: float = 2.0 / 3.0,
        lower: float = -0.1,
        upper: float = 1.1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if not 0.0 < initial_retention <= 1.0:
            raise ValueError(f"initial_retention must be in (0, 1], got {initial_retention}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if lower >= 0.0:
            raise ValueError(f"lower must be negative, got {lower}")
        if upper <= 1.0:
            raise ValueError(f"upper must be greater than 1, got {upper}")
        if not 0.0 < epsilon < 0.5:
            raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")

        self.channels = int(channels)
        self.temperature = float(temperature)
        self.lower = float(lower)
        self.upper = float(upper)
        self.epsilon = float(epsilon)
        self.inference_mode = "soft"
        self.hard_threshold = 0.5

        retention = min(float(initial_retention), 1.0 - self.epsilon)
        retention_logit = math.log(retention / (1.0 - retention))
        stretch_offset = self.temperature * math.log(-self.lower / self.upper)
        self.log_alpha = nn.Parameter(
            torch.full((self.channels,), retention_logit + stretch_offset)
        )

    def expected_active_probability(self) -> torch.Tensor:
        stretch_offset = self.temperature * math.log(-self.lower / self.upper)
        return torch.sigmoid(self.log_alpha - stretch_offset)

    def soft_gate(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.log_alpha) * (self.upper - self.lower) + self.lower
        return stretched.clamp(0.0, 1.0)

    def hard_mask(self, threshold: float | None = None) -> torch.Tensor:
        threshold = self.hard_threshold if threshold is None else float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        return self.expected_active_probability() >= threshold

    def set_inference_mode(self, mode: str) -> None:
        if mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported inference mode: {mode}")
        self.inference_mode = mode

    def _sample_gate(self) -> torch.Tensor:
        uniform = torch.rand_like(self.log_alpha).clamp(self.epsilon, 1.0 - self.epsilon)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        concrete = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
        stretched = concrete * (self.upper - self.lower) + self.lower
        return stretched.clamp(0.0, 1.0)

    def gate_values(self) -> torch.Tensor:
        if self.training:
            return self._sample_gate()
        if self.inference_mode == "hard":
            return self.hard_mask().to(dtype=self.log_alpha.dtype)
        return self.soft_gate()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"Expected channel dimension {self.channels}, got shape {tuple(inputs.shape)}"
            )
        view_shape = (1, self.channels) + (1,) * (inputs.ndim - 2)
        return inputs * self.gate_values().view(view_shape)
