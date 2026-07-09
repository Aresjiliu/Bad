from __future__ import annotations

import torch
from torch import nn

from .availability import encode_available_modalities
from .fusion import ModalityQualityEstimator, ReliabilityGatedFusion


class CompactConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(inputs)))


class CompactEncoder(nn.Module):
    def __init__(self, in_channels: int, width: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = width
        self.out_channels = c3
        self.net = nn.ModuleList(
            [
                CompactConvBlock(in_channels, c1),
                CompactConvBlock(c1, c2),
                CompactConvBlock(c2, c3, stride=2),
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.net:
            inputs = block(inputs)
        return inputs


class CompactFusionHead(nn.Module):
    def __init__(self, channels: int, width: tuple[int, int], num_classes: int) -> None:
        super().__init__()
        h1, h2 = width
        self.net = nn.ModuleList(
            [
                CompactConvBlock(channels, h1),
                CompactConvBlock(h1, h2),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(h2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.net:
            inputs = block(inputs)
        return self.linear(self.flatten(self.pool(inputs)))


class CompactBRMNet(nn.Module):
    """Gate-free BRM-Net materialized from a structured hard mask."""

    def __init__(
        self,
        main_channels: int,
        aux_channels: int,
        num_classes: int,
        main_width: tuple[int, int, int],
        aux_width: tuple[int, int, int],
        head_width: tuple[int, int],
        reliability_temperature: float = 1.0,
        fusion_mode: str = "reliability",
    ) -> None:
        super().__init__()
        if main_width[2] != aux_width[2]:
            raise ValueError("Compact encoder output widths must match for additive fusion.")
        self.main_encoder = CompactEncoder(main_channels, main_width)
        self.aux_encoder = CompactEncoder(aux_channels, aux_width)
        shared_channels = main_width[2]
        self.main_quality = ModalityQualityEstimator(shared_channels)
        self.aux_quality = ModalityQualityEstimator(shared_channels)
        self.reliability_fusion = ReliabilityGatedFusion(
            temperature=reliability_temperature,
            mode=fusion_mode,
        )
        self.classifier = CompactFusionHead(shared_channels, head_width, num_classes)

    def forward(
        self,
        main_input: torch.Tensor,
        aux_input: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        main_feature, aux_feature, q_main, q_aux = encode_available_modalities(
            self.main_encoder,
            self.aux_encoder,
            self.main_quality,
            self.aux_quality,
            main_input,
            aux_input,
            availability_mask,
        )
        fused, weights = self.reliability_fusion(
            main_feature,
            aux_feature,
            q_main,
            q_aux,
            availability_mask=availability_mask,
        )
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "q_main": q_main,
            "q_aux": q_aux,
            "fusion_weights": weights,
            "main_feature": main_feature,
            "aux_feature": aux_feature,
        }
