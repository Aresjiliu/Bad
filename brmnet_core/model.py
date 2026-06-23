from __future__ import annotations

import torch
from torch import nn

from .encoders import BudgetGatedEncoder
from .fusion import BudgetGatedFusionHead, ModalityQualityEstimator, ReliabilityGatedFusion


class BRMNet(nn.Module):
    """Minimal budget- and reliability-aware multimodal classifier."""

    def __init__(
        self,
        main_channels: int,
        aux_channels: int,
        num_classes: int,
        init_score: float = 0.85,
        reliability_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.main_encoder = BudgetGatedEncoder(main_channels, init_score=init_score)
        self.aux_encoder = BudgetGatedEncoder(aux_channels, init_score=init_score)
        self.main_quality = ModalityQualityEstimator(self.main_encoder.out_channels)
        self.aux_quality = ModalityQualityEstimator(self.aux_encoder.out_channels)
        self.reliability_fusion = ReliabilityGatedFusion(temperature=reliability_temperature)
        self.classifier = BudgetGatedFusionHead(self.main_encoder.out_channels, num_classes, init_score=init_score)

    def forward(self, main_input: torch.Tensor, aux_input: torch.Tensor) -> dict[str, torch.Tensor]:
        main_feature = self.main_encoder(main_input)
        aux_feature = self.aux_encoder(aux_input)
        q_main = self.main_quality(main_feature)
        q_aux = self.aux_quality(aux_feature)
        fused, weights = self.reliability_fusion(main_feature, aux_feature, q_main, q_aux)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "q_main": q_main,
            "q_aux": q_aux,
            "fusion_weights": weights,
            "main_feature": main_feature,
            "aux_feature": aux_feature,
        }

