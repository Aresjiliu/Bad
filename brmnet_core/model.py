from __future__ import annotations

import torch
from torch import nn

from .availability import encode_available_modalities
from .encoders import BudgetGatedEncoder
from .fusion import BudgetGatedFusionHead, ModalityQualityEstimator, ReliabilityGatedFusion
from .hard_concrete import HardConcreteGate
from .quality_probe import PreEncoderQualityProbe


class BRMNet(nn.Module):
    """Minimal budget- and reliability-aware multimodal classifier."""

    def __init__(
        self,
        main_channels: int,
        aux_channels: int,
        num_classes: int,
        init_score: float = 0.85,
        reliability_temperature: float = 1.0,
        gate_type: str = "legacy_sigmoid",
        initial_retention: float = 0.9,
        fusion_mode: str = "reliability",
        fusion_use_availability_mask: bool = True,
        use_pre_encoder_quality_probe: bool = False,
        pre_encoder_quality_hidden: int = 16,
    ) -> None:
        super().__init__()
        if gate_type not in {"legacy_sigmoid", "hard_concrete"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        self.gate_type = gate_type
        self.pre_encoder_quality_probe = (
            PreEncoderQualityProbe(main_channels, aux_channels, hidden_channels=pre_encoder_quality_hidden)
            if use_pre_encoder_quality_probe
            else None
        )
        self.shared_fusion_gate = (
            HardConcreteGate(128, initial_retention=initial_retention)
            if gate_type == "hard_concrete"
            else None
        )
        self.main_encoder = BudgetGatedEncoder(
            main_channels,
            init_score=init_score,
            gate_type=gate_type,
            initial_retention=initial_retention,
            output_gate=self.shared_fusion_gate,
        )
        self.aux_encoder = BudgetGatedEncoder(
            aux_channels,
            init_score=init_score,
            gate_type=gate_type,
            initial_retention=initial_retention,
            output_gate=self.shared_fusion_gate,
        )
        self.main_quality = ModalityQualityEstimator(self.main_encoder.out_channels)
        self.aux_quality = ModalityQualityEstimator(self.aux_encoder.out_channels)
        self.reliability_fusion = ReliabilityGatedFusion(
            temperature=reliability_temperature,
            mode=fusion_mode,
            use_availability_mask=fusion_use_availability_mask,
        )
        self.classifier = BudgetGatedFusionHead(
            self.main_encoder.out_channels,
            num_classes,
            init_score=init_score,
            gate_type=gate_type,
            initial_retention=initial_retention,
        )

    def forward(
        self,
        main_input: torch.Tensor,
        aux_input: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pre_encoder_quality = (
            self.pre_encoder_quality_probe(main_input, aux_input, availability_mask=availability_mask)
            if self.pre_encoder_quality_probe is not None
            else None
        )
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
        outputs = {
            "logits": logits,
            "q_main": q_main,
            "q_aux": q_aux,
            "fusion_weights": weights,
            "main_feature": main_feature,
            "aux_feature": aux_feature,
        }
        if pre_encoder_quality is not None:
            outputs.update(pre_encoder_quality)
        return outputs
