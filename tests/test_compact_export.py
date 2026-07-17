import unittest

import torch
from torch import nn

from brmnet_core.budget_gates import BudgetGatedConv2d
from brmnet_core.compact import CompactBRMNet
from brmnet_core.export import export_compact_brmnet
from brmnet_core.hard_concrete import HardConcreteGate
from brmnet_core.model import BRMNet


def _set_kept_channels(gate: HardConcreteGate, count: int) -> None:
    with torch.no_grad():
        gate.log_alpha.fill_(-20.0)
        gate.log_alpha[:count].fill_(20.0)


class CountingCompactEncoder(nn.Module):
    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.ones(
            inputs.shape[0],
            self.out_channels,
            4,
            4,
            device=inputs.device,
            dtype=inputs.dtype,
        )


class CompactBRMNetExportTest(unittest.TestCase):
    def test_compact_model_supports_asymmetric_branch_widths(self):
        model = CompactBRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            main_width=(4, 8, 16),
            aux_width=(3, 7, 16),
            head_width=(12, 6),
        )

        outputs = model(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))

        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["main_feature"].shape, (2, 16, 4, 4))
        self.assertEqual(outputs["aux_feature"].shape, (2, 16, 4, 4))
        self.assertFalse(any(isinstance(module, HardConcreteGate) for module in model.modules()))
        self.assertFalse(any(isinstance(module, BudgetGatedConv2d) for module in model.modules()))

    def test_compact_model_skips_fully_unavailable_auxiliary_encoder(self):
        model = CompactBRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            main_width=(4, 8, 16),
            aux_width=(3, 7, 16),
            head_width=(12, 6),
        )
        model.eval()
        model.main_encoder = CountingCompactEncoder()
        model.aux_encoder = CountingCompactEncoder()
        availability_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        outputs = model(
            torch.randn(2, 4, 7, 7),
            torch.randn(2, 1, 7, 7),
            availability_mask=availability_mask,
        )

        self.assertEqual(model.main_encoder.calls, 1)
        self.assertEqual(model.aux_encoder.calls, 0)
        torch.testing.assert_close(outputs["fusion_weights"], availability_mask)
        torch.testing.assert_close(outputs["q_aux"], torch.zeros(2, 1))

    def test_exporter_copies_all_structural_dimensions(self):
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
        )
        counts = {
            "main_1": 4,
            "main_2": 8,
            "aux_1": 3,
            "aux_2": 7,
            "shared": 16,
            "head_1": 12,
            "head_2": 6,
        }
        _set_kept_channels(source.main_encoder.net[0].gate, counts["main_1"])
        _set_kept_channels(source.main_encoder.net[1].gate, counts["main_2"])
        _set_kept_channels(source.aux_encoder.net[0].gate, counts["aux_1"])
        _set_kept_channels(source.aux_encoder.net[1].gate, counts["aux_2"])
        _set_kept_channels(source.shared_fusion_gate, counts["shared"])
        _set_kept_channels(source.classifier.net[0].gate, counts["head_1"])
        _set_kept_channels(source.classifier.net[1].gate, counts["head_2"])

        compact, metadata = export_compact_brmnet(source)

        self.assertEqual(compact.main_encoder.net[0].conv.out_channels, counts["main_1"])
        self.assertEqual(compact.main_encoder.net[1].conv.in_channels, counts["main_1"])
        self.assertEqual(compact.main_encoder.net[2].conv.in_channels, counts["main_2"])
        self.assertEqual(compact.main_encoder.net[2].conv.out_channels, counts["shared"])
        self.assertEqual(compact.aux_encoder.net[0].conv.out_channels, counts["aux_1"])
        self.assertEqual(compact.aux_encoder.net[2].conv.in_channels, counts["aux_2"])
        self.assertEqual(compact.classifier.net[0].conv.in_channels, counts["shared"])
        self.assertEqual(compact.classifier.net[1].conv.in_channels, counts["head_1"])
        self.assertEqual(compact.classifier.linear.in_features, counts["head_2"])
        self.assertEqual(compact.main_quality.net[2].in_features, counts["shared"])
        self.assertEqual(metadata["widths"]["head"], [counts["head_1"], counts["head_2"]])

    def test_exporter_matches_hard_masked_model_logits(self):
        torch.manual_seed(3)
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
        )
        for index, gate in enumerate(
            module for module in source.modules() if isinstance(module, HardConcreteGate)
        ):
            _set_kept_channels(gate, max(1, gate.channels // (index + 2)))
            gate.set_inference_mode("hard")
        source.eval()
        compact, _metadata = export_compact_brmnet(source)
        compact.eval()
        main = torch.randn(3, 4, 7, 7)
        aux = torch.randn(3, 1, 7, 7)

        with torch.no_grad():
            source_logits = source(main, aux)["logits"]
            compact_logits = compact(main, aux)["logits"]

        torch.testing.assert_close(source_logits, compact_logits, atol=1e-5, rtol=1e-5)

    def test_exporter_preserves_uniform_fusion_mode(self):
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            fusion_mode="uniform",
        )

        compact, metadata = export_compact_brmnet(source)

        self.assertEqual(compact.reliability_fusion.mode, "uniform")
        self.assertEqual(metadata["fusion_mode"], "uniform")

    def test_exporter_preserves_fusion_availability_mask_flag(self):
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            fusion_use_availability_mask=False,
        )

        compact, metadata = export_compact_brmnet(source)

        self.assertFalse(compact.reliability_fusion.use_availability_mask)
        self.assertFalse(metadata["fusion_use_availability_mask"])

    def test_exporter_supports_uniform_width_strategy(self):
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
        )

        compact, metadata = export_compact_brmnet(
            source,
            selection_strategy="uniform_width",
            uniform_width_ratio=0.5,
        )

        self.assertEqual(compact.main_encoder.net[0].conv.out_channels, 16)
        self.assertEqual(compact.main_encoder.net[1].conv.out_channels, 32)
        self.assertEqual(compact.main_encoder.net[2].conv.out_channels, 64)
        self.assertEqual(compact.aux_encoder.net[0].conv.out_channels, 16)
        self.assertEqual(compact.aux_encoder.net[1].conv.out_channels, 32)
        self.assertEqual(compact.classifier.net[0].conv.out_channels, 64)
        self.assertEqual(compact.classifier.net[1].conv.out_channels, 32)
        self.assertEqual(metadata["selection_strategy"], "uniform_width")
        self.assertEqual(metadata["uniform_width_ratio"], 0.5)

    def test_exporter_keeps_highest_probability_channel_for_empty_mask(self):
        source = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
        )
        with torch.no_grad():
            source.main_encoder.net[0].gate.log_alpha.copy_(
                torch.linspace(-30.0, -20.0, source.main_encoder.net[0].gate.channels)
            )

        compact, metadata = export_compact_brmnet(source)

        self.assertEqual(compact.main_encoder.net[0].conv.out_channels, 1)
        self.assertEqual(
            metadata["indices"]["main_1"],
            [source.main_encoder.net[0].gate.channels - 1],
        )

    def test_exporter_rejects_legacy_model(self):
        source = BRMNet(main_channels=4, aux_channels=1, num_classes=3)

        with self.assertRaisesRegex(ValueError, "hard_concrete"):
            export_compact_brmnet(source)


if __name__ == "__main__":
    unittest.main()
