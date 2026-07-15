import unittest

import torch
from torch import nn

from brmnet_core.budget_gates import BudgetGatedConv2d
from brmnet_core.fusion import ReliabilityGatedFusion
from brmnet_core.gated_blocks import HardConcreteConvBlock
from brmnet_core.hard_concrete import HardConcreteGate
from brmnet_core.model import BRMNet


class CountingEncoder(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
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


class StructuredBRMNetTest(unittest.TestCase):
    def test_hard_concrete_block_orders_gate_after_batch_norm(self):
        block = HardConcreteConvBlock(3, 5, kernel_size=3, padding=1, initial_retention=0.8)
        observed = {}

        block.bn.register_forward_hook(
            lambda _module, _inputs, output: observed.setdefault("bn_output", output.detach().clone())
        )
        block.gate.register_forward_pre_hook(
            lambda _module, inputs: observed.setdefault("gate_input", inputs[0].detach().clone())
        )

        output = block(torch.randn(2, 3, 7, 7))

        self.assertEqual(list(dict(block.named_children())), ["conv", "bn", "gate", "activation"])
        self.assertEqual(output.shape, (2, 5, 7, 7))
        torch.testing.assert_close(observed["gate_input"], observed["bn_output"])

    def test_hard_concrete_model_uses_shared_fusion_gate(self):
        model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            initial_retention=0.8,
        )

        self.assertIsNot(model.main_encoder.net[0].gate, model.aux_encoder.net[0].gate)
        self.assertIs(model.main_encoder.net[2].gate, model.shared_fusion_gate)
        self.assertIs(model.aux_encoder.net[2].gate, model.shared_fusion_gate)
        self.assertIsNot(model.classifier.net[0].gate, model.classifier.net[1].gate)

        unique_gates = {id(module) for module in model.modules() if isinstance(module, HardConcreteGate)}
        self.assertEqual(len(unique_gates), 7)

    def test_hard_concrete_model_preserves_forward_contract(self):
        model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            initial_retention=0.8,
        )
        model.eval()

        outputs = model(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))

        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["main_feature"].shape, (2, 128, 4, 4))
        self.assertEqual(outputs["aux_feature"].shape, (2, 128, 4, 4))
        self.assertEqual(outputs["fusion_weights"].shape, (2, 2))

    def test_reliability_fusion_masks_unavailable_modalities(self):
        fusion = ReliabilityGatedFusion()
        main = torch.ones(2, 3, 2, 2)
        aux = torch.full((2, 3, 2, 2), 2.0)
        q_main = torch.tensor([[0.1], [0.9]])
        q_aux = torch.tensor([[0.9], [0.1]])
        availability_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        fused, weights = fusion(main, aux, q_main, q_aux, availability_mask=availability_mask)

        torch.testing.assert_close(weights, availability_mask)
        torch.testing.assert_close(fused[0], main[0])
        torch.testing.assert_close(fused[1], aux[1])

    def test_uniform_fusion_ignores_quality_scores_but_respects_availability(self):
        fusion = ReliabilityGatedFusion(mode="uniform")
        main = torch.ones(2, 3, 2, 2)
        aux = torch.full((2, 3, 2, 2), 3.0)
        q_main = torch.tensor([[0.01], [0.99]])
        q_aux = torch.tensor([[0.99], [0.01]])
        availability_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

        fused, weights = fusion(main, aux, q_main, q_aux, availability_mask=availability_mask)

        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5], [1.0, 0.0]]))
        torch.testing.assert_close(fused[0], torch.full((3, 2, 2), 2.0))
        torch.testing.assert_close(fused[1], main[1])

    def test_rejects_unknown_fusion_mode(self):
        with self.assertRaisesRegex(ValueError, "fusion mode"):
            ReliabilityGatedFusion(mode="unknown")

    def test_model_forward_accepts_availability_mask(self):
        torch.manual_seed(0)
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)
        model.eval()
        availability_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        outputs = model(
            torch.randn(2, 4, 7, 7),
            torch.randn(2, 1, 7, 7),
            availability_mask=availability_mask,
        )

        torch.testing.assert_close(outputs["fusion_weights"], availability_mask)

    def test_model_optionally_reports_pre_encoder_quality(self):
        model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            use_pre_encoder_quality_probe=True,
            pre_encoder_quality_hidden=4,
        )

        outputs = model(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))

        self.assertEqual(outputs["pre_q_main"].shape, (2, 1))
        self.assertEqual(outputs["pre_q_aux"].shape, (2, 1))
        self.assertEqual(outputs["pre_u_main"].shape, (2, 1))
        self.assertEqual(outputs["pre_u_aux"].shape, (2, 1))

    def test_model_skips_fully_unavailable_auxiliary_encoder(self):
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)
        model.eval()
        model.main_encoder = CountingEncoder()
        model.aux_encoder = CountingEncoder()
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

    def test_model_skips_fully_unavailable_main_encoder(self):
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)
        model.eval()
        model.main_encoder = CountingEncoder()
        model.aux_encoder = CountingEncoder()
        availability_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        outputs = model(
            torch.randn(2, 4, 7, 7),
            torch.randn(2, 1, 7, 7),
            availability_mask=availability_mask,
        )

        self.assertEqual(model.main_encoder.calls, 0)
        self.assertEqual(model.aux_encoder.calls, 1)
        torch.testing.assert_close(outputs["fusion_weights"], availability_mask)
        torch.testing.assert_close(outputs["q_main"], torch.zeros(2, 1))

    def test_legacy_model_remains_available(self):
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)

        self.assertTrue(any(isinstance(module, BudgetGatedConv2d) for module in model.modules()))
        self.assertFalse(any(isinstance(module, HardConcreteGate) for module in model.modules()))

    def test_rejects_unknown_gate_type(self):
        with self.assertRaisesRegex(ValueError, "gate_type"):
            BRMNet(main_channels=4, aux_channels=1, num_classes=3, gate_type="unknown")

    def test_rejects_unknown_model_fusion_mode(self):
        with self.assertRaisesRegex(ValueError, "fusion mode"):
            BRMNet(main_channels=4, aux_channels=1, num_classes=3, fusion_mode="unknown")


if __name__ == "__main__":
    unittest.main()
