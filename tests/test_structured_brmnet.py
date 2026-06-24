import unittest

import torch

from brmnet_core.budget_gates import BudgetGatedConv2d
from brmnet_core.gated_blocks import HardConcreteConvBlock
from brmnet_core.hard_concrete import HardConcreteGate
from brmnet_core.model import BRMNet


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

    def test_legacy_model_remains_available(self):
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)

        self.assertTrue(any(isinstance(module, BudgetGatedConv2d) for module in model.modules()))
        self.assertFalse(any(isinstance(module, HardConcreteGate) for module in model.modules()))

    def test_rejects_unknown_gate_type(self):
        with self.assertRaisesRegex(ValueError, "gate_type"):
            BRMNet(main_channels=4, aux_channels=1, num_classes=3, gate_type="unknown")


if __name__ == "__main__":
    unittest.main()
