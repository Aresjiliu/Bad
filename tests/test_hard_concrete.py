import unittest

import torch

from brmnet_core.hard_concrete import HardConcreteGate


class HardConcreteGateTest(unittest.TestCase):
    def test_initial_retention_matches_expected_active_probability(self):
        gate = HardConcreteGate(4, initial_retention=0.65)

        expected = gate.expected_active_probability()

        torch.testing.assert_close(expected, torch.full((4,), 0.65), atol=1e-5, rtol=0.0)

    def test_rejects_invalid_configuration(self):
        invalid_kwargs = (
            {"channels": 0},
            {"channels": 2, "initial_retention": 0.0},
            {"channels": 2, "initial_retention": 1.1},
            {"channels": 2, "temperature": 0.0},
            {"channels": 2, "lower": 0.0},
            {"channels": 2, "upper": 0.0},
            {"channels": 2, "lower": -0.1, "upper": -0.2},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HardConcreteGate(**kwargs)

    def test_training_sample_is_bounded_and_differentiable(self):
        torch.manual_seed(0)
        gate = HardConcreteGate(3, initial_retention=0.8)
        gate.train()

        output = gate(torch.ones(2, 3, 4, 4))

        self.assertEqual(output.shape, (2, 3, 4, 4))
        self.assertTrue(bool(torch.all(output >= 0.0)))
        self.assertTrue(bool(torch.all(output <= 1.0)))
        output.sum().backward()
        self.assertIsNotNone(gate.log_alpha.grad)
        self.assertGreater(float(gate.log_alpha.grad.abs().sum()), 0.0)

    def test_soft_evaluation_is_deterministic(self):
        gate = HardConcreteGate(3, initial_retention=0.8)
        gate.eval()
        gate.set_inference_mode("soft")
        inputs = torch.ones(2, 3, 4, 4)

        first = gate(inputs)
        second = gate(inputs)

        torch.testing.assert_close(first, second)
        self.assertEqual(gate.soft_gate().shape, (3,))

    def test_hard_evaluation_uses_binary_mask(self):
        gate = HardConcreteGate(3, initial_retention=0.8)
        gate.eval()
        with torch.no_grad():
            gate.log_alpha.copy_(torch.tensor([-10.0, 0.0, 10.0]))
        gate.set_inference_mode("hard")

        mask = gate.hard_mask()
        output = gate(torch.ones(1, 3, 2, 2))

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(set(mask.tolist()), {False, True})
        self.assertEqual(set(output.unique().tolist()), {0.0, 1.0})

    def test_rejects_unknown_inference_mode_and_threshold(self):
        gate = HardConcreteGate(2)

        with self.assertRaisesRegex(ValueError, "inference mode"):
            gate.set_inference_mode("unknown")
        for threshold in (-0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "threshold"):
                gate.hard_mask(threshold)


if __name__ == "__main__":
    unittest.main()
