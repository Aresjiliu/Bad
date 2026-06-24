import unittest

import torch

from brmnet_core.model import BRMNet
from brmnet_core.resources import (
    estimate_brmnet_resources,
    estimate_compact_resources,
    find_resource_budget_threshold,
    initialize_uniform_resource_budget,
    resource_budget_loss,
)


def _manual_baseline_resources() -> tuple[int, int]:
    conv_params = (
        4 * 32 * 9
        + 32 * 64 * 9
        + 64 * 128 * 9
        + 1 * 32 * 9
        + 32 * 64 * 9
        + 64 * 128 * 9
        + 128 * 128 * 9
        + 128 * 64 * 9
    )
    bn_params = 2 * (32 + 64 + 128) * 2 + 2 * (128 + 64)
    quality_params = 2 * (128 * 64 + 64 + 64 * 1 + 1)
    classifier_params = 64 * 3 + 3
    params = conv_params + bn_params + quality_params + classifier_params

    conv_macs = (
        4 * 32 * 9 * 49
        + 32 * 64 * 9 * 49
        + 64 * 128 * 9 * 16
        + 1 * 32 * 9 * 49
        + 32 * 64 * 9 * 49
        + 64 * 128 * 9 * 16
        + 128 * 128 * 9 * 16
        + 128 * 64 * 9 * 16
    )
    quality_macs = 2 * (128 * 64 + 64)
    classifier_macs = 64 * 3
    macs = conv_macs + quality_macs + classifier_macs
    return params, macs


class BRMNetResourceTest(unittest.TestCase):
    def setUp(self):
        self.model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            initial_retention=0.8,
        )

    def test_baseline_resources_match_manual_calculation(self):
        expected_params, expected_macs = _manual_baseline_resources()

        stats = estimate_brmnet_resources(self.model, patch_size=7, mode="baseline")

        self.assertEqual(stats.params, expected_params)
        self.assertEqual(stats.macs, expected_macs)
        self.assertEqual(stats.params_ratio, 1.0)
        self.assertEqual(stats.macs_ratio, 1.0)

    def test_expected_resources_are_differentiable(self):
        stats = estimate_brmnet_resources(self.model, patch_size=7, mode="expected")

        self.assertIsInstance(stats.params, torch.Tensor)
        self.assertIsInstance(stats.macs, torch.Tensor)
        stats.macs_ratio.backward()

        gradients = [
            parameter.grad
            for name, parameter in self.model.named_parameters()
            if name.endswith("log_alpha")
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(float(gradient.abs().sum()) > 0.0 for gradient in gradients))

    def test_hard_resources_decrease_when_channels_are_disabled(self):
        before = estimate_brmnet_resources(self.model, patch_size=7, mode="hard")
        with torch.no_grad():
            self.model.main_encoder.net[0].gate.log_alpha.fill_(-20.0)
        after = estimate_brmnet_resources(self.model, patch_size=7, mode="hard")

        self.assertIsInstance(after.params, int)
        self.assertIsInstance(after.macs, int)
        self.assertLess(after.params, before.params)
        self.assertLess(after.macs, before.macs)

    def test_shared_gate_changes_both_encoders_and_classifier_input(self):
        baseline = estimate_brmnet_resources(self.model, patch_size=7, mode="hard")
        with torch.no_grad():
            self.model.shared_fusion_gate.log_alpha[:64].fill_(-20.0)
        pruned = estimate_brmnet_resources(self.model, patch_size=7, mode="hard")

        expected_param_reduction = (
            2 * 64 * 64 * 9
            + 4 * 64
            + 64 * 128 * 9
            + 2 * 64 * 64
        )
        self.assertEqual(baseline.params - pruned.params, expected_param_reduction)

    def test_resource_budget_loss_uses_requested_metric(self):
        loss, stats = resource_budget_loss(
            self.model,
            target_budget=0.65,
            patch_size=7,
            metric="macs",
        )

        torch.testing.assert_close(loss, torch.square(stats.macs_ratio - 0.65))
        with self.assertRaisesRegex(ValueError, "metric"):
            resource_budget_loss(self.model, 0.65, patch_size=7, metric="latency")

    def test_uniform_initialization_matches_resource_budget(self):
        retention = initialize_uniform_resource_budget(
            self.model,
            target_budget=0.65,
            patch_size=7,
            metric="macs",
        )
        stats = estimate_brmnet_resources(self.model, patch_size=7, mode="expected")

        self.assertGreater(retention, 0.65)
        self.assertAlmostEqual(float(stats.macs_ratio), 0.65, places=4)

    def test_uniform_initialization_accepts_full_budget_endpoint(self):
        retention = initialize_uniform_resource_budget(
            self.model,
            target_budget=1.0,
            patch_size=7,
            metric="macs",
        )
        stats = estimate_brmnet_resources(self.model, patch_size=7, mode="expected")

        self.assertGreater(retention, 0.999)
        self.assertAlmostEqual(float(stats.macs_ratio), 1.0, places=5)

    def test_compact_resource_count_matches_full_baseline(self):
        from brmnet_core.compact import CompactBRMNet

        compact = CompactBRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            main_width=(32, 64, 128),
            aux_width=(32, 64, 128),
            head_width=(128, 64),
        )
        compact_stats = estimate_compact_resources(compact, patch_size=7)
        baseline = estimate_brmnet_resources(self.model, patch_size=7, mode="baseline")

        self.assertEqual(compact_stats.params, baseline.params)
        self.assertEqual(compact_stats.macs, baseline.macs)

    def test_global_threshold_projects_hard_resources_toward_budget(self):
        with torch.no_grad():
            for index, gate in enumerate(
                module
                for module in self.model.modules()
                if hasattr(module, "log_alpha") and hasattr(module, "hard_mask")
            ):
                gate.log_alpha.copy_(
                    torch.linspace(-3.0 + index * 0.1, 3.0 + index * 0.1, gate.channels)
                )

        threshold, stats = find_resource_budget_threshold(
            self.model,
            target_budget=0.65,
            patch_size=7,
            metric="macs",
        )

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertLess(abs(float(stats.macs_ratio) - 0.65), 0.03)

    def test_rejects_legacy_models_and_invalid_modes(self):
        legacy = BRMNet(main_channels=4, aux_channels=1, num_classes=3)

        with self.assertRaisesRegex(ValueError, "hard_concrete"):
            estimate_brmnet_resources(legacy, patch_size=7, mode="baseline")
        with self.assertRaisesRegex(ValueError, "mode"):
            estimate_brmnet_resources(self.model, patch_size=7, mode="unknown")
        with self.assertRaisesRegex(ValueError, "patch_size"):
            estimate_brmnet_resources(self.model, patch_size=0, mode="baseline")


if __name__ == "__main__":
    unittest.main()
