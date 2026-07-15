import unittest

import torch

from brmnet_core.profile_router import (
    QualityBudgetRouter,
    budget_profile_routing_loss,
    evaluate_budget_profile_routing,
    oracle_budget_profile_targets,
)


class QualityBudgetRouterTest(unittest.TestCase):
    def test_router_maps_quality_features_to_budget_distribution(self):
        router = QualityBudgetRouter(profile_budgets=(0.65, 0.8, 1.0), hidden_channels=8)
        quality_features = torch.tensor(
            [
                [0.95, 0.90, 0.05, 0.10],
                [0.40, 0.35, 0.60, 0.65],
            ]
        )

        outputs = router(quality_features)

        self.assertEqual(outputs["profile_logits"].shape, (2, 3))
        self.assertEqual(outputs["profile_probs"].shape, (2, 3))
        self.assertEqual(outputs["expected_budget"].shape, (2, 1))
        self.assertEqual(outputs["selected_profile"].shape, (2,))
        torch.testing.assert_close(outputs["profile_probs"].sum(dim=1), torch.ones(2))
        self.assertTrue(bool(torch.all((0.65 <= outputs["expected_budget"]) & (outputs["expected_budget"] <= 1.0))))

    def test_oracle_targets_prefer_full_budget_for_clean_modalities(self):
        quality_features = torch.tensor(
            [
                [0.95, 0.90, 0.05, 0.10],
                [0.70, 0.45, 0.25, 0.55],
                [0.10, 0.20, 0.90, 0.80],
            ]
        )

        targets = oracle_budget_profile_targets(quality_features, profile_budgets=(0.65, 0.8, 1.0))

        torch.testing.assert_close(targets["target_indices"], torch.tensor([2, 1, 0]))
        torch.testing.assert_close(targets["target_budgets"], torch.tensor([[1.0], [0.8], [0.65]]))

    def test_routing_loss_supervises_profile_logits_and_budget_error(self):
        logits = torch.tensor([[0.0, 0.0, 3.0], [0.0, 3.0, 0.0], [3.0, 0.0, 0.0]])
        expected_budget = torch.tensor([[0.95], [0.82], [0.70]])
        targets = {
            "target_indices": torch.tensor([2, 1, 0]),
            "target_budgets": torch.tensor([[1.0], [0.8], [0.65]]),
        }

        losses = budget_profile_routing_loss(logits, expected_budget, targets)

        self.assertIn("routing_cls", losses)
        self.assertIn("routing_budget", losses)
        self.assertIn("routing_total", losses)
        self.assertGreater(float(losses["routing_total"]), 0.0)

    def test_evaluates_oracle_profile_routing_from_budget_metrics(self):
        profile_metrics = {
            0.65: {
                "full": {"oa": 0.80, "expected_macs_ratio": 0.65},
                "aux_noise": {"oa": 0.70, "expected_macs_ratio": 0.65},
                "main_only": {"oa": 0.60, "expected_macs_ratio": 0.50},
            },
            0.8: {
                "full": {"oa": 0.85, "expected_macs_ratio": 0.8},
                "aux_noise": {"oa": 0.78, "expected_macs_ratio": 0.8},
                "main_only": {"oa": 0.62, "expected_macs_ratio": 0.60},
            },
            1.0: {
                "full": {"oa": 0.90, "expected_macs_ratio": 1.0},
                "aux_noise": {"oa": 0.76, "expected_macs_ratio": 1.0},
                "main_only": {"oa": 0.64, "expected_macs_ratio": 0.78},
            },
        }
        quality_by_mode = {
            "full": [0.95, 0.90, 0.05, 0.10],
            "aux_noise": [0.70, 0.45, 0.25, 0.55],
            "main_only": [0.95, 0.00, 0.05, 1.00],
        }

        report = evaluate_budget_profile_routing(profile_metrics, quality_by_mode)

        self.assertEqual(report["per_mode"]["full"]["selected_budget"], 1.0)
        self.assertEqual(report["per_mode"]["aux_noise"]["selected_budget"], 0.8)
        self.assertEqual(report["per_mode"]["main_only"]["selected_budget"], 0.65)
        self.assertAlmostEqual(report["summary"]["mean_oa"], (0.90 + 0.78 + 0.60) / 3)
        self.assertAlmostEqual(report["summary"]["mean_selected_budget"], (1.0 + 0.8 + 0.65) / 3)


if __name__ == "__main__":
    unittest.main()
