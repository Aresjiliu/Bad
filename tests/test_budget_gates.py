import math
import unittest

import torch
from torch import nn

from brmnet_core.budget_gates import (
    BudgetGatedConv2d,
    collect_budget_stats,
    target_budget_loss,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class BudgetGateStatisticsTest(unittest.TestCase):
    def test_global_retention_is_weighted_by_output_channels(self):
        model = nn.Sequential(
            BudgetGatedConv2d(1, 2, 1, init_score=_logit(0.25), stochastic=False),
            BudgetGatedConv2d(2, 6, 1, init_score=_logit(0.75), stochastic=False),
        )

        stats = collect_budget_stats(model)

        self.assertAlmostEqual(float(stats.soft_retention), 0.625)
        self.assertEqual(stats.active_channels, 6)
        self.assertEqual(stats.total_channels, 8)
        self.assertAlmostEqual(stats.hard_retention, 0.75)
        self.assertEqual([layer.name for layer in stats.layers], ["0", "1"])
        self.assertAlmostEqual(stats.layers[0].soft_retention, 0.25)
        self.assertAlmostEqual(stats.layers[1].soft_retention, 0.75, places=6)

    def test_target_budget_loss_is_symmetric_squared_error(self):
        model = nn.Sequential(
            BudgetGatedConv2d(1, 4, 1, init_score=_logit(0.75), stochastic=False),
        )

        below = target_budget_loss(model, 0.65)
        above = target_budget_loss(model, 0.85)

        self.assertTrue(torch.isclose(below, torch.tensor(0.01), atol=1e-6))
        self.assertTrue(torch.isclose(above, torch.tensor(0.01), atol=1e-6))
        below.backward()
        self.assertIsNotNone(model[0].gate_score.grad)

    def test_target_budget_validation_and_missing_gates(self):
        model = nn.Conv2d(1, 2, 1)

        with self.assertRaisesRegex(ValueError, "no BudgetGatedConv2d"):
            target_budget_loss(model, 0.8)
        for target in (0.0, -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "target_budget"):
                target_budget_loss(nn.Sequential(BudgetGatedConv2d(1, 2, 1)), target)


if __name__ == "__main__":
    unittest.main()
