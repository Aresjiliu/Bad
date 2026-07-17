import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_validation_derived_routing import (
    build_validation_samples,
    evaluate_validation_derived_router,
    metrics_path_for_seed_budget,
    mode_target_stability_rows,
)


def _write_profile(root: Path, seed: int, budget: float, full_oa: float, degraded_oa: float) -> None:
    run_dir = root / f"houston2013_hsi-lidar_official_splitseed42_trainseed{seed}_gatehard_concrete_budget{int(budget * 100)}_metricmacs"
    run_dir.mkdir(parents=True)
    metrics = {
        "full": {
            "oa": full_oa,
            "pre_q_main": 0.95,
            "pre_q_aux": 0.94,
            "pre_u_main": 0.05,
            "pre_u_aux": 0.06,
            "expected_macs_ratio": budget,
        },
        "aux_noise_high": {
            "oa": degraded_oa,
            "pre_q_main": 0.92,
            "pre_q_aux": 0.25,
            "pre_u_main": 0.08,
            "pre_u_aux": 0.75,
            "expected_macs_ratio": budget,
        },
    }
    (run_dir / "compact_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "resource_stats.json").write_text(
        json.dumps({"compact": {"params_ratio": budget, "macs_ratio": budget}}),
        encoding="utf-8",
    )


class EvaluateValidationDerivedRoutingTest(unittest.TestCase):
    def test_finds_metrics_path_for_seed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_profile(root, seed=1, budget=0.8, full_oa=0.89, degraded_oa=0.7)

            path = metrics_path_for_seed_budget(root, seed=1, budget=0.8)

        self.assertEqual(path.name, "compact_metrics.json")

    def test_builds_validation_samples_with_pareto_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1):
                _write_profile(root, seed, 0.65, full_oa=0.80, degraded_oa=0.70)
                _write_profile(root, seed, 0.8, full_oa=0.895, degraded_oa=0.72)
                _write_profile(root, seed, 1.0, full_oa=0.900, degraded_oa=0.90)

            quality, targets, _ = build_validation_samples(root, (0, 1), (0.65, 0.8, 1.0), pareto_tolerance=0.01)

        self.assertEqual(len(quality), 4)
        self.assertEqual(targets["seed0::full"], 0.8)
        self.assertEqual(targets["seed1::aux_noise_high"], 1.0)

    def test_evaluates_heldout_seed_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (0, 1, 2):
                _write_profile(root, seed, 0.65, full_oa=0.80, degraded_oa=0.70)
                _write_profile(root, seed, 0.8, full_oa=0.895, degraded_oa=0.72)
                _write_profile(root, seed, 1.0, full_oa=0.900, degraded_oa=0.90)

            report = evaluate_validation_derived_router(
                root,
                seeds=(0, 1, 2),
                budgets=(0.65, 0.8, 1.0),
                pareto_tolerance=0.01,
                router_epochs=100,
                router_seed=3,
            )

        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(len(report["summary_rows"]), 4)
        self.assertIn("routing_accuracy_vs_pareto", report["summary_rows"][0])
        self.assertIn("0.8", report["target_distribution"])

    def test_summarizes_mode_target_stability(self):
        rows = mode_target_stability_rows(
            {
                "seed0::full": 1.0,
                "seed1::full": 0.8,
                "seed2::full": 1.0,
                "seed0::aux_only": 0.65,
                "seed1::aux_only": 0.65,
            }
        )

        by_mode = {row["mode"]: row for row in rows}
        self.assertEqual(by_mode["full"]["unique_targets"], 2)
        self.assertEqual(by_mode["full"]["majority_budget"], 1.0)
        self.assertAlmostEqual(by_mode["full"]["majority_fraction"], 2 / 3)
        self.assertEqual(by_mode["aux_only"]["unique_targets"], 1)


if __name__ == "__main__":
    unittest.main()
