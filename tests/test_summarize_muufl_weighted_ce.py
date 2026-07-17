import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_muufl_weighted_ce import (
    load_records,
    seed_from_run_dir,
    summarize_class_delta,
    summarize_records,
)


def _metrics(full_oa: float, main_oa: float, aux_oa: float, class_accuracy: list[float]) -> dict:
    modes = {
        "full": full_oa,
        "main_only": main_oa,
        "aux_only": aux_oa,
        "aux_noise_high": full_oa + 0.01,
        "aux_downsample_4": full_oa - 0.01,
        "aux_occlusion_50": full_oa + 0.02,
    }
    payload = {
        mode: {
            "oa": oa,
            "aa": oa - 0.02,
            "kappa": oa - 0.05,
            "class_accuracy": class_accuracy,
        }
        for mode, oa in modes.items()
    }
    payload["resource_stats"] = {
        "compact": {
            "full": {
                "global_params_ratio": 0.82,
                "global_macs_ratio": 0.80,
            }
        },
        "latency_ms": {
            "compact": {
                "mean": 2.5,
            }
        },
    }
    return payload


def _write_run(root: Path, seed: int, metrics: dict) -> None:
    run_dir = root / f"muufl_hsi-lidar_random_splitseed42_trainseed{seed}_gatehard_concrete_budget80_metricmacs"
    run_dir.mkdir(parents=True)
    (run_dir / "compact_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class SummarizeMUUFLWeightedCETest(unittest.TestCase):
    def test_seed_from_run_dir(self):
        self.assertEqual(seed_from_run_dir("muufl_trainseed12_budget80"), 12)

    def test_summary_and_class_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            weighted_root = root / "weighted"
            for seed in range(3):
                _write_run(baseline_root, seed, _metrics(0.80 + seed * 0.01, 0.70, 0.50, [0.9, 0.6]))
                _write_run(weighted_root, seed, _metrics(0.82 + seed * 0.01, 0.72, 0.48, [0.8, 0.8]))

            records = load_records(baseline_root, weighted_root)
            summary = summarize_records(records)
            class_delta = summarize_class_delta(records)

        baseline = next(row for row in summary if row["variant"] == "80% multi-deg.")
        weighted = next(row for row in summary if row["variant"] == "80% multi-deg. + weighted CE")
        self.assertAlmostEqual(baseline["full_oa_mean"], 81.0)
        self.assertAlmostEqual(weighted["full_oa_mean"], 83.0)
        self.assertAlmostEqual(class_delta[0]["delta_weighted_ce_minus_baseline"], -10.0)
        self.assertAlmostEqual(class_delta[1]["delta_weighted_ce_minus_baseline"], 20.0)


if __name__ == "__main__":
    unittest.main()
