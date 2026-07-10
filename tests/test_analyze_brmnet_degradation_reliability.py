from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_brmnet_degradation_reliability import (
    compute_correlations,
    load_degradation_rows,
)


class AnalyzeBRMNetDegradationReliabilityTest(unittest.TestCase):
    def test_loads_relevant_modes_and_computes_full_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            fieldnames = [
                "variant",
                "mode",
                "runs",
                "oa_mean",
                "oa_std",
                "q_aux_mean",
                "q_aux_std",
                "fusion_weight_aux_mean",
                "fusion_weight_aux_std",
                "compact_macs_ratio_mean",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "variant": "quality_multi_degradation_p025",
                        "mode": "full",
                        "runs": 3,
                        "oa_mean": 0.90,
                        "oa_std": 0.01,
                        "q_aux_mean": 0.80,
                        "q_aux_std": 0.01,
                        "fusion_weight_aux_mean": 0.50,
                        "fusion_weight_aux_std": 0.01,
                        "compact_macs_ratio_mean": 0.80,
                    }
                )
                writer.writerow(
                    {
                        "variant": "quality_multi_degradation_p025",
                        "mode": "aux_noise_high",
                        "runs": 3,
                        "oa_mean": 0.82,
                        "oa_std": 0.02,
                        "q_aux_mean": 0.60,
                        "q_aux_std": 0.01,
                        "fusion_weight_aux_mean": 0.35,
                        "fusion_weight_aux_std": 0.01,
                        "compact_macs_ratio_mean": 0.80,
                    }
                )
            rows = load_degradation_rows(path)
            high_noise = next(row for row in rows if row["mode"] == "aux_noise_high")
            self.assertAlmostEqual(high_noise["severity"], 0.50)
            self.assertAlmostEqual(high_noise["full_drop"], 0.08)

    def test_computes_noise_correlation(self):
        rows = [
            {
                "variant": "full",
                "method": "Full baseline",
                "family": "noise",
                "severity": 0.10,
                "oa": 0.90,
                "full_drop": 0.00,
                "q_aux": 0.80,
                "fusion_aux": 0.60,
            },
            {
                "variant": "full",
                "method": "Full baseline",
                "family": "noise",
                "severity": 0.50,
                "oa": 0.80,
                "full_drop": 0.10,
                "q_aux": 0.60,
                "fusion_aux": 0.40,
            },
        ]
        correlations = compute_correlations(rows)
        oa_corr = next(row for row in correlations if row["metric"] == "OA")
        self.assertLess(oa_corr["pearson_r"], 0)


if __name__ == "__main__":
    unittest.main()
