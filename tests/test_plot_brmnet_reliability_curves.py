from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.plot_brmnet_reliability_curves import build_curve_records, main


class PlotBRMNetReliabilityCurvesTest(unittest.TestCase):
    def _write_summary(self, path: Path) -> None:
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
        ]
        variants = [
            "full",
            "quality_degradation_supervised",
            "quality_multi_degradation_p025",
        ]
        modes = [
            "aux_noise_low",
            "aux_noise_mid",
            "aux_noise_high",
            "aux_downsample_2",
            "aux_downsample_4",
            "aux_occlusion_25",
            "aux_occlusion_50",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for variant in variants:
                for index, mode in enumerate(modes):
                    writer.writerow(
                        {
                            "variant": variant,
                            "mode": mode,
                            "runs": "3",
                            "oa_mean": f"{0.86 - index * 0.01:.4f}",
                            "oa_std": "0.01",
                            "q_aux_mean": f"{0.90 - index * 0.05:.4f}",
                            "q_aux_std": "0.02",
                            "fusion_weight_aux_mean": f"{0.50 - index * 0.02:.4f}",
                            "fusion_weight_aux_std": "0.01",
                        }
                    )

    def test_build_curve_records_scales_oa_to_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.csv"
            self._write_summary(summary)

            records = build_curve_records(summary)

            self.assertEqual(len(records), 3 * 7 * 3)
            first_oa = next(row for row in records if row["metric"] == "oa")
            self.assertAlmostEqual(first_oa["mean"], 86.0)

    def test_main_writes_csv_note_and_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            output_prefix = root / "curves"
            paper_prefix = root / "paper_curves"
            self._write_summary(summary)

            exit_code = main(
                [
                    "--summary-csv",
                    str(summary),
                    "--output-prefix",
                    str(output_prefix),
                    "--paper-output-prefix",
                    str(paper_prefix),
                    "--formats",
                    "png",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_prefix.with_suffix(".csv").is_file())
            self.assertTrue(output_prefix.with_suffix(".md").is_file())
            self.assertTrue(output_prefix.with_suffix(".png").is_file())
            self.assertTrue(paper_prefix.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
