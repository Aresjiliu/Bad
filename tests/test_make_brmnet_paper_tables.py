import csv
import tempfile
import unittest
from pathlib import Path

from scripts.make_brmnet_paper_tables import main


class BRMNetPaperTablesTest(unittest.TestCase):
    def test_builds_robustness_table_outputs(self):
        variants = [
            "full",
            "without_reliability_uniform_fusion",
            "quality_degradation_supervised",
            "quality_multi_degradation_supervised",
            "quality_multi_degradation_p025",
        ]
        modes = ["full", "aux_noise_high", "aux_downsample_4", "aux_occlusion_50"]
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.csv"
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "variant",
                        "mode",
                        "runs",
                        "oa_mean",
                        "compact_macs_ratio_mean",
                    ],
                )
                writer.writeheader()
                for variant_index, variant in enumerate(variants):
                    for mode_index, mode in enumerate(modes):
                        writer.writerow(
                            {
                                "variant": variant,
                                "mode": mode,
                                "runs": "3",
                                "oa_mean": str(0.8 + variant_index * 0.01 + mode_index * 0.001),
                                "compact_macs_ratio_mean": "0.8",
                            }
                        )

            output_prefix = Path(tmp) / "robustness"
            latex_output = Path(tmp) / "robustness.tex"
            rows = main(
                [
                    "--summary-csv",
                    str(summary_path),
                    "--output-prefix",
                    str(output_prefix),
                    "--latex-output",
                    str(latex_output),
                ]
            )

            self.assertEqual(len(rows), 5)
            self.assertTrue(output_prefix.with_suffix(".csv").is_file())
            self.assertTrue(output_prefix.with_suffix(".md").is_file())
            self.assertTrue(latex_output.is_file())
            self.assertIn("Adverse Avg.", latex_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
