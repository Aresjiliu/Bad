import csv
import tempfile
import unittest
from pathlib import Path

from scripts.make_paper_submission_evidence import main


class PaperSubmissionEvidenceTest(unittest.TestCase):
    def test_builds_canonical_results_and_claim_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured.csv"
            with structured.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["budget", "seed", "compact_oa", "params_ratio", "macs_ratio"],
                )
                writer.writeheader()
                for budget in ("65", "80", "90"):
                    for seed in range(3):
                        writer.writerow(
                            {
                                "budget": budget,
                                "seed": seed,
                                "compact_oa": "85.0",
                                "params_ratio": budget,
                                "macs_ratio": budget,
                            }
                        )

            priority = root / "priority.csv"
            fieldnames = [
                "variant",
                "mode",
                "runs",
                "oa_mean",
                "oa_std",
                "source_oa_mean",
                "source_oa_std",
                "compact_macs_ratio_mean",
                "compact_params_ratio_mean",
            ]
            variants = [
                "quality_multi_degradation_p025",
                "full",
                "without_reliability_uniform_fusion",
                "quality_degradation_supervised",
            ]
            modes = [
                "full",
                "main_only",
                "aux_only",
                "aux_noise_high",
                "aux_downsample_4",
                "aux_occlusion_50",
            ]
            with priority.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for variant in variants:
                    for mode in modes:
                        writer.writerow(
                            {
                                "variant": variant,
                                "mode": mode,
                                "runs": "3",
                                "oa_mean": "0.8",
                                "oa_std": "0.01",
                                "source_oa_mean": "0.82",
                                "source_oa_std": "0.02",
                                "compact_macs_ratio_mean": "0.8",
                                "compact_params_ratio_mean": "0.83",
                            }
                        )

            multidataset = root / "multidataset.csv"
            with multidataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Dataset",
                        "Full OA",
                        "Main-only OA",
                        "Aux-only OA",
                        "Occlusion50 OA",
                        "MACs",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Dataset": "Houston2013",
                        "Full OA": "86.90 +/- 1.65",
                        "Main-only OA": "80.04 +/- 1.93",
                        "Aux-only OA": "38.04 +/- 0.84",
                        "Occlusion50 OA": "85.57 +/- 1.64",
                        "MACs": "80.01 +/- 0.13",
                    }
                )

            output_csv = root / "canonical.csv"
            output_md = root / "claims.md"
            latex = root / "multidataset.tex"
            repo_latex = root / "repo_multidataset.tex"
            source_compact_prefix = root / "source_compact"
            source_compact_latex = root / "source_compact.tex"
            rows = main(
                [
                    "--priority-summary",
                    str(priority),
                    "--structured-runs",
                    str(structured),
                    "--multidataset-summary",
                    str(multidataset),
                    "--output-csv",
                    str(output_csv),
                    "--output-md",
                    str(output_md),
                    "--latex-multidataset-output",
                    str(latex),
                    "--repo-latex-multidataset-output",
                    str(repo_latex),
                    "--source-compact-output-prefix",
                    str(source_compact_prefix),
                    "--latex-source-compact-output",
                    str(source_compact_latex),
                ]
            )

            self.assertTrue(output_csv.is_file())
            self.assertTrue(output_md.is_file())
            self.assertTrue(latex.is_file())
            self.assertTrue(source_compact_prefix.with_suffix(".csv").is_file())
            self.assertTrue(source_compact_prefix.with_suffix(".md").is_file())
            self.assertTrue(source_compact_latex.is_file())
            self.assertGreater(len(rows), 10)
            self.assertIn("Routing", output_md.read_text(encoding="utf-8"))
            self.assertIn("dynamic profile routing is excluded", latex.read_text(encoding="utf-8"))
            self.assertIn("Source gated model", source_compact_latex.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
