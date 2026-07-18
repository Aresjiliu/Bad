from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.make_advisor_submission_report import main


class MakeAdvisorSubmissionReportTest(unittest.TestCase):
    def _write_canonical(self, path: Path) -> None:
        fieldnames = ["evidence_id", "mean", "formatted"]
        rows = {
            "E_BUDGET_65": ("0.65", "65.00 ± 0.10"),
            "E_BUDGET_80": ("0.80", "80.00 ± 0.10"),
            "E_BUDGET_90": ("0.90", "90.00 ± 0.10"),
            "E_SOURCE_COMPACT_FULL": ("0.86", "86.00 ± 1.00"),
            "E_UNIFORM_WIDTH_FULL": ("0.85", "85.00 ± 1.00"),
            "E_FUSION_MASK_MAIN_ONLY": ("0.76", "76.00 ± 1.00"),
            "E_ROBUST_FULL_BASELINE_AUX_DOWNSAMPLE_4": ("0.78", "78.00 ± 1.00"),
            "E_ROBUST_MULTI_P025_AUX_DOWNSAMPLE_4": ("0.84", "84.00 ± 1.00"),
            "E_MULTI_HOUSTON2013": ("0.86", "86.00 ± 1.00"),
            "E_MULTI_TRENTO": ("0.98", "98.00 ± 1.00"),
            "E_MULTI_MUUFL": ("0.87", "87.00 ± 1.00"),
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for evidence_id, (mean, formatted) in rows.items():
                writer.writerow({"evidence_id": evidence_id, "mean": mean, "formatted": formatted})

    def _write_priority(self, path: Path) -> None:
        fieldnames = ["variant", "mode", "oa_mean"]
        rows = [
            ("quality_multi_degradation_p025", "full", "0.86"),
            ("quality_multi_degradation_p025", "main_only", "0.80"),
            ("uniform_width_export", "full", "0.85"),
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for variant, mode, oa in rows:
                writer.writerow({"variant": variant, "mode": mode, "oa_mean": oa})

    def test_main_writes_advisor_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical.csv"
            priority = root / "priority.csv"
            readiness = root / "readiness.md"
            reliability = root / "reliability.md"
            pdf = root / "paper.pdf"
            output = root / "advisor.md"
            self._write_canonical(canonical)
            self._write_priority(priority)
            readiness.write_text("**Status:** PASS\n", encoding="utf-8")
            reliability.write_text(
                "\n".join(["# Curves", "", "## Main Readout", "", "- q_aux changes conservatively."]),
                encoding="utf-8",
            )
            pdf.write_bytes(b"%PDF-1.4\n")

            exit_code = main(
                [
                    "--canonical-csv",
                    str(canonical),
                    "--priority-summary",
                    str(priority),
                    "--readiness-md",
                    str(readiness),
                    "--reliability-md",
                    str(reliability),
                    "--paper-pdf",
                    str(pdf),
                    "--output-md",
                    str(output),
                ]
            )

            text = output.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertIn("Readiness scanner 状态：**PASS**", text)
            self.assertIn("建议请导师决策的问题", text)
            self.assertIn("q_aux changes conservatively", text)
            self.assertIn("MUUFL", text)


if __name__ == "__main__":
    unittest.main()
