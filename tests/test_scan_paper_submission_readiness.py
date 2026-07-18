from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.scan_paper_submission_readiness import main, scan_readiness


class ScanPaperSubmissionReadinessTest(unittest.TestCase):
    def _paper_root(self, root: Path) -> Path:
        paper = root / "paper"
        (paper / "sections").mkdir(parents=True)
        (paper / "tables").mkdir()
        (paper / "figures" / "generated").mkdir(parents=True)
        (paper / "paper.tex").write_text(
            "\\input{sections/01_intro}\n",
            encoding="utf-8",
        )
        (paper / "sections" / "01_intro.tex").write_text(
            "\n".join(
                [
                    "\\section{Intro}",
                    "Dynamic profile routing is left as future work because this paper focuses on static exported compact models.",
                    "This paragraph mentions weighted CE as a main result.",
                    "\\input{tables/existing_table}",
                    "\\includegraphics{figures/generated/existing_figure.pdf}",
                    "See Table~\\ref{tab:missing}.",
                    "\\label{sec:intro}",
                ]
            ),
            encoding="utf-8",
        )
        (paper / "tables" / "existing_table.tex").write_text("\\begin{table}\\end{table}", encoding="utf-8")
        (paper / "figures" / "generated" / "existing_figure.pdf").write_bytes(b"%PDF-1.4\n")
        (paper / "paper.log").write_text("clean log\n", encoding="utf-8")
        return paper

    def _claims_csv(self, root: Path, status: str = "ready") -> Path:
        path = root / "claims.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["evidence_id", "status", "note"])
            writer.writeheader()
            writer.writerow({"evidence_id": "E1", "status": status, "note": "ok"})
        return path

    def test_scans_terms_and_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = self._paper_root(root)
            claims = self._claims_csv(root)

            findings = scan_readiness(paper, claims)

            terms = {finding.term for finding in findings}
            self.assertIn("weighted CE", terms)
            self.assertIn("tab:missing", terms)
            self.assertNotIn("Dynamic profile routing", terms)

    def test_claim_status_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = self._paper_root(root)
            claims = self._claims_csv(root, status="needs_ablation")

            findings = scan_readiness(paper, claims)

            self.assertTrue(any(finding.category == "claims evidence" for finding in findings))

    def test_main_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = self._paper_root(root)
            claims = self._claims_csv(root)
            output_md = root / "readiness.md"
            output_csv = root / "readiness.csv"

            exit_code = main(
                [
                    "--paper-dir",
                    str(paper),
                    "--claims-csv",
                    str(claims),
                    "--output-md",
                    str(output_md),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertTrue(output_md.is_file())
            self.assertTrue(output_csv.is_file())
            self.assertIn("NEEDS_REVISION", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
