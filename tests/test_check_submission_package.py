from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_submission_package import check_submission_package, main


class CheckSubmissionPackageTest(unittest.TestCase):
    def _make_package(self, root: Path) -> Path:
        paper = root / "paper"
        (paper / "sections").mkdir(parents=True)
        (paper / "tables").mkdir()
        (paper / "figures" / "generated").mkdir(parents=True)
        (paper / "bib").mkdir()
        (paper / "code").mkdir()

        (paper / "paper.tex").write_text(
            "\n".join(
                [
                    "\\title{Example}",
                    "\\author{Anonymous Authors}",
                    "\\input{sections/00_abstract}",
                    "\\input{tables/result_table}",
                    "\\includegraphics{figures/generated/overview}",
                ]
            ),
            encoding="utf-8",
        )
        for name in [
            "00_abstract.tex",
            "01_intro.tex",
            "02_related_work.tex",
            "03_method.tex",
            "04_experiments.tex",
            "05_discussion.tex",
        ]:
            (paper / "sections" / name).write_text("Clean text.\n", encoding="utf-8")
        for idx in range(5):
            (paper / "tables" / f"table_{idx}.tex").write_text("\\begin{table}\\end{table}", encoding="utf-8")
        (paper / "tables" / "result_table.tex").write_text("\\begin{table}\\end{table}", encoding="utf-8")
        (paper / "figures" / "generated" / "overview.pdf").write_bytes(b"%PDF-1.4\n")
        (paper / "figures" / "generated" / "extra.pdf").write_bytes(b"%PDF-1.4\n")
        (paper / "bib" / "references.bib").write_text("@article{x,title={x}}\n", encoding="utf-8")
        (paper / "README.md").write_text("readme\n", encoding="utf-8")
        (paper / "AGENTS.md").write_text("agents\n", encoding="utf-8")
        (paper / "paper.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 60000)
        (paper / "paper.log").write_text("clean log\n", encoding="utf-8")
        return paper

    def test_package_passes_when_required_assets_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = self._make_package(Path(tmp))
            checks = check_submission_package(paper)

            self.assertTrue(all(item.status == "PASS" for item in checks), checks)

    def test_detects_missing_graphics_and_identity_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = self._make_package(Path(tmp))
            (paper / "paper.tex").write_text(
                "\\author{Anonymous Authors}\nD:/Academic/Aresjiliu\n\\includegraphics{figures/generated/missing}",
                encoding="utf-8",
            )

            checks = check_submission_package(paper)

            failed = {(item.category, item.item) for item in checks if item.status != "PASS"}
            self.assertIn(("latex references", "graphics files"), failed)
            self.assertIn(("anonymity", "local path or identity tokens in manuscript tex"), failed)

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = self._make_package(root)
            output_md = root / "checklist.md"
            output_csv = root / "checklist.csv"

            exit_code = main(
                [
                    "--paper-dir",
                    str(paper),
                    "--output-md",
                    str(output_md),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_md.is_file())
            self.assertTrue(output_csv.is_file())
            self.assertIn("PASS", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
