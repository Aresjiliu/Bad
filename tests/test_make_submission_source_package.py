from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.make_submission_source_package import collect_source_files, main


class MakeSubmissionSourcePackageTest(unittest.TestCase):
    def _make_paper(self, root: Path) -> Path:
        paper = root / "paper"
        (paper / "sections").mkdir(parents=True)
        (paper / "tables").mkdir()
        (paper / "figures" / "generated").mkdir(parents=True)
        (paper / "bib").mkdir()
        (paper / "__pycache__").mkdir()

        (paper / "paper.tex").write_text(
            "\n".join(
                [
                    "\\input{sections/00_abstract}",
                    "\\input{sections/01_intro}",
                    "\\bibliography{bib/references}",
                ]
            ),
            encoding="utf-8",
        )
        (paper / "sections" / "00_abstract.tex").write_text("Abstract.\n", encoding="utf-8")
        (paper / "sections" / "01_intro.tex").write_text(
            "\\input{tables/result}\n\\includegraphics{figures/generated/main_figure}\n",
            encoding="utf-8",
        )
        (paper / "tables" / "result.tex").write_text("\\begin{table}\\end{table}", encoding="utf-8")
        (paper / "tables" / "unused.tex").write_text("\\begin{table}\\end{table}", encoding="utf-8")
        (paper / "figures" / "generated" / "main_figure.pdf").write_bytes(b"%PDF-1.4\n")
        (paper / "figures" / "generated" / "unused.pdf").write_bytes(b"%PDF-1.4\n")
        (paper / "bib" / "references.bib").write_text("@article{x,title={x}}\n", encoding="utf-8")
        (paper / ".latexmkrc").write_text("$pdf_mode=1;\n", encoding="utf-8")
        (paper / "paper.aux").write_text("build artifact\n", encoding="utf-8")
        (paper / "__pycache__" / "x.pyc").write_bytes(b"x")
        return paper

    def test_collects_only_referenced_compile_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = self._make_paper(Path(tmp))
            entries = collect_source_files(paper)
            paths = {entry.path for entry in entries}

            self.assertIn("paper.tex", paths)
            self.assertIn("sections/00_abstract.tex", paths)
            self.assertIn("sections/01_intro.tex", paths)
            self.assertIn("tables/result.tex", paths)
            self.assertIn("figures/generated/main_figure.pdf", paths)
            self.assertIn("bib/references.bib", paths)
            self.assertNotIn("tables/unused.tex", paths)
            self.assertNotIn("figures/generated/unused.pdf", paths)
            self.assertNotIn("paper.aux", paths)

    def test_main_writes_zip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = self._make_paper(root)
            zip_path = root / "source.zip"
            manifest_md = root / "manifest.md"
            manifest_csv = root / "manifest.csv"

            exit_code = main(
                [
                    "--paper-dir",
                    str(paper),
                    "--zip-path",
                    str(zip_path),
                    "--manifest-md",
                    str(manifest_md),
                    "--manifest-csv",
                    str(manifest_csv),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(zip_path.is_file())
            self.assertTrue(manifest_md.is_file())
            self.assertTrue(manifest_csv.is_file())
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
            self.assertIn("paper.tex", names)
            self.assertNotIn("paper.aux", names)


if __name__ == "__main__":
    unittest.main()
