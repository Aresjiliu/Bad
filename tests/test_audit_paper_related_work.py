from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_paper_related_work import audit_related_work, main, parse_citations


class AuditPaperRelatedWorkTest(unittest.TestCase):
    def test_parse_citations_handles_multiple_keys(self):
        keys = parse_citations(r"Text~\cite{a,b, c}.")

        self.assertEqual(keys, ["a", "b", "c"])

    def test_audit_detects_missing_and_unused_bib_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper"
            (paper / "sections").mkdir(parents=True)
            (paper / "bib").mkdir()
            (paper / "sections" / "02_related_work.tex").write_text(
                r"\section{Related Work} Multimodal fusion~\cite{used,missing}.",
                encoding="utf-8",
            )
            (paper / "bib" / "references.bib").write_text(
                "\n".join(
                    [
                        "@article{used, title={Used}, year={2024}}",
                        "@article{unused, title={Unused}, year={2024}}",
                    ]
                ),
                encoding="utf-8",
            )

            audit = audit_related_work(paper)

            self.assertIn("missing", audit["missing_bib"])
            self.assertIn("unused", audit["unused_bib"])

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper"
            output = root / "audit.md"
            (paper / "sections").mkdir(parents=True)
            (paper / "bib").mkdir()
            (paper / "sections" / "02_related_work.tex").write_text(
                r"\section{Related Work} Multimodal hyperspectral LiDAR fusion pruning compact export missing degraded reliability onboard satellite edge latency resource~\cite{used}.",
                encoding="utf-8",
            )
            (paper / "bib" / "references.bib").write_text(
                "@article{used, title={Used}, year={2024}}",
                encoding="utf-8",
            )

            exit_code = main(["--paper-dir", str(paper), "--output-md", str(output)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Related Work 引用密度审查", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
