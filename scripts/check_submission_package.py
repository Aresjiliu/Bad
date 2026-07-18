from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


INPUT_PATTERN = re.compile(r"\\input\{(?P<path>[^}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]+\])?\{(?P<path>[^}]+)\}")
LOCAL_PATH_PATTERN = re.compile(r"[A-Za-z]:\\|C:/|D:/|Users[/\\]|蒋冠军|Aresjiliu", re.IGNORECASE)


@dataclass(frozen=True)
class CheckItem:
    category: str
    item: str
    status: str
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the BRM-Net submission package completeness.")
    parser.add_argument("--paper-dir", default="D:/Academic/paper_submission/brmnet_pricai2026")
    parser.add_argument("--output-md", default="docs/SUBMISSION_PACKAGE_CHECKLIST_20260719_ZH.md")
    parser.add_argument("--output-csv", default="docs/generated/submission_package_checklist.csv")
    return parser


def _status(ok: bool) -> str:
    return "PASS" if ok else "NEEDS_REVIEW"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_tex_path(root: Path, raw: str) -> Path:
    path = root / raw
    if path.suffix:
        return path
    return path.with_suffix(".tex")


def _resolve_graphics_path(root: Path, raw: str) -> Path | None:
    path = root / raw
    if path.suffix:
        return path if path.is_file() else None
    for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def check_submission_package(paper_dir: str | Path) -> list[CheckItem]:
    root = Path(paper_dir)
    checks: list[CheckItem] = []

    required_files = [
        "paper.tex",
        "paper.pdf",
        "bib/references.bib",
        "README.md",
        "AGENTS.md",
    ]
    for rel in required_files:
        path = root / rel
        detail = str(path) if path.is_file() else f"Missing: {path}"
        checks.append(CheckItem("required file", rel, _status(path.is_file()), detail))

    for rel in ["sections", "tables", "figures/generated", "code"]:
        path = root / rel
        checks.append(CheckItem("required directory", rel, _status(path.is_dir()), str(path)))

    pdf = root / "paper.pdf"
    if pdf.is_file():
        size_kb = pdf.stat().st_size / 1024
        checks.append(CheckItem("compiled pdf", "paper.pdf size", _status(size_kb > 50), f"{size_kb:.1f} KB"))

    tex_files = sorted((root / "sections").glob("*.tex")) if (root / "sections").is_dir() else []
    checks.append(CheckItem("content inventory", "section tex files", _status(len(tex_files) >= 6), str(len(tex_files))))

    table_files = sorted((root / "tables").glob("*.tex")) if (root / "tables").is_dir() else []
    checks.append(CheckItem("content inventory", "table tex files", _status(len(table_files) >= 5), str(len(table_files))))

    figure_files = []
    figures_dir = root / "figures" / "generated"
    if figures_dir.is_dir():
        figure_files = sorted(
            path for path in figures_dir.iterdir() if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
        )
    checks.append(CheckItem("content inventory", "generated figure files", _status(len(figure_files) >= 2), str(len(figure_files))))

    paper_tex = root / "paper.tex"
    manuscript_text = ""
    if paper_tex.is_file():
        manuscript_text = _read_text(paper_tex)
        for tex in tex_files:
            manuscript_text += "\n" + _read_text(tex)

        input_paths = INPUT_PATTERN.findall(manuscript_text)
        missing_inputs = [
            raw for raw in input_paths if not _resolve_tex_path(root, raw).is_file()
        ]
        checks.append(
            CheckItem(
                "latex references",
                "input files",
                _status(not missing_inputs),
                "All referenced input files exist." if not missing_inputs else ", ".join(missing_inputs),
            )
        )

        graphics_paths = GRAPHICS_PATTERN.findall(manuscript_text)
        missing_graphics = [
            raw for raw in graphics_paths if _resolve_graphics_path(root, raw) is None
        ]
        checks.append(
            CheckItem(
                "latex references",
                "graphics files",
                _status(not missing_graphics),
                "All referenced graphics exist." if not missing_graphics else ", ".join(missing_graphics),
            )
        )

        has_anonymous_author = "Anonymous Authors" in manuscript_text
        checks.append(
            CheckItem(
                "anonymity",
                "anonymous author block",
                _status(has_anonymous_author),
                "Anonymous Authors found." if has_anonymous_author else "Anonymous author block not found.",
            )
        )

        local_path_hits = sorted(set(LOCAL_PATH_PATTERN.findall(manuscript_text)))
        checks.append(
            CheckItem(
                "anonymity",
                "local path or identity tokens in manuscript tex",
                _status(not local_path_hits),
                "None." if not local_path_hits else ", ".join(local_path_hits),
            )
        )

    log = root / "paper.log"
    if log.is_file():
        log_text = _read_text(log)
        hard_patterns = ["Undefined", "LaTeX Error", "Fatal", "Overfull", "Warning: Citation"]
        hits = [pattern for pattern in hard_patterns if pattern in log_text]
        checks.append(
            CheckItem(
                "build log",
                "hard-error pattern scan",
                _status(not hits),
                "None." if not hits else ", ".join(hits),
            )
        )

    return checks


def write_outputs(checks: list[CheckItem], output_md: str | Path, output_csv: str | Path) -> None:
    output_md = Path(output_md)
    output_csv = Path(output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    failures = [item for item in checks if item.status != "PASS"]
    status = "PASS" if not failures else "NEEDS_REVIEW"

    lines = [
        "# BRM-Net 投稿包完整性检查（2026-07-19）",
        "",
        f"**Status:** {status}",
        "",
        "| Category | Item | Status | Detail |",
        "|---|---|---|---|",
    ]
    for item in checks:
        detail = item.detail.replace("|", "\\|")
        lines.append(f"| {item.category} | {item.item} | {item.status} | {detail} |")

    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 若状态为 PASS，可以进入导师审阅或最终语言润色。",
            "- 若出现 anonymity 问题，应先清理作者、用户名、本地绝对路径和仓库身份信息。",
            "- 若出现 latex references 问题，应先补齐缺失的 table/figure/input 文件再提交 Overleaf。",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "item", "status", "detail"])
        writer.writeheader()
        for item in checks:
            writer.writerow(
                {
                    "category": item.category,
                    "item": item.item,
                    "status": item.status,
                    "detail": item.detail,
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = check_submission_package(args.paper_dir)
    write_outputs(checks, args.output_md, args.output_csv)
    failures = [item for item in checks if item.status != "PASS"]
    print(f"Wrote submission package checklist with {len(failures)} findings to {args.output_md}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
