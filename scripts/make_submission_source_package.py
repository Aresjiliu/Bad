from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


INPUT_PATTERN = re.compile(r"\\input\{(?P<path>[^}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]+\])?\{(?P<path>[^}]+)\}")


@dataclass(frozen=True)
class PackageEntry:
    path: str
    kind: str
    bytes: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a clean LaTeX source package for the BRM-Net submission.")
    parser.add_argument("--paper-dir", default="D:/Academic/paper_submission/brmnet_pricai2026")
    parser.add_argument(
        "--zip-path",
        default="D:/Academic/paper_submission/brmnet_pricai2026_submission_source_20260719.zip",
    )
    parser.add_argument("--manifest-md", default="docs/SUBMISSION_SOURCE_PACKAGE_MANIFEST_20260719_ZH.md")
    parser.add_argument("--manifest-csv", default="docs/generated/submission_source_package_manifest.csv")
    return parser


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _tex_path(root: Path, raw: str) -> Path:
    path = root / raw
    return path if path.suffix else path.with_suffix(".tex")


def _graphics_path(root: Path, raw: str) -> Path:
    path = root / raw
    if path.suffix:
        return path
    for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return path.with_suffix(".pdf")


def _add_file(files: dict[Path, str], root: Path, path: Path, kind: str) -> None:
    if path.is_file():
        files[path.resolve()] = kind


def collect_source_files(paper_dir: str | Path) -> list[PackageEntry]:
    root = Path(paper_dir).resolve()
    files: dict[Path, str] = {}
    paper_tex = root / "paper.tex"
    _add_file(files, root, paper_tex, "root tex")
    _add_file(files, root, root / ".latexmkrc", "latexmk config")
    _add_file(files, root, root / "bib" / "references.bib", "bibliography")

    queue = [paper_tex]
    visited: set[Path] = set()
    while queue:
        current = queue.pop(0).resolve()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        text = _read_text(current)

        for raw in INPUT_PATTERN.findall(text):
            child = _tex_path(root, raw).resolve()
            _add_file(files, root, child, "tex input")
            queue.append(child)

        for raw in GRAPHICS_PATTERN.findall(text):
            graphic = _graphics_path(root, raw).resolve()
            _add_file(files, root, graphic, "referenced figure")

    entries = [
        PackageEntry(
            path=str(path.relative_to(root)).replace("\\", "/"),
            kind=kind,
            bytes=path.stat().st_size,
        )
        for path, kind in sorted(files.items(), key=lambda item: str(item[0]))
    ]
    return entries


def create_zip(paper_dir: str | Path, entries: list[PackageEntry], zip_path: str | Path) -> Path:
    root = Path(paper_dir).resolve()
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.write(root / entry.path, arcname=entry.path)
    return zip_path


def write_manifest(entries: list[PackageEntry], zip_path: str | Path, manifest_md: str | Path, manifest_csv: str | Path) -> None:
    zip_path = Path(zip_path)
    manifest_md = Path(manifest_md)
    manifest_csv = Path(manifest_csv)
    manifest_md.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(entry.bytes for entry in entries)
    lines = [
        "# BRM-Net 投稿源码包 Manifest（2026-07-19）",
        "",
        f"- Zip path: `{zip_path}`",
        f"- File count: {len(entries)}",
        f"- Source bytes before compression: {total_bytes}",
        "",
        "## 打包原则",
        "",
        "- 只包含 LaTeX 编译必需文件：`paper.tex`、被 `\\input{}` 引用的 sections/tables、被 `\\includegraphics{}` 引用的 figures、bibliography 和 `.latexmkrc`。",
        "- 不包含构建产物：`aux/bbl/blg/fdb_latexmk/fls/log/synctex/pdf`。",
        "- 不包含预览图、review_pages、notes、sources、pycache、私有脚本缓存和未在主文引用的 thesis-only 表图。",
        "",
        "## Files",
        "",
        "| Path | Kind | Bytes |",
        "|---|---|---:|",
    ]
    for entry in entries:
        lines.append(f"| `{entry.path}` | {entry.kind} | {entry.bytes} |")
    lines.append("")
    manifest_md.write_text("\n".join(lines), encoding="utf-8")

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "bytes"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({"path": entry.path, "kind": entry.kind, "bytes": entry.bytes})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = collect_source_files(args.paper_dir)
    zip_path = create_zip(args.paper_dir, entries, args.zip_path)
    write_manifest(entries, zip_path, args.manifest_md, args.manifest_csv)
    print(f"Wrote submission source package with {len(entries)} files to {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
