from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SECTION_GLOBS = ("sections/*.tex",)
TABLE_PATTERN = re.compile(r"\\input\{(?P<path>tables/[^}]+)\}")
FIGURE_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(?P<path>figures/[^}]+)\}")
REF_PATTERN = re.compile(r"\\(?:ref|cref|Cref)\{(?P<label>[^}]+)\}")
LABEL_PATTERN = re.compile(r"\\label\{(?P<label>[^}]+)\}")


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    file: str
    line: int
    term: str
    message: str
    excerpt: str


TERM_RULES = [
    (
        "P1",
        "scope",
        re.compile(r"\b(profile routing|budget-profile routing|patch-level routing|dynamic profile routing)\b", re.I),
        "Routing should not be presented as a submission contribution; keep it only as future work.",
    ),
    (
        "P1",
        "scope",
        re.compile(r"\b(weighted CE|class-weighted|class imbalance|per-class|confusion matrix)\b", re.I),
        "Class-imbalance diagnostics should stay out of the submission main claim unless explicitly framed as deferred analysis.",
    ),
    (
        "P1",
        "internal wording",
        re.compile(r"\b(prototype|preliminary|todo|tbd|server log|thesis only|remaining experiments)\b", re.I),
        "Internal process wording weakens the submission manuscript.",
    ),
    (
        "P2",
        "claim strength",
        re.compile(r"\b(fully calibrated|absolute physical sensor-quality|always|guarantee)\b", re.I),
        "Avoid overclaiming reliability calibration or universal behavior.",
    ),
]

ALLOWED_CONTEXTS = {
    "dynamic profile routing is left as future work",
    "intentionally excluded from the submission scope",
    "outside the submission paper",
    "deferred to supplementary or thesis material",
    "rather than fully calibrated physical quality",
    "not assumed to be fully calibrated",
    "should not be treated as an absolute physical sensor-quality",
    "not always improved",
    "future work should",
    "future work will",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan the BRM-Net submission for readiness issues.")
    parser.add_argument("--paper-dir", default="D:/Academic/paper_submission/brmnet_pricai2026")
    parser.add_argument("--claims-csv", default="docs/generated/paper_canonical_results.csv")
    parser.add_argument(
        "--output-md",
        default="docs/PAPER_SUBMISSION_READINESS_CHECK_20260719_ZH.md",
    )
    parser.add_argument(
        "--output-csv",
        default="docs/generated/paper_submission_readiness_findings.csv",
    )
    parser.add_argument("--log-file", default=None)
    return parser


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _is_allowed_context(line: str) -> bool:
    normalized = line.lower()
    return any(context in normalized for context in ALLOWED_CONTEXTS)


def _scan_terms(paper_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for pattern in SECTION_GLOBS:
        for path in sorted(paper_dir.glob(pattern)):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if _is_allowed_context(line):
                    continue
                for severity, category, regex, message in TERM_RULES:
                    match = regex.search(line)
                    if match is None:
                        continue
                    findings.append(
                        Finding(
                            severity=severity,
                            category=category,
                            file=_relative(path, paper_dir),
                            line=line_no,
                            term=match.group(0),
                            message=message,
                            excerpt=line.strip(),
                        )
                    )
    return findings


def _scan_references(paper_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    labels: set[str] = set()
    refs: list[tuple[str, str, int, str]] = []
    label_paths = sorted(paper_dir.glob("**/*.tex"))
    for path in label_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        labels.update(match.group("label") for match in LABEL_PATTERN.finditer(text))

    for path in sorted(paper_dir.glob("sections/*.tex")) + [paper_dir / "paper.tex"]:
        if not path.is_file():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            refs.extend(
                (_relative(path, paper_dir), match.group("label"), line_no, line.strip())
                for match in REF_PATTERN.finditer(line)
            )
            for match in TABLE_PATTERN.finditer(line):
                rel_path = match.group("path")
                if not (paper_dir / f"{rel_path}.tex").is_file():
                    findings.append(
                        Finding(
                            severity="P0",
                            category="missing table",
                            file=_relative(path, paper_dir),
                            line=line_no,
                            term=rel_path,
                            message="LaTeX table input does not exist.",
                            excerpt=line.strip(),
                        )
                    )
            for match in FIGURE_PATTERN.finditer(line):
                rel_path = match.group("path")
                if not (paper_dir / rel_path).is_file():
                    findings.append(
                        Finding(
                            severity="P0",
                            category="missing figure",
                            file=_relative(path, paper_dir),
                            line=line_no,
                            term=rel_path,
                            message="LaTeX figure file does not exist.",
                            excerpt=line.strip(),
                        )
                    )
    for file, label, line_no, excerpt in refs:
        if label not in labels:
            findings.append(
                Finding(
                    severity="P0",
                    category="undefined ref",
                    file=file,
                    line=line_no,
                    term=label,
                    message="Reference label is not defined in the scanned manuscript files.",
                    excerpt=excerpt,
                )
            )
    return findings


def _scan_log(paper_dir: Path, log_file: str | Path | None) -> list[Finding]:
    path = Path(log_file) if log_file else paper_dir / "paper.log"
    if not path.is_file():
        return [
            Finding(
                severity="P2",
                category="latex log",
                file=_relative(path, paper_dir),
                line=0,
                term="missing log",
                message="LaTeX log was not found; compile the paper before final readiness.",
                excerpt="",
            )
        ]
    patterns = re.compile(r"Undefined|undefined|LaTeX Error|Fatal|Overfull|Warning: Citation")
    findings: list[Finding] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if patterns.search(line):
            findings.append(
                Finding(
                    severity="P0",
                    category="latex log",
                    file=_relative(path, paper_dir),
                    line=line_no,
                    term="latex warning/error",
                    message="LaTeX log contains a blocking warning/error pattern.",
                    excerpt=line.strip(),
                )
            )
    return findings


def _scan_claims(claims_csv: str | Path) -> list[Finding]:
    path = Path(claims_csv)
    if not path.is_file():
        return [
            Finding(
                severity="P0",
                category="claims evidence",
                file=str(path),
                line=0,
                term="missing claims csv",
                message="Canonical paper evidence CSV is missing.",
                excerpt="",
            )
        ]
    findings: list[Finding] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            status = row.get("status", "")
            if status not in {"ready", "supplement_only", "thesis_only"}:
                findings.append(
                    Finding(
                        severity="P1",
                        category="claims evidence",
                        file=str(path).replace("\\", "/"),
                        line=index,
                        term=status,
                        message="Canonical evidence row is not marked ready/supplement_only/thesis_only.",
                        excerpt=f"{row.get('evidence_id', '')}: {row.get('note', '')}",
                    )
                )
    return findings


def scan_readiness(paper_dir: str | Path, claims_csv: str | Path, log_file: str | Path | None = None) -> list[Finding]:
    paper_root = Path(paper_dir)
    findings: list[Finding] = []
    findings.extend(_scan_terms(paper_root))
    findings.extend(_scan_references(paper_root))
    findings.extend(_scan_log(paper_root, log_file))
    findings.extend(_scan_claims(claims_csv))
    return sorted(findings, key=lambda item: (item.severity, item.category, item.file, item.line))


def write_findings_csv(findings: list[Finding], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding.__dict__)


def write_report(findings: list[Finding], path: str | Path) -> None:
    counts = {"P0": 0, "P1": 0, "P2": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    status = "PASS" if counts.get("P0", 0) == 0 and counts.get("P1", 0) == 0 else "NEEDS_REVISION"
    lines = [
        "# BRM-Net 投稿 Readiness Check（2026-07-19）",
        "",
        f"**Status:** {status}",
        "",
        f"- P0 blocking issues: {counts.get('P0', 0)}",
        f"- P1 scope/writing issues: {counts.get('P1', 0)}",
        f"- P2 advisory issues: {counts.get('P2', 0)}",
        "",
        "## Findings",
        "",
        "| Severity | Category | File | Line | Term | Message | Excerpt |",
        "|---|---|---|---:|---|---|---|",
    ]
    for finding in findings:
        excerpt = finding.excerpt.replace("|", "\\|")
        message = finding.message.replace("|", "\\|")
        lines.append(
            f"| {finding.severity} | {finding.category} | `{finding.file}` | {finding.line} | "
            f"`{finding.term}` | {message} | {excerpt} |"
        )
    if not findings:
        lines.append("| PASS | all | manuscript | 0 | - | No readiness issues found. | - |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- P0 必须在投稿前修复。",
            "- P1 应在导师审阅前尽量清零。",
            "- P2 是保守性提醒，不一定阻塞投稿，但应检查是否存在过度承诺。",
        ]
    )
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings = scan_readiness(args.paper_dir, args.claims_csv, args.log_file)
    write_findings_csv(findings, args.output_csv)
    write_report(findings, args.output_md)
    print(f"Wrote readiness report with {len(findings)} findings to {args.output_md}")
    return 1 if any(finding.severity == "P0" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
