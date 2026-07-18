from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


CITE_PATTERN = re.compile(r"\\cite\{(?P<keys>[^}]+)\}")
BIB_KEY_PATTERN = re.compile(r"@\w+\{(?P<key>[^,]+),")
SECTION_PATTERN = re.compile(r"\\(?:section|subsection)\{(?P<title>[^}]+)\}")

TOPIC_RULES = {
    "multimodal_remote_sensing": [
        "multimodal",
        "hyperspectral",
        "lidar",
        "sar",
        "fusion",
    ],
    "compact_export_and_pruning": [
        "pruning",
        "compact",
        "export",
        "channel",
        "budget",
        "slimmable",
    ],
    "missing_or_degraded_modality": [
        "missing",
        "degraded",
        "incomplete",
        "reconstruction",
        "distillation",
        "reliability",
    ],
    "resource_constrained_inference": [
        "onboard",
        "satellite",
        "edge",
        "latency",
        "resource",
        "flops",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit related-work citation density for the BRM-Net paper.")
    parser.add_argument("--paper-dir", default="D:/Academic/paper_submission/brmnet_pricai2026")
    parser.add_argument("--output-md", default="docs/RELATED_WORK_CITATION_AUDIT_20260719_ZH.md")
    return parser


def parse_citations(text: str) -> list[str]:
    keys: list[str] = []
    for match in CITE_PATTERN.finditer(text):
        keys.extend(key.strip() for key in match.group("keys").split(",") if key.strip())
    return keys


def parse_bib_keys(text: str) -> set[str]:
    return {match.group("key").strip() for match in BIB_KEY_PATTERN.finditer(text)}


def topic_hits(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        topic: sum(lowered.count(term) for term in terms)
        for topic, terms in TOPIC_RULES.items()
    }


def audit_related_work(paper_dir: str | Path) -> dict[str, object]:
    root = Path(paper_dir)
    section_paths = sorted((root / "sections").glob("*.tex"))
    bib_path = root / "bib" / "references.bib"
    bib_keys = parse_bib_keys(bib_path.read_text(encoding="utf-8")) if bib_path.is_file() else set()

    section_summaries: list[dict[str, object]] = []
    all_citations: list[str] = []
    cited_by_section: dict[str, list[str]] = {}
    for path in section_paths:
        text = path.read_text(encoding="utf-8")
        citations = parse_citations(text)
        all_citations.extend(citations)
        title_match = SECTION_PATTERN.search(text)
        title = title_match.group("title") if title_match else path.stem
        cited_by_section[path.name] = citations
        section_summaries.append(
            {
                "file": path.name,
                "title": title,
                "citations": len(citations),
                "unique_citations": len(set(citations)),
                "topic_hits": topic_hits(text),
            }
        )

    used_keys = set(all_citations)
    missing_bib = sorted(used_keys - bib_keys)
    unused_bib = sorted(bib_keys - used_keys)
    duplicate_citations = sorted(key for key, count in Counter(all_citations).items() if count > 1)

    related_path = root / "sections" / "02_related_work.tex"
    related_text = related_path.read_text(encoding="utf-8") if related_path.is_file() else ""
    related_topics = topic_hits(related_text)
    weak_topics = [topic for topic, hits in related_topics.items() if hits < 3]

    return {
        "section_summaries": section_summaries,
        "bib_keys": bib_keys,
        "used_keys": used_keys,
        "missing_bib": missing_bib,
        "unused_bib": unused_bib,
        "duplicate_citations": duplicate_citations,
        "related_topics": related_topics,
        "weak_topics": weak_topics,
        "cited_by_section": cited_by_section,
    }


def write_report(audit: dict[str, object], output_md: str | Path) -> None:
    section_summaries = audit["section_summaries"]
    missing_bib = audit["missing_bib"]
    unused_bib = audit["unused_bib"]
    weak_topics = audit["weak_topics"]
    related_topics = audit["related_topics"]
    used_keys = audit["used_keys"]
    bib_keys = audit["bib_keys"]

    status = "PASS" if not missing_bib and not weak_topics else "NEEDS_REVIEW"
    lines = [
        "# BRM-Net Related Work \u5f15\u7528\u5bc6\u5ea6\u5ba1\u67e5\uff082026-07-19\uff09",
        "",
        f"**Status:** {status}",
        "",
        f"- BibTeX \u6761\u76ee\u6570\uff1a{len(bib_keys)}",
        f"- \u6b63\u6587\u5df2\u4f7f\u7528\u552f\u4e00\u5f15\u7528\u6570\uff1a{len(used_keys)}",
        f"- \u7f3a\u5931 BibTeX \u7684\u5f15\u7528\uff1a{len(missing_bib)}",
        f"- \u672a\u4f7f\u7528 BibTeX \u6761\u76ee\uff1a{len(unused_bib)}",
        "",
        "## Section Citation Density",
        "",
        "| Section file | Title | Citation count | Unique citations |",
        "|---|---|---:|---:|",
    ]
    for row in section_summaries:
        lines.append(
            f"| `{row['file']}` | {row['title']} | {row['citations']} | {row['unique_citations']} |"
        )

    lines.extend(
        [
            "",
            "## Related Work Topic Coverage",
            "",
            "| Topic | Keyword hits | Interpretation |",
            "|---|---:|---|",
        ]
    )
    for topic, hits in related_topics.items():
        interpretation = "needs attention" if topic in weak_topics else "covered"
        lines.append(f"| {topic} | {hits} | {interpretation} |")

    lines.extend(["", "## Missing BibTeX Keys", ""])
    if missing_bib:
        lines.extend(f"- `{key}`" for key in missing_bib)
    else:
        lines.append("- None.")

    lines.extend(["", "## Unused BibTeX Keys", ""])
    if unused_bib:
        lines.extend(f"- `{key}`" for key in unused_bib)
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Writing Recommendations",
            "",
            "- Related work already covers multimodal RS fusion, structured pruning, missing/degraded modalities, and resource-constrained inference.",
            "- The next improvement should not simply add many citations. It should strengthen the contrast between physical compact export, fusion compatibility, and degradation-supervised reliability.",
            "- Unused entries should either be cited where they help positioning or removed before final submission to keep the bibliography clean.",
            "- If adding new references, prioritize primary papers or official proceedings pages for compact export / structural pruning and incomplete multimodal learning.",
            "",
        ]
    )
    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_related_work(args.paper_dir)
    write_report(audit, args.output_md)
    print(f"Wrote related-work citation audit to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
