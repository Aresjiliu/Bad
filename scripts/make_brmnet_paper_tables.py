from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = [
    ("Full baseline", "full"),
    ("Uniform fusion", "without_reliability_uniform_fusion"),
    ("Noise quality", "quality_degradation_supervised"),
    ("Multi-deg. p=0.50", "quality_multi_degradation_supervised"),
    ("Multi-deg. p=0.25", "quality_multi_degradation_p025"),
]

MODES = [
    ("Full", "full"),
    ("Noise-high", "aux_noise_high"),
    ("Downsample-4", "aux_downsample_4"),
    ("Occlusion-50", "aux_occlusion_50"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper tables from BRM-Net priority summaries.")
    parser.add_argument("--summary-csv", default="docs/generated/brmnet_priority_summary.csv")
    parser.add_argument("--output-prefix", default="docs/generated/brmnet_robustness_table")
    parser.add_argument(
        "--latex-output",
        default="D:/Academic/paper_submission/brmnet_pricai2026/tables/robustness_ablation.tex",
    )
    return parser


def _load_summary(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {(row["variant"], row["mode"]): row for row in csv.DictReader(handle)}


def _percent(row: dict[str, str], key: str) -> str:
    return f"{float(row[key]) * 100:.2f}"


def _rows(summary: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_label, variant in METHODS:
        mode_rows = [summary[(variant, mode)] for _, mode in MODES]
        adverse = [float(row["oa_mean"]) for row in mode_rows if row["mode"] != "full"]
        full_row = mode_rows[0]
        rows.append(
            {
                "method": method_label,
                "variant": variant,
                "runs": int(full_row["runs"]),
                "macs_percent": _percent(full_row, "compact_macs_ratio_mean"),
                "full_oa": _percent(full_row, "oa_mean"),
                "noise_high_oa": _percent(mode_rows[1], "oa_mean"),
                "downsample_4_oa": _percent(mode_rows[2], "oa_mean"),
                "occlusion_50_oa": _percent(mode_rows[3], "oa_mean"),
                "adverse_avg_oa": f"{sum(adverse) / len(adverse) * 100:.2f}",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Method",
        "MACs (%)",
        "Full OA",
        "Noise-high OA",
        "Downsample-4 OA",
        "Occlusion-50 OA",
        "Adverse Avg.",
    ]
    lines = [
        "# BRM-Net Robustness Ablation Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["macs_percent"]),
                    str(row["full_oa"]),
                    str(row["noise_high_oa"]),
                    str(row["downsample_4_oa"]),
                    str(row["occlusion_50_oa"]),
                    str(row["adverse_avg_oa"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    str(row["method"]),
                    str(row["macs_percent"]),
                    str(row["full_oa"]),
                    str(row["noise_high_oa"]),
                    str(row["downsample_4_oa"]),
                    str(row["occlusion_50_oa"]),
                    str(row["adverse_avg_oa"]),
                ]
            )
            + r" \\"
        )
    content = r"""\begin{table*}[t]
\centering
\caption{Houston2013-HS-LiDAR robustness ablation at the 80\% MAC budget. Values are compact-model OA (\%) averaged over three training seeds. Adverse Avg. averages Noise-high, Downsample-4, and Occlusion-50.}
\label{tab:robustness_ablation}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Method & MACs & Full & Noise-high & Downsample-4 & Occlusion-50 & Adverse Avg. \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> list[dict[str, object]]:
    args = build_parser().parse_args(argv)
    summary = _load_summary(args.summary_csv)
    rows = _rows(summary)
    output_prefix = Path(args.output_prefix)
    write_csv(rows, output_prefix.with_suffix(".csv"))
    write_markdown(rows, output_prefix.with_suffix(".md"))
    write_latex(rows, args.latex_output)
    return rows


if __name__ == "__main__":
    main()
