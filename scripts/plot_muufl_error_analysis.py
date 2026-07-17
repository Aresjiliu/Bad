from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MM = 1.0 / 25.4
DOUBLE_COLUMN = (183 * MM, 118 * MM)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MUUFL per-class and confusion-matrix diagnostics.")
    parser.add_argument("--runs-root", default="output/muufl_formal")
    parser.add_argument("--output-dir", default="docs/generated")
    parser.add_argument("--paper-output-dir", default="D:/Academic/paper_submission/brmnet_pricai2026/figures/generated")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def _seed_from_run_name(name: str) -> int:
    match = re.search(r"trainseed(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse train seed from run directory name: {name}")
    return int(match.group(1))


def load_full_metrics(runs_root: str | Path) -> list[dict[str, object]]:
    root = Path(runs_root)
    variant_dirs = {
        "muufl_100_baseline": root / "muufl_100_baseline",
        "muufl_80_p025_multideg": root / "muufl_80_p025_multideg",
    }
    records: list[dict[str, object]] = []
    for variant, variant_dir in variant_dirs.items():
        for metrics_path in sorted(variant_dir.glob("*/compact_metrics.json")):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            full = payload["full"]
            records.append(
                {
                    "variant": variant,
                    "seed": _seed_from_run_name(metrics_path.parent.name),
                    "run_dir": str(metrics_path.parent),
                    "class_accuracy": [float(v) for v in full["class_accuracy"]],
                    "confusion_matrix": np.asarray(full["confusion_matrix"], dtype=float),
                    "oa": float(full["oa"]),
                    "aa": float(full["aa"]),
                    "kappa": float(full["kappa"]),
                }
            )
    if len(records) != 6:
        raise ValueError(f"Expected 6 MUUFL formal records, found {len(records)}")
    return records


def summarize_per_class(records: list[dict[str, object]]) -> list[dict[str, float]]:
    variants = ["muufl_100_baseline", "muufl_80_p025_multideg"]
    class_count = len(records[0]["class_accuracy"])  # type: ignore[arg-type]
    rows: list[dict[str, float]] = []
    for class_idx in range(class_count):
        row: dict[str, float] = {"class_id": float(class_idx + 1)}
        for variant in variants:
            values = [
                float(record["class_accuracy"][class_idx]) * 100.0
                for record in records
                if record["variant"] == variant
            ]
            row[f"{variant}_mean"] = float(np.mean(values))
            row[f"{variant}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["delta_80_minus_100"] = row["muufl_80_p025_multideg_mean"] - row["muufl_100_baseline_mean"]
        rows.append(row)
    return rows


def aggregate_confusion(records: list[dict[str, object]], variant: str) -> np.ndarray:
    matrices = [record["confusion_matrix"] for record in records if record["variant"] == variant]
    if not matrices:
        raise ValueError(f"No confusion matrices found for variant {variant}")
    return np.sum(np.stack(matrices, axis=0), axis=0)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom != 0)


def write_per_class_csv(rows: list[dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "class_id",
        "muufl_100_baseline_mean",
        "muufl_100_baseline_std",
        "muufl_80_p025_multideg_mean",
        "muufl_80_p025_multideg_std",
        "delta_80_minus_100",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, float]], output_path: str | Path) -> None:
    lines = [
        "# MUUFL error analysis",
        "",
        "| Class | 100% baseline | 80% multi-deg. | Delta | Interpretation |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        delta = row["delta_80_minus_100"]
        if delta >= 2.0:
            note = "improved"
        elif delta <= -2.0:
            note = "decreased"
        else:
            note = "stable"
        lines.append(
            f"| {int(row['class_id'])} | "
            f"{row['muufl_100_baseline_mean']:.2f} +/- {row['muufl_100_baseline_std']:.2f} | "
            f"{row['muufl_80_p025_multideg_mean']:.2f} +/- {row['muufl_80_p025_multideg_std']:.2f} | "
            f"{delta:+.2f} | {note} |"
        )
    lines += [
        "",
        "The 80% multi-degradation model improves robustness states but redistributes clean full-modality class accuracy. "
        "Classes with negative deltas should be discussed as trade-off cases rather than hidden.",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def plot_muufl_error_analysis(records: list[dict[str, object]], rows: list[dict[str, float]]) -> plt.Figure:
    apply_style()
    baseline_cm = row_normalize(aggregate_confusion(records, "muufl_100_baseline")) * 100.0
    proposed_cm = row_normalize(aggregate_confusion(records, "muufl_80_p025_multideg")) * 100.0
    delta = np.asarray([row["delta_80_minus_100"] for row in rows], dtype=float)
    classes = np.arange(1, len(rows) + 1)

    fig = plt.figure(figsize=DOUBLE_COLUMN)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.8], wspace=0.28, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    im_a = ax_a.imshow(baseline_cm, cmap="viridis", vmin=0, vmax=100)
    ax_a.set_title("100% baseline")
    ax_a.set_xlabel("Predicted class")
    ax_a.set_ylabel("True class")
    ax_a.set_xticks(classes - 1)
    ax_a.set_yticks(classes - 1)
    ax_a.set_xticklabels(classes)
    ax_a.set_yticklabels(classes)
    _panel_label(ax_a, "A")

    im_b = ax_b.imshow(proposed_cm, cmap="viridis", vmin=0, vmax=100)
    ax_b.set_title("80% multi-deg.")
    ax_b.set_xlabel("Predicted class")
    ax_b.set_ylabel("True class")
    ax_b.set_xticks(classes - 1)
    ax_b.set_yticks(classes - 1)
    ax_b.set_xticklabels(classes)
    ax_b.set_yticklabels(classes)
    _panel_label(ax_b, "B")

    colorbar = fig.colorbar(im_b, ax=[ax_a, ax_b], fraction=0.035, pad=0.02)
    colorbar.set_label("Row-normalized accuracy (%)")

    colors = np.where(delta >= 0, "#0072B2", "#D55E00")
    ax_c.axhline(0, color="#5F5F5F", linewidth=0.7)
    ax_c.bar(classes, delta, color=colors, width=0.72)
    ax_c.set_xlabel("MUUFL class")
    ax_c.set_ylabel("Class accuracy delta (pp)")
    ax_c.set_xticks(classes)
    ax_c.set_title("80% multi-deg. minus 100% baseline")
    _panel_label(ax_c, "C")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    fig.align_ylabels([ax_a, ax_c])
    return fig


def save_figure(fig: plt.Figure, output_dir: str | Path, stem: str, formats: list[str]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=600, bbox_inches="tight")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    records = load_full_metrics(args.runs_root)
    rows = summarize_per_class(records)
    csv_path = output_dir / "muufl_error_analysis_per_class.csv"
    md_path = output_dir / "muufl_error_analysis.md"
    write_per_class_csv(rows, csv_path)
    write_markdown(rows, md_path)

    fig = plot_muufl_error_analysis(records, rows)
    save_figure(fig, output_dir, "muufl_error_analysis", args.formats)
    if args.paper_output_dir:
        save_figure(fig, args.paper_output_dir, "fig_muufl_error_analysis", args.formats)
    plt.close(fig)
    return {"csv": csv_path, "markdown": md_path}


if __name__ == "__main__":
    main()
