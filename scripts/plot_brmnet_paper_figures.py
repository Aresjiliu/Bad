from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MM = 1.0 / 25.4
DOUBLE_COLUMN = (183 * MM, 126 * MM)
SINGLE_COLUMN = (86 * MM, 62 * MM)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "grey": "#7A7A7A",
}
VARIANT_LABELS = {
    "full": "BRM-Net",
    "without_modality_dropout": "w/o MD",
    "without_budget_loss": "w/o budget",
    "budget_only": "budget only",
    "legacy_sigmoid_reference": "sigmoid gate",
    "default": "BRM-Net",
}
MODE_LABELS = {
    "full": "Full",
    "main_only": "HSI only",
    "aux_only": "LiDAR only",
    "aux_noise": "LiDAR noise",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready BRM-Net figures from experiment outputs."
    )
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--summary-csv", default="docs/generated/brmnet_structured_pruning.csv")
    parser.add_argument("--multiseed-csv", default="docs/generated/structured_pruning_multiseed_runs.csv")
    parser.add_argument("--output-dir", default="../paper_submission/brmnet_pricai2026/figures/generated")
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
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _percent(value: float) -> float:
    return value * 100.0 if abs(value) <= 1.5 else value


def _variant(row: dict[str, str]) -> str:
    return row.get("variant") or "default"


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " "))


def _row_for(rows: list[dict[str, str]], variant: str, mode: str) -> dict[str, str] | None:
    candidates = [row for row in rows if _variant(row) == variant and row.get("mode") == mode]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (_float(row, "target_budget"), _float(row, "epochs")))[-1]


def _available_variants(rows: list[dict[str, str]]) -> list[str]:
    priority = ["full", "without_modality_dropout", "without_budget_loss", "budget_only", "legacy_sigmoid_reference", "default"]
    available = {_variant(row) for row in rows}
    ordered = [variant for variant in priority if variant in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _load_resource_records(experiments_root: str | Path) -> list[dict[str, object]]:
    root = Path(experiments_root)
    records: list[dict[str, object]] = []
    for resource_path in root.rglob("resource_stats.json"):
        run_dir = resource_path.parent
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            continue
        try:
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            relative = run_dir.relative_to(root)
        except (json.JSONDecodeError, ValueError):
            continue
        variant = relative.parts[0] if len(relative.parts) > 1 else "default"
        compact = resource.get("compact", {})
        widths = resource.get("structure", {}).get("widths", {})
        records.append(
            {
                "variant": variant,
                "seed": int(config.get("seed", -1)),
                "target_budget": float(config.get("target_budget", 0.0)),
                "params_ratio": float(compact.get("params_ratio", 0.0)),
                "macs_ratio": float(compact.get("macs_ratio", 0.0)),
                "main_widths": compact.get("main_widths", widths.get("main", [])),
                "aux_widths": compact.get("aux_widths", widths.get("aux", [])),
                "head_widths": compact.get("head_widths", widths.get("head", [])),
            }
        )
    return records


def _width_matrix(resource_records: list[dict[str, object]], variants: list[str]) -> tuple[list[str], np.ndarray]:
    specs = [
        ("main_widths", "HSI L1", 32.0),
        ("main_widths", "HSI L2", 64.0),
        ("main_widths", "HSI L3", 128.0),
        ("aux_widths", "LiDAR L1", 32.0),
        ("aux_widths", "LiDAR L2", 64.0),
        ("aux_widths", "LiDAR L3", 128.0),
        ("head_widths", "Head L1", 128.0),
        ("head_widths", "Head L2", 64.0),
    ]
    matrix = np.full((len(specs), len(variants)), np.nan, dtype=float)
    for col, variant in enumerate(variants):
        records = [record for record in resource_records if record["variant"] == variant]
        for row, (key, _label, denom) in enumerate(specs):
            layer_idx = sum(1 for prior_key, _prior_label, _prior_denom in specs[:row] if prior_key == key)
            ratios = []
            for record in records:
                widths = record.get(key, [])
                if isinstance(widths, list) and layer_idx < len(widths):
                    ratios.append(float(widths[layer_idx]) / denom * 100.0)
            if ratios:
                matrix[row, col] = float(np.mean(ratios))
    return [label for _key, label, _denom in specs], matrix


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _plot_pareto(ax: plt.Axes, summary_rows: list[dict[str, str]], multiseed_rows: list[dict[str, str]]) -> None:
    legacy_points = []
    for row in multiseed_rows:
        macs = _percent(_float(row, "macs_ratio"))
        oa = _percent(_float(row, "compact_oa"))
        budget = _percent(_float(row, "budget"))
        if macs > 0 and oa > 0:
            legacy_points.append((macs, oa, budget))
    if legacy_points:
        xs = [item[0] for item in legacy_points]
        ys = [item[1] for item in legacy_points]
        ax.scatter(xs, ys, color="#C9C9C9", s=15, marker="o", linewidth=0, label="prior sweep", zorder=1)

    variants = _available_variants(summary_rows)
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    for idx, variant in enumerate(variants):
        row = _row_for(summary_rows, variant, "full")
        if row is None:
            continue
        macs = _percent(_float(row, "compact_macs_ratio_mean"))
        oa = _percent(_float(row, "oa_mean"))
        if macs <= 0 or oa <= 0:
            continue
        macs_std = _percent(_float(row, "compact_macs_ratio_std"))
        oa_std = _percent(_float(row, "oa_std"))
        # The current ablation points deliberately target the same 80% MAC budget.
        # Apply a small display-only offset so the uncertainty bars remain legible.
        display_macs = macs + (idx - (len(variants) - 1) / 2) * 0.55 if len(variants) > 1 else macs
        ax.errorbar(
            display_macs,
            oa,
            xerr=macs_std if macs_std > 0 else None,
            yerr=oa_std if oa_std > 0 else None,
            fmt="o",
            markersize=4.8,
            capsize=2.4,
            elinewidth=0.8,
            color=colors[idx % len(colors)],
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
            label=f"{_variant_label(variant)} (n={row.get('runs', '1')})",
        )
        ax.annotate(
            f"{oa:.1f}",
            (display_macs, oa),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6.5,
        )

    ax.axvspan(78, 82, color=OKABE_ITO["orange"], alpha=0.10, linewidth=0)
    ylo, yhi = ax.get_ylim()
    ax.text(80.25, ylo + 0.05 * (yhi - ylo), "80% target", color=OKABE_ITO["orange"], fontsize=6.5, va="bottom")
    ax.set_xlabel("Actual MAC ratio (%)")
    ax.set_ylabel("OA (%)")
    ax.set_title("Accuracy-efficiency")
    ax.grid(True, color="#E6E6E6", linewidth=0.45)
    ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.2)


def _plot_robustness(ax: plt.Axes, summary_rows: list[dict[str, str]]) -> None:
    variants = _available_variants(summary_rows)
    modes = ["full", "main_only", "aux_only", "aux_noise"]
    width = min(0.18, 0.72 / max(len(variants), 1))
    x = np.arange(len(modes))
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    hatches = ["", "///", "\\\\\\", "..."]
    for idx, variant in enumerate(variants):
        values = []
        for mode in modes:
            row = _row_for(summary_rows, variant, mode)
            values.append(_percent(_float(row, "oa_mean")) if row else np.nan)
        offset = (idx - (len(variants) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=_variant_label(variant),
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.35,
            hatch=hatches[idx % len(hatches)],
        )
    ax.set_xticks(x, [MODE_LABELS[mode] for mode in modes], rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("OA (%)")
    ax.set_title("Missing-modality robustness")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    if len(variants) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=min(len(variants), 2), columnspacing=0.8, handlelength=1.4)


def _plot_widths(ax: plt.Axes, resource_records: list[dict[str, object]], variants: list[str]) -> None:
    labels, matrix = _width_matrix(resource_records, variants)
    if not variants or np.isnan(matrix).all():
        ax.text(0.5, 0.5, "No compact width records", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    y = np.arange(len(labels))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(55, 103)
    ax.set_xlabel("Retained width (%)")
    ax.set_title("Retained width by layer")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.45)
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    if matrix.shape[1] >= 2:
        for row in range(matrix.shape[0]):
            valid = matrix[row, :]
            if np.sum(np.isfinite(valid)) >= 2:
                ax.plot(valid[:2], [row, row], color="#BDBDBD", linewidth=1.0, zorder=1)
    for col, variant in enumerate(variants):
        values = matrix[:, col]
        ax.scatter(
            values,
            y,
            s=22,
            color=colors[col % len(colors)],
            edgecolor="white",
            linewidth=0.4,
            label=_variant_label(variant),
            zorder=3,
        )
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="#D9D9D9", linewidth=0.6)
    ax.legend(loc="upper left", handlelength=1.2)


def _plot_budget(ax: plt.Axes, summary_rows: list[dict[str, str]]) -> None:
    variants = _available_variants(summary_rows)
    labels, macs, params, targets = [], [], [], []
    for variant in variants:
        row = _row_for(summary_rows, variant, "full")
        if row is None:
            continue
        labels.append(_variant_label(variant))
        macs.append(_percent(_float(row, "compact_macs_ratio_mean")))
        params.append(_percent(_float(row, "compact_params_ratio_mean")))
        targets.append(_percent(_float(row, "target_budget")))
    if not labels:
        ax.axis("off")
        return
    x = np.arange(len(labels))
    ax.bar(x - 0.16, macs, 0.32, color=OKABE_ITO["blue"], label="MACs", edgecolor="black", linewidth=0.35)
    ax.bar(x + 0.16, params, 0.32, color=OKABE_ITO["orange"], label="Params", edgecolor="black", linewidth=0.35)
    for idx, target in enumerate(targets):
        ax.plot([idx - 0.38, idx + 0.38], [target, target], color=OKABE_ITO["vermillion"], linewidth=1.0)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, max(100.0, max(macs + params + targets) * 1.12))
    ax.set_ylabel("Compact/full ratio (%)")
    ax.set_title("Budget agreement")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    ax.legend(loc="upper right")


def plot_overview(
    summary_rows: list[dict[str, str]],
    multiseed_rows: list[dict[str, str]],
    resource_records: list[dict[str, object]],
    out_dir: Path,
    formats: list[str],
) -> None:
    if not summary_rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COLUMN, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=3.0 / 72.0, h_pad=3.0 / 72.0, wspace=0.12, hspace=0.1)
    _plot_pareto(axes[0, 0], summary_rows, multiseed_rows)
    _plot_robustness(axes[0, 1], summary_rows)
    variants = _available_variants(summary_rows)
    _plot_widths(axes[1, 0], resource_records, variants)
    _plot_budget(axes[1, 1], summary_rows)
    for label, ax in zip(["A", "B", "C", "D"], axes.ravel(), strict=True):
        _panel_label(ax, label)
    _save(fig, out_dir, "fig_brmnet_results_overview", formats)


def plot_single_figures(
    summary_rows: list[dict[str, str]],
    multiseed_rows: list[dict[str, str]],
    resource_records: list[dict[str, object]],
    out_dir: Path,
    formats: list[str],
) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    _plot_pareto(ax, summary_rows, multiseed_rows)
    _save(fig, out_dir, "fig_pareto_accuracy_efficiency", formats)

    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    _plot_robustness(ax, summary_rows)
    _save(fig, out_dir, "fig_robustness_heatmap", formats)

    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    _plot_widths(ax, resource_records, _available_variants(summary_rows))
    _save(fig, out_dir, "fig_gate_retention", formats)


def main() -> int:
    args = build_parser().parse_args()
    apply_style()
    out_dir = _ensure_dir(args.output_dir)
    summary_rows = _read_csv(args.summary_csv)
    multiseed_rows = _read_csv(args.multiseed_csv)
    resource_records = _load_resource_records(args.experiments_root)
    plot_overview(summary_rows, multiseed_rows, resource_records, out_dir, args.formats)
    plot_single_figures(summary_rows, multiseed_rows, resource_records, out_dir, args.formats)
    print(f"Wrote publication figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
