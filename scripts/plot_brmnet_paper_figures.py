from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-ready BRM-Net figures from experiment outputs.")
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--summary-csv", default="docs/generated/brmnet_structured_pruning.csv")
    parser.add_argument("--multiseed-csv", default="docs/generated/structured_pruning_multiseed_runs.csv")
    parser.add_argument("--output-dir", default="../paper_submission/brmnet_pricai2026/figures/generated")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")
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


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_pareto(summary_rows: list[dict[str, str]], multiseed_rows: list[dict[str, str]], out_dir: Path, formats: list[str]) -> None:
    points = []
    for row in summary_rows:
        if row.get("mode") != "full":
            continue
        macs = _float(row, "compact_macs_ratio_mean") * 100.0
        params = _float(row, "compact_params_ratio_mean") * 100.0
        oa = _float(row, "oa_mean") * 100.0
        budget = _float(row, "target_budget") * 100.0
        if macs > 0 and oa > 0:
            points.append((budget, macs, params, oa, "summary"))
    for row in multiseed_rows:
        budget = _float(row, "budget")
        macs = _float(row, "macs_ratio")
        params = _float(row, "params_ratio")
        oa = _float(row, "compact_oa")
        if macs > 1.5:
            macs = macs
        else:
            macs *= 100.0
        if params <= 1.5:
            params *= 100.0
        points.append((budget, macs, params, oa, "multiseed"))

    if not points:
        return

    grouped: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for budget, macs, params, oa, _source in points:
        grouped[budget].append((macs, params, oa))

    xs, ys, yerr, labels = [], [], [], []
    for budget in sorted(grouped):
        values = grouped[budget]
        xs.append(np.mean([item[0] for item in values]))
        oa_values = [item[2] for item in values]
        ys.append(np.mean(oa_values))
        yerr.append(np.std(oa_values, ddof=1) if len(oa_values) > 1 else 0.0)
        labels.append(f"{int(round(budget))}%")

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.4, capsize=3, color=OKABE_ITO[0])
    for x, y, label in zip(xs, ys, labels):
        ax.text(x + 0.6, y, label, va="center", fontsize=7)
    ax.set_xlabel("Actual MAC ratio (%)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    _save(fig, out_dir, "fig_pareto_accuracy_efficiency", formats)


def plot_robustness_heatmap(summary_rows: list[dict[str, str]], out_dir: Path, formats: list[str]) -> None:
    rows = [row for row in summary_rows if abs(_float(row, "target_budget") - 0.8) < 1e-6]
    if not rows:
        rows = summary_rows
    if not rows:
        return
    mode_order = ["full", "main_only", "aux_only", "aux_noise"]
    by_mode = {row.get("mode"): _float(row, "oa_mean") * 100.0 for row in rows}
    values = [by_mode.get(mode, np.nan) for mode in mode_order]

    fig, ax = plt.subplots(figsize=(3.45, 1.65))
    matrix = np.array([values], dtype=float)
    image = ax.imshow(matrix, cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(mode_order)), ["Full", "Main only", "Aux only", "Aux noise"])
    ax.set_yticks([0], ["BRM-Net"])
    for col, value in enumerate(values):
        if np.isfinite(value):
            ax.text(col, 0, f"{value:.1f}", ha="center", va="center", color="white" if value < np.nanmean(matrix) else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("OA (%)")
    ax.set_title("Robustness under modality states")
    _save(fig, out_dir, "fig_robustness_heatmap", formats)


def load_gate_records(experiments_root: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for resource_path in Path(experiments_root).rglob("resource_stats.json"):
        config_path = resource_path.parent / "config.json"
        if not config_path.is_file():
            continue
        try:
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for gate in resource.get("gates", []):
            records.append(
                {
                    "budget": float(config.get("target_budget", 0.0)) * 100.0,
                    "seed": int(config.get("seed", -1)),
                    "name": str(gate.get("name", "")),
                    "retention": float(gate.get("active_channels", 0)) / max(float(gate.get("total_channels", 1)), 1.0) * 100.0,
                }
            )
    return records


def plot_gate_retention(gate_records: list[dict[str, object]], out_dir: Path, formats: list[str]) -> None:
    if not gate_records:
        return
    budgets = sorted({float(record["budget"]) for record in gate_records})
    names = sorted({str(record["name"]) for record in gate_records})
    name_labels = [name.replace("main_encoder.net.", "main").replace("aux_encoder.net.", "aux").replace("classifier.net.", "head") for name in names]
    matrix = np.zeros((len(names), len(budgets)), dtype=float)
    for i, name in enumerate(names):
        for j, budget in enumerate(budgets):
            values = [float(record["retention"]) for record in gate_records if record["name"] == name and float(record["budget"]) == budget]
            matrix[i, j] = np.mean(values) if values else np.nan

    fig, ax = plt.subplots(figsize=(3.45, max(2.2, 0.22 * len(names))))
    image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(budgets)), [f"{int(round(b))}%" for b in budgets])
    ax.set_yticks(range(len(names)), name_labels)
    ax.set_xlabel("Target MAC budget")
    ax.set_title("Layer-wise retained channels")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Retained channels (%)")
    _save(fig, out_dir, "fig_gate_retention", formats)


def _parse_widths(value: str) -> list[float]:
    widths = []
    for part in str(value).split("/"):
        try:
            widths.append(float(part))
        except ValueError:
            continue
    return widths


def plot_gate_retention_from_widths(multiseed_rows: list[dict[str, str]], out_dir: Path, formats: list[str]) -> None:
    if not multiseed_rows:
        return
    layer_specs = [
        ("main_widths", "Main branch", [32.0, 64.0, 128.0]),
        ("aux_widths", "Aux branch", [32.0, 64.0, 128.0]),
        ("head_widths", "Head", [128.0, 64.0]),
    ]
    budgets = sorted({_float(row, "budget") for row in multiseed_rows})
    labels: list[str] = []
    matrix_rows: list[list[float]] = []
    for key, group_label, denominators in layer_specs:
        max_layers = max((len(_parse_widths(row.get(key, ""))) for row in multiseed_rows), default=0)
        for layer_idx in range(max_layers):
            labels.append(f"{group_label} L{layer_idx + 1}")
            budget_values = []
            for budget in budgets:
                ratios = []
                for row in multiseed_rows:
                    if abs(_float(row, "budget") - budget) > 1e-6:
                        continue
                    widths = _parse_widths(row.get(key, ""))
                    if layer_idx < len(widths) and layer_idx < len(denominators):
                        ratios.append(widths[layer_idx] / denominators[layer_idx] * 100.0)
                budget_values.append(float(np.mean(ratios)) if ratios else np.nan)
            matrix_rows.append(budget_values)

    if not matrix_rows:
        return
    matrix = np.asarray(matrix_rows, dtype=float)
    fig, ax = plt.subplots(figsize=(3.45, max(2.3, 0.24 * len(labels))))
    image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(budgets)), [f"{int(round(b))}%" for b in budgets])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Target MAC budget")
    ax.set_title("Layer-wise retained width")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Retained width (%)")
    _save(fig, out_dir, "fig_gate_retention", formats)


def main() -> int:
    args = build_parser().parse_args()
    apply_style()
    out_dir = _ensure_dir(args.output_dir)
    summary_rows = _read_csv(args.summary_csv)
    multiseed_rows = _read_csv(args.multiseed_csv)
    plot_pareto(summary_rows, multiseed_rows, out_dir, args.formats)
    plot_robustness_heatmap(summary_rows, out_dir, args.formats)
    gate_records = load_gate_records(args.experiments_root)
    if gate_records:
        plot_gate_retention(gate_records, out_dir, args.formats)
    else:
        plot_gate_retention_from_widths(multiseed_rows, out_dir, args.formats)
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
