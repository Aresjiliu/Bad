from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = [
    ("full", "Compact baseline", "#0072B2", "o", "-"),
    ("quality_degradation_supervised", "Noise-only quality", "#E69F00", "^", "--"),
    ("quality_multi_degradation_p025", "Multi-deg. p=0.25", "#009E73", "s", "-."),
]

FAMILIES = {
    "Noise": [
        ("aux_noise_low", 0.10, "low"),
        ("aux_noise_mid", 0.25, "mid"),
        ("aux_noise_high", 0.50, "high"),
    ],
    "Resolution loss": [
        ("aux_downsample_2", 0.50, "2x"),
        ("aux_downsample_4", 0.75, "4x"),
    ],
    "Occlusion": [
        ("aux_occlusion_25", 0.25, "25%"),
        ("aux_occlusion_50", 0.50, "50%"),
    ],
}

METRICS = [
    ("oa", "Compact OA (%)", "oa_mean", "oa_std", 100.0),
    ("q_aux", "Aux reliability", "q_aux_mean", "q_aux_std", 1.0),
    (
        "fusion_aux",
        "Aux fusion weight",
        "fusion_weight_aux_mean",
        "fusion_weight_aux_std",
        1.0,
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot controlled-corruption reliability curves for BRM-Net."
    )
    parser.add_argument("--summary-csv", default="docs/generated/brmnet_priority_summary.csv")
    parser.add_argument(
        "--output-prefix",
        default="docs/generated/brmnet_controlled_corruption_reliability_curves",
    )
    parser.add_argument(
        "--paper-output-prefix",
        default="D:/Academic/paper_submission/brmnet_pricai2026/figures/generated/fig_controlled_corruption_reliability_curves",
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    return parser


def _read_summary(path: str | Path) -> dict[tuple[str, str], dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row["variant"], row["mode"]): row
            for row in csv.DictReader(handle)
        }


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def build_curve_records(summary_csv: str | Path) -> list[dict[str, object]]:
    lookup = _read_summary(summary_csv)
    records: list[dict[str, object]] = []
    for variant, method, _color, _marker, _style in METHODS:
        for family, mode_specs in FAMILIES.items():
            for mode, severity, severity_label in mode_specs:
                row = lookup.get((variant, mode))
                if row is None:
                    continue
                for metric, metric_label, mean_key, std_key, scale in METRICS:
                    records.append(
                        {
                            "variant": variant,
                            "method": method,
                            "family": family,
                            "mode": mode,
                            "severity": severity,
                            "severity_label": severity_label,
                            "metric": metric,
                            "metric_label": metric_label,
                            "mean": _float(row, mean_key) * scale,
                            "std": _float(row, std_key) * scale,
                            "runs": int(row["runs"]),
                        }
                    )
    return records


def write_records_csv(records: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_note(records: list[dict[str, object]], path: str | Path) -> None:
    by_key = {
        (row["variant"], row["family"], row["mode"], row["metric"]): float(row["mean"])
        for row in records
    }
    p025_noise_q_change = (
        by_key[("quality_multi_degradation_p025", "Noise", "aux_noise_high", "q_aux")]
        - by_key[("quality_multi_degradation_p025", "Noise", "aux_noise_low", "q_aux")]
    )
    p025_occ_q_change = (
        by_key[("quality_multi_degradation_p025", "Occlusion", "aux_occlusion_50", "q_aux")]
        - by_key[("quality_multi_degradation_p025", "Occlusion", "aux_occlusion_25", "q_aux")]
    )
    p025_downsample_oa = by_key[
        ("quality_multi_degradation_p025", "Resolution loss", "aux_downsample_4", "oa")
    ]
    baseline_downsample_oa = by_key[("full", "Resolution loss", "aux_downsample_4", "oa")]

    lines = [
        "# Controlled-Corruption Reliability Curves",
        "",
        "This figure uses the three-seed Houston2013 priority summary to visualize how compact OA, auxiliary reliability, and auxiliary fusion weight change as the available LiDAR branch is progressively corrupted.",
        "",
        "## Main Readout",
        "",
        f"- Under the selected multi-degradation p=0.25 setting, aux reliability changes by {p025_noise_q_change:+.3f} from low to high noise and by {p025_occ_q_change:+.3f} from 25% to 50% occlusion.",
        f"- Downsample-4 compact OA improves from {baseline_downsample_oa:.2f}% for the compact baseline to {p025_downsample_oa:.2f}% with multi-degradation supervision.",
        "- The fusion weight curves are diagnostic rather than a calibration guarantee: they expose whether the reliability branch changes behavior under controlled corruption, but they should not be interpreted as physical sensor-quality measurements.",
        "",
        "## Generated Files",
        "",
        "- `brmnet_controlled_corruption_reliability_curves.csv`: tidy source data.",
        "- `brmnet_controlled_corruption_reliability_curves.pdf/.png`: repository figure.",
        "- `fig_controlled_corruption_reliability_curves.pdf/.png`: paper figure copy.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _records_for(
    records: list[dict[str, object]],
    *,
    variant: str,
    family: str,
    metric: str,
) -> list[dict[str, object]]:
    rows = [
        row
        for row in records
        if row["variant"] == variant and row["family"] == family and row["metric"] == metric
    ]
    return sorted(rows, key=lambda row: float(row["severity"]))


def plot_curves(records: list[dict[str, object]], output_prefixes: list[str | Path], formats: list[str]) -> None:
    import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.6), sharex=False, constrained_layout=True)
    family_names = list(FAMILIES)
    for col, family in enumerate(family_names):
        axes[0, col].set_title(family)
    for row_idx, (metric, metric_label, _mean_key, _std_key, _scale) in enumerate(METRICS):
        for col, family in enumerate(family_names):
            ax = axes[row_idx, col]
            for variant, method, color, marker, linestyle in METHODS:
                rows = _records_for(records, variant=variant, family=family, metric=metric)
                if not rows:
                    continue
                x_values = [float(row["severity"]) for row in rows]
                y_values = [float(row["mean"]) for row in rows]
                y_errors = [float(row["std"]) for row in rows]
                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.35,
                    markersize=4.2,
                    capsize=2.2,
                    color=color,
                    label=method,
                )
            if col == 0:
                ax.set_ylabel(metric_label)
            if row_idx == 2:
                ax.set_xlabel("Corruption severity")
            ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.65)
            if metric == "oa":
                ax.set_ylim(65, 90)
            else:
                ax.set_ylim(-0.02, 1.02)
            specs = FAMILIES[family]
            ax.set_xticks(
                [severity for _mode, severity, _label in specs],
                [label for _mode, _severity, label in specs],
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    for prefix in output_prefixes:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(prefix.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_prefix = Path(args.output_prefix)
    records = build_curve_records(args.summary_csv)
    write_records_csv(records, output_prefix.with_suffix(".csv"))
    write_note(records, output_prefix.with_suffix(".md"))
    plot_curves(records, [output_prefix, args.paper_output_prefix], args.formats)
    print(f"Wrote controlled-corruption reliability curves to {output_prefix.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
