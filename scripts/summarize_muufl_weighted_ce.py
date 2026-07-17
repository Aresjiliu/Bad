from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


MODES = [
    "full",
    "main_only",
    "aux_only",
    "aux_noise_high",
    "aux_downsample_4",
    "aux_occlusion_50",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MUUFL weighted CE ablation against multi-degradation baseline.")
    parser.add_argument("--baseline-root", default="output/muufl_formal/muufl_80_p025_multideg")
    parser.add_argument("--weighted-root", default="output/muufl_ablation/muufl_80_p025_weighted_ce")
    parser.add_argument("--output-dir", default="docs/generated")
    return parser


def seed_from_run_dir(name: str) -> int:
    match = re.search(r"trainseed(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse train seed from run directory name: {name}")
    return int(match.group(1))


def _metric(metrics: dict, mode: str, key: str) -> float:
    return float(metrics[mode][key])


def _load_resource_stats(metrics: dict, metrics_path: Path) -> dict:
    if "resource_stats" in metrics:
        return metrics["resource_stats"]
    resource_path = metrics_path.parent / "resource_stats.json"
    if not resource_path.exists():
        raise FileNotFoundError(f"Missing resource stats file: {resource_path}")
    return json.loads(resource_path.read_text(encoding="utf-8"))


def _compact_ratio(resource_stats: dict, key: str) -> float:
    if "compact" in resource_stats and key in resource_stats["compact"]:
        return float(resource_stats["compact"][key])
    if "compact" in resource_stats and "full" in resource_stats["compact"]:
        return float(resource_stats["compact"]["full"][f"global_{key}"])
    return float(resource_stats["state_dependent"]["compact"]["full"][f"global_{key}"])


def _compact_latency(resource_stats: dict) -> float:
    compact_latency = resource_stats["latency_ms"]["compact"]
    if "mean" in compact_latency:
        return float(compact_latency["mean"])
    return float(compact_latency["full"]["mean"])


def load_variant_records(root: str | Path, variant: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for metrics_path in sorted(Path(root).glob("*/compact_metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        resource_stats = _load_resource_stats(metrics, metrics_path)
        record: dict[str, object] = {
            "variant": variant,
            "seed": seed_from_run_dir(metrics_path.parent.name),
            "run_dir": str(metrics_path.parent),
            "full_aa": _metric(metrics, "full", "aa") * 100.0,
            "full_kappa": _metric(metrics, "full", "kappa") * 100.0,
            "compact_params_ratio": _compact_ratio(resource_stats, "params_ratio") * 100.0,
            "compact_macs_ratio": _compact_ratio(resource_stats, "macs_ratio") * 100.0,
            "compact_latency_ms": _compact_latency(resource_stats),
            "class_accuracy": [float(v) * 100.0 for v in metrics["full"]["class_accuracy"]],
        }
        for mode in MODES:
            record[f"{mode}_oa"] = _metric(metrics, mode, "oa") * 100.0
        records.append(record)
    if len(records) != 3:
        raise ValueError(f"Expected 3 records under {root}, found {len(records)}")
    return records


def load_records(baseline_root: str | Path, weighted_root: str | Path) -> list[dict[str, object]]:
    return load_variant_records(baseline_root, "80% multi-deg.") + load_variant_records(weighted_root, "80% multi-deg. + weighted CE")


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_fields = ["full_aa", "full_kappa", "compact_params_ratio", "compact_macs_ratio", "compact_latency_ms"]
    numeric_fields += [f"{mode}_oa" for mode in MODES]
    rows: list[dict[str, object]] = []
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_variant[str(record["variant"])].append(record)

    for variant, variant_records in by_variant.items():
        row: dict[str, object] = {"variant": variant, "seeds": len(variant_records)}
        for field in numeric_fields:
            values = [float(record[field]) for record in variant_records]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def summarize_class_delta(records: list[dict[str, object]]) -> list[dict[str, float]]:
    variants = {str(record["variant"]) for record in records}
    baseline_name = "80% multi-deg."
    weighted_name = "80% multi-deg. + weighted CE"
    if variants != {baseline_name, weighted_name}:
        raise ValueError(f"Unexpected variants: {sorted(variants)}")

    baseline = [record for record in records if record["variant"] == baseline_name]
    weighted = [record for record in records if record["variant"] == weighted_name]
    class_count = len(baseline[0]["class_accuracy"])  # type: ignore[arg-type]
    rows: list[dict[str, float]] = []
    for class_idx in range(class_count):
        baseline_values = [float(record["class_accuracy"][class_idx]) for record in baseline]  # type: ignore[index]
        weighted_values = [float(record["class_accuracy"][class_idx]) for record in weighted]  # type: ignore[index]
        rows.append(
            {
                "class_id": float(class_idx + 1),
                "baseline_mean": mean(baseline_values),
                "weighted_ce_mean": mean(weighted_values),
                "delta_weighted_ce_minus_baseline": mean(weighted_values) - mean(baseline_values),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_records_csv(records: list[dict[str, object]], output_path: str | Path) -> None:
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"class_accuracy", "run_dir"}}
        rows.append(row)
    write_csv(rows, output_path)


def write_markdown(summary_rows: list[dict[str, object]], class_rows: list[dict[str, float]], output_path: str | Path) -> None:
    lines = [
        "# MUUFL weighted CE ablation",
        "",
        "## Three-seed summary",
        "",
        "| Variant | Full OA | Full AA | Main-only OA | Aux-only OA | Noise-high OA | Downsample4 OA | Occlusion50 OA | MACs | Params |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | "
            f"{float(row['full_oa_mean']):.2f} +/- {float(row['full_oa_std']):.2f} | "
            f"{float(row['full_aa_mean']):.2f} +/- {float(row['full_aa_std']):.2f} | "
            f"{float(row['main_only_oa_mean']):.2f} +/- {float(row['main_only_oa_std']):.2f} | "
            f"{float(row['aux_only_oa_mean']):.2f} +/- {float(row['aux_only_oa_std']):.2f} | "
            f"{float(row['aux_noise_high_oa_mean']):.2f} +/- {float(row['aux_noise_high_oa_std']):.2f} | "
            f"{float(row['aux_downsample_4_oa_mean']):.2f} +/- {float(row['aux_downsample_4_oa_std']):.2f} | "
            f"{float(row['aux_occlusion_50_oa_mean']):.2f} +/- {float(row['aux_occlusion_50_oa_std']):.2f} | "
            f"{float(row['compact_macs_ratio_mean']):.2f} +/- {float(row['compact_macs_ratio_std']):.2f} | "
            f"{float(row['compact_params_ratio_mean']):.2f} +/- {float(row['compact_params_ratio_std']):.2f} |"
        )

    delta_by_field = {}
    if len(summary_rows) == 2:
        baseline = next(row for row in summary_rows if row["variant"] == "80% multi-deg.")
        weighted = next(row for row in summary_rows if row["variant"] == "80% multi-deg. + weighted CE")
        for field in ["full_oa", "full_aa", "main_only_oa", "aux_only_oa", "aux_noise_high_oa", "aux_downsample_4_oa", "aux_occlusion_50_oa"]:
            delta_by_field[field] = float(weighted[f"{field}_mean"]) - float(baseline[f"{field}_mean"])

    lines += [
        "",
        "## Interpretation",
        "",
        f"Weighted CE changes full-modality OA by {delta_by_field.get('full_oa', 0.0):+.2f} pp and AA by {delta_by_field.get('full_aa', 0.0):+.2f} pp. "
        f"The most important side effect is aux-only OA ({delta_by_field.get('aux_only_oa', 0.0):+.2f} pp), which indicates that class balancing should not be promoted as an unconditional improvement.",
        "",
        "## Per-class full-modality delta",
        "",
        "| Class | Multi-deg. | Weighted CE | Delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in class_rows:
        lines.append(
            f"| {int(row['class_id'])} | {row['baseline_mean']:.2f} | "
            f"{row['weighted_ce_mean']:.2f} | {row['delta_weighted_ce_minus_baseline']:+.2f} |"
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    records = load_records(args.baseline_root, args.weighted_root)
    summary_rows = summarize_records(records)
    class_rows = summarize_class_delta(records)

    runs_csv = output_dir / "muufl_weighted_ce_seed0_seed1_seed2_runs.csv"
    summary_csv = output_dir / "muufl_weighted_ce_seed0_seed1_seed2_summary.csv"
    class_csv = output_dir / "muufl_weighted_ce_seed0_seed1_seed2_class_delta.csv"
    markdown = output_dir / "muufl_weighted_ce_seed0_seed1_seed2_summary.md"
    write_records_csv(records, runs_csv)
    write_csv(summary_rows, summary_csv)
    write_csv(class_rows, class_csv)
    write_markdown(summary_rows, class_rows, markdown)
    return {"runs_csv": runs_csv, "summary_csv": summary_csv, "class_csv": class_csv, "markdown": markdown}


if __name__ == "__main__":
    main()
