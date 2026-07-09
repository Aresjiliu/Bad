from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _variant_from_path(root: Path, run_dir: Path) -> str:
    try:
        relative = run_dir.relative_to(root)
    except ValueError:
        return "unknown"
    return relative.parts[0] if len(relative.parts) > 1 else "default"


def summarize_experiments(root: str | Path) -> list[dict[str, object]]:
    root = Path(root)
    groups: dict[
        tuple[object, ...],
        dict[int, tuple[int, dict[str, float]]],
    ] = defaultdict(dict)
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.json"
        if not config_path.is_file() or not history_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        compact_metrics_path = run_dir / "compact_metrics.json"
        selected_metrics_path = (
            compact_metrics_path if compact_metrics_path.is_file() else metrics_path
        )
        metrics = json.loads(selected_metrics_path.read_text(encoding="utf-8"))
        resource_path = run_dir / "resource_stats.json"
        resource_stats = (
            json.loads(resource_path.read_text(encoding="utf-8"))
            if resource_path.is_file()
            else {}
        )
        train_accuracy = float(history[-1]["train"]["accuracy"]) if history else 0.0
        train_seconds = sum(float(item["elapsed_seconds"]) for item in history)
        for mode, values in metrics.items():
            key = (
                _variant_from_path(root, run_dir),
                config["split_protocol"],
                int(config["split_seed"]),
                config.get("gate_mode", "stochastic"),
                config.get("gate_type", "legacy_sigmoid"),
                float(config.get("target_budget", -1.0)),
                config.get("budget_metric", "channels"),
                float(config.get("lambda_budget", 0.0)),
                float(config.get("modality_dropout_prob", 0.0)),
                int(config["epochs"]),
                mode,
            )
            record = {
                    "oa": float(values["oa"]),
                    "aa": float(values["aa"]),
                    "kappa": float(values["kappa"]),
                    "train_accuracy": train_accuracy,
                    "train_seconds": train_seconds,
                    "soft_retention": float(values.get("soft_retention", 0.0)),
                    "hard_retention": float(values.get("hard_retention", 0.0)),
                    "q_main": float(values.get("q_main", 0.0)),
                    "q_aux": float(values.get("q_aux", 0.0)),
                    "fusion_weight_main": float(values.get("fusion_weight_main", 0.0)),
                    "fusion_weight_aux": float(values.get("fusion_weight_aux", 0.0)),
                    "compact_params_ratio": float(
                        resource_stats.get("compact", {}).get("params_ratio", 0.0)
                    ),
                    "compact_macs_ratio": float(
                        resource_stats.get("compact", {}).get("macs_ratio", 0.0)
                    ),
                    "compact_latency_ms": float(
                        resource_stats.get("latency_ms", {})
                        .get("compact", {})
                        .get(mode, {})
                        .get("mean", 0.0)
                    ),
                }
            train_seed = int(config["seed"])
            modified = metrics_path.stat().st_mtime_ns
            previous = groups[key].get(train_seed)
            if previous is None or modified >= previous[0]:
                groups[key][train_seed] = (modified, record)

    rows = []
    for key, runs_by_seed in sorted(groups.items()):
        (
            variant,
            protocol,
            split_seed,
            gate_mode,
            gate_type,
            target_budget,
            budget_metric,
            lambda_budget,
            modality_dropout_prob,
            epochs,
            mode,
        ) = key
        runs = [item[1] for item in runs_by_seed.values()]
        row: dict[str, object] = {
            "variant": variant,
            "protocol": protocol,
            "split_seed": split_seed,
            "gate_mode": gate_mode,
            "gate_type": gate_type,
            "target_budget": target_budget,
            "budget_metric": budget_metric,
            "lambda_budget": lambda_budget,
            "modality_dropout_prob": modality_dropout_prob,
            "epochs": epochs,
            "mode": mode,
            "runs": len(runs),
        }
        for metric in (
            "oa",
            "aa",
            "kappa",
            "train_accuracy",
            "train_seconds",
            "soft_retention",
            "hard_retention",
            "q_main",
            "q_aux",
            "fusion_weight_main",
            "fusion_weight_aux",
            "compact_params_ratio",
            "compact_macs_ratio",
            "compact_latency_ms",
        ):
            values = [run[metric] for run in runs]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_std"] = _std(values)
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, object]], output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# BRM-Net Experiment Summary\n\n")
        handle.write("| Variant | Protocol | Gate | Metric | Budget | Lambda | Dropout | Epochs | Mode | Runs | OA | AA | Kappa | Params Ratio | MACs Ratio | Latency ms |\n")
        handle.write("|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['variant']} | {row['protocol']} | {row['gate_type']} | {row['budget_metric']} | "
                f"{row['target_budget']:.2f} | {row['lambda_budget']:.2f} | "
                f"{row['modality_dropout_prob']:.2f} | {row['epochs']} | "
                f"{row['mode']} | {row['runs']} | "
                f"{row['oa_mean']:.4f} +/- {row['oa_std']:.4f} | "
                f"{row['aa_mean']:.4f} +/- {row['aa_std']:.4f} | "
                f"{row['kappa_mean']:.4f} +/- {row['kappa_std']:.4f} | "
                f"{row['compact_params_ratio_mean']:.4f} | "
                f"{row['compact_macs_ratio_mean']:.4f} | "
                f"{row['compact_latency_ms_mean']:.3f} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    rows = summarize_experiments(args.root)
    write_summary(rows, args.output_prefix)
    print(f"Summarized {len(rows)} protocol/mode groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
