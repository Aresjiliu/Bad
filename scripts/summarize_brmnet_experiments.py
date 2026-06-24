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


def summarize_experiments(root: str | Path) -> list[dict[str, object]]:
    groups: dict[
        tuple[object, ...],
        dict[int, tuple[int, dict[str, float]]],
    ] = defaultdict(dict)
    for metrics_path in Path(root).rglob("metrics.json"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.json"
        if not config_path.is_file() or not history_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_accuracy = float(history[-1]["train"]["accuracy"]) if history else 0.0
        train_seconds = sum(float(item["elapsed_seconds"]) for item in history)
        for mode, values in metrics.items():
            key = (
                config["split_protocol"],
                int(config["split_seed"]),
                config.get("gate_mode", "stochastic"),
                int(config["epochs"]),
                mode,
            )
            record = {
                    "oa": float(values["oa"]),
                    "aa": float(values["aa"]),
                    "kappa": float(values["kappa"]),
                    "train_accuracy": train_accuracy,
                    "train_seconds": train_seconds,
                }
            train_seed = int(config["seed"])
            modified = metrics_path.stat().st_mtime_ns
            previous = groups[key].get(train_seed)
            if previous is None or modified >= previous[0]:
                groups[key][train_seed] = (modified, record)

    rows = []
    for key, runs_by_seed in sorted(groups.items()):
        protocol, split_seed, gate_mode, epochs, mode = key
        runs = [item[1] for item in runs_by_seed.values()]
        row: dict[str, object] = {
            "protocol": protocol,
            "split_seed": split_seed,
            "gate_mode": gate_mode,
            "epochs": epochs,
            "mode": mode,
            "runs": len(runs),
        }
        for metric in ("oa", "aa", "kappa", "train_accuracy", "train_seconds"):
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
        handle.write("| Protocol | Gate | Epochs | Mode | Runs | OA | AA | Kappa |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['protocol']} | {row['gate_mode']} | {row['epochs']} | "
                f"{row['mode']} | {row['runs']} | "
                f"{row['oa_mean']:.4f} +/- {row['oa_std']:.4f} | "
                f"{row['aa_mean']:.4f} +/- {row['aa_std']:.4f} | "
                f"{row['kappa_mean']:.4f} +/- {row['kappa_std']:.4f} |\n"
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
