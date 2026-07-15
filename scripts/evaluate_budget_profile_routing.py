from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brmnet_core.profile_router import evaluate_budget_profile_routing


QUALITY_KEYS = ("pre_q_main", "pre_q_aux", "pre_u_main", "pre_u_aux")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _missing_or_zero(metrics: dict[str, float], key: str) -> bool:
    return key not in metrics or float(metrics[key]) == 0.0


def _backfill_resource_ratios(
    metrics_by_mode: dict[str, dict[str, float]],
    metrics_path: str | Path,
) -> dict[str, dict[str, float]]:
    resource_path = Path(metrics_path).with_name("resource_stats.json")
    if not resource_path.exists():
        return metrics_by_mode

    resource_stats = _load_json(resource_path)
    compact_stats = resource_stats.get("compact", {})
    state_stats = resource_stats.get("state_dependent", {}).get("compact", {})
    for mode, metrics in metrics_by_mode.items():
        mode_stats = state_stats.get(mode, {})
        macs_ratio = mode_stats.get("global_macs_ratio", compact_stats.get("macs_ratio"))
        params_ratio = mode_stats.get("global_params_ratio", compact_stats.get("params_ratio"))
        if macs_ratio is not None and _missing_or_zero(metrics, "expected_macs_ratio"):
            metrics["expected_macs_ratio"] = float(macs_ratio)
        if params_ratio is not None and _missing_or_zero(metrics, "expected_params_ratio"):
            metrics["expected_params_ratio"] = float(params_ratio)
    return metrics_by_mode


def load_profile_metrics(profile_specs: list[str]) -> dict[float, dict[str, dict[str, float]]]:
    metrics: dict[float, dict[str, dict[str, float]]] = {}
    for spec in profile_specs:
        if ":" not in spec:
            raise ValueError(f"profile metric spec must be '<budget>:<path>', got {spec!r}")
        budget_text, path_text = spec.split(":", 1)
        budget = float(budget_text)
        metrics[budget] = _backfill_resource_ratios(_load_json(path_text), path_text)
    return metrics


def load_quality_features(path: str | Path) -> dict[str, list[float]]:
    metrics_by_mode = _load_json(path)
    features: dict[str, list[float]] = {}
    for mode, metrics in metrics_by_mode.items():
        missing = [key for key in QUALITY_KEYS if key not in metrics]
        if missing:
            raise KeyError(f"Mode {mode!r} is missing quality keys: {', '.join(missing)}")
        features[mode] = [float(metrics[key]) for key in QUALITY_KEYS]
    return features


def write_routing_csv(path: str | Path, report: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_mode = report["per_mode"]
    fieldnames = ["mode", "selected_budget", "oa", "expected_macs_ratio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in per_mode.items():
            writer.writerow({"mode": mode, **{key: metrics.get(key, "") for key in fieldnames[1:]}})


def write_routing_json(path: str | Path, report: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate oracle routing over fixed BRM-Net budget profiles.")
    parser.add_argument(
        "--profile-metrics",
        action="append",
        required=True,
        help="Budget profile metrics as '<budget>:<metrics.json>'. Repeat for 0.65/0.8/1.0.",
    )
    parser.add_argument("--quality-metrics", required=True, help="Metrics JSON containing pre_q/pre_u fields by mode.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_metrics = load_profile_metrics(args.profile_metrics)
    quality_features = load_quality_features(args.quality_metrics)
    report = evaluate_budget_profile_routing(profile_metrics, quality_features)
    write_routing_json(args.output_json, report)
    write_routing_csv(args.output_csv, report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
