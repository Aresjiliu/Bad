from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brmnet_core.profile_router import (
    evaluate_budget_profile_routing,
    predict_budget_profile_selection,
    select_pareto_tolerance_budget_profiles,
    train_quality_budget_router,
)
from scripts.evaluate_budget_profile_routing import _add_tradeoff_diagnostics, load_profile_metrics


QUALITY_KEYS = ("pre_q_main", "pre_q_aux", "pre_u_main", "pre_u_aux")
FALLBACK_QUALITY_KEYS = ("q_main", "q_aux")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate validation-derived budget routing from multi-seed profile metrics.",
    )
    parser.add_argument(
        "--profile-root",
        default="output/routing_profiles/quality_routing_profile_formal/quality_routing_profile",
    )
    parser.add_argument("--budgets", default="0.65,0.8,1.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--pareto-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--label-strategy",
        choices=("per_seed_pareto", "mean_profile_pareto"),
        default="per_seed_pareto",
        help="per_seed_pareto uses each seed's own Pareto labels; mean_profile_pareto derives one stable label set from seed-averaged profile metrics.",
    )
    parser.add_argument("--router-epochs", type=int, default=400)
    parser.add_argument("--router-seed", type=int, default=7)
    parser.add_argument("--output-dir", default="docs/generated")
    parser.add_argument("--output-stem", default="brmnet_validation_derived_pareto_routing")
    return parser


def parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) < 2:
        raise ValueError("At least two budget values are required")
    return values


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one seed is required")
    return values


def _budget_label(budget: float) -> str:
    return f"budget{int(round(float(budget) * 100))}"


def metrics_path_for_seed_budget(profile_root: str | Path, seed: int, budget: float) -> Path:
    root = Path(profile_root)
    pattern = f"*trainseed{int(seed)}*{_budget_label(budget)}*/compact_metrics.json"
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one metrics file for seed={seed}, budget={budget}, found {len(matches)}")
    return matches[0]


def load_seed_profile_metrics(
    profile_root: str | Path,
    seed: int,
    budgets: tuple[float, ...],
) -> dict[float, dict[str, dict[str, float]]]:
    specs = [f"{budget}:{metrics_path_for_seed_budget(profile_root, seed, budget)}" for budget in budgets]
    return load_profile_metrics(specs)


def average_profile_metrics(
    metrics_by_seed: dict[int, dict[float, dict[str, dict[str, float]]]],
    budgets: tuple[float, ...],
) -> dict[float, dict[str, dict[str, float]]]:
    averaged: dict[float, dict[str, dict[str, float]]] = {}
    for budget in budgets:
        modes = sorted(
            {
                mode
                for profile_metrics in metrics_by_seed.values()
                for mode in profile_metrics.get(budget, {})
            }
        )
        averaged[budget] = {}
        for mode in modes:
            keys = sorted(
                {
                    key
                    for profile_metrics in metrics_by_seed.values()
                    for key, value in profile_metrics.get(budget, {}).get(mode, {}).items()
                    if isinstance(value, (int, float))
                }
            )
            averaged[budget][mode] = {}
            for key in keys:
                values = [
                    float(profile_metrics[budget][mode][key])
                    for profile_metrics in metrics_by_seed.values()
                    if mode in profile_metrics.get(budget, {}) and key in profile_metrics[budget][mode]
                ]
                if values:
                    averaged[budget][mode][key] = sum(values) / len(values)
    return averaged


def quality_features_from_metrics(metrics_by_mode: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    for mode, metrics in metrics_by_mode.items():
        if all(key in metrics for key in QUALITY_KEYS):
            values = [float(metrics[key]) for key in QUALITY_KEYS]
            if max(abs(value) for value in values) == 0.0 and all(key in metrics for key in FALLBACK_QUALITY_KEYS):
                q_main = float(metrics["q_main"])
                q_aux = float(metrics["q_aux"])
                values = [q_main, q_aux, 1.0 - q_main, 1.0 - q_aux]
        elif all(key in metrics for key in FALLBACK_QUALITY_KEYS):
            q_main = float(metrics["q_main"])
            q_aux = float(metrics["q_aux"])
            values = [q_main, q_aux, 1.0 - q_main, 1.0 - q_aux]
        else:
            missing = [key for key in QUALITY_KEYS if key not in metrics]
            raise KeyError(f"Mode {mode!r} is missing quality keys: {', '.join(missing)}")
        features[mode] = values
    return features


def sample_key(seed: int, mode: str) -> str:
    return f"seed{int(seed)}::{mode}"


def unqualified_mode(key: str) -> str:
    return key.split("::", 1)[1] if "::" in key else key


def build_validation_samples(
    profile_root: str | Path,
    seeds: tuple[int, ...],
    budgets: tuple[float, ...],
    pareto_tolerance: float,
    label_strategy: str = "per_seed_pareto",
) -> tuple[
    dict[str, list[float]],
    dict[str, float],
    dict[int, dict[float, dict[str, dict[str, float]]]],
]:
    quality_by_sample: dict[str, list[float]] = {}
    target_by_sample: dict[str, float] = {}
    metrics_by_seed: dict[int, dict[float, dict[str, dict[str, float]]]] = {}
    for seed in seeds:
        profile_metrics = load_seed_profile_metrics(profile_root, seed, budgets)
        metrics_by_seed[seed] = profile_metrics

    if label_strategy == "mean_profile_pareto":
        mean_profile_metrics = average_profile_metrics(metrics_by_seed, budgets)
        stable_targets = select_pareto_tolerance_budget_profiles(
            mean_profile_metrics,
            profile_budgets=budgets,
            metric_tolerance=pareto_tolerance,
        )
    elif label_strategy == "per_seed_pareto":
        stable_targets = {}
    else:
        raise ValueError(f"Unknown label_strategy: {label_strategy}")

    for seed in seeds:
        profile_metrics = metrics_by_seed[seed]
        seed_quality = quality_features_from_metrics(profile_metrics[budgets[-1]])
        if label_strategy == "mean_profile_pareto":
            seed_targets = stable_targets
        else:
            seed_targets = select_pareto_tolerance_budget_profiles(
                profile_metrics,
                profile_budgets=budgets,
                metric_tolerance=pareto_tolerance,
            )
        for mode, features in seed_quality.items():
            if mode not in seed_targets:
                continue
            key = sample_key(seed, mode)
            quality_by_sample[key] = features
            target_by_sample[key] = seed_targets[mode]
    return quality_by_sample, target_by_sample, metrics_by_seed


def _training_subset(
    quality_by_sample: dict[str, list[float]],
    target_by_sample: dict[str, float],
    heldout_seed: int,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    prefix = f"seed{int(heldout_seed)}::"
    train_quality = {key: value for key, value in quality_by_sample.items() if not key.startswith(prefix)}
    train_targets = {key: value for key, value in target_by_sample.items() if key in train_quality}
    if not train_quality:
        raise ValueError("Training subset is empty; use at least two seeds for held-out-seed evaluation")
    return train_quality, train_targets


def _selection_for_seed(
    selected_by_sample: dict[str, float],
    seed: int,
) -> dict[str, float]:
    prefix = f"seed{int(seed)}::"
    return {
        unqualified_mode(key): budget
        for key, budget in selected_by_sample.items()
        if key.startswith(prefix)
    }


def _target_for_seed(target_by_sample: dict[str, float], seed: int) -> dict[str, float]:
    return _selection_for_seed(target_by_sample, seed)


def _routing_accuracy(predicted: dict[str, float], target: dict[str, float]) -> float:
    if not target:
        return 0.0
    correct = sum(1 for mode, budget in target.items() if float(predicted.get(mode, -1.0)) == float(budget))
    return correct / len(target)


def evaluate_validation_derived_router(
    profile_root: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    pareto_tolerance: float = 0.01,
    label_strategy: str = "per_seed_pareto",
    router_epochs: int = 400,
    router_seed: int = 7,
) -> dict[str, object]:
    quality_by_sample, target_by_sample, metrics_by_seed = build_validation_samples(
        profile_root,
        seeds,
        budgets,
        pareto_tolerance,
        label_strategy=label_strategy,
    )
    per_seed_reports: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, float | int]] = []
    all_predictions: dict[str, float] = {}

    for heldout_seed in seeds:
        train_quality, train_targets = _training_subset(quality_by_sample, target_by_sample, heldout_seed)
        router, history = train_quality_budget_router(
            train_quality,
            train_targets,
            profile_budgets=budgets,
            epochs=router_epochs,
            seed=router_seed,
        )
        heldout_quality = {
            key: value
            for key, value in quality_by_sample.items()
            if key.startswith(f"seed{int(heldout_seed)}::")
        }
        heldout_predictions = predict_budget_profile_selection(router, heldout_quality)
        all_predictions.update(heldout_predictions)
        selected_by_mode = _selection_for_seed(heldout_predictions, heldout_seed)
        target_by_mode = _target_for_seed(target_by_sample, heldout_seed)
        report = evaluate_budget_profile_routing(
            metrics_by_seed[heldout_seed],
            quality_features_from_metrics(metrics_by_seed[heldout_seed][budgets[-1]]),
            profile_budgets=budgets,
            selected_budgets_by_mode=selected_by_mode,
        )
        report["summary"]["heldout_seed"] = int(heldout_seed)
        report["summary"]["training_samples"] = len(train_quality)
        report["summary"]["heldout_samples"] = len(heldout_quality)
        report["summary"]["routing_accuracy_vs_pareto"] = _routing_accuracy(selected_by_mode, target_by_mode)
        report["summary"]["training_final_loss"] = history[-1]
        _add_tradeoff_diagnostics(report, metrics_by_seed[heldout_seed])
        for mode, metrics in report["per_mode"].items():
            metrics["target_budget"] = float(target_by_mode[mode])
        per_seed_reports[str(heldout_seed)] = report
        summary_rows.append(
            {
                "seed": int(heldout_seed),
                "mean_oa": float(report["summary"]["mean_oa"]),
                "mean_macs": float(report["summary"]["mean_expected_macs_ratio"]),
                "routing_accuracy_vs_pareto": float(report["summary"]["routing_accuracy_vs_pareto"]),
                "mean_regret": float(report["summary"]["mean_metric_regret"]),
                "mean_saving": float(report["summary"]["mean_resource_saving_vs_reference"]),
                "training_samples": int(report["summary"]["training_samples"]),
                "heldout_samples": int(report["summary"]["heldout_samples"]),
            }
        )

    mean_row = {
        "seed": "mean",
        "mean_oa": sum(float(row["mean_oa"]) for row in summary_rows) / len(summary_rows),
        "mean_macs": sum(float(row["mean_macs"]) for row in summary_rows) / len(summary_rows),
        "routing_accuracy_vs_pareto": sum(float(row["routing_accuracy_vs_pareto"]) for row in summary_rows) / len(summary_rows),
        "mean_regret": sum(float(row["mean_regret"]) for row in summary_rows) / len(summary_rows),
        "mean_saving": sum(float(row["mean_saving"]) for row in summary_rows) / len(summary_rows),
        "training_samples": sum(int(row["training_samples"]) for row in summary_rows) / len(summary_rows),
        "heldout_samples": sum(int(row["heldout_samples"]) for row in summary_rows) / len(summary_rows),
    }
    summary_rows.append(mean_row)
    target_distribution = Counter(float(value) for value in target_by_sample.values())
    prediction_distribution = Counter(float(value) for value in all_predictions.values())
    return {
        "config": {
            "profile_root": str(profile_root),
            "seeds": list(seeds),
            "budgets": list(budgets),
            "pareto_tolerance": pareto_tolerance,
            "label_strategy": label_strategy,
            "router_epochs": router_epochs,
            "router_seed": router_seed,
        },
        "sample_count": len(quality_by_sample),
        "target_distribution": {str(key): value for key, value in sorted(target_distribution.items())},
        "prediction_distribution": {str(key): value for key, value in sorted(prediction_distribution.items())},
        "summary_rows": summary_rows,
        "per_seed": per_seed_reports,
    }


def write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_csv(
    path: str | Path,
    quality_by_sample: dict[str, list[float]],
    target_by_sample: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "seed", "mode", *QUALITY_KEYS, "target_budget"],
        )
        writer.writeheader()
        for key, features in quality_by_sample.items():
            match = re.match(r"seed(\d+)::(.+)", key)
            seed = int(match.group(1)) if match else ""
            mode = match.group(2) if match else key
            writer.writerow(
                {
                    "sample": key,
                    "seed": seed,
                    "mode": mode,
                    **{feature_key: value for feature_key, value in zip(QUALITY_KEYS, features)},
                    "target_budget": target_by_sample[key],
                }
            )


def mode_target_stability_rows(target_by_sample: dict[str, float]) -> list[dict[str, object]]:
    by_mode: dict[str, list[float]] = {}
    for key, budget in target_by_sample.items():
        by_mode.setdefault(unqualified_mode(key), []).append(float(budget))
    rows: list[dict[str, object]] = []
    for mode, budgets in sorted(by_mode.items()):
        counts = Counter(budgets)
        majority_budget, majority_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        rows.append(
            {
                "mode": mode,
                "samples": len(budgets),
                "unique_targets": len(counts),
                "majority_budget": majority_budget,
                "majority_fraction": majority_count / len(budgets),
                "target_distribution": json.dumps({str(key): value for key, value in sorted(counts.items())}),
            }
        )
    return rows


def write_mode_stability_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "samples", "unique_targets", "majority_budget", "majority_fraction", "target_distribution"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(report: dict[str, object], path: str | Path) -> None:
    rows = report["summary_rows"]
    stability_rows = report.get("mode_target_stability", [])
    unstable_modes = [
        str(row["mode"])
        for row in stability_rows
        if int(row["unique_targets"]) > 1
    ]
    lines = [
        "# Validation-derived Pareto routing",
        "",
        f"Label strategy: {report['config'].get('label_strategy', 'per_seed_pareto')}",
        f"Samples: {report['sample_count']}",
        f"Target distribution: {report['target_distribution']}",
        f"Prediction distribution: {report['prediction_distribution']}",
        f"Unstable target modes: {', '.join(unstable_modes) if unstable_modes else 'none'}",
        "",
        "| Held-out seed | Mean OA | Mean MACs | Routing acc. | Mean regret | Mean saving | Train samples | Held-out samples |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {float(row['mean_oa']):.4f} | {float(row['mean_macs']):.4f} | "
            f"{float(row['routing_accuracy_vs_pareto']):.4f} | {float(row['mean_regret']):.4f} | "
            f"{float(row['mean_saving']):.4f} | {float(row['training_samples']):.1f} | {float(row['heldout_samples']):.1f} |"
        )
    lines += [
        "",
        "## Label stability",
        "",
        "| Mode | Unique targets | Majority budget | Majority fraction | Distribution |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in stability_rows:
        lines.append(
            f"| {row['mode']} | {row['unique_targets']} | {float(row['majority_budget']):.2f} | "
            f"{float(row['majority_fraction']):.2f} | `{row['target_distribution']}` |"
        )
    lines += [
        "",
        "This is a validation-state-derived router: each seed/state pair is treated as one supervised sample. "
        "It is a stronger training protocol than fitting only eleven state-level samples, but it is still not a true patch-level router because patch-wise quality features are not stored by the current formal runs. "
        "The mean-profile label strategy reduces seed-to-seed label noise by deriving labels from seed-averaged profile metrics.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    budgets = parse_float_tuple(args.budgets)
    seeds = parse_int_tuple(args.seeds)
    quality_by_sample, target_by_sample, _ = build_validation_samples(
        args.profile_root,
        seeds,
        budgets,
        args.pareto_tolerance,
        label_strategy=args.label_strategy,
    )
    report = evaluate_validation_derived_router(
        args.profile_root,
        seeds=seeds,
        budgets=budgets,
        pareto_tolerance=args.pareto_tolerance,
        label_strategy=args.label_strategy,
        router_epochs=args.router_epochs,
        router_seed=args.router_seed,
    )
    stability_rows = mode_target_stability_rows(target_by_sample)
    report["mode_target_stability"] = stability_rows
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    json_path = output_dir / f"{stem}.json"
    summary_csv = output_dir / f"{stem}_summary.csv"
    sample_csv = output_dir / f"{stem}_samples.csv"
    stability_csv = output_dir / f"{stem}_label_stability.csv"
    markdown = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_csv(summary_csv, report["summary_rows"])
    write_sample_csv(sample_csv, quality_by_sample, target_by_sample)
    write_mode_stability_csv(stability_csv, stability_rows)
    write_markdown(report, markdown)
    print(json.dumps(report["summary_rows"][-1], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
