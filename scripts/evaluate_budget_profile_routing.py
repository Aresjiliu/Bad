from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brmnet_core.profile_router import (
    evaluate_budget_profile_routing,
    predict_budget_profile_selection,
    select_utility_budget_profiles,
    train_quality_budget_router,
)


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


def _static_budget_selection(quality_features: dict[str, list[float]], budget: float) -> dict[str, float]:
    return {mode: float(budget) for mode in quality_features}


def _oracle_budget_selection(report: dict[str, object]) -> dict[str, float]:
    return {
        mode: float(metrics["selected_budget"])
        for mode, metrics in report["per_mode"].items()
    }


def _routing_accuracy(
    predicted_budgets_by_mode: dict[str, float],
    oracle_budgets_by_mode: dict[str, float],
) -> float:
    if not oracle_budgets_by_mode:
        return 0.0
    correct = sum(
        1
        for mode, oracle_budget in oracle_budgets_by_mode.items()
        if float(predicted_budgets_by_mode[mode]) == float(oracle_budget)
    )
    return correct / len(oracle_budgets_by_mode)


def build_routing_reports(
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    quality_features: dict[str, list[float]],
    profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    include_learned: bool = False,
    router_epochs: int = 200,
    router_seed: int = 0,
    utility_resource_penalty: float | None = None,
) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for budget in profile_budgets:
        reports[f"static_{budget}"] = evaluate_budget_profile_routing(
            profile_metrics,
            quality_features,
            profile_budgets=profile_budgets,
            selected_budgets_by_mode=_static_budget_selection(quality_features, budget),
        )
    oracle_report = evaluate_budget_profile_routing(
        profile_metrics,
        quality_features,
        profile_budgets=profile_budgets,
    )
    reports["oracle"] = oracle_report
    if include_learned:
        oracle_selection = _oracle_budget_selection(oracle_report)
        router, history = train_quality_budget_router(
            quality_features,
            oracle_selection,
            profile_budgets=profile_budgets,
            epochs=router_epochs,
            seed=router_seed,
        )
        learned_selection = predict_budget_profile_selection(router, quality_features)
        learned_report = evaluate_budget_profile_routing(
            profile_metrics,
            quality_features,
            profile_budgets=profile_budgets,
            selected_budgets_by_mode=learned_selection,
        )
        learned_report["summary"]["routing_accuracy_vs_oracle"] = _routing_accuracy(
            learned_selection,
            oracle_selection,
        )
        learned_report["summary"]["training_final_loss"] = history[-1]
        reports["learned"] = learned_report
    if utility_resource_penalty is not None:
        utility_selection = select_utility_budget_profiles(
            profile_metrics,
            profile_budgets=profile_budgets,
            resource_penalty=utility_resource_penalty,
        )
        utility_report = evaluate_budget_profile_routing(
            profile_metrics,
            quality_features,
            profile_budgets=profile_budgets,
            selected_budgets_by_mode=utility_selection,
        )
        utility_report["summary"]["utility_resource_penalty"] = float(utility_resource_penalty)
        reports["utility"] = utility_report
        if include_learned:
            router, history = train_quality_budget_router(
                quality_features,
                utility_selection,
                profile_budgets=profile_budgets,
                epochs=router_epochs,
                seed=router_seed,
            )
            learned_selection = predict_budget_profile_selection(router, quality_features)
            learned_report = evaluate_budget_profile_routing(
                profile_metrics,
                quality_features,
                profile_budgets=profile_budgets,
                selected_budgets_by_mode=learned_selection,
            )
            learned_report["summary"]["routing_accuracy_vs_utility"] = _routing_accuracy(
                learned_selection,
                utility_selection,
            )
            learned_report["summary"]["training_final_loss"] = history[-1]
            learned_report["summary"]["utility_resource_penalty"] = float(utility_resource_penalty)
            reports["learned_utility"] = learned_report
    return reports


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


def write_comparison_csv(path: str | Path, reports: dict[str, dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "mean_oa",
        "mean_selected_budget",
        "mean_expected_macs_ratio",
        "routing_accuracy_vs_oracle",
        "routing_accuracy_vs_utility",
        "utility_resource_penalty",
        "modes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, report in reports.items():
            summary = report["summary"]
            writer.writerow(
                {
                    "policy": policy,
                    **{field: summary.get(field, "") for field in fieldnames[1:]},
                }
            )


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
    parser.add_argument("--output-comparison-json")
    parser.add_argument("--output-comparison-csv")
    parser.add_argument("--learned-routing", action="store_true")
    parser.add_argument("--router-epochs", type=int, default=200)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--utility-resource-penalty", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_metrics = load_profile_metrics(args.profile_metrics)
    quality_features = load_quality_features(args.quality_metrics)
    report = evaluate_budget_profile_routing(profile_metrics, quality_features)
    write_routing_json(args.output_json, report)
    write_routing_csv(args.output_csv, report)
    if args.output_comparison_json or args.output_comparison_csv:
        reports = build_routing_reports(
            profile_metrics,
            quality_features,
            include_learned=args.learned_routing,
            router_epochs=args.router_epochs,
            router_seed=args.router_seed,
            utility_resource_penalty=args.utility_resource_penalty,
        )
        if args.output_comparison_json:
            write_routing_json(args.output_comparison_json, reports)
        if args.output_comparison_csv:
            write_comparison_csv(args.output_comparison_csv, reports)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
