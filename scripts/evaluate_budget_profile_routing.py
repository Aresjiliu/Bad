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
    select_pareto_tolerance_budget_profiles,
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


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("float list must contain at least one value")
    return values


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


def _best_metrics_by_mode(
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    metric_key: str = "oa",
    resource_key: str = "expected_macs_ratio",
    reference_budget: float = 1.0,
) -> dict[str, dict[str, float]]:
    modes = sorted({mode for metrics_by_mode in profile_metrics.values() for mode in metrics_by_mode})
    best_by_mode: dict[str, dict[str, float]] = {}
    reference_metrics = profile_metrics.get(float(reference_budget), {})
    for mode in modes:
        available = [
            metrics_by_mode[mode]
            for metrics_by_mode in profile_metrics.values()
            if mode in metrics_by_mode
        ]
        if not available:
            continue
        best_metric = max(float(metrics[metric_key]) for metrics in available)
        reference_resource = reference_metrics.get(mode, {}).get(resource_key)
        best_by_mode[mode] = {
            f"best_{metric_key}": best_metric,
            f"reference_{resource_key}": float(reference_resource) if reference_resource is not None else 0.0,
        }
    return best_by_mode


def _add_tradeoff_diagnostics(
    report: dict[str, object],
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    metric_key: str = "oa",
    resource_key: str = "expected_macs_ratio",
    reference_budget: float = 1.0,
) -> None:
    best_by_mode = _best_metrics_by_mode(
        profile_metrics,
        metric_key=metric_key,
        resource_key=resource_key,
        reference_budget=reference_budget,
    )
    regrets: list[float] = []
    savings: list[float] = []
    for mode, metrics in report["per_mode"].items():
        best = best_by_mode.get(mode)
        if not best:
            continue
        regret = float(best[f"best_{metric_key}"]) - float(metrics[metric_key])
        reference_resource = float(best[f"reference_{resource_key}"])
        saving = reference_resource - float(metrics[resource_key]) if reference_resource else 0.0
        metrics["metric_regret"] = regret
        metrics["resource_saving_vs_reference"] = saving
        regrets.append(regret)
        savings.append(saving)
    count = max(len(regrets), 1)
    report["summary"]["mean_metric_regret"] = sum(regrets) / count
    report["summary"]["mean_resource_saving_vs_reference"] = sum(savings) / count


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


def _format_sweep_value(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m")


def build_sweep_reports(
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    quality_features: dict[str, list[float]],
    profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    pareto_tolerances: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03),
    utility_resource_penalties: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for tolerance in pareto_tolerances:
        selection = select_pareto_tolerance_budget_profiles(
            profile_metrics,
            profile_budgets=profile_budgets,
            metric_tolerance=tolerance,
        )
        report = evaluate_budget_profile_routing(
            profile_metrics,
            quality_features,
            profile_budgets=profile_budgets,
            selected_budgets_by_mode=selection,
        )
        report["summary"]["pareto_metric_tolerance"] = float(tolerance)
        _add_tradeoff_diagnostics(report, profile_metrics)
        reports[f"pareto_delta_{_format_sweep_value(tolerance)}"] = report
    for penalty in utility_resource_penalties:
        selection = select_utility_budget_profiles(
            profile_metrics,
            profile_budgets=profile_budgets,
            resource_penalty=penalty,
        )
        report = evaluate_budget_profile_routing(
            profile_metrics,
            quality_features,
            profile_budgets=profile_budgets,
            selected_budgets_by_mode=selection,
        )
        report["summary"]["utility_resource_penalty"] = float(penalty)
        _add_tradeoff_diagnostics(report, profile_metrics)
        reports[f"utility_lambda_{_format_sweep_value(penalty)}"] = report
    return reports


def build_leave_one_state_out_report(
    profile_metrics: dict[float, dict[str, dict[str, float]]],
    quality_features: dict[str, list[float]],
    profile_budgets: tuple[float, ...] = (0.65, 0.8, 1.0),
    pareto_tolerance: float = 0.01,
    router_epochs: int = 200,
    router_seed: int = 0,
) -> dict[str, object]:
    target_selection = select_pareto_tolerance_budget_profiles(
        profile_metrics,
        profile_budgets=profile_budgets,
        metric_tolerance=pareto_tolerance,
    )
    predicted_selection: dict[str, float] = {}
    modes = list(quality_features)
    for heldout_mode in modes:
        train_quality = {
            mode: features
            for mode, features in quality_features.items()
            if mode != heldout_mode
        }
        train_targets = {
            mode: target_selection[mode]
            for mode in train_quality
            if mode in target_selection
        }
        if not train_quality or heldout_mode not in target_selection:
            continue
        router, _ = train_quality_budget_router(
            train_quality,
            train_targets,
            profile_budgets=profile_budgets,
            epochs=router_epochs,
            seed=router_seed,
        )
        prediction = predict_budget_profile_selection(
            router,
            {heldout_mode: quality_features[heldout_mode]},
        )
        predicted_selection[heldout_mode] = prediction[heldout_mode]

    report = evaluate_budget_profile_routing(
        profile_metrics,
        quality_features,
        profile_budgets=profile_budgets,
        selected_budgets_by_mode=predicted_selection,
    )
    correct = 0
    for mode, metrics in report["per_mode"].items():
        target_budget = float(target_selection[mode])
        selected_budget = float(metrics["selected_budget"])
        metrics["target_budget"] = target_budget
        if selected_budget == target_budget:
            correct += 1
    heldout_count = len(report["per_mode"])
    report["summary"]["heldout_modes"] = heldout_count
    report["summary"]["pareto_metric_tolerance"] = float(pareto_tolerance)
    report["summary"]["routing_accuracy_vs_pareto"] = correct / heldout_count if heldout_count else 0.0
    _add_tradeoff_diagnostics(report, profile_metrics)
    return report


def write_routing_csv(path: str | Path, report: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    per_mode = report["per_mode"]
    fieldnames = [
        "mode",
        "selected_budget",
        "target_budget",
        "oa",
        "expected_macs_ratio",
        "metric_regret",
        "resource_saving_vs_reference",
    ]
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
        "routing_accuracy_vs_pareto",
        "utility_resource_penalty",
        "pareto_metric_tolerance",
        "mean_metric_regret",
        "mean_resource_saving_vs_reference",
        "heldout_modes",
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
    parser.add_argument("--output-sweep-json")
    parser.add_argument("--output-sweep-csv")
    parser.add_argument("--pareto-tolerances", default="0,0.005,0.01,0.02,0.03")
    parser.add_argument("--utility-resource-penalties", default="0,0.05,0.1,0.15,0.2,0.3")
    parser.add_argument("--output-loo-json")
    parser.add_argument("--output-loo-csv")
    parser.add_argument("--loo-pareto-tolerance", type=float, default=0.01)
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
    if args.output_sweep_json or args.output_sweep_csv:
        sweep_reports = build_sweep_reports(
            profile_metrics,
            quality_features,
            pareto_tolerances=_parse_float_list(args.pareto_tolerances),
            utility_resource_penalties=_parse_float_list(args.utility_resource_penalties),
        )
        if args.output_sweep_json:
            write_routing_json(args.output_sweep_json, sweep_reports)
        if args.output_sweep_csv:
            write_comparison_csv(args.output_sweep_csv, sweep_reports)
    if args.output_loo_json or args.output_loo_csv:
        loo_report = build_leave_one_state_out_report(
            profile_metrics,
            quality_features,
            pareto_tolerance=args.loo_pareto_tolerance,
            router_epochs=args.router_epochs,
            router_seed=args.router_seed,
        )
        if args.output_loo_json:
            write_routing_json(args.output_loo_json, loo_report)
        if args.output_loo_csv:
            write_routing_csv(args.output_loo_csv, loo_report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
