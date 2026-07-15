import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_budget_profile_routing import (
    build_routing_reports,
    build_sweep_reports,
    load_profile_metrics,
    load_quality_features,
    main,
    write_comparison_csv,
)


class EvaluateBudgetProfileRoutingScriptTest(unittest.TestCase):
    def test_loads_profile_metrics_from_budget_specs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps({"full": {"oa": 0.9}}), encoding="utf-8")

            metrics = load_profile_metrics([f"1.0:{metrics_path}"])

        self.assertEqual(metrics[1.0]["full"]["oa"], 0.9)

    def test_load_profile_metrics_backfills_compact_resource_ratios(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "compact_metrics.json"
            metrics_path.write_text(
                json.dumps({"full": {"oa": 0.9, "expected_macs_ratio": 0.0}}),
                encoding="utf-8",
            )
            (root / "resource_stats.json").write_text(
                json.dumps(
                    {
                        "compact": {"params_ratio": 0.7, "macs_ratio": 0.65},
                        "state_dependent": {
                            "compact": {
                                "full": {"global_params_ratio": 0.7, "global_macs_ratio": 0.65}
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = load_profile_metrics([f"0.65:{metrics_path}"])

        self.assertEqual(metrics[0.65]["full"]["expected_macs_ratio"], 0.65)
        self.assertEqual(metrics[0.65]["full"]["expected_params_ratio"], 0.7)

    def test_loads_quality_features_from_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quality.json"
            path.write_text(
                json.dumps(
                    {
                        "full": {
                            "pre_q_main": 0.9,
                            "pre_q_aux": 0.8,
                            "pre_u_main": 0.1,
                            "pre_u_aux": 0.2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            features = load_quality_features(path)

        self.assertEqual(features["full"], [0.9, 0.8, 0.1, 0.2])

    def test_main_writes_json_and_csv_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quality_path = root / "quality.json"
            quality_path.write_text(
                json.dumps(
                    {
                        "full": {
                            "pre_q_main": 0.9,
                            "pre_q_aux": 0.9,
                            "pre_u_main": 0.1,
                            "pre_u_aux": 0.1,
                        },
                        "main_only": {
                            "pre_q_main": 0.9,
                            "pre_q_aux": 0.0,
                            "pre_u_main": 0.1,
                            "pre_u_aux": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            profile_specs = []
            for budget, full_oa, main_oa in ((0.65, 0.8, 0.6), (0.8, 0.85, 0.62), (1.0, 0.9, 0.64)):
                path = root / f"profile_{budget}.json"
                path.write_text(
                    json.dumps(
                        {
                            "full": {"oa": full_oa, "expected_macs_ratio": budget},
                            "main_only": {"oa": main_oa, "expected_macs_ratio": budget},
                        }
                    ),
                    encoding="utf-8",
                )
                profile_specs.extend(["--profile-metrics", f"{budget}:{path}"])
            output_json = root / "routing.json"
            output_csv = root / "routing.csv"

            exit_code = main(
                [
                    *profile_specs,
                    "--quality-metrics",
                    str(quality_path),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ]
            )

            report = json.loads(output_json.read_text(encoding="utf-8"))
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["per_mode"]["full"]["selected_budget"], 1.0)
        self.assertEqual(report["per_mode"]["main_only"]["selected_budget"], 0.65)
        self.assertEqual(rows[0]["mode"], "full")

    def test_main_writes_sweep_reports_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quality_path = root / "quality.json"
            quality_path.write_text(
                json.dumps(
                    {
                        "full": {
                            "pre_q_main": 0.9,
                            "pre_q_aux": 0.9,
                            "pre_u_main": 0.1,
                            "pre_u_aux": 0.1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            profile_specs = []
            for budget, full_oa in ((0.65, 0.80), (0.8, 0.895), (1.0, 0.900)):
                path = root / f"profile_{budget}.json"
                path.write_text(
                    json.dumps({"full": {"oa": full_oa, "expected_macs_ratio": budget}}),
                    encoding="utf-8",
                )
                profile_specs.extend(["--profile-metrics", f"{budget}:{path}"])
            sweep_json = root / "sweep.json"
            sweep_csv = root / "sweep.csv"

            exit_code = main(
                [
                    *profile_specs,
                    "--quality-metrics",
                    str(quality_path),
                    "--output-json",
                    str(root / "routing.json"),
                    "--output-csv",
                    str(root / "routing.csv"),
                    "--output-sweep-json",
                    str(sweep_json),
                    "--output-sweep-csv",
                    str(sweep_csv),
                    "--pareto-tolerances",
                    "0,0.01",
                    "--utility-resource-penalties",
                    "0,0.2",
                ]
            )

            reports = json.loads(sweep_json.read_text(encoding="utf-8"))
            with sweep_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertIn("pareto_delta_0.01", reports)
        self.assertTrue(any(row["policy"] == "utility_lambda_0.2" for row in rows))

    def test_builds_static_oracle_and_learned_comparison_reports(self):
        quality_features = {
            "full": [0.95, 0.90, 0.05, 0.10],
            "aux_noise": [0.70, 0.45, 0.25, 0.55],
            "main_only": [0.95, 0.00, 0.05, 1.00],
        }
        profile_metrics = {}
        for budget in (0.65, 0.8, 1.0):
            profile_metrics[budget] = {
                "full": {"oa": 0.70 + budget / 10.0, "expected_macs_ratio": budget},
                "aux_noise": {"oa": 0.65 + budget / 10.0, "expected_macs_ratio": budget},
                "main_only": {"oa": 0.60 + budget / 10.0, "expected_macs_ratio": budget},
            }

        reports = build_routing_reports(
            profile_metrics,
            quality_features,
            include_learned=True,
            router_epochs=120,
            router_seed=7,
            utility_resource_penalty=0.2,
        )

        self.assertEqual(
            set(reports),
            {"static_0.65", "static_0.8", "static_1.0", "oracle", "learned", "utility", "learned_utility"},
        )
        self.assertEqual(reports["oracle"]["per_mode"]["full"]["selected_budget"], 1.0)
        self.assertEqual(reports["learned"]["per_mode"]["full"]["selected_budget"], 1.0)
        self.assertIn("routing_accuracy_vs_utility", reports["learned_utility"]["summary"])
        self.assertIn("routing_accuracy_vs_oracle", reports["learned"]["summary"])

    def test_comparison_csv_keeps_utility_accuracy_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.csv"
            write_comparison_csv(
                path,
                {
                    "learned_utility": {
                        "summary": {
                            "mean_oa": 0.8,
                            "mean_selected_budget": 0.65,
                            "mean_expected_macs_ratio": 0.61,
                            "routing_accuracy_vs_utility": 0.9,
                            "modes": 11,
                        }
                    }
                },
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["routing_accuracy_vs_utility"], "0.9")

    def test_builds_pareto_and_utility_sweep_reports(self):
        quality_features = {
            "full": [0.95, 0.90, 0.05, 0.10],
            "aux_noise": [0.70, 0.45, 0.25, 0.55],
        }
        profile_metrics = {
            0.65: {
                "full": {"oa": 0.80, "expected_macs_ratio": 0.65},
                "aux_noise": {"oa": 0.70, "expected_macs_ratio": 0.61},
            },
            0.8: {
                "full": {"oa": 0.895, "expected_macs_ratio": 0.75},
                "aux_noise": {"oa": 0.76, "expected_macs_ratio": 0.74},
            },
            1.0: {
                "full": {"oa": 0.900, "expected_macs_ratio": 0.94},
                "aux_noise": {"oa": 0.80, "expected_macs_ratio": 0.93},
            },
        }

        reports = build_sweep_reports(
            profile_metrics,
            quality_features,
            pareto_tolerances=(0.0, 0.01),
            utility_resource_penalties=(0.0, 0.2),
        )

        self.assertEqual(
            set(reports),
            {"pareto_delta_0", "pareto_delta_0.01", "utility_lambda_0", "utility_lambda_0.2"},
        )
        self.assertEqual(reports["pareto_delta_0.01"]["per_mode"]["full"]["selected_budget"], 0.8)
        self.assertIn("mean_metric_regret", reports["pareto_delta_0.01"]["summary"])
        self.assertEqual(reports["utility_lambda_0"]["per_mode"]["full"]["selected_budget"], 1.0)

    def test_script_can_run_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quality_path = root / "quality.json"
            quality_path.write_text(
                json.dumps(
                    {
                        "full": {
                            "pre_q_main": 0.9,
                            "pre_q_aux": 0.9,
                            "pre_u_main": 0.1,
                            "pre_u_aux": 0.1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            profile_args = []
            for budget in (0.65, 0.8, 1.0):
                path = root / f"profile_{budget}.json"
                path.write_text(
                    json.dumps({"full": {"oa": 0.8 + budget / 10.0, "expected_macs_ratio": budget}}),
                    encoding="utf-8",
                )
                profile_args.extend(["--profile-metrics", f"{budget}:{path}"])
            output_json = root / "routing.json"
            output_csv = root / "routing.csv"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_budget_profile_routing.py",
                    *profile_args,
                    "--quality-metrics",
                    str(quality_path),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
