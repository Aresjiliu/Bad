import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_brmnet_experiments import summarize_experiments


class SummarizeBRMNetExperimentsTest(unittest.TestCase):
    def test_groups_runs_and_computes_mean_and_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, oa in ((0, 0.8), (1, 1.0)):
                run = root / f"run{seed}"
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps(
                        {
                            "split_protocol": "random",
                            "split_seed": 42,
                            "seed": seed,
                            "gate_mode": "deterministic",
                            "target_budget": 0.8,
                            "epochs": 20,
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "history.json").write_text(
                    json.dumps([{"elapsed_seconds": 2.0, "train": {"accuracy": 0.9}}]),
                    encoding="utf-8",
                )
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "full": {"oa": oa, "aa": oa - 0.1, "kappa": oa - 0.2},
                            "main_only": {"oa": 0.5, "aa": 0.4, "kappa": 0.3},
                        }
                    ),
                    encoding="utf-8",
                )
            duplicate = root / "duplicate-seed0"
            duplicate.mkdir()
            (duplicate / "config.json").write_text(
                json.dumps(
                    {
                        "split_protocol": "random",
                        "split_seed": 42,
                        "seed": 0,
                        "gate_mode": "deterministic",
                        "target_budget": 0.8,
                        "epochs": 20,
                    }
                ),
                encoding="utf-8",
            )
            (duplicate / "history.json").write_text(
                json.dumps([{"elapsed_seconds": 2.0, "train": {"accuracy": 0.9}}]),
                encoding="utf-8",
            )
            (duplicate / "metrics.json").write_text(
                json.dumps(
                    {
                        "full": {"oa": 0.8, "aa": 0.7, "kappa": 0.6},
                        "main_only": {"oa": 0.5, "aa": 0.4, "kappa": 0.3},
                    }
                ),
                encoding="utf-8",
            )

            rows = summarize_experiments(root)

        full = next(row for row in rows if row["mode"] == "full")
        self.assertEqual(full["protocol"], "random")
        self.assertEqual(full["gate_mode"], "deterministic")
        self.assertEqual(full["target_budget"], 0.8)
        self.assertEqual(full["runs"], 2)
        self.assertAlmostEqual(full["oa_mean"], 0.9)
        self.assertAlmostEqual(full["oa_std"], 0.1414213562)

    def test_does_not_mix_different_target_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for budget in (0.65, 0.9):
                run = root / f"budget-{budget}"
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps(
                        {
                            "split_protocol": "official",
                            "split_seed": 42,
                            "seed": 0,
                            "gate_mode": "deterministic",
                            "target_budget": budget,
                            "epochs": 20,
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "history.json").write_text(
                    json.dumps([{"elapsed_seconds": 1.0, "train": {"accuracy": 0.8}}]),
                    encoding="utf-8",
                )
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "full": {
                                "oa": budget,
                                "aa": budget,
                                "kappa": budget,
                                "soft_retention": budget,
                                "hard_retention": 1.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            rows = summarize_experiments(root)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["target_budget"] for row in rows}, {0.65, 0.9})


if __name__ == "__main__":
    unittest.main()
