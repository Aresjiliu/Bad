import tempfile
import unittest
from pathlib import Path

import torch

from brmnet_core import BRMNet
from scripts.run_brmnet_houston import (
    build_parser,
    build_run_paths,
    retention_to_gate_score,
    write_structured_pruning_artifacts,
)


class BRMNetHoustonRunnerTest(unittest.TestCase):
    def test_raw_data_arguments(self):
        args = build_parser().parse_args(
            [
                "--data-format",
                "raw",
                "--data-root",
                "D:/data/Houston",
                "--split-protocol",
                "random",
                "--split-seed",
                "9",
                "--split-file",
                "D:/splits/random.npz",
                "--dataset-only",
                "--num-workers",
                "0",
            ]
        )

        self.assertEqual(args.data_format, "raw")
        self.assertEqual(args.split_protocol, "random")
        self.assertEqual(args.split_seed, 9)
        self.assertEqual(args.split_file, "D:/splits/random.npz")
        self.assertTrue(args.dataset_only)
        self.assertEqual(args.num_workers, 0)

    def test_raw_protocol_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.data_format, "raw")
        self.assertEqual(args.split_protocol, "official")
        self.assertEqual(args.split_seed, 42)
        self.assertEqual(args.output_dir, "output/experiments")
        self.assertEqual(args.gate_mode, "deterministic")
        self.assertEqual(args.gate_type, "hard_concrete")
        self.assertEqual(args.budget_metric, "macs")
        self.assertIsNone(args.gate_threshold)
        self.assertEqual(args.compact_finetune_epochs, 10)
        self.assertEqual(args.target_budget, 1.0)
        self.assertEqual(args.lambda_budget, 1.0)
        self.assertIsNone(args.gate_init_retention)
        self.assertFalse(args.dataset_only)

    def test_retention_to_gate_score_round_trips_probability(self):
        import math

        for retention in (0.65, 0.8, 0.9):
            score = retention_to_gate_score(retention)
            probability = 1.0 / (1.0 + math.exp(-score))
            self.assertAlmostEqual(probability, retention)

        self.assertGreater(retention_to_gate_score(1.0), 0.0)
        with self.assertRaisesRegex(ValueError, "retention"):
            retention_to_gate_score(0.0)

    def test_build_run_paths_names_protocol_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_run_paths(
                output_dir=tmp,
                pair_modalities="hsi+lidar",
                protocol="random",
                split_seed=42,
                train_seed=0,
                target_budget=0.65,
                gate_type="hard_concrete",
                budget_metric="macs",
            )

            self.assertEqual(
                paths["run_dir"].name,
                "houston2013_hsi-lidar_random_splitseed42_trainseed0_gatehard_concrete_budget65_metricmacs",
            )
            self.assertEqual(paths["metrics"].name, "metrics.csv")
            self.assertEqual(paths["checkpoint"].name, "checkpoint.pt")
            self.assertEqual(paths["config"].name, "config.json")
            self.assertEqual(paths["history"].name, "history.json")
            self.assertEqual(paths["metrics_json"].name, "metrics.json")
            self.assertEqual(paths["compact_metrics_json"].name, "compact_metrics.json")
            self.assertEqual(paths["resource_stats"].name, "resource_stats.json")
            self.assertEqual(paths["compact_model"].name, "compact_model.pt")
            self.assertEqual(paths["compact_history"].name, "compact_history.json")
            self.assertEqual(paths["run_dir"].parent, Path(tmp))

            deterministic = build_run_paths(
                output_dir=tmp,
                pair_modalities="hsi+lidar",
                protocol="random",
                split_seed=42,
                train_seed=1,
                gate_mode="deterministic",
                target_budget=0.9,
                gate_type="legacy_sigmoid",
                budget_metric="params",
            )
            self.assertEqual(
                deterministic["run_dir"].name,
                "houston2013_hsi-lidar_random_splitseed42_trainseed1_gatelegacy_sigmoid-deterministic_budget90_metricparams",
            )

    def test_structured_artifact_writer_creates_compact_outputs(self):
        model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            initial_retention=0.8,
        )
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            compact, stats = write_structured_pruning_artifacts(
                model=model,
                run_dir=Path(tmp),
                patch_size=7,
                threshold=0.5,
                sample_main=torch.randn(2, 4, 7, 7),
                sample_aux=torch.randn(2, 1, 7, 7),
            )

            self.assertTrue((Path(tmp) / "resource_stats.json").is_file())
            self.assertTrue((Path(tmp) / "compact_model.pt").is_file())
            self.assertTrue((Path(tmp) / "compact_config.json").is_file())
            self.assertLess(stats["equivalence_max_abs_error"], 1e-5)
            self.assertLess(stats["equivalence_l2_relative_error"], 1e-5)
            self.assertGreater(sum(parameter.numel() for parameter in compact.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
