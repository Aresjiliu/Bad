import tempfile
import unittest
from pathlib import Path

from scripts.run_brmnet_houston import build_parser, build_run_paths, retention_to_gate_score


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
            )

            self.assertEqual(
                paths["run_dir"].name,
                "houston2013_hsi-lidar_random_splitseed42_trainseed0_gatedeterministic_budget65",
            )
            self.assertEqual(paths["metrics"].name, "metrics.csv")
            self.assertEqual(paths["checkpoint"].name, "checkpoint.pt")
            self.assertEqual(paths["config"].name, "config.json")
            self.assertEqual(paths["history"].name, "history.json")
            self.assertEqual(paths["metrics_json"].name, "metrics.json")
            self.assertEqual(paths["run_dir"].parent, Path(tmp))

            deterministic = build_run_paths(
                output_dir=tmp,
                pair_modalities="hsi+lidar",
                protocol="random",
                split_seed=42,
                train_seed=1,
                gate_mode="deterministic",
                target_budget=0.9,
            )
            self.assertEqual(
                deterministic["run_dir"].name,
                "houston2013_hsi-lidar_random_splitseed42_trainseed1_gatedeterministic_budget90",
            )


if __name__ == "__main__":
    unittest.main()
