import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat

from brmnet_core import BRMNet
from brmnet_core.data import CoordinateSplit, save_coordinate_split
from scripts.run_brmnet_houston import (
    build_parser,
    build_run_paths,
    compact_selection_score,
    inverse_frequency_class_weights,
    main,
    split_loader_for_validation,
    retention_to_gate_score,
    write_structured_pruning_artifacts,
)


class BRMNetHoustonRunnerTest(unittest.TestCase):
    def test_raw_data_arguments(self):
        args = build_parser().parse_args(
            [
                "--dataset",
                "houston2013",
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

        self.assertEqual(args.dataset, "houston2013")
        self.assertEqual(args.data_format, "raw")
        self.assertEqual(args.split_protocol, "random")
        self.assertEqual(args.split_seed, 9)
        self.assertEqual(args.split_file, "D:/splits/random.npz")
        self.assertTrue(args.dataset_only)
        self.assertEqual(args.num_workers, 0)

    def test_inverse_frequency_class_weights_normalize_present_classes(self):
        dataset = TensorDataset(
            torch.randn(6, 4, 3, 3),
            torch.randn(6, 1, 3, 3),
            torch.tensor([0, 0, 0, 1, 1, 2]),
        )

        weights = inverse_frequency_class_weights(dataset, num_classes=3)

        torch.testing.assert_close(weights, torch.tensor([6 / 11, 9 / 11, 18 / 11]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_trento_dataset_only_uses_trento_channels_and_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trento"
            root.mkdir()
            hsi = np.ones((5, 6, 63), dtype=np.float32)
            aux = np.ones((5, 6, 2), dtype=np.float32)
            gt = np.zeros((5, 6), dtype=np.uint8)
            gt[:2, :3] = 1
            gt[2:4, :3] = 2
            savemat(root / "Italy_hsi.mat", {"data": hsi})
            savemat(root / "Italy_lidar.mat", {"data": aux})
            savemat(root / "allgrd.mat", {"mask_test": gt})
            split = CoordinateSplit(
                protocol="random",
                seed=5,
                train_coords=np.array([[0, 0], [2, 0]], dtype=np.int64),
                train_labels=np.array([1, 2], dtype=np.int64),
                test_coords=np.array([[0, 1], [2, 1]], dtype=np.int64),
                test_labels=np.array([1, 2], dtype=np.int64),
            )
            split_path = Path(tmpdir) / "split.npz"
            save_coordinate_split(split_path, split)

            result = main(
                [
                    "--dataset",
                    "trento",
                    "--data-root",
                    str(root),
                    "--split-file",
                    str(split_path),
                    "--split-protocol",
                    "random",
                    "--class-num",
                    "6",
                    "--aux-channel-mode",
                    "first",
                    "--dataset-only",
                    "--latency-warmup",
                    "0",
                    "--latency-iterations",
                    "1",
                    "--output-dir",
                    str(Path(tmpdir) / "runs"),
                ]
            )

        self.assertEqual(result["dataset"]["dataset"], "trento")
        self.assertEqual(result["dataset"]["train_samples"], 2)
        self.assertEqual(result["dataset"]["test_samples"], 2)
        self.assertEqual(result["dataset"]["aux_channel_mode"], "first")
        self.assertEqual(result["channels"], [63, 1])
        self.assertTrue(Path(result["run_dir"]).name.startswith("trento_"))

    def test_muufl_dataset_only_uses_muufl_channels_and_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "muufl"
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir(parents=True)
            hsi = np.ones((5, 6, 64), dtype=np.float32)
            lidar = np.ones((5, 6, 2), dtype=np.float32)
            gt = np.zeros((5, 6), dtype=np.int16)
            gt[:2, :3] = 1
            gt[2:4, :3] = 2
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {
                    "hsi": {
                        "Data": hsi,
                        "Lidar": lidar,
                        "sceneLabels": {"labels": gt},
                    }
                },
            )
            split = CoordinateSplit(
                protocol="random",
                seed=5,
                train_coords=np.array([[0, 0], [2, 0]], dtype=np.int64),
                train_labels=np.array([1, 2], dtype=np.int64),
                test_coords=np.array([[0, 1], [2, 1]], dtype=np.int64),
                test_labels=np.array([1, 2], dtype=np.int64),
            )
            split_path = Path(tmpdir) / "muufl_split.npz"
            save_coordinate_split(split_path, split)

            result = main(
                [
                    "--dataset",
                    "muufl",
                    "--data-root",
                    str(root),
                    "--split-file",
                    str(split_path),
                    "--split-protocol",
                    "random",
                    "--class-num",
                    "11",
                    "--aux-channel-mode",
                    "both",
                    "--dataset-only",
                    "--latency-warmup",
                    "0",
                    "--latency-iterations",
                    "1",
                    "--output-dir",
                    str(Path(tmpdir) / "runs"),
                ]
            )

        self.assertEqual(result["dataset"]["dataset"], "muufl")
        self.assertEqual(result["dataset"]["train_samples"], 2)
        self.assertEqual(result["dataset"]["test_samples"], 2)
        self.assertEqual(result["dataset"]["aux_channel_mode"], "both")
        self.assertEqual(result["channels"], [64, 2])
        self.assertTrue(Path(result["run_dir"]).name.startswith("muufl_"))

    def test_raw_protocol_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.dataset, "houston2013")
        self.assertEqual(args.data_format, "raw")
        self.assertEqual(args.split_protocol, "official")
        self.assertEqual(args.split_seed, 42)
        self.assertEqual(args.output_dir, "output/experiments")
        self.assertEqual(args.gate_mode, "deterministic")
        self.assertEqual(args.gate_type, "hard_concrete")
        self.assertEqual(args.fusion_mode, "reliability")
        self.assertFalse(args.disable_fusion_availability_mask)
        self.assertEqual(args.budget_metric, "macs")
        self.assertIsNone(args.gate_threshold)
        self.assertEqual(args.compact_finetune_epochs, 10)
        self.assertEqual(args.target_budget, 1.0)
        self.assertEqual(args.lambda_budget, 1.0)
        self.assertEqual(args.lambda_pre_quality, 0.0)
        self.assertEqual(args.modality_dropout_prob, 0.0)
        self.assertEqual(args.min_active_ratio, 0.0)
        self.assertEqual(args.val_fraction, 0.1)
        self.assertEqual(args.val_split_strategy, "class_balanced")
        self.assertEqual(args.selection_metric, "oa")
        self.assertEqual(args.compact_val_fraction, 0.1)
        self.assertEqual(args.compact_val_split_strategy, "class_balanced")
        self.assertEqual(args.compact_selection_metric, "oa")
        self.assertIsNone(args.gate_init_retention)
        self.assertEqual(args.latency_warmup, 5)
        self.assertEqual(args.latency_iterations, 20)
        self.assertFalse(args.dataset_only)

    def test_modality_dropout_argument(self):
        args = build_parser().parse_args(["--modality-dropout-prob", "0.25"])

        self.assertEqual(args.modality_dropout_prob, 0.25)

    def test_pre_encoder_quality_arguments(self):
        args = build_parser().parse_args(["--lambda-pre-quality", "0.5", "--pre-encoder-quality-hidden", "8"])

        self.assertEqual(args.lambda_pre_quality, 0.5)
        self.assertEqual(args.pre_encoder_quality_hidden, 8)

    def test_aux_quality_degradation_arguments(self):
        args = build_parser().parse_args(
            [
                "--aux-quality-degradation-prob",
                "0.5",
                "--aux-quality-degradation-target",
                "0.4",
                "--aux-quality-degradation-types",
                "noise,downsample_4,occlusion_50",
            ]
        )

        self.assertEqual(args.aux_quality_degradation_prob, 0.5)
        self.assertEqual(args.aux_quality_degradation_target, 0.4)
        self.assertEqual(args.aux_quality_degradation_types, "noise,downsample_4,occlusion_50")

    def test_uniform_fusion_argument(self):
        args = build_parser().parse_args(["--fusion-mode", "uniform"])

        self.assertEqual(args.fusion_mode, "uniform")

    def test_disable_fusion_availability_mask_argument(self):
        args = build_parser().parse_args(["--disable-fusion-availability-mask"])

        self.assertTrue(args.disable_fusion_availability_mask)

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
                min_active_ratio=0.25,
                latency_warmup=0,
                latency_iterations=1,
            )

            self.assertTrue((Path(tmp) / "resource_stats.json").is_file())
            self.assertTrue((Path(tmp) / "compact_model.pt").is_file())
            self.assertTrue((Path(tmp) / "compact_config.json").is_file())
            for gate in stats["gates"]:
                self.assertGreaterEqual(
                    gate["active_channels"],
                    int(gate["total_channels"] * 0.25),
                )
            self.assertLess(stats["equivalence_max_abs_error"], 1e-5)
            self.assertLess(stats["equivalence_l2_relative_error"], 1e-5)
            for family in ("hard", "compact"):
                self.assertLess(
                    stats["state_dependent"][family]["main_only"]["macs_ratio"],
                    stats["state_dependent"][family]["full"]["macs_ratio"],
                )
                self.assertLess(
                    stats["state_dependent"][family]["aux_only"]["macs_ratio"],
                    stats["state_dependent"][family]["full"]["macs_ratio"],
                )
                for state in ("full", "main_only", "aux_only"):
                    self.assertGreaterEqual(
                        stats["latency_ms"][family][state]["mean"],
                        0.0,
                    )
                    self.assertEqual(stats["latency_ms"][family][state]["iterations"], 1)
            self.assertGreater(sum(parameter.numel() for parameter in compact.parameters()), 0)

    def test_split_loader_for_validation_uses_train_subset_and_deterministic_val_subset(self):
        dataset = TensorDataset(
            torch.arange(20).view(10, 2).float(),
            torch.arange(10).view(10, 1).float(),
            torch.arange(10),
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        train_loader, val_loader = split_loader_for_validation(loader, val_fraction=0.3, seed=7)
        _train_loader_again, val_loader_again = split_loader_for_validation(loader, val_fraction=0.3, seed=7)

        self.assertEqual(len(train_loader.dataset), 7)
        self.assertEqual(len(val_loader.dataset), 3)
        self.assertEqual(val_loader.dataset.indices, val_loader_again.dataset.indices)
        self.assertEqual(train_loader.batch_size, 4)
        self.assertFalse(val_loader.drop_last)

    def test_split_loader_for_validation_can_balance_classes(self):
        labels = torch.tensor([0] * 8 + [1] * 4 + [2] * 2)
        dataset = TensorDataset(
            torch.arange(len(labels)).view(-1, 1).float(),
            torch.arange(len(labels)).view(-1, 1).float(),
            labels,
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        train_loader, val_loader = split_loader_for_validation(
            loader,
            val_fraction=0.25,
            seed=3,
            strategy="class_balanced",
        )

        val_labels = labels[val_loader.dataset.indices]
        train_labels = labels[train_loader.dataset.indices]
        self.assertEqual(set(val_labels.tolist()), {0, 1, 2})
        self.assertEqual(len(val_loader.dataset), 4)
        self.assertTrue(all(label in train_labels.tolist() for label in (0, 1, 2)))

    def test_split_loader_for_validation_can_be_disabled(self):
        dataset = TensorDataset(torch.arange(6), torch.arange(6), torch.arange(6))
        loader = DataLoader(dataset, batch_size=2)

        train_loader, val_loader = split_loader_for_validation(loader, val_fraction=0.0, seed=0)

        self.assertIs(train_loader, loader)
        self.assertIsNone(val_loader)

    def test_compact_selection_score_supports_accuracy_metrics_and_loss(self):
        metrics = {"oa": 0.7, "accuracy": 0.6, "loss": 1.2}

        self.assertEqual(compact_selection_score(metrics, "oa"), 0.7)
        self.assertEqual(compact_selection_score(metrics, "accuracy"), 0.6)
        self.assertEqual(compact_selection_score(metrics, "loss"), -1.2)
        with self.assertRaisesRegex(KeyError, "kappa"):
            compact_selection_score(metrics, "kappa")


if __name__ == "__main__":
    unittest.main()
