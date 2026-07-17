import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brmnet_core import BRMNet
from brmnet_core.budget_gates import iter_budget_gates, set_gate_stochastic
from brmnet_core.engine import (
    apply_aux_quality_degradation,
    apply_degradation,
    apply_modality_dropout,
    evaluate,
    evaluate_degradation_matrix,
    train_one_epoch,
    unpack_batch,
)
from brmnet_core.losses import brmnet_loss


class BRMNetEngineTest(unittest.TestCase):
    def test_set_gate_stochastic_updates_all_budget_gates(self):
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3)

        set_gate_stochastic(model, False)

        self.assertTrue(all(not gate.stochastic for gate in iter_budget_gates(model)))

    def test_unpack_batch_accepts_tuple_and_dict(self):
        main = torch.randn(2, 4, 7, 7)
        aux = torch.randn(2, 1, 7, 7)
        labels = torch.tensor([0, 1])

        tuple_batch = unpack_batch((main, aux, labels), torch.device("cpu"))
        dict_batch = unpack_batch({"main": main, "aux": aux, "label": labels}, torch.device("cpu"))
        legacy_batch = unpack_batch({"m_1": main, "m_2": aux, "label": labels}, torch.device("cpu"))

        self.assertTrue(torch.equal(tuple_batch.main, main))
        self.assertTrue(torch.equal(tuple_batch.aux, aux))
        self.assertTrue(torch.equal(tuple_batch.labels, labels))
        self.assertTrue(torch.equal(dict_batch.main, main))
        self.assertTrue(torch.equal(dict_batch.aux, aux))
        self.assertTrue(torch.equal(dict_batch.labels, labels))
        self.assertTrue(torch.equal(legacy_batch.main, main))
        self.assertTrue(torch.equal(legacy_batch.aux, aux))
        self.assertTrue(torch.equal(legacy_batch.labels, labels))

    def test_apply_degradation_sets_availability_mask(self):
        batch = unpack_batch(
            (
                torch.ones(2, 4, 7, 7),
                torch.ones(2, 1, 7, 7),
                torch.tensor([0, 1]),
            ),
            torch.device("cpu"),
        )

        main_only = apply_degradation(batch, degradation="main_only")
        aux_only = apply_degradation(batch, degradation="aux_only")

        torch.testing.assert_close(main_only.availability_mask, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        torch.testing.assert_close(aux_only.availability_mask, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        self.assertTrue(torch.equal(main_only.aux, torch.zeros_like(batch.aux)))
        self.assertTrue(torch.equal(aux_only.main, torch.zeros_like(batch.main)))

    def test_apply_degradation_generates_quality_targets_for_p0_modes(self):
        batch = unpack_batch(
            (
                torch.ones(2, 4, 8, 8),
                torch.arange(1, 129, dtype=torch.float32).reshape(2, 1, 8, 8),
                torch.tensor([0, 1]),
            ),
            torch.device("cpu"),
        )

        noisy = apply_degradation(batch, degradation="aux_noise_mid")
        downsampled = apply_degradation(batch, degradation="aux_downsample_2")
        occluded = apply_degradation(batch, degradation="aux_occlusion_50")

        torch.testing.assert_close(noisy.quality_targets[0], torch.ones(2, 1))
        torch.testing.assert_close(noisy.quality_targets[1], torch.full((2, 1), 0.5))
        torch.testing.assert_close(downsampled.quality_targets[1], torch.full((2, 1), 0.5))
        torch.testing.assert_close(occluded.quality_targets[1], torch.full((2, 1), 0.5))
        self.assertFalse(torch.equal(downsampled.aux, batch.aux))
        self.assertEqual(float((occluded.aux == 0).float().mean()), 0.5)

    def test_aux_noise_degradation_uses_monotonic_noise_levels(self):
        batch = unpack_batch(
            (
                torch.ones(4, 4, 8, 8),
                torch.zeros(4, 1, 8, 8),
                torch.tensor([0, 1, 2, 1]),
            ),
            torch.device("cpu"),
        )

        means = []
        for mode in ("aux_noise_low", "aux_noise_mid", "aux_noise_high"):
            torch.manual_seed(7)
            degraded = apply_degradation(batch, degradation=mode, aux_noise_std=0.1)
            means.append(float(degraded.aux.abs().mean()))

        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])

    def test_apply_modality_dropout_drops_one_modality_per_selected_sample(self):
        batch = unpack_batch(
            (
                torch.ones(6, 4, 7, 7),
                torch.ones(6, 1, 7, 7),
                torch.tensor([0, 1, 2, 0, 1, 2]),
            ),
            torch.device("cpu"),
        )

        dropped = apply_modality_dropout(
            batch,
            probability=1.0,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertTrue(torch.equal(dropped.availability_mask.sum(dim=1), torch.ones(6)))
        dropped_main = dropped.availability_mask[:, 0] == 0
        dropped_aux = dropped.availability_mask[:, 1] == 0
        self.assertTrue(torch.equal(dropped.main[dropped_main], torch.zeros_like(dropped.main[dropped_main])))
        self.assertTrue(torch.equal(dropped.aux[dropped_aux], torch.zeros_like(dropped.aux[dropped_aux])))

    def test_apply_modality_dropout_generates_quality_targets_from_availability(self):
        batch = unpack_batch(
            (
                torch.ones(6, 4, 7, 7),
                torch.ones(6, 1, 7, 7),
                torch.tensor([0, 1, 2, 0, 1, 2]),
            ),
            torch.device("cpu"),
        )

        dropped = apply_modality_dropout(
            batch,
            probability=1.0,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertIsNotNone(dropped.quality_targets)
        torch.testing.assert_close(dropped.quality_targets[0], dropped.availability_mask[:, 0:1])
        torch.testing.assert_close(dropped.quality_targets[1], dropped.availability_mask[:, 1:2])

    def test_apply_modality_dropout_masks_existing_quality_targets(self):
        batch = unpack_batch(
            (
                torch.ones(6, 4, 7, 7),
                torch.ones(6, 1, 7, 7),
                torch.tensor([0, 1, 2, 0, 1, 2]),
                torch.full((6, 1), 0.9),
                torch.full((6, 1), 0.8),
            ),
            torch.device("cpu"),
        )

        dropped = apply_modality_dropout(
            batch,
            probability=1.0,
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(dropped.quality_targets[0], torch.full((6, 1), 0.9) * dropped.availability_mask[:, 0:1])
        torch.testing.assert_close(dropped.quality_targets[1], torch.full((6, 1), 0.8) * dropped.availability_mask[:, 1:2])

    def test_apply_aux_quality_degradation_keeps_aux_available_but_lowers_quality_target(self):
        batch = unpack_batch(
            (
                torch.ones(3, 4, 7, 7),
                torch.zeros(3, 1, 7, 7),
                torch.tensor([0, 1, 2]),
            ),
            torch.device("cpu"),
        )

        degraded = apply_aux_quality_degradation(
            batch,
            probability=1.0,
            aux_noise_std=0.2,
            aux_quality_target=0.4,
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(degraded.availability_mask, torch.ones(3, 2))
        torch.testing.assert_close(degraded.quality_targets[0], torch.ones(3, 1))
        torch.testing.assert_close(degraded.quality_targets[1], torch.full((3, 1), 0.4))
        self.assertFalse(torch.equal(degraded.aux, batch.aux))

    def test_apply_aux_quality_degradation_respects_unavailable_aux_mask(self):
        batch = unpack_batch(
            {
                "main": torch.ones(2, 4, 7, 7),
                "aux": torch.ones(2, 1, 7, 7),
                "labels": torch.tensor([0, 1]),
                "availability_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            },
            torch.device("cpu"),
        )

        degraded = apply_aux_quality_degradation(
            batch,
            probability=1.0,
            aux_noise_std=0.2,
            aux_quality_target=0.4,
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(degraded.availability_mask, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        torch.testing.assert_close(degraded.quality_targets[1], torch.tensor([[0.0], [0.4]]))

    def test_apply_aux_quality_degradation_supports_resolution_loss_type(self):
        aux = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
        batch = unpack_batch(
            (
                torch.ones(2, 4, 8, 8),
                aux,
                torch.tensor([0, 1]),
            ),
            torch.device("cpu"),
        )

        degraded = apply_aux_quality_degradation(
            batch,
            probability=1.0,
            degradation_types=("downsample_4",),
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(degraded.availability_mask, torch.ones(2, 2))
        torch.testing.assert_close(degraded.quality_targets[1], torch.full((2, 1), 0.25))
        self.assertFalse(torch.equal(degraded.aux, batch.aux))

    def test_apply_aux_quality_degradation_supports_occlusion_type(self):
        batch = unpack_batch(
            (
                torch.ones(2, 4, 8, 8),
                torch.ones(2, 1, 8, 8),
                torch.tensor([0, 1]),
            ),
            torch.device("cpu"),
        )

        degraded = apply_aux_quality_degradation(
            batch,
            probability=1.0,
            degradation_types=("occlusion_50",),
            generator=torch.Generator().manual_seed(0),
        )

        torch.testing.assert_close(degraded.quality_targets[1], torch.full((2, 1), 0.5))
        self.assertTrue(torch.all(degraded.aux[:, :, :4, :] == 0.0))
        self.assertTrue(torch.all(degraded.aux[:, :, 4:, :] == 1.0))

    def test_train_one_epoch_updates_model_and_reports_metrics(self):
        torch.manual_seed(0)
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3, init_score=0.5)
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 4, 7, 7),
                torch.randn(4, 1, 7, 7),
                torch.tensor([0, 1, 2, 1]),
            ),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = next(model.parameters()).detach().clone()

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            loss_kwargs={"target_budget": 0.8, "lambda_budget": 1.0},
        )

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("soft_retention", metrics)
        self.assertIn("hard_retention", metrics)
        self.assertAlmostEqual(metrics["target_budget"], 0.8)
        self.assertGreater(metrics["samples"], 0)
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))

    def test_train_one_epoch_reports_hard_concrete_resource_metrics(self):
        torch.manual_seed(0)
        model = BRMNet(
            main_channels=4,
            aux_channels=1,
            num_classes=3,
            gate_type="hard_concrete",
            initial_retention=0.8,
        )
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 4, 7, 7),
                torch.randn(4, 1, 7, 7),
                torch.tensor([0, 1, 2, 1]),
            ),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            loss_kwargs={
                "target_budget": 0.8,
                "lambda_budget": 1.0,
                "patch_size": 7,
                "budget_metric": "macs",
            },
        )

        self.assertGreater(metrics["resource_ratio"], 0.0)
        self.assertGreater(metrics["expected_params_ratio"], 0.0)
        self.assertGreater(metrics["expected_macs_ratio"], 0.0)
        self.assertAlmostEqual(metrics["target_budget"], 0.8)

    def test_brmnet_loss_accepts_class_weights(self):
        class NoGateModel(nn.Module):
            pass

        outputs = {
            "logits": torch.tensor(
                [
                    [3.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 3.0],
                ]
            )
        }
        labels = torch.tensor([0, 1, 1])

        unweighted = brmnet_loss(NoGateModel(), outputs, labels, lambda_budget=0.0)["cls"]
        weighted = brmnet_loss(
            NoGateModel(),
            outputs,
            labels,
            lambda_budget=0.0,
            class_weights=torch.tensor([1.0, 4.0]),
        )["cls"]

        self.assertGreater(float(weighted), float(unweighted))

    def test_evaluate_supports_degradation_modes(self):
        torch.manual_seed(0)
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3, init_score=0.5)
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 4, 7, 7),
                torch.randn(4, 1, 7, 7),
                torch.tensor([0, 1, 2, 1]),
            ),
            batch_size=2,
        )

        for mode in ("full", "main_only", "aux_only", "aux_noise"):
            metrics = evaluate(model, loader, torch.device("cpu"), degradation=mode, aux_noise_std=0.1)
            self.assertIn("loss", metrics)
            self.assertIn("accuracy", metrics)
            self.assertEqual(metrics["samples"], 4)

    def test_evaluate_reports_quality_and_routing_statistics(self):
        class FixedRoutingModel(nn.Module):
            def forward(self, main, aux, availability_mask=None):
                batch_size = main.shape[0]
                weights = torch.tensor([[0.25, 0.75]], dtype=main.dtype).repeat(batch_size, 1)
                return {
                    "logits": torch.ones(batch_size, 3),
                    "q_main": torch.full((batch_size, 1), 0.25),
                    "q_aux": torch.full((batch_size, 1), 0.75),
                    "fusion_weights": weights,
                }

        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 4, 7, 7),
                torch.randn(4, 1, 7, 7),
                torch.tensor([0, 1, 2, 1]),
            ),
            batch_size=2,
        )

        metrics = evaluate(FixedRoutingModel(), loader, torch.device("cpu"))

        self.assertAlmostEqual(metrics["q_main"], 0.25)
        self.assertAlmostEqual(metrics["q_aux"], 0.75)
        self.assertAlmostEqual(metrics["fusion_weight_main"], 0.25)
        self.assertAlmostEqual(metrics["fusion_weight_aux"], 0.75)

    def test_evaluate_degradation_matrix_returns_report_modes(self):
        torch.manual_seed(0)
        model = BRMNet(main_channels=4, aux_channels=1, num_classes=3, init_score=0.5)
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 4, 7, 7),
                torch.randn(4, 1, 7, 7),
                torch.tensor([0, 1, 2, 1]),
            ),
            batch_size=2,
        )

        matrix = evaluate_degradation_matrix(model, loader, torch.device("cpu"), aux_noise_std=0.1)

        self.assertEqual(
            set(matrix),
            {
                "full",
                "main_only",
                "aux_only",
                "aux_noise",
                "aux_noise_low",
                "aux_noise_mid",
                "aux_noise_high",
                "aux_downsample_2",
                "aux_downsample_4",
                "aux_occlusion_25",
                "aux_occlusion_50",
            },
        )
        self.assertTrue(all(metrics["samples"] == 4 for metrics in matrix.values()))

    def test_evaluate_reports_remote_sensing_classification_metrics(self):
        class FixedModel(nn.Module):
            def forward(self, main, aux):
                return {
                    "logits": main[:, :, 0, 0],
                    "quality": torch.ones(len(main), 2),
                }

        logits = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 2.0, 3.0],
                [0.0, 2.0, 3.0],
            ]
        ).reshape(4, 3, 1, 1)
        loader = DataLoader(
            TensorDataset(
                logits,
                torch.zeros(4, 1, 1, 1),
                torch.tensor([0, 1, 1, 2]),
            ),
            batch_size=2,
        )

        metrics = evaluate(FixedModel(), loader, torch.device("cpu"))

        self.assertAlmostEqual(metrics["oa"], 0.75)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["aa"], (1.0 + 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(metrics["kappa"], 0.6363636363)
        self.assertEqual(metrics["class_accuracy"], [1.0, 0.5, 1.0])
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        )


if __name__ == "__main__":
    unittest.main()
