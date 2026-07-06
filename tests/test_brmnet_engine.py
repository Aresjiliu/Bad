import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from brmnet_core import BRMNet
from brmnet_core.budget_gates import iter_budget_gates, set_gate_stochastic
from brmnet_core.engine import (
    apply_degradation,
    apply_modality_dropout,
    evaluate,
    evaluate_degradation_matrix,
    train_one_epoch,
    unpack_batch,
)


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

        self.assertEqual(set(matrix), {"full", "main_only", "aux_only", "aux_noise"})
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
