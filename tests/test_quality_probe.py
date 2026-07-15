import unittest

import torch

from brmnet_core.losses import brmnet_loss, pre_encoder_quality_loss
from brmnet_core.quality_probe import PreEncoderQualityProbe


class PreEncoderQualityProbeTest(unittest.TestCase):
    def test_probe_returns_bounded_scores_for_both_modalities(self):
        probe = PreEncoderQualityProbe(main_channels=4, aux_channels=1, hidden_channels=4)

        outputs = probe(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))

        self.assertEqual(set(outputs), {"pre_q_main", "pre_q_aux", "pre_u_main", "pre_u_aux"})
        for value in outputs.values():
            self.assertEqual(value.shape, (2, 1))
            self.assertTrue(bool(torch.all((0.0 <= value) & (value <= 1.0))))

    def test_probe_marks_unavailable_modalities_as_low_quality_and_high_uncertainty(self):
        probe = PreEncoderQualityProbe(main_channels=4, aux_channels=1, hidden_channels=4)
        availability_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        outputs = probe(
            torch.randn(2, 4, 7, 7),
            torch.randn(2, 1, 7, 7),
            availability_mask=availability_mask,
        )

        torch.testing.assert_close(outputs["pre_q_main"], torch.stack([outputs["pre_q_main"][0], torch.zeros(1)]))
        torch.testing.assert_close(outputs["pre_q_aux"], torch.stack([torch.zeros(1), outputs["pre_q_aux"][1]]))
        torch.testing.assert_close(outputs["pre_u_main"], torch.stack([outputs["pre_u_main"][0], torch.ones(1)]))
        torch.testing.assert_close(outputs["pre_u_aux"], torch.stack([torch.ones(1), outputs["pre_u_aux"][1]]))

    def test_pre_encoder_loss_supervises_quality_and_complementary_uncertainty(self):
        quality = torch.tensor([[0.9], [0.2]])
        uncertainty = torch.tensor([[0.1], [0.8]])
        targets = torch.tensor([[1.0], [0.0]])

        loss = pre_encoder_quality_loss(quality, quality, uncertainty, uncertainty, targets, targets)

        self.assertGreater(float(loss), 0.0)

    def test_brmnet_loss_includes_pre_encoder_quality_term_when_enabled(self):
        model = torch.nn.Linear(2, 2)
        labels = torch.tensor([0, 1])
        outputs = {
            "logits": torch.tensor([[2.0, 0.1], [0.2, 1.5]]),
            "pre_q_main": torch.tensor([[0.9], [0.2]]),
            "pre_q_aux": torch.tensor([[0.8], [0.3]]),
            "pre_u_main": torch.tensor([[0.1], [0.8]]),
            "pre_u_aux": torch.tensor([[0.2], [0.7]]),
        }
        quality_targets = (torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0], [0.0]]))

        losses = brmnet_loss(
            model,
            outputs,
            labels,
            lambda_budget=0.0,
            lambda_pre_quality=0.5,
            quality_targets=quality_targets,
        )

        self.assertIn("pre_quality", losses)
        self.assertGreater(float(losses["pre_quality"]), 0.0)
        torch.testing.assert_close(losses["total"], losses["cls"] + 0.5 * losses["pre_quality"])


if __name__ == "__main__":
    unittest.main()
