import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
