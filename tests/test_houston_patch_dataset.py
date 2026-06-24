import unittest

import numpy as np
import torch

from brmnet_core.data.patch_dataset import HoustonPatchDataset, normalize_scene


class HoustonPatchDatasetTest(unittest.TestCase):
    def test_scene_normalization_uses_all_unlabeled_pixels(self):
        hsi = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
        lidar = np.array([[[2.0], [4.0]], [[6.0], [8.0]]], dtype=np.float32)

        normalized, stats = normalize_scene(hsi, lidar)

        self.assertAlmostEqual(float(normalized.hsi.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.hsi.std()), 1.0, places=6)
        self.assertAlmostEqual(float(normalized.lidar.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.lidar.std()), 1.0, places=6)
        self.assertEqual(stats.hsi_mean.shape, (1,))
        self.assertEqual(stats.hsi_std.shape, (1,))
        self.assertEqual(stats.lidar_mean.shape, (1,))
        self.assertEqual(stats.lidar_std.shape, (1,))
        self.assertEqual(normalized.hsi.dtype, np.float32)
        self.assertEqual(normalized.lidar.dtype, np.float32)

    def test_scene_normalization_replaces_zero_variance_standard_deviation(self):
        hsi = np.ones((2, 3, 2), dtype=np.float32)
        lidar = np.full((2, 3, 1), 4.0, dtype=np.float32)

        normalized, stats = normalize_scene(hsi, lidar)

        np.testing.assert_array_equal(normalized.hsi, np.zeros_like(hsi))
        np.testing.assert_array_equal(normalized.lidar, np.zeros_like(lidar))
        np.testing.assert_array_equal(stats.hsi_std, np.ones(2, dtype=np.float32))
        np.testing.assert_array_equal(stats.lidar_std, np.ones(1, dtype=np.float32))
        self.assertTrue(np.isfinite(normalized.hsi).all())
        self.assertTrue(np.isfinite(normalized.lidar).all())

    def test_extracts_reflection_padded_boundary_patch_and_zero_based_label(self):
        hsi = np.arange(3 * 3, dtype=np.float32).reshape(3, 3, 1)
        lidar = (hsi + 100).copy()
        dataset = HoustonPatchDataset(
            hsi,
            lidar,
            coords=np.array([[0, 0]]),
            labels=np.array([2]),
            patch_size=3,
            augment=False,
        )

        sample = dataset[0]

        self.assertEqual(tuple(sample["m_1"].shape), (1, 3, 3))
        self.assertEqual(tuple(sample["m_2"].shape), (1, 3, 3))
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["label"].dtype, torch.int64)
        self.assertEqual(sample["m_1"].dtype, torch.float32)
        self.assertEqual(sample["m_2"].dtype, torch.float32)
        torch.testing.assert_close(sample["coord"], torch.tensor([0, 0]))
        torch.testing.assert_close(
            sample["m_1"],
            torch.tensor([[[4.0, 3.0, 4.0], [1.0, 0.0, 1.0], [4.0, 3.0, 4.0]]]),
        )

    def test_rejects_invalid_patch_sizes_and_coordinate_contract(self):
        hsi = np.zeros((3, 3, 2), dtype=np.float32)
        lidar = np.zeros((3, 3, 1), dtype=np.float32)
        cases = (
            ("positive odd", {"patch_size": 0}),
            ("positive odd", {"patch_size": 2}),
            ("length", {"coords": np.array([[0, 0], [1, 1]])}),
            ("shape", {"coords": np.array([0, 0])}),
            ("bounds", {"coords": np.array([[3, 0]])}),
            ("labels", {"labels": np.array([0])}),
            ("coords.*numeric", {"coords": np.array([["x", "y"]])}),
            ("coords.*finite", {"coords": np.array([[np.nan, 0.0]])}),
            ("labels.*numeric", {"labels": np.array(["class-1"])}),
            ("labels.*finite", {"labels": np.array([np.inf])}),
        )
        for message, overrides in cases:
            with self.subTest(message=message, overrides=overrides):
                kwargs = {
                    "coords": np.array([[0, 0]]),
                    "labels": np.array([1]),
                    "patch_size": 3,
                }
                kwargs.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    HoustonPatchDataset(hsi, lidar, **kwargs)

    def test_augmentation_applies_identical_spatial_flips_to_both_modalities(self):
        hsi = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
        lidar = (hsi + 100).copy()
        dataset = HoustonPatchDataset(
            hsi,
            lidar,
            coords=np.array([[1, 1]]),
            labels=np.array([1]),
            patch_size=3,
            augment=True,
        )

        torch.manual_seed(3)
        sample = dataset[0]

        torch.testing.assert_close(sample["m_2"] - sample["m_1"], torch.full((1, 3, 3), 100.0))


if __name__ == "__main__":
    unittest.main()
