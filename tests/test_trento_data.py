import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from brmnet_core.data.trento import load_trento_scene


class TrentoDataTest(unittest.TestCase):
    def test_loads_trento_scene_with_aux_channel_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hsi = np.ones((4, 5, 63), dtype=np.float32)
            lidar = np.stack(
                [
                    np.full((4, 5), 2, dtype=np.float32),
                    np.full((4, 5), 4, dtype=np.float32),
                ],
                axis=-1,
            )
            labels = np.arange(20, dtype=np.uint8).reshape(4, 5) % 7
            savemat(root / "Italy_hsi.mat", {"data": hsi})
            savemat(root / "Italy_lidar.mat", {"data": lidar})
            savemat(root / "allgrd.mat", {"mask_test": labels})

            first = load_trento_scene(root, aux_channel_mode="first")
            both = load_trento_scene(root, aux_channel_mode="both")
            mean = load_trento_scene(root, aux_channel_mode="mean")

        self.assertEqual(first.hsi.shape, (4, 5, 63))
        self.assertEqual(first.aux.shape, (4, 5, 1))
        self.assertTrue(np.all(first.aux == 2))
        self.assertEqual(both.aux.shape, (4, 5, 2))
        self.assertEqual(mean.aux.shape, (4, 5, 1))
        self.assertTrue(np.all(mean.aux == 3))
        self.assertEqual(first.gt.shape, (4, 5))

    def test_rejects_spatial_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            savemat(root / "Italy_hsi.mat", {"data": np.ones((4, 5, 63), dtype=np.float32)})
            savemat(root / "Italy_lidar.mat", {"data": np.ones((4, 4, 2), dtype=np.float32)})
            savemat(root / "allgrd.mat", {"mask_test": np.ones((4, 5), dtype=np.uint8)})

            with self.assertRaisesRegex(ValueError, "spatial shapes must match"):
                load_trento_scene(root)


if __name__ == "__main__":
    unittest.main()
