import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from brmnet_core.data.muufl import load_muufl_scene


def _hsi_struct(hsi: np.ndarray, lidar: np.ndarray, labels: np.ndarray):
    return {
        "Data": hsi,
        "Lidar": {
            "Intensity": lidar[..., 0],
            "Elevation": lidar[..., 1],
        },
        "sceneLabels": {
            "labels": labels,
        },
    }


class MUUFLDataTest(unittest.TestCase):
    def test_loads_scene_label_file_with_lidar_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir()
            hsi = np.ones((4, 5, 64), dtype=np.float32)
            lidar = np.stack(
                [
                    np.full((4, 5), 3, dtype=np.float32),
                    np.full((4, 5), 7, dtype=np.float32),
                ],
                axis=-1,
            )
            labels = np.arange(20, dtype=np.uint8).reshape(4, 5) % 12
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {"hsi": _hsi_struct(hsi, lidar, labels)},
            )

            first = load_muufl_scene(root, aux_channel_mode="first")
            both = load_muufl_scene(root, aux_channel_mode="both")
            mean = load_muufl_scene(root, aux_channel_mode="mean")

        self.assertEqual(first.hsi.shape, (4, 5, 64))
        self.assertEqual(first.aux.shape, (4, 5, 1))
        self.assertTrue(np.all(first.aux == 3))
        self.assertEqual(both.aux.shape, (4, 5, 2))
        self.assertEqual(mean.aux.shape, (4, 5, 1))
        self.assertTrue(np.all(mean.aux == 5))
        self.assertEqual(first.gt.shape, (4, 5))
        self.assertEqual(first.aux_channel_mode, "first")

    def test_loads_real_muufl_lidar_struct_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir()
            hsi = np.ones((4, 5, 64), dtype=np.float32)
            labels = np.arange(20, dtype=np.uint8).reshape(4, 5) % 12
            lidar = np.array(
                [
                    {
                        "x": np.arange(5, dtype=np.float32),
                        "y": np.arange(4, dtype=np.float32),
                        "z": np.full((4, 5), 3, dtype=np.float32),
                        "info": "first lidar raster",
                    },
                    {
                        "x": np.arange(5, dtype=np.float32),
                        "y": np.arange(4, dtype=np.float32),
                        "z": np.full((4, 5), 7, dtype=np.float32),
                        "info": "second lidar raster",
                    },
                ],
                dtype=object,
            )
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {"hsi": {"Data": hsi, "Lidar": lidar, "sceneLabels": {"labels": labels}}},
            )

            scene = load_muufl_scene(root, aux_channel_mode="both")

        self.assertEqual(scene.aux.shape, (4, 5, 2))
        self.assertTrue(np.all(scene.aux[..., 0] == 3))
        self.assertTrue(np.all(scene.aux[..., 1] == 7))

    def test_loads_real_muufl_lidar_z_cube_without_duplicate_structs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir()
            hsi = np.ones((4, 5, 64), dtype=np.float32)
            labels = np.arange(20, dtype=np.uint8).reshape(4, 5) % 12
            lidar_cube = np.stack(
                [
                    np.full((4, 5), 3, dtype=np.float32),
                    np.full((4, 5), 7, dtype=np.float32),
                ],
                axis=-1,
            )
            lidar = np.array(
                [
                    {
                        "x": np.arange(5, dtype=np.float32),
                        "y": np.arange(4, dtype=np.float32),
                        "z": lidar_cube,
                        "info": "first lidar cube",
                    },
                    {
                        "x": np.arange(5, dtype=np.float32),
                        "y": np.arange(4, dtype=np.float32),
                        "z": lidar_cube + 100,
                        "info": "duplicate lidar cube from source packaging",
                    },
                ],
                dtype=object,
            )
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {"hsi": {"Data": hsi, "Lidar": lidar, "sceneLabels": {"labels": labels}}},
            )

            scene = load_muufl_scene(root, aux_channel_mode="both")

        self.assertEqual(scene.aux.shape, (4, 5, 2))
        self.assertTrue(np.all(scene.aux[..., 0] == 3))
        self.assertTrue(np.all(scene.aux[..., 1] == 7))

    def test_rejects_mismatched_spatial_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir()
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {
                    "hsi": _hsi_struct(
                        np.ones((4, 5, 64), dtype=np.float32),
                        np.ones((4, 4, 2), dtype=np.float32),
                        np.ones((4, 5), dtype=np.uint8),
                    )
                },
            )

            with self.assertRaisesRegex(ValueError, "spatial shapes must match"):
                load_muufl_scene(root)

    def test_maps_unlabeled_negative_one_to_background_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir()
            labels = np.array(
                [
                    [-1, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, -1, 1, 2],
                    [3, 4, 5, 6, 7],
                ],
                dtype=np.int16,
            )
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {
                    "hsi": _hsi_struct(
                        np.ones((4, 5, 64), dtype=np.float32),
                        np.ones((4, 5, 2), dtype=np.float32),
                        labels,
                    )
                },
            )

            scene = load_muufl_scene(root)

        self.assertEqual(int(scene.gt.min()), 0)
        self.assertEqual(int(scene.gt.max()), 11)
        self.assertEqual(int((scene.gt == 0).sum()), 2)


if __name__ == "__main__":
    unittest.main()
