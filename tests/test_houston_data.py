import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from brmnet_core.data.houston import (
    HOUSTON_CLASS_NAMES,
    HOUSTON_TRAIN_COUNTS,
    HoustonScene,
    _load_known_key,
    load_houston_scene,
)


HSI_FILENAME = "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat"
LIDAR_FILENAME = "2013_IEEE_GRSS_DF_Contest_LiDAR.mat"
GT_FILENAME = "GRSS2013.mat"
ROI_FILENAME = "2013_IEEE_GRSS_DF_Contest_Samples_TR.mat"


class HoustonDataTest(unittest.TestCase):
    def _write_required_files(
        self,
        root: Path,
        *,
        hsi: np.ndarray | None = None,
        lidar: np.ndarray | None = None,
        gt: np.ndarray | None = None,
    ) -> None:
        savemat(
            root / HSI_FILENAME,
            {"ans": np.ones((3, 4, 2), dtype=np.uint16) if hsi is None else hsi},
        )
        savemat(
            root / LIDAR_FILENAME,
            {"LiDAR_data": np.ones((3, 4), dtype=np.float64) if lidar is None else lidar},
        )
        savemat(
            root / GT_FILENAME,
            {
                "name": (
                    np.array(
                        [[0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 0, 0]],
                        dtype=np.uint8,
                    )
                    if gt is None
                    else gt
                )
            },
        )

    def test_loads_known_mat_keys_and_normalizes_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_files(root)

            scene = load_houston_scene(root, require_roi=False)

            self.assertIsInstance(scene, HoustonScene)
            self.assertEqual(scene.hsi.shape, (3, 4, 2))
            self.assertEqual(scene.lidar.shape, (3, 4, 1))
            self.assertEqual(scene.gt.shape, (3, 4))
            self.assertEqual(scene.hsi.dtype, np.float32)
            self.assertEqual(scene.lidar.dtype, np.float32)
            self.assertEqual(scene.gt.dtype, np.int64)
            self.assertIsNone(scene.roi_records)
            self.assertEqual(
                scene.source_paths,
                {
                    "hsi": root / HSI_FILENAME,
                    "lidar": root / LIDAR_FILENAME,
                    "gt": root / GT_FILENAME,
                },
            )

    def test_defines_official_counts_and_envi_class_names(self):
        self.assertEqual(sum(HOUSTON_TRAIN_COUNTS.values()), 2832)
        self.assertEqual(HOUSTON_TRAIN_COUNTS[1], 198)
        self.assertEqual(HOUSTON_TRAIN_COUNTS[15], 187)
        self.assertEqual(HOUSTON_CLASS_NAMES[1], "grass_healthy")
        self.assertEqual(HOUSTON_CLASS_NAMES[12], "parking_lot1")
        self.assertEqual(HOUSTON_CLASS_NAMES[15], "running_track")

    def test_missing_known_key_reports_expected_and_available_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / HSI_FILENAME
            savemat(path, {"wrong": np.ones((3, 4, 2)), "other": np.zeros(1)})

            with self.assertRaises(KeyError) as context:
                _load_known_key(path, "ans")

            message = str(context.exception)
            self.assertIn("ans", message)
            self.assertIn("other", message)
            self.assertIn("wrong", message)
            self.assertNotIn("__header__", message)

    def test_rejects_spatial_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_files(
                root,
                lidar=np.ones((2, 4), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "spatial shapes"):
                load_houston_scene(root, require_roi=False)

    def test_require_roi_rejects_missing_roi_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_files(root)

            with self.assertRaisesRegex(FileNotFoundError, "Samples_TR"):
                load_houston_scene(root, require_roi=True)

    def test_loads_roi_records_when_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_files(root)
            records = np.array(["; ROI name: grass_healthy", "1 2 3"], dtype=object)
            savemat(root / ROI_FILENAME, {"TR_Samples": records})

            scene = load_houston_scene(root, require_roi=True)

            np.testing.assert_array_equal(scene.roi_records, records)
            self.assertEqual(scene.source_paths["roi"], root / ROI_FILENAME)

    def test_rejects_unexpected_array_dimensions(self):
        cases = (
            ("HSI", {"hsi": np.ones((3, 4), dtype=np.float32)}),
            ("LiDAR", {"lidar": np.ones((3,), dtype=np.float32)}),
            ("GT", {"gt": np.ones((3, 4, 2), dtype=np.int64)}),
        )
        for expected_name, overrides in cases:
            with self.subTest(array=expected_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_required_files(root, **overrides)

                with self.assertRaisesRegex(ValueError, expected_name):
                    load_houston_scene(root, require_roi=False)


if __name__ == "__main__":
    unittest.main()
