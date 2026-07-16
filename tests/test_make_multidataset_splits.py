import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from brmnet_core.data import load_coordinate_split
from scripts.make_multidataset_splits import main


class MakeMultidatasetSplitsTest(unittest.TestCase):
    def test_writes_trento_split_from_downloaded_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trento"
            root.mkdir()
            labels = np.array(
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                ],
                dtype=np.uint8,
            )
            savemat(root / "Italy_hsi.mat", {"data": np.ones((6, 6, 63), dtype=np.float32)})
            savemat(root / "Italy_lidar.mat", {"data": np.ones((6, 6, 2), dtype=np.float32)})
            savemat(root / "allgrd.mat", {"mask_test": labels})
            output = Path(tmpdir) / "split.npz"

            exit_code = main(
                [
                    "--dataset",
                    "trento",
                    "--root",
                    str(root),
                    "--train-counts",
                    "1:2,2:2,3:2,4:2,5:2,6:2",
                    "--seed",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            split = load_coordinate_split(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(split.seed, 3)
        self.assertEqual(len(split.train_coords), 12)
        self.assertEqual(len(split.test_coords), 24)

    def test_writes_muufl_split_from_scene_label_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "muufl"
            scene_dir = root / "MUUFLGulfportSceneLabels"
            scene_dir.mkdir(parents=True)
            labels = np.array(
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                ],
                dtype=np.int16,
            )
            savemat(
                scene_dir / "muufl_gulfport_campus_1_hsi_220_label.mat",
                {
                    "hsi": {
                        "Data": np.ones((6, 6, 64), dtype=np.float32),
                        "Lidar": np.ones((6, 6, 2), dtype=np.float32),
                        "sceneLabels": {"labels": labels},
                    }
                },
            )
            output = Path(tmpdir) / "muufl_split.npz"

            exit_code = main(
                [
                    "--dataset",
                    "muufl",
                    "--root",
                    str(root),
                    "--train-counts",
                    "1:2,2:2,3:2,4:2,5:2,6:2",
                    "--seed",
                    "4",
                    "--output",
                    str(output),
                ]
            )
            split = load_coordinate_split(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(split.seed, 4)
        self.assertEqual(len(split.train_coords), 12)
        self.assertEqual(len(split.test_coords), 24)


if __name__ == "__main__":
    unittest.main()
