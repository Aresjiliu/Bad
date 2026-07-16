import tempfile
import unittest
from pathlib import Path

import numpy as np

from brmnet_core.data import (
    CoordinateSplit,
    build_trento_raw_loaders,
    save_coordinate_split,
)
from brmnet_core.data.trento import TrentoScene


class TrentoFactoryTest(unittest.TestCase):
    def test_builds_trento_loaders_from_fixed_split_file(self):
        hsi = np.ones((5, 6, 63), dtype=np.float32)
        aux = np.ones((5, 6, 1), dtype=np.float32)
        gt = np.zeros((5, 6), dtype=np.int64)
        gt[:2, :3] = 1
        gt[2:4, :3] = 2
        split = CoordinateSplit(
            protocol="random",
            seed=7,
            train_coords=np.array([[0, 0], [2, 0]], dtype=np.int64),
            train_labels=np.array([1, 2], dtype=np.int64),
            test_coords=np.array([[0, 1], [2, 1]], dtype=np.int64),
            test_labels=np.array([1, 2], dtype=np.int64),
        )
        scene = TrentoScene(
            hsi=hsi,
            aux=aux,
            gt=gt,
            source_paths={},
            aux_channel_mode="first",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.npz"
            save_coordinate_split(split_path, split)
            bundle = build_trento_raw_loaders(
                scene=scene,
                split_seed=7,
                patch_size=3,
                batch_size=2,
                num_workers=0,
                split_file=split_path,
            )

        batch = next(iter(bundle.train_loader))
        self.assertEqual(batch["m_1"].shape[1:], (63, 3, 3))
        self.assertEqual(batch["m_2"].shape[1:], (1, 3, 3))
        self.assertEqual(bundle.metadata["dataset"], "trento")
        self.assertEqual(bundle.metadata["train_samples"], 2)
        self.assertEqual(bundle.metadata["test_samples"], 2)
        self.assertEqual(bundle.metadata["aux_channel_mode"], "first")


if __name__ == "__main__":
    unittest.main()
