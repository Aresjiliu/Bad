import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np

from brmnet_core.data.factory import (
    build_houston_raw_loaders,
    write_houston_data_artifacts,
)
from brmnet_core.data.houston import HoustonScene
from brmnet_core.data.splits import CoordinateSplit, save_coordinate_split


class HoustonFactoryTest(unittest.TestCase):
    @staticmethod
    def _scene(root: Path | None = None, roi_records=None) -> HoustonScene:
        source_paths = {}
        if root is not None:
            for name in ("hsi", "lidar", "gt"):
                path = root / f"{name}.mat"
                path.write_bytes(f"{name}-data".encode("ascii"))
                source_paths[name] = path
        return HoustonScene(
            hsi=np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2),
            lidar=np.arange(4 * 4, dtype=np.float32).reshape(4, 4, 1),
            gt=np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ],
                dtype=np.int64,
            ),
            roi_records=roi_records,
            source_paths=MappingProxyType(source_paths),
        )

    def test_builds_random_loaders_and_metadata(self):
        bundle = build_houston_raw_loaders(
            scene=self._scene(),
            protocol="random",
            split_seed=7,
            patch_size=3,
            batch_size=2,
            num_workers=0,
            train_counts={1: 1, 2: 1},
        )

        batch = next(iter(bundle.train_loader))

        self.assertEqual(tuple(batch["m_1"].shape[1:]), (2, 3, 3))
        self.assertEqual(tuple(batch["m_2"].shape[1:]), (1, 3, 3))
        self.assertEqual(bundle.split.protocol, "random")
        self.assertEqual(bundle.metadata["train_samples"], 2)
        self.assertEqual(bundle.metadata["test_samples"], 14)
        self.assertEqual(bundle.metadata["train_class_counts"], {"1": 1, "2": 1})
        self.assertEqual(bundle.metadata["test_class_counts"], {"1": 7, "2": 7})

    def test_official_protocol_requires_roi_records(self):
        with self.assertRaisesRegex(ValueError, "ROI"):
            build_houston_raw_loaders(
                scene=self._scene(),
                protocol="official",
                split_seed=42,
                patch_size=3,
                batch_size=2,
                num_workers=0,
                train_counts={1: 1, 2: 1},
            )

    def test_split_file_must_match_requested_protocol(self):
        split = CoordinateSplit(
            protocol="official",
            seed=None,
            train_coords=np.array([[0, 0]]),
            train_labels=np.array([1]),
            test_coords=np.array([[0, 1]]),
            test_labels=np.array([1]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.npz"
            save_coordinate_split(path, split)

            with self.assertRaisesRegex(ValueError, "protocol"):
                build_houston_raw_loaders(
                    scene=self._scene(),
                    protocol="random",
                    split_seed=7,
                    patch_size=3,
                    batch_size=2,
                    num_workers=0,
                    train_counts={1: 1, 2: 1},
                    split_file=path,
                )

    def test_writes_split_summary_normalization_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = self._scene(root)
            bundle = build_houston_raw_loaders(
                scene=scene,
                protocol="random",
                split_seed=7,
                patch_size=3,
                batch_size=2,
                num_workers=0,
                train_counts={1: 1, 2: 1},
            )

            paths = write_houston_data_artifacts(root / "run", bundle, scene)

            self.assertEqual(
                set(paths),
                {"split", "split_summary", "normalization", "data_fingerprint"},
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
            fingerprints = json.loads(paths["data_fingerprint"].read_text(encoding="utf-8"))
            self.assertEqual(summary["protocol"], "random")
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["train_samples"], 2)
            self.assertEqual(summary["test_samples"], 14)
            self.assertEqual(set(fingerprints["files"]), {"hsi", "lidar", "gt"})
            for details in fingerprints["files"].values():
                self.assertEqual(len(details["sha256"]), 64)
                self.assertGreater(details["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
