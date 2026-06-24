import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from brmnet_core.data import (
    CoordinateSplit as PublicCoordinateSplit,
    HOUSTON_CLASS_NAMES,
    HOUSTON_TRAIN_COUNTS,
    build_official_split as public_build_official_split,
    build_random_split as public_build_random_split,
    load_houston_scene,
    load_coordinate_split as public_load_coordinate_split,
    parse_envi_roi_records as public_parse_envi_roi_records,
    save_coordinate_split as public_save_coordinate_split,
)
from brmnet_core.data.splits import (
    CoordinateSplit,
    build_official_split,
    build_random_split,
    load_coordinate_split,
    parse_envi_roi_records,
    save_coordinate_split,
)


class CoordinateSplitTest(unittest.TestCase):
    def test_normalizes_arrays_to_copied_read_only_int64(self):
        train_coords = np.array([[0, 1]], dtype=np.int32)
        train_labels = np.array([1], dtype=np.int16)
        split = CoordinateSplit(
            protocol="official",
            seed=None,
            train_coords=train_coords,
            train_labels=train_labels,
            test_coords=np.array([[1, 0]], dtype=np.int32),
            test_labels=np.array([2], dtype=np.int16),
        )

        train_coords[0, 0] = 9
        train_labels[0] = 9

        self.assertEqual(split.train_coords.dtype, np.int64)
        self.assertEqual(split.train_labels.dtype, np.int64)
        np.testing.assert_array_equal(split.train_coords, [[0, 1]])
        np.testing.assert_array_equal(split.train_labels, [1])
        self.assertFalse(split.train_coords.flags.writeable)
        self.assertFalse(split.train_labels.flags.writeable)
        self.assertFalse(split.test_coords.flags.writeable)
        self.assertFalse(split.test_labels.flags.writeable)
        with self.assertRaises(ValueError):
            split.train_coords[0, 0] = 3

    def test_array_writeability_cannot_be_restored(self):
        split = CoordinateSplit(
            protocol="official",
            seed=None,
            train_coords=np.array([[0, 0]]),
            train_labels=np.array([1]),
            test_coords=np.array([[0, 1]]),
            test_labels=np.array([1]),
        )

        for field in (
            "train_coords",
            "train_labels",
            "test_coords",
            "test_labels",
        ):
            with self.subTest(field=field):
                array = getattr(split, field)
                with self.assertRaises(ValueError):
                    array.setflags(write=True)

    def test_rejects_invalid_shapes_lengths_and_protocol(self):
        valid = {
            "protocol": "random",
            "seed": 7,
            "train_coords": np.array([[0, 0]]),
            "train_labels": np.array([1]),
            "test_coords": np.array([[0, 1]]),
            "test_labels": np.array([1]),
        }
        invalid_cases = (
            ("protocol", {"protocol": "legacy"}),
            ("shape", {"train_coords": np.array([0, 0])}),
            ("length", {"test_labels": np.array([], dtype=np.int64)}),
            ("non-negative", {"test_coords": np.array([[-1, 0]])}),
        )
        for message, overrides in invalid_cases:
            with self.subTest(message=message):
                values = dict(valid)
                values.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    CoordinateSplit(**values)

    def test_normalizes_integer_seed_and_rejects_invalid_seed_type(self):
        split = CoordinateSplit(
            protocol="random",
            seed=np.int64(7),
            train_coords=np.array([[0, 0]]),
            train_labels=np.array([1]),
            test_coords=np.array([[0, 1]]),
            test_labels=np.array([1]),
        )
        self.assertEqual(split.seed, 7)
        self.assertIs(type(split.seed), int)

        with self.assertRaisesRegex(ValueError, "seed"):
            CoordinateSplit(
                protocol="random",
                seed="7",
                train_coords=np.array([[0, 0]]),
                train_labels=np.array([1]),
                test_coords=np.array([[0, 1]]),
                test_labels=np.array([1]),
            )

    def test_rejects_duplicate_coordinates_and_train_test_overlap(self):
        with self.assertRaisesRegex(ValueError, "duplicate.*train"):
            CoordinateSplit(
                protocol="random",
                seed=1,
                train_coords=np.array([[0, 0], [0, 0]]),
                train_labels=np.array([1, 1]),
                test_coords=np.array([[1, 0]]),
                test_labels=np.array([1]),
            )

        with self.assertRaisesRegex(ValueError, "overlap"):
            CoordinateSplit(
                protocol="official",
                seed=None,
                train_coords=np.array([[0, 0]]),
                train_labels=np.array([1]),
                test_coords=np.array([[0, 0]]),
                test_labels=np.array([1]),
            )


class RandomSplitTest(unittest.TestCase):
    def test_is_deterministic_class_balanced_and_class_ordered(self):
        gt = np.array(
            [
                [2, 2, 2, 2, 0],
                [1, 1, 1, 1, 0],
            ]
        )
        counts = {2: 2, 1: 2}

        first = build_random_split(gt, counts, seed=42)
        second = build_random_split(gt, counts, seed=42)

        self.assertEqual(first.protocol, "random")
        self.assertEqual(first.seed, 42)
        np.testing.assert_array_equal(first.train_coords, second.train_coords)
        np.testing.assert_array_equal(first.test_coords, second.test_coords)
        np.testing.assert_array_equal(first.train_labels, [1, 1, 2, 2])
        np.testing.assert_array_equal(first.test_labels, [1, 1, 2, 2])

    def test_does_not_modify_numpy_global_rng(self):
        gt = np.array([[1, 1, 1], [2, 2, 2]])
        np.random.seed(123)
        expected = np.random.random(4)

        np.random.seed(123)
        build_random_split(gt, {1: 1, 2: 1}, seed=9)
        actual = np.random.random(4)

        np.testing.assert_array_equal(actual, expected)

    def test_rejects_boolean_seed(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            build_random_split(
                np.array([[1, 1]]),
                {1: 1},
                seed=True,
            )

    def test_rejects_invalid_gt_class_counts_and_insufficient_samples(self):
        valid_gt = np.array([[1, 1], [2, 2]])
        invalid_cases = (
            ("2-D", np.array([1, 1]), {1: 1}),
            ("class id", valid_gt, {0: 1}),
            ("class id", valid_gt, {3: 1}),
            ("count", valid_gt, {1: 0}),
            ("count", valid_gt, {1: 1.5}),
            ("samples", valid_gt, {1: 3}),
        )
        for message, gt, counts in invalid_cases:
            with self.subTest(message=message, counts=counts):
                with self.assertRaisesRegex(ValueError, message):
                    build_random_split(gt, counts, seed=1)


class OfficialSplitTest(unittest.TestCase):
    @staticmethod
    def _records() -> np.ndarray:
        records = np.empty(11, dtype=object)
        records[0] = ";2013_IEEE_GRSS_DF_Contest_Samples_TR"
        records[1] = np.array([], dtype=np.float64)
        records[2] = np.array(
            [";            ", "ROI          ", "name:        ", "grass_healthy"]
        )
        records[3] = np.array([";    ", "ROI  ", "npts:", "2  "])
        records[4] = np.array([";  ", "ID ", "X  ", "Y  ", "Lat", "Lon"])
        records[5] = np.array(["1 ", "2 ", "1 ", "29.7", "-95.3"])
        records[6] = np.array(["2 ", "3 ", "1 ", "29.7", "-95.3"])
        records[7] = np.array(
            [";", "ROI", "name:", "grass_stressed"], dtype="<U16"
        )
        records[8] = np.array([";", "ROI", "npts:", "1"])
        records[9] = np.array([";", "ID", "X", "Y", "Lat", "Lon"])
        records[10] = np.array(["1", "1", "2", "0", "0"])
        return records

    def test_parser_returns_raw_one_based_xy_and_labels(self):
        coords, labels = parse_envi_roi_records(
            self._records(),
            {1: "grass_healthy", 2: "grass_stressed"},
        )

        np.testing.assert_array_equal(coords, [[2, 1], [3, 1], [1, 2]])
        np.testing.assert_array_equal(labels, [1, 1, 2])
        self.assertEqual(coords.dtype, np.int64)
        self.assertEqual(labels.dtype, np.int64)

    def test_official_split_converts_xy_validates_gt_and_builds_test_set(self):
        gt = np.array(
            [
                [1, 1, 1],
                [2, 2, 0],
            ]
        )

        split = build_official_split(
            gt,
            self._records(),
            {1: "grass_healthy", 2: "grass_stressed"},
            train_counts={1: 2, 2: 1},
        )

        self.assertEqual(split.protocol, "official")
        self.assertIsNone(split.seed)
        np.testing.assert_array_equal(split.train_coords, [[0, 1], [0, 2], [1, 0]])
        np.testing.assert_array_equal(split.train_labels, [1, 1, 2])
        np.testing.assert_array_equal(split.test_coords, [[0, 0], [1, 1]])
        np.testing.assert_array_equal(split.test_labels, [1, 2])

    def test_parser_rejects_unknown_duplicate_mismatched_and_contextless_records(self):
        unknown = self._records()
        unknown[2] = np.array([";", "ROI", "name:", "not_a_class"])

        duplicate = self._records()
        duplicate[7] = np.array([";", "ROI", "name:", "grass_healthy"])

        mismatched = self._records()
        mismatched[3] = np.array([";", "ROI", "npts:", "3"])

        contextless = np.empty(1, dtype=object)
        contextless[0] = np.array(["1", "1", "1", "0", "0"])

        cases = (
            ("unknown ROI", unknown),
            ("duplicate ROI", duplicate),
            ("npts", mismatched),
            ("without ROI context", contextless),
        )
        for message, records in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_envi_roi_records(
                        records,
                        {1: "grass_healthy", 2: "grass_stressed"},
                    )

    def test_official_split_rejects_gt_mismatch_bounds_and_expected_counts(self):
        valid_gt = np.array([[1, 1, 1], [2, 2, 0]])
        mismatch_gt = valid_gt.copy()
        mismatch_gt[0, 1] = 2

        out_of_bounds = self._records()
        out_of_bounds[5] = np.array(["1", "4", "1", "0", "0"])

        cases = (
            ("GT label mismatch", mismatch_gt, self._records(), None),
            ("out of bounds", valid_gt, out_of_bounds, None),
            (
                "expected 3",
                valid_gt,
                self._records(),
                {1: 3, 2: 1},
            ),
        )
        for message, gt, records, counts in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_official_split(
                        gt,
                        records,
                        {1: "grass_healthy", 2: "grass_stressed"},
                        train_counts=counts,
                    )

    def test_official_split_rejects_incomplete_or_invalid_expected_counts(self):
        gt = np.array([[1, 1, 1], [2, 2, 0]])
        invalid_counts = (
            ("class ids", {1: 2}),
            ("class ids", {1: 2, 2: 1, 3: 1}),
            ("positive integer", {1: 2, 2: 0}),
        )
        for message, counts in invalid_counts:
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(ValueError, message):
                    build_official_split(
                        gt,
                        self._records(),
                        {1: "grass_healthy", 2: "grass_stressed"},
                        train_counts=counts,
                    )


class CoordinateSplitPersistenceTest(unittest.TestCase):
    def test_public_api_exports_split_operations(self):
        self.assertIs(PublicCoordinateSplit, CoordinateSplit)
        self.assertIs(public_build_official_split, build_official_split)
        self.assertIs(public_build_random_split, build_random_split)
        self.assertIs(public_load_coordinate_split, load_coordinate_split)
        self.assertIs(public_parse_envi_roi_records, parse_envi_roi_records)
        self.assertIs(public_save_coordinate_split, save_coordinate_split)

    def test_round_trip_preserves_fields_dtype_and_read_only_arrays(self):
        splits = (
            CoordinateSplit(
                protocol="official",
                seed=None,
                train_coords=np.array([[0, 0]]),
                train_labels=np.array([1]),
                test_coords=np.array([[0, 1]]),
                test_labels=np.array([1]),
            ),
            CoordinateSplit(
                protocol="random",
                seed=42,
                train_coords=np.array([[1, 0]]),
                train_labels=np.array([2]),
                test_coords=np.array([[1, 1]]),
                test_labels=np.array([2]),
            ),
            CoordinateSplit(
                protocol="random",
                seed=-1,
                train_coords=np.array([[2, 0]]),
                train_labels=np.array([3]),
                test_coords=np.array([[2, 1]]),
                test_labels=np.array([3]),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, split in enumerate(splits):
                with self.subTest(protocol=split.protocol, seed=split.seed):
                    path = Path(tmp) / str(index) / "split.npz"
                    save_coordinate_split(path, split)
                    loaded = load_coordinate_split(path)

                    self.assertTrue(path.is_file())
                    with np.load(path, allow_pickle=False) as data:
                        self.assertIn("seed_is_none", data.files)
                        self.assertEqual(
                            bool(data["seed_is_none"].item()),
                            split.seed is None,
                        )
                    self.assertEqual(loaded.protocol, split.protocol)
                    self.assertEqual(loaded.seed, split.seed)
                    for field in (
                        "train_coords",
                        "train_labels",
                        "test_coords",
                        "test_labels",
                    ):
                        actual = getattr(loaded, field)
                        np.testing.assert_array_equal(actual, getattr(split, field))
                        self.assertEqual(actual.dtype, np.int64)
                        self.assertFalse(actual.flags.writeable)

    def test_load_reapplies_coordinate_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            np.savez_compressed(
                path,
                protocol=np.array("official"),
                seed=np.array(0, dtype=np.int64),
                seed_is_none=np.array(True, dtype=np.bool_),
                train_coords=np.array([[0, 0]], dtype=np.int64),
                train_labels=np.array([1], dtype=np.int64),
                test_coords=np.array([[0, 0]], dtype=np.int64),
                test_labels=np.array([1], dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "overlap"):
                load_coordinate_split(path)


class HoustonRealDataIntegrationTest(unittest.TestCase):
    def test_official_split_matches_houston_dataset_counts(self):
        data_root = os.environ.get("BRMNET_HOUSTON_DATA_ROOT")
        if not data_root:
            self.skipTest("BRMNET_HOUSTON_DATA_ROOT is not set")

        scene = load_houston_scene(data_root)
        split = build_official_split(
            scene.gt,
            scene.roi_records,
            HOUSTON_CLASS_NAMES,
            HOUSTON_TRAIN_COUNTS,
        )

        self.assertEqual(int(np.count_nonzero(scene.gt)), 15029)
        self.assertEqual(len(split.train_coords), 2832)
        self.assertEqual(len(split.test_coords), 12197)
        actual_counts = {
            class_id: int(np.count_nonzero(split.train_labels == class_id))
            for class_id in HOUSTON_TRAIN_COUNTS
        }
        self.assertEqual(actual_counts, HOUSTON_TRAIN_COUNTS)


if __name__ == "__main__":
    unittest.main()
