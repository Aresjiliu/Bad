import unittest

from brmnet_core.data.dataset_specs import (
    DATASET_SPECS,
    recommended_dataset_sequence,
)


class DatasetSpecsTest(unittest.TestCase):
    def test_registry_contains_priority_multimodal_datasets(self):
        self.assertIn("houston2013_hs_lidar", DATASET_SPECS)
        self.assertIn("trento", DATASET_SPECS)
        self.assertIn("muufl", DATASET_SPECS)

        trento = DATASET_SPECS["trento"]
        self.assertEqual(trento.hsi_bands, 63)
        self.assertEqual(trento.aux_bands, 1)
        self.assertEqual(trento.num_classes, 6)
        self.assertEqual(trento.priority, "p0")

        muufl = DATASET_SPECS["muufl"]
        self.assertEqual(muufl.aux_bands, 2)
        self.assertIn(64, muufl.accepted_hsi_bands)
        self.assertIn(72, muufl.accepted_hsi_bands)
        self.assertEqual(muufl.priority, "p1")

    def test_recommended_sequence_excludes_deferred_datasets_by_default(self):
        names = [spec.name for spec in recommended_dataset_sequence()]
        self.assertEqual(
            names,
            [
                "Houston2013-HS-LiDAR",
                "Trento",
                "MUUFL Gulfport",
            ],
        )

        with_deferred = [spec.name for spec in recommended_dataset_sequence(include_deferred=True)]
        self.assertIn("Augsburg", with_deferred)
        self.assertIn("Houston2018", with_deferred)


if __name__ == "__main__":
    unittest.main()
