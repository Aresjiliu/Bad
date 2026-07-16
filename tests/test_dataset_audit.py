import unittest

import numpy as np

from brmnet_core.data.dataset_audit import (
    audit_label_map,
    recommend_dataset_usage,
)


class DatasetAuditTest(unittest.TestCase):
    def test_audits_label_map_ignoring_background_and_invalid_values(self):
        labels = np.array(
            [
                [0, 1, 1, 2],
                [-1, 2, 2, 3],
                [3, 3, 3, 3],
            ]
        )

        audit = audit_label_map(labels, background_values=(0, -1))

        self.assertEqual(audit["labeled_pixels"], 10)
        self.assertEqual(audit["class_count"], 3)
        self.assertEqual(audit["class_counts"], {1: 2, 2: 3, 3: 5})
        self.assertEqual(audit["imbalance_ratio"], 2.5)

    def test_recommends_smoke_then_full_when_split_is_missing(self):
        audit = {
            "class_count": 6,
            "labeled_pixels": 30000,
            "imbalance_ratio": 20.0,
            "empty_classes": [],
        }

        decision = recommend_dataset_usage(
            expected_classes=6,
            audit=audit,
            has_fixed_split=False,
        )

        self.assertEqual(decision["status"], "smoke_then_protocolize")
        self.assertIn("fixed train/test split is missing", decision["reasons"])

    def test_rejects_when_class_count_mismatches(self):
        audit = {
            "class_count": 5,
            "labeled_pixels": 30000,
            "imbalance_ratio": 3.0,
            "empty_classes": [],
        }

        decision = recommend_dataset_usage(
            expected_classes=6,
            audit=audit,
            has_fixed_split=True,
        )

        self.assertEqual(decision["status"], "do_not_use_yet")
        self.assertIn("class count mismatch: expected 6, found 5", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
