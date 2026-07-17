import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.plot_muufl_error_analysis import (
    aggregate_confusion,
    load_full_metrics,
    row_normalize,
    summarize_per_class,
)


class PlotMUUFLErrorAnalysisTest(unittest.TestCase):
    def _write_metrics(self, root: Path, variant: str, seed: int, class_accuracy, confusion):
        run_dir = root / variant / f"muufl_hsi-lidar_random_splitseed42_trainseed{seed}_budget"
        run_dir.mkdir(parents=True)
        payload = {
            "full": {
                "oa": 0.8,
                "aa": 0.75,
                "kappa": 0.7,
                "class_accuracy": class_accuracy,
                "confusion_matrix": confusion,
            }
        }
        (run_dir / "compact_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_load_and_summarize_per_class_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in range(3):
                self._write_metrics(root, "muufl_100_baseline", seed, [0.8, 0.6], [[8, 2], [4, 6]])
                self._write_metrics(root, "muufl_80_p025_multideg", seed, [0.7, 0.9], [[7, 3], [1, 9]])

            records = load_full_metrics(root)
            rows = summarize_per_class(records)

        self.assertEqual(len(records), 6)
        self.assertEqual(rows[0]["delta_80_minus_100"], -10.0)
        self.assertEqual(rows[1]["delta_80_minus_100"], 30.0)

    def test_aggregate_and_row_normalize_confusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in range(3):
                self._write_metrics(root, "muufl_100_baseline", seed, [0.8, 0.6], [[8, 2], [4, 6]])
                self._write_metrics(root, "muufl_80_p025_multideg", seed, [0.7, 0.9], [[7, 3], [1, 9]])

            records = load_full_metrics(root)
            matrix = aggregate_confusion(records, "muufl_80_p025_multideg")
            normalized = row_normalize(matrix)

        np.testing.assert_array_equal(matrix, np.asarray([[21, 9], [3, 27]], dtype=float))
        np.testing.assert_allclose(normalized, np.asarray([[0.7, 0.3], [0.1, 0.9]]))


if __name__ == "__main__":
    unittest.main()
