import csv
import tempfile
import unittest
from pathlib import Path

import json

from brmnet_core.reporting import write_metrics_csv, write_metrics_json


class BRMNetReportingTest(unittest.TestCase):
    def test_write_metrics_csv_flattens_degradation_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.csv"
            write_metrics_csv(
                output_path,
                {
                    "full": {"loss": 1.2, "accuracy": 0.8, "samples": 10},
                    "main_only": {
                        "loss": 1.4,
                        "accuracy": 0.7,
                        "samples": 10,
                        "class_accuracy": [0.8, 0.6],
                    },
                },
            )

            with output_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["mode"], "full")
        self.assertEqual(rows[0]["accuracy"], "0.8")
        self.assertEqual(rows[1]["mode"], "main_only")
        self.assertEqual(rows[1]["samples"], "10")
        self.assertEqual(json.loads(rows[1]["class_accuracy"]), [0.8, 0.6])

    def test_write_metrics_json_preserves_nested_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            metrics = {
                "full": {
                    "oa": 0.8,
                    "class_accuracy": [0.9, 0.7],
                    "confusion_matrix": [[9, 1], [3, 7]],
                }
            }

            write_metrics_json(output_path, metrics)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                metrics,
            )


if __name__ == "__main__":
    unittest.main()
