import csv
import tempfile
import unittest
from pathlib import Path

from brmnet_core.reporting import write_metrics_csv


class BRMNetReportingTest(unittest.TestCase):
    def test_write_metrics_csv_flattens_degradation_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.csv"
            write_metrics_csv(
                output_path,
                {
                    "full": {"loss": 1.2, "accuracy": 0.8, "samples": 10},
                    "main_only": {"loss": 1.4, "accuracy": 0.7, "samples": 10},
                },
            )

            with output_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["mode"], "full")
        self.assertEqual(rows[0]["accuracy"], "0.8")
        self.assertEqual(rows[1]["mode"], "main_only")
        self.assertEqual(rows[1]["samples"], "10")


if __name__ == "__main__":
    unittest.main()
