import unittest

from scripts.make_brmnet_experiment_matrix import build_parser, command_for_row, experiment_rows


class BRMNetExperimentMatrixTest(unittest.TestCase):
    def test_core_matrix_includes_uniform_fusion_ablation(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        uniform = next(row for row in rows if row["variant"] == "without_reliability_uniform_fusion")

        self.assertEqual(uniform["fusion_mode"], "uniform")
        self.assertEqual(uniform["target_budget"], 0.8)
        self.assertEqual(uniform["modality_dropout_prob"], 0.25)
        self.assertEqual(uniform["lambda_budget"], 1.0)

        command = command_for_row(uniform, args)

        self.assertIn("--fusion-mode uniform", command)


if __name__ == "__main__":
    unittest.main()
