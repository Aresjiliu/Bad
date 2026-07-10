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

    def test_core_matrix_includes_quality_supervised_reliability_variant(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        supervised = next(row for row in rows if row["variant"] == "quality_supervised")

        self.assertEqual(supervised["fusion_mode"], "reliability")
        self.assertEqual(supervised["lambda_quality"], 1.0)
        self.assertEqual(supervised["modality_dropout_prob"], 0.25)

        command = command_for_row(supervised, args)

        self.assertIn("--lambda-quality 1.0", command)

    def test_core_matrix_includes_quality_degradation_supervised_variant(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        supervised = next(row for row in rows if row["variant"] == "quality_degradation_supervised")

        self.assertEqual(supervised["fusion_mode"], "reliability")
        self.assertEqual(supervised["lambda_quality"], 1.0)
        self.assertEqual(supervised["aux_quality_degradation_prob"], 0.5)

        command = command_for_row(supervised, args)

        self.assertIn("--aux-quality-degradation-prob 0.5", command)

    def test_core_matrix_includes_multi_degradation_supervised_variant(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        supervised = next(row for row in rows if row["variant"] == "quality_multi_degradation_supervised")

        self.assertEqual(supervised["fusion_mode"], "reliability")
        self.assertEqual(supervised["lambda_quality"], 1.0)
        self.assertEqual(supervised["aux_quality_degradation_types"], "noise,downsample_4,occlusion_50")

        command = command_for_row(supervised, args)

        self.assertIn("--aux-quality-degradation-types 'noise,downsample_4,occlusion_50'", command)

    def test_core_matrix_includes_light_multi_degradation_variant(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        supervised = next(row for row in rows if row["variant"] == "quality_multi_degradation_p025")

        self.assertEqual(supervised["fusion_mode"], "reliability")
        self.assertEqual(supervised["lambda_quality"], 1.0)
        self.assertEqual(supervised["aux_quality_degradation_prob"], 0.25)
        self.assertEqual(supervised["aux_quality_degradation_types"], "noise,downsample_4,occlusion_50")

        command = command_for_row(supervised, args)

        self.assertIn("--aux-quality-degradation-prob 0.25", command)


if __name__ == "__main__":
    unittest.main()
