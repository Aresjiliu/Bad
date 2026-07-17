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

    def test_core_matrix_includes_fusion_availability_mask_ablation(self):
        args = build_parser().parse_args(["--ablation", "core", "--seeds", "0"])

        rows = experiment_rows(args)
        ablation = next(row for row in rows if row["variant"] == "without_fusion_availability_mask")

        self.assertTrue(ablation["disable_fusion_availability_mask"])
        self.assertEqual(ablation["fusion_mode"], "reliability")
        self.assertEqual(ablation["lambda_quality"], 1.0)
        self.assertEqual(ablation["aux_quality_degradation_prob"], 0.25)

        command = command_for_row(ablation, args)

        self.assertIn("--disable-fusion-availability-mask", command)

    def test_routing_profile_matrix_generates_fixed_budget_profiles(self):
        args = build_parser().parse_args(
            [
                "--ablation",
                "routing_profiles",
                "--seeds",
                "0",
                "--data-format",
                "legacy",
                "--python",
                "conda run -n hslinets python",
            ]
        )

        rows = experiment_rows(args)

        self.assertEqual([row["target_budget"] for row in rows], [0.65, 0.8, 1.0])
        self.assertEqual({row["variant"] for row in rows}, {"quality_routing_profile"})
        self.assertTrue(all(row["lambda_pre_quality"] == 0.5 for row in rows))
        self.assertTrue(all(row["aux_quality_degradation_prob"] == 0.25 for row in rows))

        command = command_for_row(rows[0], args)

        self.assertIn("--lambda-pre-quality 0.5", command)
        self.assertIn("--pre-encoder-quality-hidden 8", command)
        self.assertIn("--data-format legacy", command)
        self.assertTrue(command.startswith("conda run -n hslinets python scripts/run_brmnet_houston.py"))


if __name__ == "__main__":
    unittest.main()
