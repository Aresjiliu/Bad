import unittest

import torch
from torch import nn

from brmnet_core.projection import FeatureProjection


class FeatureProjectionTest(unittest.TestCase):
    def test_projection_maps_branch_features_to_common_width(self):
        projection = FeatureProjection(in_channels=12, out_channels=8)

        outputs = projection(torch.randn(2, 12, 4, 4))

        self.assertEqual(outputs.shape, (2, 8, 4, 4))

    def test_equal_width_projection_has_no_learned_adapter(self):
        projection = FeatureProjection(in_channels=8, out_channels=8)

        self.assertIsInstance(projection.net, nn.Identity)
        self.assertEqual(sum(parameter.numel() for parameter in projection.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
