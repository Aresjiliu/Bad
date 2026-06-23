import unittest

from brmnet_core.legacy import MODALITY_CHANNELS, build_houston_args, infer_channels


class BRMNetLegacyTest(unittest.TestCase):
    def test_build_houston_args_matches_old_dataloader_contract(self):
        args = build_houston_args(
            data_root="D:/data/Huston2013",
            pair_modalities="hsi+lidar",
            batch_size=8,
            patch_size=7,
            num_workers=0,
        )

        self.assertEqual(args.data_root, "D:/data/Huston2013")
        self.assertEqual(args.pair_modalities, ["hsi", "lidar"])
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.patch_size, 7)
        self.assertEqual(args.num_workers, 0)

    def test_infer_channels_uses_known_remote_sensing_modalities(self):
        self.assertEqual(MODALITY_CHANNELS["hsi"], 144)
        self.assertEqual(infer_channels(["hsi", "lidar"]), (144, 1))
        self.assertEqual(infer_channels("hsi+ms"), (144, 8))


if __name__ == "__main__":
    unittest.main()
