import random
import unittest

from data.datasets.driving_parquet import (
    DrivingParquetDataset,
    normalize_available_camera_group_weights,
)


class CameraDirectionSamplingTest(unittest.TestCase):
    """验证前视优先采样的目标比例和组内均匀性。"""

    def test_full_surround_keeps_configured_weights(self):
        configured = {
            "front": 0.60,
            "front_side": 0.15,
            "side": 0.15,
            "rear": 0.10,
        }
        actual = normalize_available_camera_group_weights(configured, configured)
        for group, expected in configured.items():
            self.assertAlmostEqual(actual[group], expected)

    def test_missing_group_keeps_front_at_sixty_percent(self):
        configured = {
            "front": 0.60,
            "front_side": 0.15,
            "side": 0.15,
            "rear": 0.10,
        }
        actual = normalize_available_camera_group_weights(
            ["front", "front_side", "rear"], configured
        )
        self.assertAlmostEqual(actual["front"], 0.60)
        self.assertAlmostEqual(actual["front_side"], 0.24)
        self.assertAlmostEqual(actual["rear"], 0.16)
        self.assertAlmostEqual(sum(actual.values()), 1.0)

    def test_sampling_matches_weights_and_is_uniform_inside_group(self):
        dataset = object.__new__(DrivingParquetDataset)
        dataset.sequence_list = []
        dataset.active_camera_group_weights = {
            "front": 0.60,
            "front_side": 0.15,
            "side": 0.15,
            "rear": 0.10,
        }
        dataset.sequences_by_camera_group = {
            "front": {"CAM_F0": [("scene", "CAM_F0")]},
            "front_side": {
                "CAM_L0": [("scene", "CAM_L0")],
                "CAM_R0": [("scene", "CAM_R0")],
            },
            "side": {
                "CAM_L1": [("scene", "CAM_L1")],
                "CAM_R1": [("scene", "CAM_R1")],
            },
            "rear": {
                "CAM_L2": [("scene", "CAM_L2")],
                "CAM_B0": [("scene", "CAM_B0")],
                "CAM_R2": [("scene", "CAM_R2")],
            },
        }

        random.seed(17)
        counts: dict[str, int] = {}
        for _ in range(100_000):
            camera = dataset._sample_training_sequence_key()[1]
            counts[camera] = counts.get(camera, 0) + 1

        self.assertAlmostEqual(counts["CAM_F0"] / 100_000, 0.60, delta=0.01)
        self.assertAlmostEqual(counts["CAM_L0"] / 100_000, 0.075, delta=0.005)
        self.assertAlmostEqual(counts["CAM_R0"] / 100_000, 0.075, delta=0.005)
        self.assertAlmostEqual(counts["CAM_L1"] / 100_000, 0.075, delta=0.005)
        self.assertAlmostEqual(counts["CAM_R1"] / 100_000, 0.075, delta=0.005)
        self.assertAlmostEqual(counts["CAM_L2"] / 100_000, 0.10 / 3, delta=0.004)
        self.assertAlmostEqual(counts["CAM_B0"] / 100_000, 0.10 / 3, delta=0.004)
        self.assertAlmostEqual(counts["CAM_R2"] / 100_000, 0.10 / 3, delta=0.004)


if __name__ == "__main__":
    unittest.main()
