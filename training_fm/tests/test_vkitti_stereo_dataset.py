# 本次修改：用最小合成 VKITTI 目录验证双目 time-major 顺序、深度单位、标定读取和最终 tensor schema。
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from training_fm.vkitti_stereo_dataset import VKittiStereoDataset


# 本次修改：生成标准 SceneXX/clone RGB、depth、intrinsic.txt 和 extrinsic.txt，避免测试依赖真实远端 bucket。
def _write_synthetic_sequence(root: Path, scene_name: str):
    sequence_root = root / scene_name / "clone"
    for modality in ("rgb", "depth"):
        for camera_id in (0, 1):
            (sequence_root / "frames" / modality / f"Camera_{camera_id}").mkdir(
                parents=True, exist_ok=True
            )

    extrinsic_rows = []
    intrinsic_rows = []
    for frame_id in range(4):
        for camera_id in (0, 1):
            image = np.full(
                (24, 48, 3),
                20 + frame_id * 30 + camera_id * 10,
                dtype=np.uint8,
            )
            depth = np.full((24, 48), 1000 + frame_id * 100, dtype=np.uint16)
            cv2.imwrite(
                str(
                    sequence_root
                    / "frames"
                    / "rgb"
                    / f"Camera_{camera_id}"
                    / f"rgb_{frame_id:05d}.jpg"
                ),
                image,
            )
            cv2.imwrite(
                str(
                    sequence_root
                    / "frames"
                    / "depth"
                    / f"Camera_{camera_id}"
                    / f"depth_{frame_id:05d}.png"
                ),
                depth,
            )

            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[0, 3] = frame_id + camera_id * 0.5
            extrinsic_rows.append(
                [frame_id, camera_id, *extrinsic.reshape(-1).tolist()]
            )
            intrinsic_rows.append([frame_id, camera_id, 20.0, 20.0, 24.0, 12.0])

    np.savetxt(
        sequence_root / "extrinsic.txt",
        np.asarray(extrinsic_rows),
        header="frame cameraID r00 r01 r02 r03 r10 r11 r12 r13 r20 r21 r22 r23 r30 r31 r32 r33",
        comments="",
    )
    np.savetxt(
        sequence_root / "intrinsic.txt",
        np.asarray(intrinsic_rows),
        header="frame cameraID fx fy cx cy",
        comments="",
    )


# 本次修改：构造 source bucket 会注入的最小 common_config，固定为 2 目、2 历史、2 未来。
def _common_config():
    return SimpleNamespace(
        views_per_timestep=2,
        history_timesteps=2,
        future_timesteps=2,
        img_size=28,
        patch_size=7,
    )


# 本次修改：检查一个完整样本的 8 槽顺序和所有几何字段首维，防止相机优先或单目误标回归。
class VKittiStereoDatasetTest(unittest.TestCase):
    def test_time_major_stereo_window_and_metric_depth(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            _write_synthetic_sequence(root, "Scene01")
            dataset = VKittiStereoDataset(
                common_config=_common_config(),
                VKitti_DIR=str(root),
                split="train",
                train_scenes=["Scene01"],
                val_scenes=["Scene18"],
                conditions=["clone"],
                cameras=[0, 1],
                temporal_strides=[1],
            )

            sample = dataset[(0, 8, 0.5)]
            self.assertEqual(sample["images"].shape, (8, 3, 14, 28))
            self.assertEqual(sample["depths"].shape, (8, 14, 28))
            self.assertEqual(sample["extrinsics"].shape, (8, 3, 4))
            self.assertEqual(sample["intrinsics"].shape, (8, 3, 3))
            self.assertEqual(sample["cam_points"].shape, (8, 14, 28, 3))
            self.assertEqual(sample["world_points"].shape, (8, 14, 28, 3))
            self.assertEqual(sample["point_masks"].shape, (8, 14, 28))
            self.assertEqual(sample["frame_indices"].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
            self.assertEqual(sample["camera_ids"].tolist(), [0, 1, 0, 1, 0, 1, 0, 1])
            self.assertAlmostEqual(float(sample["depths"][0].mean()), 10.0)
            self.assertAlmostEqual(float(sample["depths"][6].mean()), 13.0)
            # 本次修改：adapter 必须向 batch sampler 暴露 stride 对应的合法窗口索引。
            self.assertEqual(
                dataset.get_temporal_stride_indices(),
                {1: (0,)},
            )


if __name__ == "__main__":
    unittest.main()
