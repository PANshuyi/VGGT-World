import unittest

import torch

from loss import prepare_mixed_metric_loss_inputs
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


class MetricScaleTest(unittest.TestCase):
    """验证米制固定倍率和非米制尺度不变分支的核心约定。"""

    def test_preprocess_only_scales_metric_sample(self):
        # 两个样本共用同一组几何，仅 is_metric 不同。
        extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(2, 2, 1, 1)[..., :3, :]
        extrinsics[:, 1, 0, 3] = 2.0
        world_points = torch.tensor([1.0, 0.0, 2.0]).view(1, 1, 1, 1, 3).repeat(2, 2, 1, 1, 1)
        cam_points = world_points.clone()
        depths = torch.full((2, 2, 1, 1), 2.0)
        masks = torch.ones((2, 2, 1, 1), dtype=torch.bool)

        out_ext, out_cam, out_world, out_depth = normalize_camera_extrinsics_and_points_batch(
            extrinsics=extrinsics,
            cam_points=cam_points,
            world_points=world_points,
            depths=depths,
            point_masks=masks,
            scale_mode="metric_mixed",
            is_metric=torch.tensor([True, False]),
            metric_scale_factor=0.1,
        )

        self.assertTrue(torch.allclose(out_world[0], world_points[0] * 0.1))
        self.assertTrue(torch.allclose(out_depth[0], depths[0] * 0.1))
        self.assertTrue(torch.allclose(out_cam[0], cam_points[0] * 0.1))
        self.assertAlmostEqual(out_ext[0, 1, 0, 3].item(), 0.2)

        self.assertTrue(torch.allclose(out_world[1], world_points[1]))
        self.assertTrue(torch.allclose(out_depth[1], depths[1]))
        self.assertTrue(torch.allclose(out_cam[1], cam_points[1]))
        self.assertAlmostEqual(out_ext[1, 1, 0, 3].item(), 2.0)

    def test_nonmetric_point_depth_and_translation_share_one_scale(self):
        masks = torch.ones((2, 1, 1, 2), dtype=torch.bool)
        # metric 样本应保持原数值；non-metric GT 点的平均距离为 2，
        # prediction 点的平均距离为 4。
        gt_points = torch.tensor(
            [
                [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
                [[[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]],
            ]
        ).reshape(2, 1, 1, 2, 3)
        pred_points = torch.tensor(
            [
                [[[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]],
                [[[[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]],
            ]
        ).reshape(2, 1, 1, 2, 3)

        gt_extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(2, 1, 1, 1)[..., :3, :]
        gt_extrinsics[:, :, 0, 3] = torch.tensor([0.5, 6.0]).view(2, 1)
        pred_pose = torch.zeros((2, 1, 9))
        pred_pose[:, :, 0] = torch.tensor([0.7, 8.0]).view(2, 1)

        batch = {
            "is_metric": torch.tensor([True, False]),
            "world_points": gt_points,
            "cam_points": gt_points.clone(),
            "depths": torch.tensor([1.0, 10.0]).view(2, 1, 1, 1),
            "extrinsics": gt_extrinsics,
            "point_masks": masks,
        }
        predictions = {
            "world_points": pred_points,
            "depth": torch.tensor([2.0, 20.0]).view(2, 1, 1, 1, 1),
            "pose_enc_list": [pred_pose],
            "pose_enc": pred_pose,
        }

        pred_loss, gt_loss = prepare_mixed_metric_loss_inputs(predictions, batch)

        # metric 样本直接使用固定倍率空间，loss 不再二次归一化。
        self.assertTrue(torch.equal(pred_loss["world_points"][0], pred_points[0]))
        self.assertTrue(torch.equal(gt_loss["depths"][0], batch["depths"][0]))

        # non-metric 样本的 point/depth/translation 分别共用
        # prediction scale=4 与 GT scale=2。
        self.assertTrue(torch.allclose(pred_loss["world_points"][1], pred_points[1] / 4.0))
        self.assertTrue(torch.allclose(pred_loss["depth"][1], predictions["depth"][1] / 4.0))
        self.assertAlmostEqual(pred_loss["pose_enc_list"][0][1, 0, 0].item(), 2.0)
        self.assertTrue(torch.allclose(gt_loss["world_points"][1], gt_points[1] / 2.0))
        self.assertTrue(torch.allclose(gt_loss["depths"][1], batch["depths"][1] / 2.0))
        self.assertAlmostEqual(gt_loss["extrinsics"][1, 0, 0, 3].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
