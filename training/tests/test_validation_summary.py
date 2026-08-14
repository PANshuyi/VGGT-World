import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from train_utils.general import AverageMeter, distributed_average_meter_values


class DistributedValidationSummaryTest(unittest.TestCase):
    def test_uses_global_sum_and_count_instead_of_mean_of_rank_means(self):
        """模拟 rank 0 有 2 个样本、rank 1 有 1 个样本的情况。"""
        meter = AverageMeter("Loss/val_loss_objective")
        meter.update(2.0, n=2)  # rank 0: sum=4, count=2

        def add_remote_rank(statistics, op=None):
            statistics.add_(torch.tensor([[8.0, 1.0]], dtype=statistics.dtype))

        with (
            mock.patch("train_utils.general.dist.is_available", return_value=True),
            mock.patch("train_utils.general.dist.is_initialized", return_value=True),
            mock.patch("train_utils.general.dist.all_reduce", side_effect=add_remote_rank),
        ):
            result = distributed_average_meter_values(
                {meter.name: meter}, torch.device("cpu")
            )

        # 正确值是 (4 + 8) / (2 + 1) = 4，不是 (2 + 8) / 2 = 5。
        self.assertEqual(result[meter.name], 4.0)

    def test_skips_meter_without_any_observation(self):
        meter = AverageMeter("Loss/val_missing")
        result = distributed_average_meter_values(
            {meter.name: meter}, torch.device("cpu")
        )
        self.assertEqual(result, {})


class PerDatasetValidationSummaryTest(unittest.TestCase):
    def _build_trainer_without_runtime_setup(self):
        # 只测试纯统计方法，不初始化 DDP、模型或 TensorBoard。
        from trainer import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.loss = SimpleNamespace(
            camera={"weight": 5.0},
            depth={"weight": 2.0},
            point={"weight": 3.0},
        )
        return trainer

    def test_weighted_components_reconstruct_objective(self):
        trainer = self._build_trainer_without_runtime_setup()
        losses = {
            "loss_camera": torch.tensor(1.0),
            "loss_conf_depth": torch.tensor(2.0),
            "loss_reg_depth": torch.tensor(3.0),
            "loss_grad_depth": torch.tensor(4.0),
            "loss_conf_point": torch.tensor(5.0),
            "loss_reg_point": torch.tensor(6.0),
            "loss_grad_point": torch.tensor(7.0),
        }
        expected_objective = (
            losses["loss_camera"] * 5.0
            + sum(losses[key] for key in (
                "loss_conf_depth", "loss_reg_depth", "loss_grad_depth"
            )) * 2.0
            + sum(losses[key] for key in (
                "loss_conf_point", "loss_reg_point", "loss_grad_point"
            )) * 3.0
        )
        losses["objective"] = expected_objective

        totals = trainer._get_weighted_validation_loss_totals(losses)

        self.assertEqual(totals["camera"].item(), 5.0)
        self.assertEqual(totals["depth_total"].item(), 18.0)
        self.assertEqual(totals["point_total"].item(), 54.0)
        self.assertEqual(totals["objective"].item(), 77.0)
        self.assertEqual(
            totals["objective"].item(),
            totals["camera"].item()
            + totals["depth_total"].item()
            + totals["point_total"].item(),
        )

    def test_validation_row_keeps_exact_clip_id_and_its_own_dataset(self):
        trainer = self._build_trainer_without_runtime_setup()
        trainer._get_scalar_log_keys = lambda phase: [
            "loss_objective",
            "loss_camera",
        ]
        batch = {
            "dataset_name": ["kitti"],
            "scene_name": ["2011_09_26/drive_0001"],
            "camera_name": ["image_02"],
            "clip_start": torch.tensor([12]),
            "extrinsics": torch.zeros(1, 2, 3, 4),
        }
        losses = {
            "objective": torch.tensor(4.0),
            "loss_camera": torch.tensor(0.2),
        }
        totals = {
            "objective": torch.tensor(4.0),
            "camera": torch.tensor(1.0),
            "depth_total": torch.tensor(2.0),
            "point_total": torch.tensor(1.0),
        }

        row = trainer._build_validation_loss_row(batch, losses, totals)

        self.assertEqual(
            row["eval_id"],
            "kitti|2011_09_26/drive_0001|image_02|12",
        )
        self.assertEqual(row["dataset_name"], "kitti")
        self.assertEqual(row["loss_objective"], 4.0)
        self.assertEqual(row["weighted_depth_total"], 2.0)


if __name__ == "__main__":
    unittest.main()
