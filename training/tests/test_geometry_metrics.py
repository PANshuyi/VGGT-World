"""七数据集验证几何指标的回归测试。"""

import csv
import math

import torch

from train_utils.geometry_metrics import (
    GeometryMetricEvaluator,
    build_eval_id,
    deduplicate_metric_rows,
    summarize_metric_rows,
    write_metric_reports,
)


def _perfect_two_frame_batch():
    """构造一个处于训练 x0.1 空间的完美预测样本。"""
    batch_size, frames, height, width = 1, 2, 11, 10
    # 真实 2m，进入训练前固定乘 0.1，因此张量为 0.2。
    points = torch.zeros(batch_size, frames, height, width, 3)
    points[..., 0] = torch.linspace(-0.05, 0.05, width).view(1, 1, 1, width)
    points[..., 1] = torch.linspace(-0.05, 0.05, height).view(1, 1, height, 1)
    points[..., 2] = 0.2

    extrinsics = torch.eye(4).repeat(batch_size, frames, 1, 1)[..., :3, :]
    # W2C translation -0.1 对应真实相机中心向 +x 移动 1m。
    extrinsics[:, 1, 0, 3] = -0.1

    # GT camera points必须用各帧 W2C 将第一帧坐标系点投影回去。
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    cam_points = torch.matmul(
        extrinsics.unsqueeze(2).unsqueeze(3), points_h.unsqueeze(-1)
    ).squeeze(-1)

    pose_enc = torch.zeros(batch_size, frames, 9)
    pose_enc[..., :3] = extrinsics[..., :3, 3]
    pose_enc[..., 6] = 1.0  # quaternion xyzw 的 w=1

    batch = {
        "images": torch.zeros(batch_size, frames, 3, height, width),
        "world_points": points,
        "cam_points": cam_points,
        "extrinsics": extrinsics,
        "point_masks": torch.ones(
            batch_size, frames, height, width, dtype=torch.bool
        ),
        "is_metric": torch.tensor([True]),
        "dataset_name": ["vkitti"],
        "scene_name": ["Scene20/clone"],
        "camera_name": ["Camera_0"],
        "clip_start": torch.tensor([25]),
        "seq_name": ["vkitti/Scene20/clone/Camera_0/clip_0001"],
    }
    predictions = {"world_points": points.clone(), "pose_enc": pose_enc}
    return batch, predictions


def test_perfect_prediction_has_perfect_or_zero_metrics():
    batch, predictions = _perfect_two_frame_batch()
    evaluator = GeometryMetricEvaluator(
        {
            "metric_scale_factor": 0.1,
            "max_points_per_frame": None,
            "cdist_chunk_size": 64,
            "min_pose_valid_pixels": 100,
        }
    )

    row = evaluator.compute_batch(batch, predictions)[0]

    assert row["eval_id"] == "vkitti|Scene20/clone|Camera_0|25"
    assert math.isclose(row["first_camera_accuracy"], 0.0, abs_tol=1e-7)
    assert math.isclose(row["first_camera_completeness"], 0.0, abs_tol=1e-7)
    assert math.isclose(row["first_camera_chamfer"], 0.0, abs_tol=1e-7)
    assert math.isclose(row["camera_ray_depth_abs_rel"], 0.0, abs_tol=1e-7)
    assert math.isclose(row["camera_ray_depth_delta_1_25"], 1.0, abs_tol=1e-7)
    assert math.isclose(row["camera_to_first_camera_auc_30"], 100.0, abs_tol=1e-6)


def test_ray_depth_uses_gt_pose_while_pose_auc_uses_predicted_pose():
    """Point/ray metric and Camera Head metric must remain disentangled."""
    batch, predictions = _perfect_two_frame_batch()
    # Keep the Point Map perfect, but deliberately predict an identity pose
    # for frame 1.  Ray depth should still be perfect because DVGT evaluates
    # Point Map with GT poses; Pose AUC must expose the Camera Head error.
    predictions["pose_enc"][:, 1, :3] = 0.0
    evaluator = GeometryMetricEvaluator(
        {
            "metric_scale_factor": 0.1,
            "max_points_per_frame": None,
            "cdist_chunk_size": 64,
            "min_pose_valid_pixels": 100,
        }
    )

    row = evaluator.compute_batch(batch, predictions)[0]

    assert math.isclose(row["camera_ray_depth_abs_rel"], 0.0, abs_tol=1e-7)
    assert math.isclose(row["camera_ray_depth_delta_1_25"], 1.0, abs_tol=1e-7)
    assert row["camera_to_first_camera_auc_30"] < 100.0


def test_unique_id_retains_distinct_clips_and_drops_only_exact_padding_duplicate():
    batch, _ = _perfect_two_frame_batch()
    first_id = build_eval_id(batch, 0)
    batch["clip_start"] = torch.tensor([37])
    second_id = build_eval_id(batch, 0)
    assert first_id != second_id

    base = {
        "dataset_name": "vkitti",
        "scene_name": "Scene20/clone",
        "camera_name": "Camera_0",
        "clip_start": 25,
        "seq_name": "same_physical_sequence",
        "num_frames": 12,
        "first_camera_accuracy": 1.0,
        "first_camera_completeness": 1.0,
        "first_camera_chamfer": 2.0,
        "camera_ray_depth_abs_rel": 0.1,
        "camera_ray_depth_delta_1_25": 0.9,
        "camera_to_first_camera_auc_30": 80.0,
    }
    rows = [
        {**base, "eval_id": first_id},
        {**base, "eval_id": first_id},  # DDP padding duplicate
        {**base, "eval_id": second_id, "clip_start": 37},
    ]
    assert len(deduplicate_metric_rows(rows)) == 2


def test_reports_keep_all_seven_dataset_groups(tmp_path):
    dataset_names = (
        "openscene",
        "waymo",
        "ddad",
        "mvs_synth",
        "nuscene",
        "kitti",
        "vkitti",
    )
    batch, predictions = _perfect_two_frame_batch()
    row = GeometryMetricEvaluator(
        {"metric_scale_factor": 0.1, "max_points_per_frame": 16}
    ).compute_batch(batch, predictions)[0]
    summary = summarize_metric_rows([row], dataset_names)
    csv_path, summary_path = write_metric_reports([row], summary, tmp_path)

    with summary_path.open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {item["dataset_name"] for item in summary_rows} == {
        *dataset_names,
        "all",
    }
    assert csv_path.exists()
    assert summary_path.exists()
