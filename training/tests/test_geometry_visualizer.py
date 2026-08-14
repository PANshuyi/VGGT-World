"""固定验证探针和几何绘图核心坐标约定的回归测试。"""

import json

import torch

from train_utils.geometry_visualizer import (
    FixedProbeRegistry,
    GeometryProbeVisualizer,
    camera_centers_from_extrinsics,
    colorize_depth,
    point_map_to_ray_depth,
    rasterize_bev,
)


def test_point_map_ray_depth_uses_each_frames_world_to_camera_pose():
    # 两帧 point map 都位于第一帧相机 C0。第二帧的 W2C 沿 z 平移 -1，
    # 因此 C0 中 z=2m 的点在第二帧相机里距离为 1m。
    points = torch.tensor(
        [
            [[[[0.0, 0.0, 2.0]]]],
            [[[[0.0, 0.0, 2.0]]]],
        ]
    ).squeeze(1)
    extrinsics = torch.eye(4).repeat(2, 1, 1)[:, :3]
    extrinsics[1, 2, 3] = -1.0

    ray_depth = point_map_to_ray_depth(points, extrinsics)
    torch.testing.assert_close(ray_depth.flatten(), torch.tensor([2.0, 1.0]))


def test_camera_centers_invert_world_to_camera_translation():
    extrinsics = torch.eye(4).repeat(2, 1, 1)[:, :3]
    extrinsics[1, 0, 3] = -3.0
    centers = camera_centers_from_extrinsics(extrinsics)
    torch.testing.assert_close(centers[1], torch.tensor([3.0, 0.0, 0.0]))


def test_depth_colormap_keeps_invalid_pixels_black_and_handles_nan():
    depth = torch.tensor([[[1.0, float("nan")], [10.0, 100.0]]])
    mask = torch.tensor([[[True, False], [True, True]]])
    image = colorize_depth(depth, mask, minimum=1.0, maximum=100.0)
    assert image.shape == (1, 3, 2, 2)
    assert torch.equal(image[0, :, 0, 1], torch.zeros(3))
    assert torch.isfinite(image).all()


def test_bev_uses_x_right_and_z_forward_ranges():
    points = torch.tensor([[[[[-1.0, 0.0, 2.0], [1.0, 0.0, 8.0]]]]])
    mask = torch.ones(points.shape[:-1], dtype=torch.bool)
    bev = rasterize_bev(
        points,
        mask,
        x_range=(-2.0, 2.0),
        z_range=(0.0, 10.0),
        resolution=1.0,
    )
    assert bev.shape == (3, 10, 4)
    assert torch.count_nonzero(bev) > 0


def test_fixed_probe_manifest_persists_normal_and_fallback_slots(tmp_path):
    manifest = tmp_path / "visual_probe_manifest.json"
    plan = {"openscene": ["front_normal", "front_fallback"]}
    registry = FixedProbeRegistry(manifest, plan)

    normal_id = "openscene|s0|CAM_F0|0"
    fallback_id = "openscene|s1|CAM_F0|12"
    assert registry.consider(
        "openscene", normal_id, "openscene/s0/CAM_F0/clip_0000", 0.0
    ) == (
        0,
        "front_normal",
    )
    assert registry.consider(
        "openscene", fallback_id, "openscene/s1/CAM_F0/clip_0000", 1.0
    ) == (
        1,
        "front_fallback",
    )
    saved_slot = json.loads(manifest.read_text())["slots"]["openscene"][1]
    assert saved_slot["seq_name"].endswith("clip_0000")
    assert saved_slot["eval_id"] == fallback_id

    reloaded = FixedProbeRegistry(manifest, plan)
    reloaded.start_validation()
    # 同 seq_name 但 clip_start 不同时不能命中旧 probe。
    assert reloaded.consider(
        "openscene",
        "openscene|s1|CAM_F0|24",
        "openscene/s1/CAM_F0/clip_0000",
        1.0,
    ) is None
    assert reloaded.consider(
        "openscene", fallback_id, "openscene/s1/CAM_F0/clip_0000", 1.0
    ) == (
        1,
        "front_fallback",
    )


def test_version_one_probe_manifest_is_bound_to_an_exact_clip_on_first_match(tmp_path):
    manifest = tmp_path / "visual_probe_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "slots": {
                    "vkitti": [
                        {
                            "seq_name": "vkitti/Scene20/clone/Camera_0/clip_0000",
                            "role": "camera_0",
                        }
                    ]
                },
            }
        )
    )
    registry = FixedProbeRegistry(manifest, {"vkitti": ["camera_0"]})
    exact_id = "vkitti|Scene20/clone|Camera_0|0"
    assert registry.consider(
        "vkitti",
        exact_id,
        "vkitti/Scene20/clone/Camera_0/clip_0000",
        0.0,
    ) == (0, "camera_0")
    payload = json.loads(manifest.read_text())
    assert payload["version"] == 2
    assert payload["slots"]["vkitti"][0]["eval_id"] == exact_id


def test_full_probe_render_emits_all_six_visual_groups():
    frame_count, height, width = 2, 3, 4
    z_depth = torch.full((1, frame_count, height, width), 0.2)  # 训练空间0.2=真实2m
    world_points = torch.zeros(1, frame_count, height, width, 3)
    world_points[..., 2] = z_depth
    extrinsics = torch.eye(4).repeat(1, frame_count, 1, 1)[..., :3, :]
    point_masks = torch.ones(1, frame_count, height, width, dtype=torch.bool)
    # pose encoding 采用 [T_xyz, quaternion_xyzw, fov_h, fov_w]。
    pose_enc = torch.zeros(1, frame_count, 9)
    pose_enc[..., 6] = 1.0

    batch = {
        "images": torch.rand(1, frame_count, 3, height, width),
        "depths": z_depth,
        "world_points": world_points,
        "extrinsics": extrinsics,
        "point_masks": point_masks,
        "is_metric": torch.tensor([True]),
    }
    predictions = {
        "depth": z_depth.unsqueeze(-1),
        "depth_conf": torch.full_like(z_depth, 2.0),
        "world_points": world_points.clone(),
        "world_points_conf": torch.full_like(z_depth, 2.0),
        "pose_enc": pose_enc,
    }
    visualizer = GeometryProbeVisualizer(
        {
            "metric_scale_factor": 0.1,
            "max_frames": 12,
            "bev_x_range_m": (-4, 4),
            "bev_z_range_m": (-1, 5),
            "bev_resolution_m": 1.0,
        }
    )
    visuals = visualizer.render(batch, predictions, sample_index=0)

    assert set(visuals) == {
        "01_rgb_clip",
        "02_depth_head",
        "03_point_ray_depth",
        "04_point_fusion_bev",
        "05_point_fusion_time",
        "06_camera_trajectory",
    }
    for image in visuals.values():
        assert image.ndim == 3 and image.shape[0] == 3
        assert torch.isfinite(image).all()
