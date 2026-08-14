"""VGGT 驾驶微调的固定验证探针与几何可视化。

训练空间中的可信米制几何统一乘了 ``metric_scale_factor``（默认 0.1）。
本模块只在绘图前把 GT 和预测同时恢复为米，绝不改动 forward、loss 或反向传播。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vggt.utils.geometry import project_world_points_to_camera_points_batch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


_FRONT_CAMERA_TOKENS = {
    "openscene": ("/CAM_F0/",),
    "waymo": ("/FRONT/",),
    "nuscene": ("/CAM_FRONT/",),
    "ddad": ("/camera_01/",),
    "kitti": ("/image_02/", "/image_03/"),
    "vkitti": ("/Camera_0/",),
    "mvs_synth": (),
}


class FixedProbeRegistry:
    """在首轮验证中选出固定样本，并把选择写入 JSON 供续训复用。

    probe plan 的每一项是一个角色：``front``、``front_normal``、
    ``front_fallback``、``image_02``、``image_03``、``camera_0`` 或 ``any``。
    角色只决定首轮如何选样本；之后严格按
    ``dataset|scene|camera|clip_start`` 唯一 ID 匹配，因此同一
    物理序列被切成多个 clip 时不会误匹配，不同 checkpoint 的
    TensorBoard 图像也可纵向对比。
    """

    VERSION = 2

    def __init__(self, manifest_path: str | Path, probe_plan: Mapping[str, Sequence[str]]):
        self.manifest_path = Path(manifest_path)
        self.probe_plan = {
            str(name): [str(role) for role in roles]
            for name, roles in probe_plan.items()
        }
        self.slots: dict[str, list[dict[str, str] | None]] = {
            name: [None] * len(roles) for name, roles in self.probe_plan.items()
        }
        self._logged_this_validation: set[tuple[str, int]] = set()
        self._load()

    def _load(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            saved_version = int(payload.get("version", -1))
            if saved_version not in (1, self.VERSION):
                logging.warning("Ignoring incompatible visual probe manifest: %s", self.manifest_path)
                return
            saved_slots = payload.get("slots", {})
            for dataset_name, current_slots in self.slots.items():
                saved = saved_slots.get(dataset_name, [])
                for index in range(min(len(current_slots), len(saved))):
                    item = saved[index]
                    if not isinstance(item, dict) or not item.get("seq_name"):
                        continue
                    loaded_item = {
                        "seq_name": str(item["seq_name"]),
                        "role": str(item.get("role", self.probe_plan[dataset_name][index])),
                    }
                    if item.get("eval_id"):
                        loaded_item["eval_id"] = str(item["eval_id"])
                    else:
                        # V1 清单只有 seq_name。不猜 clip_start；在它下次
                        # 精确命中同名样本时再绑定唯一 ID 并升级清单。
                        loaded_item["legacy_seq_name"] = str(item["seq_name"])
                    current_slots[index] = loaded_item
            logging.info("Loaded fixed visual probes from %s", self.manifest_path)
        except (OSError, ValueError, TypeError) as error:
            logging.warning("Could not read visual probe manifest %s: %s", self.manifest_path, error)

    def _save(self) -> None:
        """直接写最终文件，兼容不支持临时文件 rename 的 bucket。"""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": self.VERSION, "slots": self.slots}
            self.manifest_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as error:
            # manifest 写失败不应中断长时间验证；本轮仍可正常输出图像，
            # 只是下次启动时会重新挑选 probe。
            logging.warning(
                "Could not write visual probe manifest %s: %s",
                self.manifest_path,
                error,
            )

    def start_validation(self) -> None:
        self._logged_this_validation.clear()

    @staticmethod
    def _is_front(dataset_name: str, seq_name: str) -> bool:
        tokens = _FRONT_CAMERA_TOKENS.get(dataset_name, ())
        return not tokens or any(token in seq_name for token in tokens)

    def _matches_role(
        self,
        dataset_name: str,
        seq_name: str,
        role: str,
        fallback_ratio: float,
    ) -> bool:
        role = role.lower()
        if role == "any":
            return True
        if role == "front":
            return self._is_front(dataset_name, seq_name)
        if role == "front_normal":
            return self._is_front(dataset_name, seq_name) and fallback_ratio == 0.0
        if role == "front_fallback":
            return self._is_front(dataset_name, seq_name) and fallback_ratio > 0.0
        if role == "image_02":
            return "/image_02/" in seq_name
        if role == "image_03":
            return "/image_03/" in seq_name
        if role == "camera_0":
            return "/Camera_0/" in seq_name
        raise ValueError(f"Unknown visual probe role: {role}")

    def consider(
        self,
        dataset_name: str,
        eval_id: str,
        seq_name: str,
        fallback_ratio: float = 0.0,
    ) -> tuple[int, str] | None:
        """返回当前样本对应的固定槽位；本轮已记录则返回 ``None``。"""
        dataset_name = str(dataset_name)
        eval_id = str(eval_id)
        seq_name = str(seq_name)
        if dataset_name not in self.slots:
            return None

        # V2 manifest 只认精确 clip ID，保证纵向对比稳定。
        for slot_index, item in enumerate(self.slots[dataset_name]):
            slot_key = (dataset_name, slot_index)
            if (
                item is not None
                and item.get("eval_id") == eval_id
                and slot_key not in self._logged_this_validation
            ):
                self._logged_this_validation.add(slot_key)
                return slot_index, item["role"]

        # 兼容 V1：旧清单首次命中原 seq_name 后，立即绑定
        # 当前的唯一 clip ID；从下一轮起不再依赖 seq_name。
        for slot_index, item in enumerate(self.slots[dataset_name]):
            slot_key = (dataset_name, slot_index)
            if (
                item is not None
                and not item.get("eval_id")
                and item.get("legacy_seq_name") == seq_name
                and slot_key not in self._logged_this_validation
            ):
                item["eval_id"] = eval_id
                item.pop("legacy_seq_name", None)
                self._logged_this_validation.add(slot_key)
                self._save()
                logging.info(
                    "Upgraded legacy visual probe %s[%d] to eval_id=%s",
                    dataset_name,
                    slot_index,
                    eval_id,
                )
                return slot_index, item["role"]

        # 首轮验证按预设角色填空；同一个 clip 不重复占两个槽位。
        # 同一物理序列被切成多个 clip 时，只选其中一个，避免 Waymo/
        # nuScenes 的两个 probe 实际只是同一场景的相邻片段。
        physical_sequence = eval_id.rsplit("|", maxsplit=1)[0]
        used_sequences = {
            item["eval_id"].rsplit("|", maxsplit=1)[0]
            for item in self.slots[dataset_name]
            if item is not None and item.get("eval_id")
        }
        if physical_sequence in used_sequences:
            return None
        for slot_index, item in enumerate(self.slots[dataset_name]):
            role = self.probe_plan[dataset_name][slot_index]
            if item is None and self._matches_role(
                dataset_name, seq_name, role, fallback_ratio
            ):
                self.slots[dataset_name][slot_index] = {
                    "eval_id": eval_id,
                    "seq_name": seq_name,
                    "role": role,
                }
                self._logged_this_validation.add((dataset_name, slot_index))
                self._save()
                logging.info(
                    "Registered fixed visual probe %s[%d] (%s): %s (%s)",
                    dataset_name,
                    slot_index,
                    role,
                    seq_name,
                    eval_id,
                )
                return slot_index, role
        return None

    def missing_slots(self) -> list[str]:
        missing = []
        for dataset_name, slots in self.slots.items():
            for index, item in enumerate(slots):
                if item is None:
                    missing.append(f"{dataset_name}[{index}]={self.probe_plan[dataset_name][index]}")
        return missing


def _as_rgb(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().float().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    return images.clamp(0.0, 1.0)


def _resize_rgb(images: torch.Tensor, tile_height: int) -> torch.Tensor:
    """等比例缩放一组 ``Sx3xHxW`` 图像。"""
    height, width = images.shape[-2:]
    tile_width = max(1, int(round(width * tile_height / max(height, 1))))
    return F.interpolate(
        images, size=(tile_height, tile_width), mode="bilinear", align_corners=False
    )


def _draw_labels(canvas: torch.Tensor, labels: Sequence[str], row_height: int, label_width: int) -> torch.Tensor:
    array = (canvas.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for row_index, label in enumerate(labels):
        draw.text((5, row_index * row_height + 5), str(label), fill=(255, 255, 255))
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255.0)


def labeled_sequence_grid(
    rows: Sequence[tuple[str, torch.Tensor]],
    *,
    max_frames: int = 12,
    tile_height: int = 96,
    label_width: int = 92,
) -> torch.Tensor:
    """把若干 ``Sx3xHxW`` 序列按行拼成带英文行名的 TensorBoard 图。"""
    if not rows:
        raise ValueError("rows must not be empty")
    frame_count = min(max_frames, min(images.shape[0] for _, images in rows))
    resized_rows = []
    for _, images in rows:
        resized = _resize_rgb(_as_rgb(images[:frame_count]), tile_height)
        resized_rows.append(torch.cat(list(resized), dim=-1))
    content_width = max(row.shape[-1] for row in resized_rows)
    canvas = torch.zeros(3, tile_height * len(rows), label_width + content_width)
    for row_index, row in enumerate(resized_rows):
        top = row_index * tile_height
        canvas[:, top : top + tile_height, label_width : label_width + row.shape[-1]] = row
    return _draw_labels(canvas, [label for label, _ in rows], tile_height, label_width)


def _turbo(values: torch.Tensor) -> torch.Tensor:
    """轻量连续色表；输入 ``...`` in [0,1]，输出 ``...x3``。"""
    anchors = torch.tensor(
        [
            [0.08, 0.05, 0.35],
            [0.10, 0.45, 0.95],
            [0.10, 0.85, 0.55],
            [0.95, 0.90, 0.15],
            [0.90, 0.15, 0.05],
        ],
        dtype=torch.float32,
    )
    # 无效 depth/confidence 会在着色后被 mask 为黑色；这里先清零，
    # 避免 NaN 转 long 后产生越界索引。
    scaled = torch.nan_to_num(values.float(), nan=0.0, posinf=1.0, neginf=0.0)
    scaled = scaled.clamp(0, 1) * (len(anchors) - 1)
    lower = scaled.floor().long().clamp(min=0, max=len(anchors) - 2)
    fraction = (scaled - lower.float()).unsqueeze(-1)
    return anchors[lower] * (1.0 - fraction) + anchors[lower + 1] * fraction


def colorize_depth(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """按固定米制范围着色；无效像素保持黑色，便于辨认稀疏 fallback。"""
    depth = depth.detach().float().cpu()
    mask = mask.detach().bool().cpu() & torch.isfinite(depth) & (depth > 0)
    # log 距离能同时显示近处结构和 100m 远端，所有 checkpoint 共用范围。
    log_min = math.log(max(minimum, 1e-6))
    log_max = math.log(max(maximum, minimum + 1e-6))
    normalized = (torch.log(depth.clamp(min=minimum, max=maximum)) - log_min) / (log_max - log_min)
    rgb = _turbo(normalized)
    rgb[~mask] = 0.0
    return rgb.permute(0, 3, 1, 2)


def colorize_error(error: torch.Tensor, mask: torch.Tensor, maximum: float) -> torch.Tensor:
    error = error.detach().float().cpu()
    mask = mask.detach().bool().cpu() & torch.isfinite(error)
    rgb = _turbo((error / max(maximum, 1e-6)).clamp(0, 1))
    rgb[~mask] = 0.0
    return rgb.permute(0, 3, 1, 2)


def colorize_confidence(
    confidence: torch.Tensor,
    *,
    minimum: float = 1.0,
    maximum: float = 10.0,
) -> torch.Tensor:
    """按固定范围着色 VGGT 的 ``expp1`` confidence。

    不做每张图 p1/p99 自适应，确保不同训练 step 的颜色可以直接纵向比较。
    """
    confidence = confidence.detach().float().cpu()
    finite = torch.isfinite(confidence)
    log_min = math.log(max(minimum, 1e-6))
    log_max = math.log(max(maximum, minimum + 1e-6))
    normalized = (
        torch.log(confidence.clamp(min=minimum, max=maximum)) - log_min
    ) / (log_max - log_min)
    rgb = _turbo(normalized)
    rgb[~finite] = 0.0
    return rgb.permute(0, 3, 1, 2)


def camera_centers_from_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """由 OpenCV world-to-camera ``[R|t]`` 得到 world/C0 中的相机中心。"""
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return -torch.matmul(rotation.transpose(-1, -2), translation.unsqueeze(-1)).squeeze(-1)


def point_map_to_ray_depth(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    """把 C0 point map 变换到各自相机坐标系，再计算欧氏 ray distance。"""
    camera_points = project_world_points_to_camera_points_batch(
        world_points.unsqueeze(0), world_to_camera.unsqueeze(0)
    ).squeeze(0)
    return torch.linalg.vector_norm(camera_points, dim=-1)


def _bev_indices(
    points: torch.Tensor,
    mask: torch.Tensor,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    points = points.detach().float().cpu().reshape(-1, 3)
    mask = mask.detach().bool().cpu().reshape(-1)
    valid = mask & torch.isfinite(points).all(dim=-1)
    x, z = points[:, 0], points[:, 2]
    valid &= (x >= x_range[0]) & (x < x_range[1]) & (z >= z_range[0]) & (z < z_range[1])
    width = int(math.ceil((x_range[1] - x_range[0]) / resolution))
    height = int(math.ceil((z_range[1] - z_range[0]) / resolution))
    ix = ((x[valid] - x_range[0]) / resolution).long().clamp(0, width - 1)
    iy = ((z_range[1] - z[valid]) / resolution).long().clamp(0, height - 1)
    return iy, ix, height, width


def rasterize_bev(
    points: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    iy, ix, height, width = _bev_indices(points, mask, x_range, z_range, resolution)
    density = torch.zeros(height * width)
    if iy.numel():
        density.index_add_(0, iy * width + ix, torch.ones_like(iy, dtype=torch.float32))
    density = torch.log1p(density).reshape(height, width)
    maximum = density.max().clamp(min=1e-6)
    rgb = _turbo(density / maximum)
    # 空栅格保持黑色，避免最低色标覆盖真实占据区域。
    rgb[density == 0] = 0.0
    return rgb.permute(2, 0, 1)


def rasterize_time_bev(
    points: torch.Tensor,
    mask: torch.Tensor,
    *,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    """按帧序号着色融合点云；颜色断裂可直观看出跨帧漂移。"""
    frame_count = points.shape[0]
    height = int(math.ceil((z_range[1] - z_range[0]) / resolution))
    width = int(math.ceil((x_range[1] - x_range[0]) / resolution))
    color_sum = torch.zeros(height * width, 3)
    count = torch.zeros(height * width, 1)
    frame_colors = _turbo(torch.linspace(0, 1, max(frame_count, 2)))[:frame_count]
    for frame_index in range(frame_count):
        iy, ix, _, _ = _bev_indices(
            points[frame_index], mask[frame_index], x_range, z_range, resolution
        )
        if iy.numel() == 0:
            continue
        flat = iy * width + ix
        color_sum.index_add_(0, flat, frame_colors[frame_index].expand(flat.numel(), 3))
        count.index_add_(0, flat, torch.ones(flat.numel(), 1))
    rgb = (color_sum / count.clamp(min=1)).reshape(height, width, 3)
    return rgb.permute(2, 0, 1)


def overlay_bev(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    matched_mask: torch.Tensor,
    *,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    """GT=绿色，Pred=洋红，重合栅格=白色。"""
    gt_y, gt_x, height, width = _bev_indices(
        gt_points, matched_mask, x_range, z_range, resolution
    )
    pred_y, pred_x, _, _ = _bev_indices(
        pred_points, matched_mask, x_range, z_range, resolution
    )
    gt_occ = torch.zeros(height * width, dtype=torch.bool)
    pred_occ = torch.zeros(height * width, dtype=torch.bool)
    gt_occ[gt_y * width + gt_x] = True
    pred_occ[pred_y * width + pred_x] = True
    rgb = torch.zeros(height * width, 3)
    rgb[gt_occ] = torch.tensor([0.1, 0.9, 0.2])
    rgb[pred_occ] = torch.tensor([0.95, 0.1, 0.8])
    rgb[gt_occ & pred_occ] = 1.0
    return rgb.reshape(height, width, 3).permute(2, 0, 1)


def _labeled_panel(images: Sequence[tuple[str, torch.Tensor]], label_height: int = 22) -> torch.Tensor:
    resized = []
    target_height = max(image.shape[-2] for _, image in images)
    for _, image in images:
        if image.shape[-2] != target_height:
            target_width = max(1, int(round(image.shape[-1] * target_height / image.shape[-2])))
            image = F.interpolate(
                image.unsqueeze(0), size=(target_height, target_width), mode="bilinear", align_corners=False
            ).squeeze(0)
        resized.append(image.float().clamp(0, 1))
    width = sum(image.shape[-1] for image in resized)
    canvas = torch.zeros(3, label_height + target_height, width)
    cursor = 0
    label_positions = []
    for (label, _), image in zip(images, resized):
        canvas[:, label_height:, cursor : cursor + image.shape[-1]] = image
        label_positions.append((cursor + 4, label))
        cursor += image.shape[-1]
    array = (canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    draw = ImageDraw.Draw(pil_image)
    for x, label in label_positions:
        draw.text((x, 4), label, fill=(255, 255, 255))
    return torch.from_numpy(np.asarray(pil_image).copy()).permute(2, 0, 1).float().div(255)


def trajectory_panel(
    gt_extrinsics: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    *,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution: float,
) -> torch.Tensor:
    gt_centers = camera_centers_from_extrinsics(gt_extrinsics.detach().float().cpu())
    pred_centers = camera_centers_from_extrinsics(pred_extrinsics.detach().float().cpu())
    height = int(math.ceil((z_range[1] - z_range[0]) / resolution))
    width = int(math.ceil((x_range[1] - x_range[0]) / resolution))
    canvas = torch.zeros(height, width, 3)

    def draw_path(centers: torch.Tensor, color: torch.Tensor) -> None:
        if centers.numel() == 0:
            return
        pieces = []
        for index in range(max(centers.shape[0] - 1, 1)):
            start = centers[min(index, centers.shape[0] - 1)]
            end = centers[min(index + 1, centers.shape[0] - 1)]
            pieces.append(start[None] + torch.linspace(0, 1, 40)[:, None] * (end - start)[None])
        path = torch.cat(pieces, dim=0)
        mask = torch.ones(path.shape[0], dtype=torch.bool)
        iy, ix, _, _ = _bev_indices(path, mask, x_range, z_range, resolution)
        canvas[iy, ix] = color

    draw_path(gt_centers, torch.tensor([0.1, 0.9, 0.2]))
    draw_path(pred_centers, torch.tensor([0.95, 0.1, 0.8]))
    return _labeled_panel([("GT green / Pred magenta", canvas.permute(2, 0, 1))])


class GeometryProbeVisualizer:
    """把一个固定验证 clip 渲染成若干 TensorBoard 图片。"""

    def __init__(self, config: Mapping[str, Any]):
        self.metric_scale_factor = float(config.get("metric_scale_factor", 0.1))
        self.max_frames = int(config.get("max_frames", 12))
        self.depth_min_m = float(config.get("depth_min_m", 1.0))
        self.depth_max_m = float(config.get("depth_max_m", 100.0))
        self.error_max_m = float(config.get("error_max_m", 10.0))
        self.confidence_min = float(config.get("confidence_min", 1.0))
        self.confidence_max = float(config.get("confidence_max", 10.0))
        self.x_range = tuple(float(value) for value in config.get("bev_x_range_m", (-40, 40)))
        self.z_range = tuple(float(value) for value in config.get("bev_z_range_m", (-10, 100)))
        self.bev_resolution = float(config.get("bev_resolution_m", 0.25))

    def _scale_multiplier(self, is_metric: bool) -> float:
        if is_metric:
            if self.metric_scale_factor <= 0:
                raise ValueError("metric_scale_factor must be positive")
            return 1.0 / self.metric_scale_factor
        return 1.0

    def render(
        self,
        batch: Mapping[str, Any],
        predictions: Mapping[str, torch.Tensor],
        sample_index: int,
    ) -> dict[str, torch.Tensor]:
        """渲染一个样本；所有输出都是 ``3xHxW``、[0,1] CPU tensor。"""
        images = batch["images"][sample_index].detach().float().cpu()[: self.max_frames]
        frame_count, _, height, width = images.shape
        is_metric_raw = batch.get("is_metric", True)
        if torch.is_tensor(is_metric_raw):
            is_metric = bool(is_metric_raw.reshape(-1)[sample_index].item())
        elif isinstance(is_metric_raw, (list, tuple)):
            is_metric = bool(is_metric_raw[sample_index])
        else:
            is_metric = bool(is_metric_raw)
        scale = self._scale_multiplier(is_metric)

        gt_depth = batch["depths"][sample_index, :frame_count].detach().float().cpu() * scale
        gt_mask = batch["point_masks"][sample_index, :frame_count].detach().bool().cpu()
        gt_mask &= torch.isfinite(gt_depth) & (gt_depth > 0)
        pred_depth = predictions["depth"][sample_index, :frame_count].detach().float().cpu().squeeze(-1) * scale
        depth_valid = gt_mask & torch.isfinite(pred_depth)

        visuals: dict[str, torch.Tensor] = {
            "01_rgb_clip": labeled_sequence_grid(
                [("RGB", images)], max_frames=self.max_frames
            )
        }
        depth_rows = [
            ("GT z-depth (m)", colorize_depth(gt_depth, gt_mask, minimum=self.depth_min_m, maximum=self.depth_max_m)),
            ("Pred z-depth (m)", colorize_depth(pred_depth, torch.isfinite(pred_depth) & (pred_depth > 0), minimum=self.depth_min_m, maximum=self.depth_max_m)),
            ("Abs error (m)", colorize_error((pred_depth - gt_depth).abs(), depth_valid, self.error_max_m)),
        ]
        if "depth_conf" in predictions:
            depth_rows.append(
                (
                    "Depth confidence",
                    colorize_confidence(
                        predictions["depth_conf"][sample_index, :frame_count],
                        minimum=self.confidence_min,
                        maximum=self.confidence_max,
                    ),
                )
            )
        visuals["02_depth_head"] = labeled_sequence_grid(
            depth_rows, max_frames=self.max_frames
        )

        gt_world = batch["world_points"][sample_index, :frame_count].detach().float().cpu() * scale
        pred_world = predictions["world_points"][sample_index, :frame_count].detach().float().cpu() * scale
        gt_extrinsics = batch["extrinsics"][sample_index, :frame_count].detach().float().cpu()

        if "pose_enc" in predictions:
            pred_extrinsics, _ = pose_encoding_to_extri_intri(
                predictions["pose_enc"][sample_index : sample_index + 1, :frame_count].detach().float().cpu(),
                image_size_hw=(height, width),
                build_intrinsics=False,
            )
            pred_extrinsics = pred_extrinsics.squeeze(0)
        else:
            pred_extrinsics = gt_extrinsics.clone()

        # 点和 W2C translation 必须使用同一个单位后再进行坐标变换。
        gt_extrinsics_metric = gt_extrinsics.clone()
        pred_extrinsics_metric = pred_extrinsics.clone()
        gt_extrinsics_metric[..., :3, 3] *= scale
        pred_extrinsics_metric[..., :3, 3] *= scale
        gt_ray = point_map_to_ray_depth(gt_world, gt_extrinsics_metric)
        # 用 GT pose 转回各自相机可隔离 Point Head 本身的误差；再用预测
        # pose 转一次，则展示 Point Head + Camera Head 的端到端几何误差。
        pred_ray_gt_pose = point_map_to_ray_depth(pred_world, gt_extrinsics_metric)
        pred_ray_pred_pose = point_map_to_ray_depth(pred_world, pred_extrinsics_metric)
        ray_valid_gt_pose = gt_mask & torch.isfinite(pred_ray_gt_pose)
        ray_valid_pred_pose = gt_mask & torch.isfinite(pred_ray_pred_pose)
        ray_rows = [
            ("GT point ray (m)", colorize_depth(gt_ray, gt_mask, minimum=self.depth_min_m, maximum=self.depth_max_m)),
            (
                "Pred ray / GT pose",
                colorize_depth(
                    pred_ray_gt_pose,
                    torch.isfinite(pred_ray_gt_pose) & (pred_ray_gt_pose > 0),
                    minimum=self.depth_min_m,
                    maximum=self.depth_max_m,
                ),
            ),
            (
                "Error / GT pose",
                colorize_error(
                    (pred_ray_gt_pose - gt_ray).abs(),
                    ray_valid_gt_pose,
                    self.error_max_m,
                ),
            ),
            (
                "Pred ray / Pred pose",
                colorize_depth(
                    pred_ray_pred_pose,
                    torch.isfinite(pred_ray_pred_pose) & (pred_ray_pred_pose > 0),
                    minimum=self.depth_min_m,
                    maximum=self.depth_max_m,
                ),
            ),
            (
                "Error / Pred pose",
                colorize_error(
                    (pred_ray_pred_pose - gt_ray).abs(),
                    ray_valid_pred_pose,
                    self.error_max_m,
                ),
            ),
        ]
        if "world_points_conf" in predictions:
            ray_rows.append(
                (
                    "Point confidence",
                    colorize_confidence(
                        predictions["world_points_conf"][sample_index, :frame_count],
                        minimum=self.confidence_min,
                        maximum=self.confidence_max,
                    ),
                )
            )
        visuals["03_point_ray_depth"] = labeled_sequence_grid(
            ray_rows, max_frames=self.max_frames
        )

        pred_full_mask = torch.isfinite(pred_world).all(dim=-1)
        gt_bev = rasterize_bev(
            gt_world, gt_mask, x_range=self.x_range, z_range=self.z_range, resolution=self.bev_resolution
        )
        pred_full_bev = rasterize_bev(
            pred_world, pred_full_mask, x_range=self.x_range, z_range=self.z_range, resolution=self.bev_resolution
        )
        pred_matched_bev = rasterize_bev(
            pred_world, gt_mask, x_range=self.x_range, z_range=self.z_range, resolution=self.bev_resolution
        )
        matched_overlay = overlay_bev(
            gt_world,
            pred_world,
            gt_mask,
            x_range=self.x_range,
            z_range=self.z_range,
            resolution=self.bev_resolution,
        )
        visuals["04_point_fusion_bev"] = _labeled_panel(
            [
                ("GT", gt_bev),
                ("Pred full", pred_full_bev),
                ("Pred GT-mask", pred_matched_bev),
                ("Overlay", matched_overlay),
            ]
        )
        visuals["05_point_fusion_time"] = _labeled_panel(
            [
                (
                    "GT time",
                    rasterize_time_bev(
                        gt_world,
                        gt_mask,
                        x_range=self.x_range,
                        z_range=self.z_range,
                        resolution=self.bev_resolution,
                    ),
                ),
                (
                    "Pred time",
                    rasterize_time_bev(
                        pred_world,
                        gt_mask,
                        x_range=self.x_range,
                        z_range=self.z_range,
                        resolution=self.bev_resolution,
                    ),
                ),
            ]
        )
        visuals["06_camera_trajectory"] = trajectory_panel(
            gt_extrinsics_metric,
            pred_extrinsics_metric,
            x_range=self.x_range,
            z_range=self.z_range,
            resolution=self.bev_resolution,
        )
        return visuals


def fallback_ratio_from_batch(batch: Mapping[str, Any], sample_index: int) -> float:
    fallback = batch.get("use_lidar_proj_depth")
    if fallback is None:
        return 0.0
    if torch.is_tensor(fallback):
        values = fallback[sample_index].detach().float()
        return float(values.mean().item()) if values.numel() else 0.0
    values = fallback[sample_index] if isinstance(fallback, (list, tuple)) else fallback
    array = np.asarray(values, dtype=np.float32)
    return float(array.mean()) if array.size else 0.0


def batch_string_value(batch: Mapping[str, Any], key: str, sample_index: int) -> str:
    value = batch[key]
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return str(value[sample_index])
    return str(value)
