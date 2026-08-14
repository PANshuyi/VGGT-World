"""Validation-only geometry metrics for the seven-dataset VGGT fine-tune.

The training target lives in a fixed ``x0.1`` coordinate space.  This module is
called only under ``torch.no_grad()`` and restores metric samples to metres
before computing DVGT-compatible point-map metrics.  It never mutates the
batch/predictions and is deliberately isolated from forward, loss and backward.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist

from vggt.utils.geometry import project_world_points_to_camera_points_batch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


METRIC_COLUMNS = (
    "first_camera_accuracy",
    "first_camera_completeness",
    "first_camera_chamfer",
    "camera_ray_depth_abs_rel",
    "camera_ray_depth_delta_1_25",
    "camera_to_first_camera_auc_30",
)

IDENTITY_COLUMNS = (
    "eval_id",
    "dataset_name",
    "scene_name",
    "camera_name",
    "clip_start",
    "seq_name",
    "num_frames",
)


def _batch_string(batch: Mapping[str, Any], key: str, index: int) -> str:
    value = batch.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return str(value[index])
    return str(value)


def _batch_integer(batch: Mapping[str, Any], key: str, index: int) -> int:
    value = batch.get(key)
    if torch.is_tensor(value):
        return int(value[index].detach().cpu().item())
    if isinstance(value, (list, tuple)):
        return int(value[index])
    if value is None:
        ids = batch.get("ids")
        if torch.is_tensor(ids):
            return int(ids[index, 0].detach().cpu().item())
        return -1
    return int(value)


def build_eval_id(batch: Mapping[str, Any], index: int) -> str:
    """Build a collision-resistant ID for one validation clip.

    DDP sequential samplers may pad with duplicated samples.  Unlike DVGT's
    ``seq_name``-only deduplication, this ID preserves different clips cut from
    the same physical camera sequence.
    """
    dataset_name = _batch_string(batch, "dataset_name", index)
    scene_name = _batch_string(batch, "scene_name", index)
    camera_name = _batch_string(batch, "camera_name", index)
    clip_start = _batch_integer(batch, "clip_start", index)
    return f"{dataset_name}|{scene_name}|{camera_name}|{clip_start}"


def _metric_flags(batch: Mapping[str, Any], batch_size: int) -> torch.Tensor:
    flags = batch.get("is_metric")
    if flags is None:
        return torch.ones(batch_size, dtype=torch.bool, device=batch["images"].device)
    if torch.is_tensor(flags):
        return flags.to(device=batch["images"].device, dtype=torch.bool).reshape(batch_size)
    return torch.as_tensor(flags, dtype=torch.bool, device=batch["images"].device).reshape(batch_size)


def _restore_metric_tensor(
    tensor: torch.Tensor,
    metric_flags: torch.Tensor,
    metric_scale_factor: float,
) -> torch.Tensor:
    """Return a restored copy; non-metric samples retain their native scale."""
    restored = tensor.detach().float().clone()
    view_shape = (restored.shape[0],) + (1,) * (restored.ndim - 1)
    factors = torch.where(
        metric_flags,
        torch.full_like(metric_flags, 1.0 / metric_scale_factor, dtype=restored.dtype),
        torch.ones_like(metric_flags, dtype=restored.dtype),
    )
    return restored * factors.view(view_shape)


def _restore_extrinsics(
    extrinsics: torch.Tensor,
    metric_flags: torch.Tensor,
    metric_scale_factor: float,
) -> torch.Tensor:
    restored = extrinsics.detach().float().clone()
    factors = torch.where(
        metric_flags,
        torch.full_like(metric_flags, 1.0 / metric_scale_factor, dtype=restored.dtype),
        torch.ones_like(metric_flags, dtype=restored.dtype),
    )
    restored[..., :3, 3] *= factors.view(-1, 1, 1)
    return restored


def _homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    bottom = torch.zeros(*extrinsics.shape[:-2], 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
    bottom[..., 0, 3] = 1.0
    return torch.cat([extrinsics, bottom], dim=-2)


def _deterministic_subsample(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    # Evenly-spaced indices are deterministic across ranks/runs and do not
    # consume training RNG state.
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(max_points),
        device=points.device,
    ).round().long()
    return points.index_select(0, indices)


def _nearest_mean(source: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    minima = []
    for chunk in source.split(max(1, int(chunk_size)), dim=0):
        distances = torch.cdist(chunk.unsqueeze(0), target.unsqueeze(0)).squeeze(0)
        minima.append(distances.amin(dim=1))
    return torch.cat(minima).mean()


def point_map_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points_per_frame: int | None,
    cdist_chunk_size: int,
) -> tuple[float, float, float]:
    """DVGT definitions: pred->GT accuracy and GT->pred completeness."""
    accuracy_values = []
    completeness_values = []
    for frame_index in range(prediction.shape[0]):
        frame_mask = mask[frame_index].bool()
        pred = prediction[frame_index][frame_mask]
        gt = target[frame_index][frame_mask]
        finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
        pred = pred[finite]
        gt = gt[finite]
        if pred.numel() == 0 or gt.numel() == 0:
            continue
        pred = _deterministic_subsample(pred, max_points_per_frame)
        gt = _deterministic_subsample(gt, max_points_per_frame)
        accuracy_values.append(_nearest_mean(pred, gt, cdist_chunk_size))
        completeness_values.append(_nearest_mean(gt, pred, cdist_chunk_size))
    if not accuracy_values:
        return math.nan, math.nan, math.nan
    accuracy = torch.stack(accuracy_values).mean().item()
    completeness = torch.stack(completeness_values).mean().item()
    return accuracy, completeness, accuracy + completeness


def ray_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    valid = (
        mask.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (target > 1e-6)
    )
    if not torch.any(valid):
        return math.nan, math.nan
    # DVGT clamps both prediction and target before evaluating Abs Rel and
    # delta.  In particular, a negative/zero predicted ray must be treated as
    # an invalid near-zero depth rather than accidentally satisfying delta.
    pred = prediction[valid].clamp_min(1e-6)
    gt = target[valid].clamp_min(1e-6)
    abs_rel = ((pred - gt).abs() / gt).mean().item()
    ratio = torch.maximum(pred / gt, gt / pred)
    delta = (ratio < 1.25).float().mean().item()
    return abs_rel, delta


def _rotation_error_degrees(pred_rotation: torch.Tensor, gt_rotation: torch.Tensor) -> torch.Tensor:
    delta = pred_rotation @ gt_rotation.transpose(-1, -2)
    cosine = ((delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _translation_error_degrees(pred_translation: torch.Tensor, gt_translation: torch.Tensor) -> torch.Tensor:
    pred_norm = pred_translation.norm(dim=-1)
    gt_norm = gt_translation.norm(dim=-1)
    cosine = (
        (pred_translation * gt_translation).sum(dim=-1)
        / (pred_norm * gt_norm).clamp_min(1e-8)
    ).abs().clamp(0.0, 1.0)
    # This also matches DVGT for a zero-length translation: its epsilon-based
    # normalization produces cosine=0 and therefore a 90-degree error.  Such
    # pairs stay in the AUC denominator rather than being silently discarded.
    return torch.rad2deg(torch.acos(cosine))


def camera_pose_auc_30(
    pred_w2c: torch.Tensor,
    gt_w2c: torch.Tensor,
    point_masks: torch.Tensor,
    *,
    min_valid_pixels: int,
) -> float:
    """All-frame-pair rotation/translation angular AUC, matching DVGT AUC@30."""
    valid_frames = point_masks.reshape(point_masks.shape[0], -1).sum(dim=-1) > int(min_valid_pixels)
    pred_w2c = _homogeneous(pred_w2c[valid_frames])
    gt_w2c = _homogeneous(gt_w2c[valid_frames])
    if pred_w2c.shape[0] < 2:
        return math.nan

    pairs = torch.combinations(torch.arange(pred_w2c.shape[0], device=pred_w2c.device), r=2)
    first, second = pairs[:, 0], pairs[:, 1]
    pred_relative = pred_w2c[first] @ torch.linalg.inv(pred_w2c[second])
    gt_relative = gt_w2c[first] @ torch.linalg.inv(gt_w2c[second])
    rotation_error = _rotation_error_degrees(pred_relative[:, :3, :3], gt_relative[:, :3, :3])
    translation_error = _translation_error_degrees(pred_relative[:, :3, 3], gt_relative[:, :3, 3])
    pair_error = torch.maximum(rotation_error, translation_error)
    pair_error = pair_error[torch.isfinite(pair_error)]
    if pair_error.numel() == 0:
        return math.nan
    thresholds = torch.arange(1, 31, device=pair_error.device, dtype=pair_error.dtype)
    recall = (pair_error[:, None] < thresholds[None]).float().mean(dim=0)
    # DVGT uses ``np.histogram(errors, bins=np.arange(31))``.  NumPy includes
    # the right edge only for the final bin, so an error exactly equal to 30
    # degrees contributes to the last recall value while other integer bin
    # boundaries remain left-inclusive.  Preserve that small edge-case here.
    recall[-1] = (pair_error <= thresholds[-1]).float().mean()
    return (recall.mean() * 100.0).item()


class GeometryMetricEvaluator:
    """Compute one metric row per validation sample without changing training state."""

    def __init__(self, config: Mapping[str, Any]):
        self.metric_scale_factor = float(config.get("metric_scale_factor", 0.1))
        self.max_points_per_frame = config.get("max_points_per_frame", 2048)
        if self.max_points_per_frame is not None:
            self.max_points_per_frame = int(self.max_points_per_frame)
        self.cdist_chunk_size = int(config.get("cdist_chunk_size", 512))
        self.min_pose_valid_pixels = int(config.get("min_pose_valid_pixels", 100))
        if self.metric_scale_factor <= 0:
            raise ValueError("metric_scale_factor must be positive")

    @torch.no_grad()
    def compute_batch(
        self,
        batch: Mapping[str, Any],
        predictions: Mapping[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        batch_size = int(batch["images"].shape[0])
        metric_flags = _metric_flags(batch, batch_size)
        gt_points = _restore_metric_tensor(
            batch["world_points"], metric_flags, self.metric_scale_factor
        )
        gt_extrinsics = _restore_extrinsics(
            batch["extrinsics"], metric_flags, self.metric_scale_factor
        )
        gt_cam_points = _restore_metric_tensor(
            batch["cam_points"], metric_flags, self.metric_scale_factor
        )
        masks = batch["point_masks"].detach().bool()

        pred_points = predictions.get("world_points")
        pred_pose = predictions.get("pose_enc")
        pred_pose_list = predictions.get("pose_enc_list")
        if pred_pose is None and pred_pose_list is not None:
            pred_pose = pred_pose_list[-1]
        pred_extrinsics = None
        if pred_pose is not None:
            pred_extrinsics, _ = pose_encoding_to_extri_intri(
                pred_pose.detach().float(), build_intrinsics=False
            )
            pred_extrinsics = _restore_extrinsics(
                pred_extrinsics, metric_flags, self.metric_scale_factor
            )
        if pred_points is not None:
            pred_points = _restore_metric_tensor(
                pred_points, metric_flags, self.metric_scale_factor
            )

        rows = []
        for sample_index in range(batch_size):
            row: dict[str, Any] = {
                "eval_id": build_eval_id(batch, sample_index),
                "dataset_name": _batch_string(batch, "dataset_name", sample_index),
                "scene_name": _batch_string(batch, "scene_name", sample_index),
                "camera_name": _batch_string(batch, "camera_name", sample_index),
                "clip_start": _batch_integer(batch, "clip_start", sample_index),
                "seq_name": _batch_string(batch, "seq_name", sample_index),
                "num_frames": int(batch["images"].shape[1]),
            }
            row.update({name: math.nan for name in METRIC_COLUMNS})

            if pred_points is not None:
                accuracy, completeness, chamfer = point_map_metrics(
                    pred_points[sample_index],
                    gt_points[sample_index],
                    masks[sample_index],
                    max_points_per_frame=self.max_points_per_frame,
                    cdist_chunk_size=self.cdist_chunk_size,
                )
                row["first_camera_accuracy"] = accuracy
                row["first_camera_completeness"] = completeness
                row["first_camera_chamfer"] = chamfer

                # Match DVGT's ray-depth metric: use the GT camera pose to
                # transform the predicted first-camera Point Map back to each
                # frame camera.  This isolates Point Head geometry.  The
                # Camera Head is evaluated independently by pose AUC below.
                pred_ray = project_world_points_to_camera_points_batch(
                    pred_points[sample_index : sample_index + 1],
                    gt_extrinsics[sample_index : sample_index + 1],
                ).norm(dim=-1)[0]
                gt_ray = gt_cam_points[sample_index].norm(dim=-1)
                abs_rel, delta = ray_depth_metrics(
                    pred_ray, gt_ray, masks[sample_index]
                )
                row["camera_ray_depth_abs_rel"] = abs_rel
                row["camera_ray_depth_delta_1_25"] = delta

            if pred_extrinsics is not None:
                row["camera_to_first_camera_auc_30"] = camera_pose_auc_30(
                    pred_extrinsics[sample_index],
                    gt_extrinsics[sample_index],
                    masks[sample_index],
                    min_valid_pixels=self.min_pose_valid_pixels,
                )
            rows.append(row)
        return rows


def deduplicate_metric_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop only exact DDP-padding duplicates, retaining every distinct clip."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        eval_id = str(row["eval_id"])
        if eval_id not in unique:
            unique[eval_id] = dict(row)
    return list(unique.values())


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    dataset_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    groups = list(dataset_names) + ["all"]
    summary: dict[str, dict[str, float]] = {}
    for group in groups:
        selected = rows if group == "all" else [row for row in rows if row["dataset_name"] == group]
        values: dict[str, float] = {"num_clips": float(len(selected))}
        for metric_name in METRIC_COLUMNS:
            finite = [float(row[metric_name]) for row in selected if math.isfinite(float(row[metric_name]))]
            values[metric_name] = sum(finite) / len(finite) if finite else math.nan
        summary[group] = values
    return summary


def gather_metric_rows(local_rows: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]] | None:
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_rows, gathered, dst=0)
        if rank != 0:
            return None
        return [row for rank_rows in gathered for row in (rank_rows or [])]
    return list(local_rows) if rank == 0 else None


def write_metric_reports(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Mapping[str, float]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write final files directly (no temp-file rename, JuiceFS friendly)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_results.csv"
    summary_path = output_dir / "summary_metrics.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*IDENTITY_COLUMNS, *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset_name", "num_clips", *METRIC_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for dataset_name, values in summary.items():
            writer.writerow({"dataset_name": dataset_name, **values})
    logging.info("Validation geometry metrics saved to %s", output_dir)
    return csv_path, summary_path
