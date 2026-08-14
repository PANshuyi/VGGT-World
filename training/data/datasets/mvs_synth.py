"""Raw MVS-Synth (GTAV_1080_new) loader for metric VGGT fine-tuning.

这里复用 MapAnything 对 MVS-Synth 的尺度约定：原始 depth 和
camera translation 都是 GTA game unit，两者同步除以 10 转成米。
每帧还会在正有效 depth 上计算 P95，将更远的地平线/异常点置为
0，使后续 depth 和反投影 point 共用同一无效 mask。
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Iterable

# OpenCV 要求在首次 import cv2 前开启 OpenEXR codec。
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2
from data.datasets.driving_parquet import (
    sample_fixed_stride_positions,
    split_nonoverlapping_positions,
)


def split_mvs_synth_scenes(
    all_scenes: Iterable[Path],
    val_ratio: float = 0.05,
    split_seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """按 MapAnything 的协议固定随机划分完整场景。

    MapAnything 对 MVS-Synth 使用 ``val_ratio=0.05``，并在聚合场景
    列表前把 NumPy 随机种子设为 42。这里先排序再用独立 RNG，既保持
    相同的 95%/5% 场景级协议，又不受进程哈希或训练期随机状态影响。
    """
    scenes = sorted(all_scenes)
    if len(scenes) < 2:
        raise ValueError("MVS-Synth scene split requires at least two scenes")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")

    num_val_scenes = max(1, int(len(scenes) * val_ratio))
    # 至少保留一个训练场景，避免极小测试目录产生空训练集。
    num_val_scenes = min(num_val_scenes, len(scenes) - 1)
    rng = np.random.RandomState(int(split_seed))
    val_indices = set(
        int(index)
        for index in rng.choice(len(scenes), num_val_scenes, replace=False)
    )
    train_scenes = [scene for index, scene in enumerate(scenes) if index not in val_indices]
    val_scenes = [scene for index, scene in enumerate(scenes) if index in val_indices]
    return train_scenes, val_scenes


class MVSSynthDataset(BaseDataset):
    """MVS-Synth 每个数字场景是一条单相机时序。"""

    def __init__(
        self,
        common_conf,
        split: str = "train",
        MVS_DIR: str = "MVS-Synth/GTAV_1080_new",
        len_train: int = 144_000,
        len_test: int = 12_000,
        val_ratio: float = 0.05,
        split_seed: int = 42,
        is_metric: bool = True,
        game_units_per_meter: float = 10.0,
        far_depth_percentile: float = 95.0,
        min_frame_stride: int = 2,
        max_frame_stride: int = 5,
        validation_frame_stride: int = 5,
    ) -> None:
        super().__init__(common_conf=common_conf)
        self.root = Path(MVS_DIR)
        self.training = bool(common_conf.training)
        self.inside_random = bool(common_conf.inside_random)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.is_metric = bool(is_metric)
        self.game_units_per_meter = float(game_units_per_meter)
        self.far_depth_percentile = float(far_depth_percentile)
        self.min_frame_stride = int(min_frame_stride)
        self.max_frame_stride = int(max_frame_stride)
        # 训练仍在 2~5 中随机选一个 stride；验证则在
        # Dataset 初始化时固定按 stride=5 下采样并切 clip，
        # 与五个真实数据集的顺序、可重复验证语义一致。
        self.validation_frame_stride = int(validation_frame_stride)
        if self.game_units_per_meter <= 0:
            raise ValueError("game_units_per_meter must be positive")
        if not 0 < self.far_depth_percentile <= 100:
            raise ValueError("far_depth_percentile must be in (0, 100]")
        if self.min_frame_stride <= 0 or self.max_frame_stride < self.min_frame_stride:
            raise ValueError("frame stride must satisfy 0 < min_frame_stride <= max_frame_stride")
        if self.validation_frame_stride <= 0:
            raise ValueError("validation_frame_stride must be positive")
        self.dataset_name = "mvs_synth"

        all_scenes = sorted(
            path for path in self.root.iterdir() if path.is_dir() and path.name.isdigit()
        )
        if len(all_scenes) < 2:
            raise FileNotFoundError(f"No MVS-Synth numeric scenes found under {self.root}")
        train_scenes, val_scenes = split_mvs_synth_scenes(
            all_scenes,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
        if split in {"train", "training"}:
            self.scene_list = train_scenes
            self.len_train = int(len_train)
        elif split in {"val", "test", "validation"}:
            self.scene_list = val_scenes
            # 验证使用真实场景数，不再把 6 个 val scene 虚拟重复
            # 成 len_test 个样本。这样才能与 DVGT 的顺序完整验证一致。
            self.len_train = len(self.scene_list)
        else:
            raise ValueError(f"Unknown split: {split}")
        if not self.scene_list:
            raise ValueError(f"MVS-Synth split {split} is empty")

        self.frames = {}
        for scene in self.scene_list:
            image_stems = {path.stem for path in (scene / "images").glob("*.png")}
            depth_stems = {path.stem for path in (scene / "depths").glob("*.exr")}
            pose_stems = {path.stem for path in (scene / "poses").glob("*.json")}
            common_stems = sorted(image_stems & depth_stems & pose_stems)
            if len(common_stems) >= 2:
                self.frames[scene.name] = common_stems
        self.scene_list = [scene for scene in self.scene_list if scene.name in self.frames]
        if not self.scene_list:
            raise ValueError(f"No complete MVS-Synth sequences found under {self.root}")
        self.validation_samples: list[tuple[Path, np.ndarray, int]] = []
        if not self.training:
            clip_len = int(common_conf.fix_img_num)
            if clip_len <= 0:
                raise ValueError("sequential validation requires fix_img_num > 0")
            for scene in self.scene_list:
                clips = split_nonoverlapping_positions(
                    len(self.frames[scene.name]),
                    clip_len,
                    stride=self.validation_frame_stride,
                )
                # 直接保存每个 clip 的确切帧位置；后续每轮验证
                # 不会重新随机 stride 或起点。尾段 2~11 帧保留，
                # 仅 1 帧尾段由 split_nonoverlapping_positions 丢弃。
                for clip_index, positions in enumerate(clips):
                    self.validation_samples.append((scene, positions, clip_index))
            if not self.validation_samples:
                raise ValueError("No valid MVS-Synth validation clips")
            self.len_train = len(self.validation_samples)
        logging.info(
            "[mvs_synth] %d scenes, virtual length=%d, is_metric=%s, "
            "game_units_per_meter=%g, far_depth_percentile=%g, split_seed=%d",
            len(self.scene_list),
            self.len_train,
            self.is_metric,
            self.game_units_per_meter,
            self.far_depth_percentile,
            split_seed,
        )

    def _load_camera(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            camera = json.load(handle)
        intrinsic = np.array(
            [
                [camera["f_x"], 0.0, camera["c_x"]],
                [0.0, camera["f_y"], camera["c_y"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        raw_w2c = np.asarray(camera["extrinsic"], dtype=np.float64).reshape(4, 4)

        # 原始 GTA pose 是左手系。与已核验的 MapAnything 转换保持一致：
        # 先求 C2W，将 world Y 轴反向，再把 camera translation 从
        # game unit 同步转成米，最后转回 VGGT 需要的 OpenCV W2C。
        flip_y = np.diag([1.0, -1.0, 1.0, 1.0])
        c2w_opencv = flip_y @ np.linalg.inv(raw_w2c)
        c2w_opencv[:3, 3] /= self.game_units_per_meter
        w2c_opencv = np.linalg.inv(c2w_opencv)
        if not np.isclose(np.linalg.det(w2c_opencv[:3, :3]), 1.0, atol=1e-3):
            raise ValueError(f"Invalid handedness after conversion: {path}")
        return intrinsic, w2c_opencv[:3]

    @staticmethod
    def _convert_depth_to_metric(
        depth: np.ndarray,
        game_units_per_meter: float,
        far_depth_percentile: float,
    ) -> np.ndarray:
        """Convert raw GTA depth to metres and mask invalid/farthest pixels.

        P95 只在正的有限值上计算，避免天空的 Inf 或已置零像素
        改变阈值。返回值中 0 表示无效，BaseDataset 反投影时会让
        depth mask 和 point mask 保持一致。
        """
        depth = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(depth) & (depth > 0)
        depth[~valid] = 0.0
        depth[valid] /= float(game_units_per_meter)

        valid_metric_depth = depth[depth > 0]
        if valid_metric_depth.size:
            far_threshold = np.percentile(valid_metric_depth, far_depth_percentile)
            depth[depth > far_threshold] = 0.0
        return depth

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot read EXR depth: {path}")
        if depth.ndim == 3:
            # 某些 OpenEXR 后端把单通道重复为 RGB，三通道等价时取第一个。
            depth = depth[..., 0]
        return self._convert_depth_to_metric(
            depth,
            game_units_per_meter=self.game_units_per_meter,
            far_depth_percentile=self.far_depth_percentile,
        )

    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name: str | None = None,
        ids: Iterable[int] | None = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if img_per_seq is None:
            raise ValueError("img_per_seq is required")
        validation_clip_index = None
        preset_positions = None
        if seq_name is None:
            if self.inside_random and self.training:
                seq_index = random.randrange(len(self.scene_list))
            if not self.training:
                scene, preset_positions, validation_clip_index = self.validation_samples[int(seq_index)]
            else:
                scene = self.scene_list[int(seq_index) % len(self.scene_list)]
        else:
            scene = self.root / str(seq_name)
        stems = self.frames[scene.name]
        if ids is None:
            if preset_positions is not None:
                positions = preset_positions
            else:
                # 正常训练路径在 2~5 中随机选 stride；显式按
                # seq_name 调用验证样本时也固定 stride=5、起点=0，
                # 避免绕过 validation_samples 后重新引入随机性。
                sample_min_stride = (
                    self.min_frame_stride if self.training else self.validation_frame_stride
                )
                sample_max_stride = (
                    self.max_frame_stride if self.training else self.validation_frame_stride
                )
                positions = sample_fixed_stride_positions(
                    len(stems),
                    int(img_per_seq),
                    min_stride=sample_min_stride,
                    max_stride=sample_max_stride,
                    training=self.training,
                    allow_duplicates=self.allow_duplicate_img,
                    randomize=self.training,
                )
        else:
            positions = np.asarray(list(ids), dtype=np.int64)

        target_shape = self.get_target_shape(aspect_ratio)
        output = {
            "images": [],
            "depths": [],
            "extrinsics": [],
            "intrinsics": [],
            "cam_points": [],
            "world_points": [],
            "point_masks": [],
            "original_sizes": [],
        }
        sampled_ids = []
        for position in positions:
            stem = stems[int(position)]
            image_path = scene / "images" / f"{stem}.png"
            depth_path = scene / "depths" / f"{stem}.exr"
            pose_path = scene / "poses" / f"{stem}.json"
            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read MVS-Synth image: {image_path}")
            depth = self._load_depth(depth_path)
            if image.shape[:2] != depth.shape:
                raise ValueError(f"Image/depth shape mismatch: {image_path} vs {depth_path}")
            intrinsic, extrinsic = self._load_camera(pose_path)
            original_size = np.asarray(image.shape[:2])
            processed = self.process_one_image(
                image,
                depth,
                extrinsic,
                intrinsic,
                original_size,
                target_shape,
                filepath=str(image_path),
            )
            image, depth, extrinsic, intrinsic, world, cam, mask, _ = processed
            output["images"].append(image)
            output["depths"].append(depth)
            output["extrinsics"].append(extrinsic)
            output["intrinsics"].append(intrinsic)
            output["world_points"].append(world)
            output["cam_points"].append(cam)
            output["point_masks"].append(mask)
            output["original_sizes"].append(original_size)
            sampled_ids.append(int(stem) if stem.isdigit() else int(position))

        clip_suffix = "" if validation_clip_index is None else f"/clip_{validation_clip_index:04d}"
        output.update(
            {
                "seq_name": f"mvs_synth/{scene.name}{clip_suffix}",
                "dataset_name": self.dataset_name,
                # MVS-Synth 只有一路物理相机；显式保存 clip 起始帧
                # 便于多卡验证汇总时精确去除 sampler padding 重复。
                "scene_name": str(scene.name),
                "camera_name": "camera_0",
                "clip_start": int(sampled_ids[0]),
                "ids": np.asarray(sampled_ids, dtype=np.int64),
                "frame_num": len(sampled_ids),
                "is_metric": self.is_metric,
            }
        )
        return output
