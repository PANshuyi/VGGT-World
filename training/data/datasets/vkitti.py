"""Virtual KITTI 2 loader：Camera_0/Camera_1 分别是独立时序。"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2
from data.datasets.driving_parquet import (
    sample_fixed_stride_positions,
    split_nonoverlapping_positions,
)


def split_vkitti_camera_sequences(
    all_sequences: Iterable[Path],
    root: Path,
    split: str,
    val_scene: str = "Scene20",
) -> list[Path]:
    """按物理场景划分 VKITTI，并完整保留 Scene20 做验证。

    一个物理 Scene 下的全部天气/扰动条件和 Camera_0/1 必须在同一
    split，避免道路布局与轨迹通过不同 condition 或相机发生泄漏。
    """
    sequences = sorted(all_sequences)
    scene_names = {path.relative_to(root).parts[0] for path in sequences}
    if val_scene not in scene_names:
        raise ValueError(
            f"VKITTI validation scene {val_scene!r} not found; "
            f"available scenes: {sorted(scene_names)}"
        )
    if split in {"train", "training"}:
        return [
            path for path in sequences if path.relative_to(root).parts[0] != val_scene
        ]
    if split in {"val", "test", "validation"}:
        return [
            path for path in sequences if path.relative_to(root).parts[0] == val_scene
        ]
    raise ValueError(f"Unknown split: {split}")


class VKittiDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKitti_DIR: str = "/path/to/vkitti",
        len_train: int = 100_000,
        len_test: int = 10_000,
        depth_max: float = 80.0,
        is_metric: bool = True,
        val_scene: str = "Scene20",
        min_frame_stride: int = 2,
        max_frame_stride: int = 5,
        validation_frame_stride: int = 5,
    ) -> None:
        super().__init__(common_conf=common_conf)
        self.root = Path(VKitti_DIR)
        self.training = bool(common_conf.training)
        self.inside_random = bool(common_conf.inside_random)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.depth_max = float(depth_max)
        self.is_metric = bool(is_metric)
        self.min_frame_stride = int(min_frame_stride)
        self.max_frame_stride = int(max_frame_stride)
        # 训练仍在 2~5 中随机选 stride；验证在初始化时
        # 固定按 stride=5 下采样并切为不重叠 clip。
        self.validation_frame_stride = int(validation_frame_stride)
        if self.min_frame_stride <= 0 or self.max_frame_stride < self.min_frame_stride:
            raise ValueError("frame stride must satisfy 0 < min_frame_stride <= max_frame_stride")
        if self.validation_frame_stride <= 0:
            raise ValueError("validation_frame_stride must be positive")
        self.dataset_name = "vkitti"

        # 路径类似 Scene01/clone/frames/rgb/Camera_0。每一路相机单独入列。
        all_sequences = sorted(path for path in self.root.glob("*/*/*/rgb/*") if path.is_dir())
        if not all_sequences:
            raise FileNotFoundError(f"No VKITTI camera sequences found under {self.root}")

        # 参考公开工作的 Scene20 留出协议：以物理 Scene 为单位划分，
        # 该 Scene 的全部 condition 和两路相机始终位于验证侧。
        self.sequence_list = split_vkitti_camera_sequences(
            all_sequences,
            root=self.root,
            split=split,
            val_scene=val_scene,
        )
        self.validation_samples: list[tuple[Path, np.ndarray, int]] = []
        if split in {"train", "training"}:
            self.len_train = int(len_train)
        elif split in {"val", "test", "validation"}:
            clip_len = int(common_conf.fix_img_num)
            if clip_len <= 0:
                raise ValueError("sequential validation requires fix_img_num > 0")
            # 与五个真实数据集一样，这里保存确切帧位置，
            # 后续每轮验证都使用同一批 clip。
            for sequence_path in self.sequence_list:
                camera_id = int(sequence_path.name.rsplit("_", 1)[-1])
                sequence_root = sequence_path.parents[2]
                extrinsic_table = self._load_table(sequence_root / "extrinsic.txt", camera_id)
                intrinsic_table = self._load_table(sequence_root / "intrinsic.txt", camera_id)
                frame_count = len(
                    {int(row[0]) for row in extrinsic_table}
                    & {int(row[0]) for row in intrinsic_table}
                )
                clips = split_nonoverlapping_positions(
                    frame_count,
                    clip_len,
                    stride=self.validation_frame_stride,
                )
                for clip_index, positions in enumerate(clips):
                    self.validation_samples.append((sequence_path, positions, clip_index))
            if not self.validation_samples:
                raise ValueError("No valid VKITTI validation clips")
            self.len_train = len(self.validation_samples)
        else:
            raise ValueError(f"Unknown split: {split}")
        if not self.sequence_list:
            raise ValueError(f"VKITTI split {split} is empty under {self.root}")
        logging.info(
            "[vkitti] %d camera sequences, virtual length=%d, depth_max=%sm, "
            "val_scene=%s",
            len(self.sequence_list),
            self.len_train,
            self.depth_max,
            val_scene,
        )

    @staticmethod
    def _load_table(path: Path, camera_id: int):
        table = np.loadtxt(path, delimiter=" ", skiprows=1)
        table = np.atleast_2d(table)
        table = table[table[:, 1].astype(int) == camera_id]
        return table[np.argsort(table[:, 0], kind="stable")]

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
                seq_index = random.randrange(len(self.sequence_list))
            if not self.training:
                sequence_path, preset_positions, validation_clip_index = self.validation_samples[int(seq_index)]
            else:
                sequence_path = self.sequence_list[int(seq_index) % len(self.sequence_list)]
        else:
            sequence_path = self.root / str(seq_name)

        camera_id = int(sequence_path.name.rsplit("_", 1)[-1])
        # Scene/variant 是 Camera_x 往上三层。
        sequence_root = sequence_path.parents[2]
        extrinsic_table = self._load_table(sequence_root / "extrinsic.txt", camera_id)
        intrinsic_table = self._load_table(sequence_root / "intrinsic.txt", camera_id)
        # 不按“表中的第 N 行”盲目配对，而是显式按 frame_id 取交集，
        # 防止某张表缺一帧后，后续所有 K 和 pose 都发生错位。
        extrinsics_by_frame = {int(row[0]): row for row in extrinsic_table}
        intrinsics_by_frame = {int(row[0]): row for row in intrinsic_table}
        common_frame_ids = sorted(extrinsics_by_frame.keys() & intrinsics_by_frame.keys())
        frame_count = len(common_frame_ids)
        if frame_count < 2:
            raise ValueError(f"Not enough calibrated VKITTI frames: {sequence_path}")
        if ids is None:
            if preset_positions is not None:
                positions = preset_positions
            else:
                # 正常训练路径在 2~5 中随机选 stride；显式按
                # seq_name 调用验证样本时仍固定 stride=5、起点=0。
                sample_min_stride = (
                    self.min_frame_stride if self.training else self.validation_frame_stride
                )
                sample_max_stride = (
                    self.max_frame_stride if self.training else self.validation_frame_stride
                )
                positions = sample_fixed_stride_positions(
                    frame_count,
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
        frame_ids = []
        for position in positions:
            frame_id = common_frame_ids[int(position)]
            extrinsic_row = extrinsics_by_frame[frame_id]
            intrinsic_row = intrinsics_by_frame[frame_id]
            image_path = sequence_path / f"rgb_{frame_id:05d}.jpg"
            depth_path = Path(str(image_path).replace("/rgb/", "/depth/"))
            depth_path = depth_path.with_name(f"depth_{frame_id:05d}.png")

            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read VKITTI image: {image_path}")
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if raw_depth is None:
                raise FileNotFoundError(f"Cannot read VKITTI depth: {depth_path}")
            # VKITTI uint16 单位是 cm，65535 是天空/无效。先显式清理
            # invalid 再转米，不依赖 depth_max 恰好把 655.35m 过滤掉。
            invalid = raw_depth == np.iinfo(np.uint16).max
            depth = raw_depth.astype(np.float32) / 100.0
            depth[invalid | ~np.isfinite(depth) | (depth <= 0) | (depth > self.depth_max)] = 0.0

            extrinsic = extrinsic_row[2:].reshape(4, 4)[:3].astype(np.float64)
            intrinsic = np.eye(3, dtype=np.float64)
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = intrinsic_row[-4:]
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
            frame_ids.append(frame_id)

        relative_name = sequence_path.relative_to(self.root).as_posix()
        # relative_name 为 Scene/condition/frames/rgb/Camera_N；用物理场景+
        # condition 作为 scene_name，将 Camera_0/1 保留为独立时序。
        relative_parts = sequence_path.relative_to(self.root).parts
        scene_name = "/".join(relative_parts[:2])
        clip_suffix = "" if validation_clip_index is None else f"/clip_{validation_clip_index:04d}"
        output.update(
            {
                "seq_name": f"vkitti/{relative_name}{clip_suffix}",
                "dataset_name": self.dataset_name,
                "scene_name": scene_name,
                "camera_name": sequence_path.name,
                "clip_start": int(frame_ids[0]),
                "ids": np.asarray(frame_ids, dtype=np.int64),
                "frame_num": len(frame_ids),
                "is_metric": self.is_metric,
            }
        )
        return output
