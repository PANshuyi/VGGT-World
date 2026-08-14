"""DVGT 预处理产物到单路 VGGT 时序样本的通用 loader。

这里只复用 DVGT 的 parquet、校准和 MoGe/LiDAR depth 产物，不复用
DVGT 的 ``T x V`` 多相机 ego 样本组织。每个 ``(scene_id, cam_type)``
都被拆成一条独立且按时间排序的物理相机序列，符合本项目
第一阶段及后续 FM 的单路设定。
"""

from __future__ import annotations

import glob
import io
import logging
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2


# 多路驾驶数据集沿用 DVGT 的相机语义，但每一路仍是一条独立时序。
# 这里的分组只控制“选择哪条物理相机序列”的概率，不会把多路图像
# 拼进同一个 VGGT 样本。
CAMERA_DIRECTION_MAPS: dict[str, dict[str, str]] = {
    "openscene": {
        "CAM_F0": "front",
        "CAM_L0": "front_side",
        "CAM_R0": "front_side",
        "CAM_L1": "side",
        "CAM_R1": "side",
        "CAM_L2": "rear",
        "CAM_B0": "rear",
        "CAM_R2": "rear",
    },
    "waymo": {
        "FRONT": "front",
        "FRONT_LEFT": "front_side",
        "FRONT_RIGHT": "front_side",
        "SIDE_LEFT": "side",
        "SIDE_RIGHT": "side",
    },
    "nuscene": {
        "CAM_FRONT": "front",
        "CAM_FRONT_LEFT": "front_side",
        "CAM_FRONT_RIGHT": "front_side",
        "CAM_BACK": "rear",
        "CAM_BACK_LEFT": "rear",
        "CAM_BACK_RIGHT": "rear",
    },
    "ddad": {
        "camera_01": "front",
        "camera_05": "front_side",
        "camera_06": "front_side",
        "camera_07": "rear",
        "camera_08": "rear",
        "camera_09": "rear",
    },
}


def normalize_available_camera_group_weights(
    available_groups: Iterable[str],
    configured_weights: dict[str, float],
) -> dict[str, float]:
    """把方向组目标比例适配到当前数据集实际具备的相机组。

    OpenScene 四组齐全时严格保持 60/15/15/10。Waymo 没有后向、
    nuScenes/DDAD 没有正侧，因此保留正前 60%，将剩余 40% 按实际
    存在的非正前组原始相对权重重新归一化。
    """
    available = list(dict.fromkeys(available_groups))
    if not available:
        return {}

    positive = {
        group: float(configured_weights.get(group, 0.0))
        for group in available
        if float(configured_weights.get(group, 0.0)) > 0.0
    }
    if not positive:
        uniform = 1.0 / len(available)
        return {group: uniform for group in available}

    if "front" in positive and len(positive) > 1:
        front_weight = min(max(positive["front"], 0.0), 1.0)
        non_front = {key: value for key, value in positive.items() if key != "front"}
        non_front_total = sum(non_front.values())
        if non_front_total > 0.0:
            result = {"front": front_weight}
            result.update(
                {
                    key: (1.0 - front_weight) * value / non_front_total
                    for key, value in non_front.items()
                }
            )
            return result

    total = sum(positive.values())
    return {key: value / total for key, value in positive.items()}


def sample_ordered_positions(
    frame_count: int,
    image_count: int,
    *,
    fps_step: int = 1,
    training: bool,
    allow_duplicates: bool,
) -> np.ndarray:
    """从一条物理相机序列里取有序局部时间窗。

    VGGT 的动态 batch 会随机给出 2~24 帧。DDAD 等短序列不足时
    允许按时间重复，但始终保证输出按帧号非递减，避免把随机视图
    错当成驾驶时序。
    """
    if frame_count <= 0 or image_count <= 0:
        raise ValueError("frame_count and image_count must be positive")

    candidate = np.arange(0, frame_count, max(1, int(fps_step)), dtype=np.int64)
    if candidate.size >= image_count:
        max_start = candidate.size - image_count
        start = random.randint(0, max_start) if training and max_start > 0 else 0
        return candidate[start : start + image_count]

    if not allow_duplicates:
        raise ValueError(
            f"Sequence has only {candidate.size} usable frames but {image_count} were requested"
        )

    # 等间隔重复比单纯 padding 最后一帧更能保留整条短序列。
    return candidate[np.linspace(0, candidate.size - 1, image_count).round().astype(np.int64)]


def sample_fixed_stride_positions(
    frame_count: int,
    image_count: int,
    *,
    min_stride: int = 2,
    max_stride: int = 5,
    training: bool,
    allow_duplicates: bool,
    randomize: bool | None = None,
) -> np.ndarray:
    """按一个样本内固定的随机 stride 采样有序时序。

    默认保持原语义：训练时随机，验证时固定。如果显式传入
    ``randomize=True``，验证样本也会从当前序列真正可行的
    ``[min_stride, max_stride]`` 中均匀采样，再均匀采样不越界的起点。

    边界降级顺序：
    1. 优先缩小到仍在配置区间内的 stride；
    2. 序列不足以支持 ``min_stride`` 时降级为 stride=1；
    3. 连续帧也不足时，仅在 allow_duplicates=True 时均匀重复。

    这里不采用 StreamVGGT VKITTI 的“10% 逐段随机间隔”，
    因此返回结果始终有序，且非降级样本的相邻帧差恒定。
    """
    if frame_count <= 0 or image_count <= 0:
        raise ValueError("frame_count and image_count must be positive")
    min_stride = int(min_stride)
    max_stride = int(max_stride)
    if min_stride <= 0 or max_stride < min_stride:
        raise ValueError("stride range must satisfy 0 < min_stride <= max_stride")
    should_randomize = training if randomize is None else bool(randomize)

    if image_count == 1:
        start = random.randrange(frame_count) if should_randomize and frame_count > 1 else 0
        return np.asarray([start], dtype=np.int64)

    max_feasible_stride = (frame_count - 1) // (image_count - 1)
    feasible_max = min(max_stride, max_feasible_stride)
    if feasible_max >= min_stride:
        stride = random.randint(min_stride, feasible_max) if should_randomize else min_stride
        max_start = frame_count - 1 - stride * (image_count - 1)
        start = random.randint(0, max_start) if should_randomize and max_start > 0 else 0
        return start + np.arange(image_count, dtype=np.int64) * stride

    # 超短序列不能满足最小 stride，但仍尽量保留不重复的连续时序。
    if frame_count >= image_count:
        max_start = frame_count - image_count
        start = random.randint(0, max_start) if should_randomize and max_start > 0 else 0
        return start + np.arange(image_count, dtype=np.int64)

    if not allow_duplicates:
        raise ValueError(
            f"Sequence has only {frame_count} frames but {image_count} were requested"
        )

    # 均匀重复比只 padding 最后一帧更能保留整条短序列的运动。
    return np.linspace(0, frame_count - 1, image_count).round().astype(np.int64)


def split_nonoverlapping_positions(
    frame_count: int,
    clip_len: int,
    *,
    stride: int = 1,
    min_clip_len: int = 2,
) -> list[np.ndarray]:
    """把完整时序切成确定性的非重叠验证片段。

    语义对应 DVGT ``split_scenes_into_clips``：先按数据集帧率下采样，
    再按固定最大帧数切片，尾段也保留。由于 VGGT 当前验证最少要求
    2 张图，唯一差别是只含 1 帧的尾段会被丢弃。
    """
    if frame_count <= 0 or clip_len <= 0 or stride <= 0:
        raise ValueError("frame_count, clip_len and stride must be positive")
    positions = np.arange(0, frame_count, stride, dtype=np.int64)
    return [
        positions[start : start + clip_len]
        for start in range(0, len(positions), clip_len)
        if len(positions[start : start + clip_len]) >= min_clip_len
    ]


def read_dvgt_depth(path: str | Path) -> np.ndarray:
    """解码 DVGT ``write_depth`` 产生的对数 16-bit PNG，输出 float32。"""
    path = Path(path)
    data = path.read_bytes()
    pil_image = Image.open(io.BytesIO(data))
    if "near" not in pil_image.info or "far" not in pil_image.info:
        raise ValueError(f"Depth PNG misses DVGT near/far metadata: {path}")

    near = float(pil_image.info["near"])
    far = float(pil_image.info["far"])
    encoded = np.asarray(pil_image)
    invalid_nan = encoded == 0
    invalid_inf = encoded == 65535
    normalized = (encoded.astype(np.float32) - 1.0) / 65533.0
    depth = near ** (1.0 - normalized) * far**normalized
    depth[invalid_nan | invalid_inf] = 0.0
    return depth.astype(np.float32, copy=False)


def _expand_parquet_paths(paths: str | Path | Sequence[str | Path]) -> list[str]:
    """支持单文件、glob 或文件列表，便于 OpenScene 分 slice 读取。"""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded: list[str] = []
    for item in paths:
        item = str(item)
        matches = sorted(glob.glob(item)) if glob.has_magic(item) else [item]
        expanded.extend(matches)
    missing = [path for path in expanded if not Path(path).is_file()]
    if not expanded or missing:
        raise FileNotFoundError(f"Missing parquet metadata: {missing or paths}")
    return expanded


def sample_metadata_scenes(
    metadata: pd.DataFrame,
    max_scenes: int | None,
    seed: int = 42,
    min_unique_frames: int | None = None,
) -> pd.DataFrame:
    """按 scene_id 固定随机抽取完整场景。

    抽样发生在拆分物理相机之前，因此同一 scene 的所有相机和帧
    要么全部保留，要么全部移除。这对应 DVGT 的
    ``openscene_test_sample_200`` 场景级验证语义。
    """
    if max_scenes is None or int(max_scenes) <= 0:
        return metadata
    frame_counts = metadata.groupby("scene_id")["frame_idx"].nunique()
    if min_unique_frames is not None:
        # 与 DVGT 脚本完全一致：阈值是 >=40 个唯一时间帧，
        # 不是 parquet 行数（同一时刻通常有多路相机）。
        frame_counts = frame_counts[frame_counts >= int(min_unique_frames)]
    scene_ids = frame_counts.index.astype(str).tolist()
    max_scenes = int(max_scenes)
    if not scene_ids:
        raise ValueError("No scene satisfies the validation scene filter")
    if len(scene_ids) <= max_scenes:
        selected_scenes = set(scene_ids)
    else:
        # DVGT 的生成脚本使用 pandas.Series.sample；沿用相同 API、
        # random_state 和输入顺序，才能复现其 sample_200 选择逻辑。
        selected_scenes = set(
            pd.Series(scene_ids).sample(n=max_scenes, random_state=int(seed)).tolist()
        )
    return metadata[metadata["scene_id"].astype(str).isin(selected_scenes)].copy()


class DrivingParquetDataset(BaseDataset):
    """五个 DVGT 真实驾驶数据集的共享 VGGT loader。"""

    def __init__(
        self,
        common_conf,
        *,
        dataset_name: str,
        parquet_path: str | Sequence[str],
        image_root: str,
        align_depth_root: str,
        proj_depth_root: str,
        split: str = "train",
        original_fps: int = 2,
        target_fps: int = 2,
        len_train: int = 100_000,
        len_test: int = 10_000,
        is_metric: bool = True,
        min_depth: float = 1.0,
        # 五个真实驾驶数据集统一只使用 [1, 100] 米的有效深度监督。
        # 这是 depth 的范围 mask，不是 DVGT 在当前 ego 坐标系下的 ray-distance mask。
        max_depth: float = 100.0,
        max_scenes: int | None = None,
        scene_sample_seed: int = 42,
        min_scene_frames: int | None = None,
    ) -> None:
        super().__init__(common_conf=common_conf)
        self.dataset_name = str(dataset_name)
        self.training = bool(common_conf.training)
        self.debug = bool(common_conf.debug)
        self.inside_random = bool(common_conf.inside_random)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.fps_step = max(1, int(original_fps) // max(1, int(target_fps)))
        self.is_metric = bool(is_metric)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        # 只在训练集 inside_random 路径启用方向偏置。验证集保持自然
        # 相机分布，便于分别构造 front-val 与 all-view-val。
        camera_sampling = common_conf.get("camera_sampling", {})
        self.camera_sampling_enabled = bool(camera_sampling.get("enabled", False))
        self.camera_group_weights = {
            "front": float(camera_sampling.get("front", 0.60)),
            "front_side": float(camera_sampling.get("front_side", 0.15)),
            "side": float(camera_sampling.get("side", 0.15)),
            "rear": float(camera_sampling.get("rear", 0.10)),
        }

        # 路径根目录不包含数据集名；与 DVGT SceneDataset 的约定一致。
        self.image_root = Path(image_root) / self.dataset_name
        self.align_depth_root = Path(align_depth_root) / self.dataset_name
        self.proj_depth_root = Path(proj_depth_root) / self.dataset_name

        parquet_files = _expand_parquet_paths(parquet_path)
        logging.info("[%s] loading metadata: %s", self.dataset_name, parquet_files)
        frames = [pd.read_parquet(path) for path in parquet_files]
        metadata = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        required = {
            "scene_id",
            "frame_idx",
            "cam_type",
            "filename",
            "intrinsics",
            "extrinsics",
        }
        missing_columns = sorted(required.difference(metadata.columns))
        if missing_columns:
            raise ValueError(f"{self.dataset_name} parquet misses columns: {missing_columns}")
        if "use_lidar_proj_depth" not in metadata:
            metadata["use_lidar_proj_depth"] = False
        # parquet 中的空值不能直接 bool(np.nan)，否则会被误判成 True，
        # 进而错误地去读取稀疏 LiDAR fallback 深度。
        metadata["use_lidar_proj_depth"] = (
            metadata["use_lidar_proj_depth"].fillna(False).astype(bool)
        )

        metadata["scene_id"] = metadata["scene_id"].astype(str)
        metadata["cam_type"] = metadata["cam_type"].astype(str)
        metadata = metadata.drop_duplicates(subset=["scene_id", "frame_idx", "cam_type"])
        metadata = sample_metadata_scenes(
            metadata,
            max_scenes=max_scenes,
            seed=scene_sample_seed,
            min_unique_frames=min_scene_frames,
        )
        logging.info(
            "[%s] selected %d complete scenes (max_scenes=%s, min_frames=%s, seed=%d)",
            self.dataset_name,
            metadata["scene_id"].nunique(),
            max_scenes,
            min_scene_frames,
            scene_sample_seed,
        )

        # 核心约束：多路数据不在同一样本中展平，而是拆为多条
        # 单路序列。这也防止同时刻不同相机被误解为时间连续帧。
        self.sequences: dict[tuple[str, str], pd.DataFrame] = {}
        for key, group in metadata.groupby(["scene_id", "cam_type"], sort=True):
            group = group.sort_values("frame_idx", kind="stable").reset_index(drop=True)
            if len(group) >= 2:
                self.sequences[(str(key[0]), str(key[1]))] = group
        self.sequence_list = sorted(self.sequences)
        if not self.sequence_list:
            raise ValueError(f"No valid single-camera sequences found for {self.dataset_name}")

        # cam_type 大小写在不同数据源中可能不一致，统一 casefold 后匹配。
        direction_map = {
            camera_name.casefold(): group
            for camera_name, group in CAMERA_DIRECTION_MAPS.get(self.dataset_name, {}).items()
        }
        self.sequences_by_camera_group: dict[
            str, dict[str, list[tuple[str, str]]]
        ] = {}
        unknown_camera_types: set[str] = set()
        for key in self.sequence_list:
            camera_type = key[1]
            group = direction_map.get(camera_type.casefold())
            if group is None:
                unknown_camera_types.add(camera_type)
                continue
            self.sequences_by_camera_group.setdefault(group, {}).setdefault(
                camera_type, []
            ).append(key)

        self.active_camera_group_weights: dict[str, float] = {}
        if self.camera_sampling_enabled and self.training and direction_map:
            if unknown_camera_types:
                raise ValueError(
                    f"{self.dataset_name} has unmapped camera types while biased camera "
                    f"sampling is enabled: {sorted(unknown_camera_types)}"
                )
            self.active_camera_group_weights = normalize_available_camera_group_weights(
                self.sequences_by_camera_group,
                self.camera_group_weights,
            )
            if "front" not in self.active_camera_group_weights:
                raise ValueError(
                    f"{self.dataset_name} has no front-camera sequence for biased sampling"
                )
            logging.info(
                "[%s] camera-direction sampling weights: %s",
                self.dataset_name,
                self.active_camera_group_weights,
            )

        self.validation_samples: list[tuple[tuple[str, str], np.ndarray, int]] = []
        if split in {"train", "training"}:
            self.len_train = int(len_train)
        elif split in {"val", "test", "validation"}:
            # 与 DVGT 一样把每条完整时序切成不重叠固定长度片段，
            # 顺序验证会覆盖全部片段，而不是每条长时序只看开头一次。
            clip_len = int(common_conf.fix_img_num)
            if clip_len <= 0:
                raise ValueError("sequential validation requires fix_img_num > 0")
            for key in self.sequence_list:
                clips = split_nonoverlapping_positions(
                    len(self.sequences[key]), clip_len, stride=self.fps_step
                )
                for clip_index, positions in enumerate(clips):
                    self.validation_samples.append((key, positions, clip_index))
            if not self.validation_samples:
                raise ValueError(f"No valid validation clips for {self.dataset_name}")
            self.len_train = len(self.validation_samples)
        else:
            raise ValueError(f"Unknown split: {split}")
        logging.info(
            "[%s] %d physical-camera sequences, logical samples=%d",
            self.dataset_name,
            len(self.sequence_list),
            self.len_train,
        )

    def _sample_training_sequence_key(self) -> tuple[str, str]:
        """先选方向组，再在组内均匀选择一条单物理相机序列。"""
        if not self.active_camera_group_weights:
            return random.choice(self.sequence_list)

        groups = list(self.active_camera_group_weights)
        weights = [self.active_camera_group_weights[group] for group in groups]
        selected_group = random.choices(groups, weights=weights, k=1)[0]
        # 先在组内均匀选择物理相机，再在该相机的场景序列中均匀选择。
        # 不能直接从组内所有 sequence 采样，否则缺帧/缺场景较少的
        # 相机会因 sequence 数量更多而获得更高概率。
        camera_sequences = self.sequences_by_camera_group[selected_group]
        selected_camera = random.choice(list(camera_sequences))
        return random.choice(camera_sequences[selected_camera])

    @staticmethod
    def _matrix(value, shape) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def _storage_path(self, root: Path, filename, suffix: str) -> Path:
        relative = Path(str(filename))
        if relative.suffix:
            relative = relative.with_suffix("")
        return root / f"{relative}{suffix}"

    def get_data(
        self,
        seq_index: int | None = None,
        img_per_seq: int | None = None,
        seq_name=None,
        ids: Iterable[int] | None = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if img_per_seq is None:
            raise ValueError("img_per_seq is required")
        preset_positions = None
        validation_clip_index = None
        if seq_name is None:
            if seq_index is None:
                raise ValueError("seq_index or seq_name is required")
            if self.inside_random and self.training:
                key = self._sample_training_sequence_key()
            elif not self.training:
                key, preset_positions, validation_clip_index = self.validation_samples[int(seq_index)]
            else:
                key = self.sequence_list[int(seq_index) % len(self.sequence_list)]
        else:
            key = tuple(seq_name) if not isinstance(seq_name, tuple) else seq_name

        sequence = self.sequences[key]
        if ids is None:
            positions = (
                preset_positions
                if preset_positions is not None
                else sample_ordered_positions(
                    len(sequence),
                    int(img_per_seq),
                    fps_step=self.fps_step,
                    training=self.training,
                    allow_duplicates=self.allow_duplicate_img,
                )
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
            # 供固定验证探针区分 MoGe 稠密监督与 LiDAR fallback。
            # 该字段只用于日志，不参与模型输入或 loss。
            "use_lidar_proj_depth": [],
        }
        frame_ids = []

        for position in positions:
            row = sequence.iloc[int(position)]
            image_path = self._storage_path(self.image_root, row["filename"], ".jpg")
            use_lidar = bool(row.get("use_lidar_proj_depth", False))
            depth_root = self.proj_depth_root if use_lidar else self.align_depth_root
            depth_path = self._storage_path(depth_root, row["filename"], ".png")

            image = read_image_cv2(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            depth = read_dvgt_depth(depth_path)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0
            if image.shape[:2] != depth.shape:
                raise ValueError(f"Image/depth shape mismatch: {image_path} vs {depth_path}")

            intrinsics = self._matrix(row["intrinsics"], (3, 3))
            extrinsics = self._matrix(row["extrinsics"], (3, 4))
            original_size = np.asarray(image.shape[:2])
            processed = self.process_one_image(
                image,
                depth,
                extrinsics,
                intrinsics,
                original_size,
                target_shape,
                filepath=str(image_path),
            )
            (
                image,
                depth,
                extrinsics,
                intrinsics,
                world_points,
                cam_points,
                point_mask,
                _,
            ) = processed

            output["images"].append(image)
            output["depths"].append(depth)
            output["extrinsics"].append(extrinsics)
            output["intrinsics"].append(intrinsics)
            output["world_points"].append(world_points)
            output["cam_points"].append(cam_points)
            output["point_masks"].append(point_mask)
            output["original_sizes"].append(original_size)
            output["use_lidar_proj_depth"].append(use_lidar)
            frame_ids.append(int(row["frame_idx"]))

        scene_id, cam_type = key
        clip_suffix = "" if validation_clip_index is None else f"/clip_{validation_clip_index:04d}"
        output.update(
            {
                "seq_name": f"{self.dataset_name}/{scene_id}/{cam_type}{clip_suffix}",
                "dataset_name": self.dataset_name,
                # 验证指标使用结构化元数据生成唯一 clip ID，
                # 不再像 DVGT 那样仅按 seq_name 去重而丢掉同序列的后续 clip。
                "scene_name": str(scene_id),
                "camera_name": str(cam_type),
                "clip_start": int(frame_ids[0]),
                "ids": np.asarray(frame_ids, dtype=np.int64),
                "frame_num": len(frame_ids),
                "is_metric": self.is_metric,
            }
        )
        return output
