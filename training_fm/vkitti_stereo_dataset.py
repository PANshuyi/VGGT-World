"""VKITTI 2 stereo sequence adapter for VGGT-World joint fine-tuning.

The adapter is intentionally self-contained so the source-aware loader can use
VKITTI even when the original VGGT ``training/data`` package is absent from
this checkout.  It follows the official VGGT VKITTI conventions for RGB,
metric depth, OpenCV camera-from-world extrinsics, and per-pixel 3D points, but
changes sampling from one camera at a time to synchronized time-major stereo.
"""

# 本次修改：集中导入 VKITTI 索引、图像处理、张量输出和 VGGT 几何反投影所需依赖。
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vggt.utils.geometry import depth_to_world_coords_points


# 本次修改：用不可变记录保存一个场景条件及一个合法时序窗口，确保采样后可以精确复现帧序列。
@dataclass(frozen=True)
class _SequenceRecord:
    scene_name: str
    condition: str
    root: Path
    frame_ids: Tuple[int, ...]
    extrinsics: Mapping[Tuple[int, int], np.ndarray]
    intrinsics: Mapping[Tuple[int, int], np.ndarray]


@dataclass(frozen=True)
class _WindowRecord:
    sequence_index: int
    frame_ids: Tuple[int, ...]
    temporal_stride: int


# 本次修改：将 Hydra/OmegaConf 中的标量或列表统一转换为普通 Python 列表，便于严格校验配置。
def _as_list(value, name: str) -> List:
    if value is None:
        raise ValueError(f"{name} must not be null.")
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    converted = int(value)
    if float(value) != converted or converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return converted


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    converted = int(value)
    if float(value) != converted or converted < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return converted


def _config_value(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


# 本次修改：解析官方 intrinsic/extrinsic 文本表，并按 (frame_id, camera_id) 建立显式索引，避免依赖行号顺序。
def _load_camera_tables(sequence_root: Path):
    extrinsic_path = sequence_root / "extrinsic.txt"
    intrinsic_path = sequence_root / "intrinsic.txt"
    missing = [
        str(path)
        for path in (extrinsic_path, intrinsic_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "VKITTI camera supervision requires the text-GT files; missing "
            f"{missing}. Expected them beside the frames/ directory."
        )

    extrinsic_rows = np.atleast_2d(
        np.loadtxt(extrinsic_path, skiprows=1)
    )
    intrinsic_rows = np.atleast_2d(
        np.loadtxt(intrinsic_path, skiprows=1)
    )
    if extrinsic_rows.shape[1] < 18:
        raise ValueError(
            f"{extrinsic_path} must contain frame, camera and 16 matrix values; "
            f"got shape {extrinsic_rows.shape}."
        )
    if intrinsic_rows.shape[1] < 6:
        raise ValueError(
            f"{intrinsic_path} must contain frame, camera, fx, fy, cx, cy; "
            f"got shape {intrinsic_rows.shape}."
        )

    extrinsics: Dict[Tuple[int, int], np.ndarray] = {}
    for row in extrinsic_rows:
        key = (int(row[0]), int(row[1]))
        if key in extrinsics:
            raise ValueError(f"Duplicate extrinsic row for frame/camera {key}.")
        matrix = row[2:18].reshape(4, 4)
        if not np.isfinite(matrix).all() or not np.allclose(
            matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4
        ):
            raise ValueError(f"Invalid homogeneous extrinsic for {key}.")
        extrinsics[key] = matrix[:3].astype(np.float32)

    intrinsics: Dict[Tuple[int, int], np.ndarray] = {}
    for row in intrinsic_rows:
        key = (int(row[0]), int(row[1]))
        if key in intrinsics:
            raise ValueError(f"Duplicate intrinsic row for frame/camera {key}.")
        fx, fy, cx, cy = row[-4:]
        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            raise ValueError(f"Invalid intrinsic row for {key}: {row[-4:]}.")
        intrinsics[key] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return extrinsics, intrinsics


# 本次修改：从文件名提取实际 frame id，严格检查 RGB/depth/双目/标定是否逐帧对齐。
def _frame_ids(directory: Path, prefix: str, suffix: str) -> set:
    ids = set()
    for path in directory.glob(f"{prefix}_*{suffix}"):
        numeric = path.stem.removeprefix(f"{prefix}_")
        if not numeric.isdigit():
            continue
        ids.add(int(numeric))
    return ids


def _discover_shared_frames(
    sequence_root: Path,
    cameras: Sequence[int],
    extrinsics: Mapping[Tuple[int, int], np.ndarray],
    intrinsics: Mapping[Tuple[int, int], np.ndarray],
    strict_metadata: bool,
) -> Tuple[int, ...]:
    per_camera_ids = []
    for camera_id in cameras:
        rgb_dir = sequence_root / "frames" / "rgb" / f"Camera_{camera_id}"
        depth_dir = sequence_root / "frames" / "depth" / f"Camera_{camera_id}"
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Expected VKITTI stereo directories {rgb_dir} and {depth_dir}."
            )
        rgb_ids = _frame_ids(rgb_dir, "rgb", ".jpg")
        depth_ids = _frame_ids(depth_dir, "depth", ".png")
        if strict_metadata and rgb_ids != depth_ids:
            raise ValueError(
                f"RGB/depth frame mismatch in {sequence_root}, camera {camera_id}: "
                f"rgb={len(rgb_ids)}, depth={len(depth_ids)}."
            )
        image_ids = rgb_ids & depth_ids
        camera_table_ids = {
            frame_id
            for frame_id, table_camera_id in extrinsics
            if table_camera_id == camera_id
            and (frame_id, camera_id) in intrinsics
        }
        if strict_metadata and not image_ids.issubset(camera_table_ids):
            missing = sorted(image_ids - camera_table_ids)[:10]
            raise ValueError(
                f"Missing intrinsic/extrinsic rows in {sequence_root}, camera "
                f"{camera_id}, example frame ids={missing}."
            )
        per_camera_ids.append(image_ids & camera_table_ids)

    if strict_metadata and any(ids != per_camera_ids[0] for ids in per_camera_ids[1:]):
        raise ValueError(
            f"Configured cameras {list(cameras)} do not have identical valid "
            f"frame sets in {sequence_root}."
        )
    shared_ids = set.intersection(*per_camera_ids)
    if not shared_ids:
        raise ValueError(f"No synchronized RGB/depth/camera frames in {sequence_root}.")
    return tuple(sorted(shared_ids))


# 本次修改：采用确定性的等比缩放和主点中心裁剪，并同步更新 K；不施加会破坏双目几何的独立随机裁剪。
def _resize_and_crop(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    target_hw: Tuple[int, int],
):
    source_h, source_w = image_rgb.shape[:2]
    target_h, target_w = target_hw
    if depth_m.shape != (source_h, source_w):
        raise ValueError(
            f"RGB/depth shape mismatch: rgb={(source_h, source_w)}, "
            f"depth={depth_m.shape}."
        )

    scale = max(target_h / source_h, target_w / source_w)
    resized_h = max(target_h, int(math.ceil(source_h * scale)))
    resized_w = max(target_w, int(math.ceil(source_w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    image_rgb = cv2.resize(
        image_rgb, (resized_w, resized_h), interpolation=interpolation
    )
    depth_m = cv2.resize(
        depth_m, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST
    )

    intrinsic = intrinsic.astype(np.float32, copy=True)
    intrinsic[0, :] *= resized_w / source_w
    intrinsic[1, :] *= resized_h / source_h
    crop_x = int(round(float(intrinsic[0, 2]) - target_w / 2.0))
    crop_y = int(round(float(intrinsic[1, 2]) - target_h / 2.0))
    crop_x = min(max(crop_x, 0), resized_w - target_w)
    crop_y = min(max(crop_y, 0), resized_h - target_h)
    image_rgb = image_rgb[crop_y : crop_y + target_h, crop_x : crop_x + target_w]
    depth_m = depth_m[crop_y : crop_y + target_h, crop_x : crop_x + target_w]
    intrinsic[0, 2] -= crop_x
    intrinsic[1, 2] -= crop_y
    return image_rgb, depth_m, intrinsic


# 本次修改：实现一个直接满足 source-aware 四元索引契约的 VKITTI 双目 Dataset，输出完整几何监督张量。
class VKittiStereoDataset(Dataset):
    """Return synchronized time-major VKITTI slots for joint FM/geometry loss.

    Slot order is always ``[t0-c0, t0-c1, t1-c0, t1-c1, ...]``.  With the
    default two history and two future timesteps this produces eight images.
    """

    def __init__(
        self,
        common_config,
        VKitti_DIR: str,
        split: str,
        conditions: Sequence[str] = ("clone",),
        cameras: Sequence[int] = (0, 1),
        train_scenes: Sequence[str] = (
            "Scene01",
            "Scene02",
            "Scene06",
            "Scene20",
        ),
        val_scenes: Sequence[str] = ("Scene18",),
        temporal_strides: Sequence[int] = (1,),
        start_step: int = 1,
        depth_scale: float = 100.0,
        max_depth_m: float = 80.0,
        strict_metadata: bool = True,
    ):
        self.root = Path(VKitti_DIR).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"VKitti_DIR does not exist or is not mounted: {self.root}. "
                "Override vkitti_root in the Hydra config on the training host."
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}.")

        self.split = split
        self.conditions = tuple(str(x) for x in _as_list(conditions, "conditions"))
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError(
                f"conditions must be non-empty and unique, got {self.conditions}."
            )
        self.cameras = tuple(
            _nonnegative_int(x, "camera_id")
            for x in _as_list(cameras, "cameras")
        )
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"cameras contains duplicates: {self.cameras}.")
        self.temporal_strides = tuple(
            _positive_int(x, "temporal_stride")
            for x in _as_list(temporal_strides, "temporal_strides")
        )
        if len(set(self.temporal_strides)) != len(self.temporal_strides):
            raise ValueError(
                f"temporal_strides contains duplicates: {self.temporal_strides}."
            )
        self.start_step = _positive_int(start_step, "start_step")
        self.depth_scale = float(depth_scale)
        self.max_depth_m = float(max_depth_m)
        self.strict_metadata = bool(strict_metadata)
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive.")

        # 本次修改：从 source bucket 注入的布局读取物理时间数并交叉验证双目路数，杜绝把 8 张图误解释成 8 个时刻。
        self.views_per_timestep = _positive_int(
            _config_value(common_config, "views_per_timestep", len(self.cameras)),
            "views_per_timestep",
        )
        self.history_timesteps = _positive_int(
            _config_value(common_config, "history_timesteps", 2),
            "history_timesteps",
        )
        self.future_timesteps = _positive_int(
            _config_value(common_config, "future_timesteps", 2),
            "future_timesteps",
        )
        if self.views_per_timestep != len(self.cameras):
            raise ValueError(
                f"Bucket declares V={self.views_per_timestep}, but adapter has "
                f"cameras={self.cameras}."
            )
        self.physical_timesteps = self.history_timesteps + self.future_timesteps
        self.total_slots = self.physical_timesteps * self.views_per_timestep
        self.img_size = _positive_int(
            _config_value(common_config, "img_size", 448), "img_size"
        )
        self.patch_size = _positive_int(
            _config_value(common_config, "patch_size", 14), "patch_size"
        )

        # 本次修改：默认以 Scene20 验证、其余四个场景训练；条件固定 clone，后续可通过配置扩展但不会混入同一相机布局之外的数据。
        train_scenes = tuple(str(x) for x in _as_list(train_scenes, "train_scenes"))
        val_scenes = tuple(str(x) for x in _as_list(val_scenes, "val_scenes"))
        overlap = sorted(set(train_scenes) & set(val_scenes))
        if overlap:
            raise ValueError(
                "VKITTI scene-level split must be disjoint; overlapping scenes="
                f"{overlap}."
            )
        selected_scenes = train_scenes if split == "train" else val_scenes
        self.sequences: List[_SequenceRecord] = []
        for scene_name in sorted(str(x) for x in selected_scenes):
            for condition in self.conditions:
                sequence_root = self.root / scene_name / condition
                if not sequence_root.is_dir():
                    raise FileNotFoundError(
                        f"Configured VKITTI sequence is missing: {sequence_root}."
                    )
                extrinsics, intrinsics = _load_camera_tables(sequence_root)
                frame_ids = _discover_shared_frames(
                    sequence_root,
                    self.cameras,
                    extrinsics,
                    intrinsics,
                    self.strict_metadata,
                )
                self.sequences.append(
                    _SequenceRecord(
                        scene_name=scene_name,
                        condition=condition,
                        root=sequence_root,
                        frame_ids=frame_ids,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                    )
                )

        # 本次修改：预先枚举所有合法 start/stride 窗口；DataLoader 只负责打乱索引，不会在双目之间产生不同的时间采样。
        self.windows: List[_WindowRecord] = []
        for sequence_index, sequence in enumerate(self.sequences):
            available = set(sequence.frame_ids)
            for temporal_stride in self.temporal_strides:
                for start_frame in sequence.frame_ids[:: self.start_step]:
                    frame_ids = tuple(
                        start_frame + offset * temporal_stride
                        for offset in range(self.physical_timesteps)
                    )
                    if all(frame_id in available for frame_id in frame_ids):
                        self.windows.append(
                            _WindowRecord(
                                sequence_index=sequence_index,
                                frame_ids=frame_ids,
                                temporal_stride=temporal_stride,
                            )
                        )
        if not self.windows:
            raise ValueError(
                "No valid VKITTI windows were found. Check scenes, conditions, "
                "temporal_strides and text-GT files."
            )
        # 本次修改：建立 stride -> window indices 映射，供 source-aware sampler 每个 batch 选择统一的时间间隔。
        temporal_stride_indices: Dict[int, List[int]] = {
            temporal_stride: [] for temporal_stride in self.temporal_strides
        }
        for window_index, window in enumerate(self.windows):
            temporal_stride_indices[window.temporal_stride].append(window_index)
        missing_strides = [
            temporal_stride
            for temporal_stride, indices in temporal_stride_indices.items()
            if not indices
        ]
        if missing_strides:
            raise ValueError(
                "No valid VKITTI windows were found for temporal strides "
                f"{missing_strides}."
            )
        self.temporal_stride_indices = {
            temporal_stride: tuple(indices)
            for temporal_stride, indices in temporal_stride_indices.items()
        }
        self.epoch = 0
        logging.info(
            "VKITTI %s: root=%s, sequences=%d, windows=%d, cameras=%s, "
            "physical_timesteps=%d, slots=%d",
            self.split,
            self.root,
            len(self.sequences),
            len(self.windows),
            self.cameras,
            self.physical_timesteps,
            self.total_slots,
        )

    # 本次修改：数据集长度使用真实合法窗口数，source sampler 会按该长度计算默认混训曝光权重。
    def __len__(self):
        return len(self.windows)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    # 本次修改：向 batch sampler 暴露只读的 stride 分组，实际 sample index 仍直接索引 self.windows。
    def get_temporal_stride_indices(self):
        return dict(self.temporal_stride_indices)

    # 本次修改：将 sampler 的目标宽高比转换为 patch-size 整除的 VGGT 输入分辨率。
    def _target_hw(self, aspect_ratio: float) -> Tuple[int, int]:
        aspect_ratio = float(aspect_ratio)
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError(f"aspect_ratio must be positive, got {aspect_ratio}.")
        target_w = (self.img_size // self.patch_size) * self.patch_size
        target_h = int(self.img_size * aspect_ratio)
        target_h = max(
            self.patch_size, (target_h // self.patch_size) * self.patch_size
        )
        return target_h, target_w

    # 本次修改：加载一个 time-major 双目窗口，所有 RGB/depth/K/pose/point 使用完全相同的槽位顺序。
    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 3:
            raise ValueError(
                "VKittiStereoDataset expects "
                "(sample_index, image_num, aspect_ratio)."
            )
        sample_index, image_num, aspect_ratio = index
        if int(image_num) != self.total_slots:
            raise ValueError(
                f"Adapter expects {self.total_slots} slots, got {image_num}."
            )
        window = self.windows[int(sample_index)]
        sequence = self.sequences[window.sequence_index]
        target_hw = self._target_hw(float(aspect_ratio))

        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        cam_points = []
        world_points = []
        point_masks = []
        frame_indices = []
        camera_ids = []
        original_sizes = []

        for frame_id in window.frame_ids:
            for camera_id in self.cameras:
                rgb_path = (
                    sequence.root
                    / "frames"
                    / "rgb"
                    / f"Camera_{camera_id}"
                    / f"rgb_{frame_id:05d}.jpg"
                )
                depth_path = (
                    sequence.root
                    / "frames"
                    / "depth"
                    / f"Camera_{camera_id}"
                    / f"depth_{frame_id:05d}.png"
                )
                image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if image_bgr is None or depth_raw is None:
                    raise FileNotFoundError(
                        f"Failed to read paired VKITTI files {rgb_path} and "
                        f"{depth_path}."
                    )
                if depth_raw.ndim != 2:
                    raise ValueError(
                        f"VKITTI depth must be a single-channel image, got "
                        f"shape={depth_raw.shape} at {depth_path}."
                    )
                if self.strict_metadata and not np.issubdtype(
                    depth_raw.dtype, np.integer
                ):
                    raise TypeError(
                        f"Expected integer VKITTI depth PNG, got {depth_raw.dtype} "
                        f"at {depth_path}."
                    )
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                depth_m = depth_raw.astype(np.float32) / self.depth_scale
                depth_m[~np.isfinite(depth_m)] = 0.0
                if self.max_depth_m > 0:
                    depth_m[depth_m > self.max_depth_m] = 0.0
                original_sizes.append(np.asarray(image_rgb.shape[:2], dtype=np.int64))

                key = (frame_id, camera_id)
                extrinsic = sequence.extrinsics[key].astype(np.float32, copy=True)
                intrinsic = sequence.intrinsics[key].astype(np.float32, copy=True)
                image_rgb, depth_m, intrinsic = _resize_and_crop(
                    image_rgb, depth_m, intrinsic, target_hw
                )
                world_xyz, camera_xyz, valid_mask = depth_to_world_coords_points(
                    depth_m, extrinsic, intrinsic
                )

                images.append(np.ascontiguousarray(image_rgb))
                depths.append(np.ascontiguousarray(depth_m, dtype=np.float32))
                extrinsics.append(extrinsic)
                intrinsics.append(intrinsic)
                cam_points.append(np.ascontiguousarray(camera_xyz, dtype=np.float32))
                world_points.append(np.ascontiguousarray(world_xyz, dtype=np.float32))
                point_masks.append(np.ascontiguousarray(valid_mask, dtype=np.bool_))
                frame_indices.append(frame_id)
                camera_ids.append(camera_id)

        # 本次修改：直接返回 ComposedDataset 原本会生成的 tensor schema，并附加可审计的帧/相机/layout 元数据。
        sample = {
            "seq_name": f"vkitti_{sequence.scene_name}_{sequence.condition}",
            "images": torch.from_numpy(np.stack(images))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(torch.float32)
            .div_(255.0),
            "depths": torch.from_numpy(np.stack(depths)).to(torch.float32),
            "extrinsics": torch.from_numpy(np.stack(extrinsics)).to(torch.float32),
            "intrinsics": torch.from_numpy(np.stack(intrinsics)).to(torch.float32),
            "cam_points": torch.from_numpy(np.stack(cam_points)).to(torch.float32),
            "world_points": torch.from_numpy(np.stack(world_points)).to(torch.float32),
            "point_masks": torch.from_numpy(np.stack(point_masks)).to(torch.bool),
            "ids": torch.tensor(frame_indices, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "camera_ids": torch.tensor(camera_ids, dtype=torch.long),
            "original_sizes": torch.from_numpy(np.stack(original_sizes)),
            "temporal_stride": int(window.temporal_stride),
            "views_per_timestep": self.views_per_timestep,
            "history_timesteps": self.history_timesteps,
            "future_timesteps": self.future_timesteps,
        }
        return sample
