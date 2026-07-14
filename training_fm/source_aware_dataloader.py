"""Source-aware dynamic batching for variable-camera VGGT-World training.

This module deliberately keeps the original VGGT dataset contract: an
underlying composed dataset receives ``(sample_index, image_num, aspect)``.
The difference is that a whole batch first chooses one source/layout bucket,
so single-, stereo-, and multi-camera samples are never collated together.
"""

# 本次修改：新增 source-aware 变长混训模块，统一管理同质 bucket、动态 batch size、DDP 分片与运行时布局元数据。
import copy
import logging
import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler


# 本次修改：提供 Hydra 配置读取、严格正整数校验、无副作用克隆及 ComposedDataset 子集选择工具。
def _config_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _strict_positive_int(value: Any, name: str) -> int:
    """Reject booleans/fractional layout values instead of silently truncating."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}."
        ) from error
    if not math.isfinite(numeric) or numeric != converted or converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return converted


def _clone_config(config: Any):
    """Clone a Hydra/OmegaConf config without mutating the composed config."""
    if OmegaConf.is_config(config):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    return OmegaConf.create(copy.deepcopy(config))


def _select_dataset_configs(base_dataset_config: Any, dataset_indices: Sequence[int]):
    """Clone a ComposedDataset config and keep only one layout-compatible subset."""
    selected = _clone_config(base_dataset_config)
    dataset_configs = _config_get(selected, "dataset_configs", None)
    if dataset_configs is None:
        raise ValueError(
            "source_buckets[*].dataset_indices requires the base dataset config "
            "to contain dataset_configs. Alternatively put a complete dataset "
            "config in source_buckets[*].dataset."
        )

    indices = [int(index) for index in dataset_indices]
    if not indices:
        raise ValueError("dataset_indices must contain at least one index.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"dataset_indices contains duplicates: {indices}.")
    if min(indices) < 0 or max(indices) >= len(dataset_configs):
        raise IndexError(
            f"dataset_indices={indices} is outside [0, {len(dataset_configs) - 1}]."
        )
    selected.dataset_configs = [dataset_configs[index] for index in indices]
    return selected


# 本次修改：保存一个同质 source/layout bucket 的相机布局、采样权重和显存预算元数据。
@dataclass(frozen=True)
class SourceBucketSpec:
    """Runtime metadata for one homogeneous camera-layout bucket."""

    name: str
    views_per_timestep: int
    history_timesteps: int
    future_timesteps: int
    total_slots: int
    sampling_weight: Optional[float]
    cost_per_sample: float
    aspect_min: float
    aspect_max: float
    batch_size_override: Optional[int]
    max_batch_size: Optional[int]
    # 本次修改：可选的 batch 级时间采样间隔；非空时 sampler 每步只从一个 stride 的索引池取样。
    temporal_strides: tuple = ()

    @property
    def history_slots(self) -> int:
        return self.views_per_timestep * self.history_timesteps

    @property
    def future_slots(self) -> int:
        return self.views_per_timestep * self.future_timesteps


# 本次修改：按 bucket 分别实例化底层 adapter，并把 sampler 四元索引路由到正确数据源。
class SourceBucketDataset(Dataset):
    """Route sampler indices to one of several independently composed datasets."""

    _TEMPORAL_KEYS = (
        "images",
        "depths",
        "extrinsics",
        "intrinsics",
        "cam_points",
        "world_points",
        "point_masks",
        "ids",
        "frame_indices",
        "camera_ids",
        "original_sizes",
    )

    def __init__(self, dataset, common_config, source_buckets):
        if not source_buckets:
            raise ValueError(
                "source_buckets must be a non-empty list. Each entry declares "
                "one homogeneous source/camera layout."
            )

        self.datasets: List[Dataset] = []
        self.bucket_specs: List[SourceBucketSpec] = []
        self.bucket_lengths: List[int] = []
        # 本次修改：按 bucket 保存 stride -> dataset indices，保证一个 batch 的物理时间间隔一致。
        self.bucket_temporal_stride_indices: List[Dict[int, tuple]] = []
        names = set()

        for bucket_index, bucket in enumerate(source_buckets):
            # 本次修改：解析并严格校验 V、历史/未来物理时刻数及等长 FM 所需的总槽位 S。
            name = str(_config_get(bucket, "name", f"bucket_{bucket_index}"))
            if name in names:
                raise ValueError(f"Duplicate source bucket name: {name!r}.")
            names.add(name)

            views = _strict_positive_int(
                _config_get(bucket, "views_per_timestep", 0),
                f"{name}.views_per_timestep",
            )
            history_timesteps = _strict_positive_int(
                _config_get(bucket, "history_timesteps", 0),
                f"{name}.history_timesteps",
            )
            future_timesteps = _strict_positive_int(
                _config_get(bucket, "future_timesteps", 0),
                f"{name}.future_timesteps",
            )
            history_slots = views * history_timesteps
            future_slots = views * future_timesteps
            if history_slots != future_slots:
                raise ValueError(
                    "The current FM requires equal condition/target slot counts; "
                    f"bucket {name!r} has history={history_slots}, "
                    f"future={future_slots}."
                )
            total_slots = history_slots + future_slots

            # 本次修改：每个 bucket 可直接给完整 dataset config，也可按索引复用原 ComposedDataset 配置。
            bucket_dataset_config = _config_get(bucket, "dataset", None)
            dataset_indices = _config_get(bucket, "dataset_indices", None)
            if bucket_dataset_config is not None and dataset_indices is not None:
                raise ValueError(
                    f"Bucket {name!r} must set only one of dataset or dataset_indices."
                )
            if bucket_dataset_config is None:
                if dataset is None:
                    raise ValueError(
                        f"Bucket {name!r} has no dataset config and no base dataset."
                    )
                bucket_dataset_config = (
                    _clone_config(dataset)
                    if dataset_indices is None
                    else _select_dataset_configs(dataset, dataset_indices)
                )

            # 本次修改：底层 adapter 继续接收原三元索引，但 img_nums 被限制为本 bucket 的合法 S。
            bucket_common_config = _clone_config(common_config)
            bucket_common_config.img_nums = [total_slots, total_slots]
            bucket_common_config.fix_img_num = -1
            # Expose the structured meaning of S to a multi-camera adapter;
            # image_num alone cannot distinguish 8 mono times from 4 stereo times.
            bucket_common_config.source_bucket_name = name
            bucket_common_config.views_per_timestep = views
            bucket_common_config.history_timesteps = history_timesteps
            bucket_common_config.future_timesteps = future_timesteps
            # Global inside_random would ignore rank/source indices. Randomness is
            # now owned by the source-aware sampler, so force deterministic routing.
            bucket_common_config.inside_random = False
            bucket_dataset = instantiate(
                bucket_dataset_config,
                common_config=bucket_common_config,
                _recursive_=False,
            )
            bucket_length = len(bucket_dataset)
            if bucket_length <= 0:
                raise ValueError(f"Bucket {name!r} dataset is empty.")

            # 本次修改：若 bucket 声明 batch_temporal_strides，则要求 adapter 暴露每个 stride 的合法窗口索引。
            # The batch sampler chooses one of these groups per optimizer step,
            # avoiding samples with different physical time gaps in one batch.
            raw_batch_temporal_strides = _config_get(
                bucket, "batch_temporal_strides", None
            )
            temporal_strides = ()
            temporal_stride_indices: Dict[int, tuple] = {}
            if raw_batch_temporal_strides is not None:
                temporal_strides = tuple(
                    _strict_positive_int(value, f"{name}.batch_temporal_strides")
                    for value in raw_batch_temporal_strides
                )
                if not temporal_strides or len(set(temporal_strides)) != len(
                    temporal_strides
                ):
                    raise ValueError(
                        f"Bucket {name!r} batch_temporal_strides must be non-empty "
                        f"and unique, got {temporal_strides}."
                    )
                group_getter = getattr(
                    bucket_dataset, "get_temporal_stride_indices", None
                )
                if not callable(group_getter):
                    raise TypeError(
                        f"Bucket {name!r} requests batch-uniform temporal strides, "
                        "but its dataset does not implement "
                        "get_temporal_stride_indices()."
                    )
                available_groups = group_getter()
                for temporal_stride in temporal_strides:
                    indices = tuple(
                        int(index)
                        for index in available_groups.get(temporal_stride, ())
                    )
                    if not indices:
                        raise ValueError(
                            f"Bucket {name!r} has no valid windows for temporal "
                            f"stride {temporal_stride}."
                        )
                    if min(indices) < 0 or max(indices) >= bucket_length:
                        raise IndexError(
                            f"Bucket {name!r} returned out-of-range indices for "
                            f"temporal stride {temporal_stride}."
                        )
                    temporal_stride_indices[temporal_stride] = indices

            # 本次修改：解析该 bucket 的宽高比范围、step 权重、样本成本及可选 batch size 限制。
            aspect_range = _config_get(bucket, "aspect_ratio_range", None)
            if aspect_range is None:
                aspect_range = bucket_common_config.augs.aspects
            if len(aspect_range) != 2:
                raise ValueError(
                    f"Bucket {name!r} aspect_ratio_range must be [min,max]."
                )
            aspect_min, aspect_max = map(float, aspect_range)
            if aspect_min <= 0 or aspect_min > aspect_max:
                raise ValueError(
                    f"Bucket {name!r} has invalid aspect range {aspect_range}."
                )

            sampling_weight = _config_get(bucket, "sampling_weight", None)
            if sampling_weight is not None:
                sampling_weight = float(sampling_weight)
            cost_per_sample = float(
                _config_get(bucket, "cost_per_sample", total_slots)
            )
            if (
                (sampling_weight is not None and sampling_weight <= 0)
                or cost_per_sample <= 0
            ):
                raise ValueError(
                    f"Bucket {name!r} requires positive optional "
                    "sampling_weight and cost_per_sample, got "
                    f"{sampling_weight}, {cost_per_sample}."
                )
            batch_size_override = _config_get(bucket, "batch_size", None)
            max_batch_size = _config_get(bucket, "max_batch_size", None)
            if batch_size_override is not None:
                batch_size_override = int(batch_size_override)
                if batch_size_override <= 0:
                    raise ValueError(f"Bucket {name!r} batch_size must be positive.")
            if max_batch_size is not None:
                max_batch_size = int(max_batch_size)
                if max_batch_size <= 0:
                    raise ValueError(
                        f"Bucket {name!r} max_batch_size must be positive."
                    )

            # 本次修改：登记底层 dataset、真实长度和布局 spec，供采样器计算权重及动态 local B。
            self.datasets.append(bucket_dataset)
            self.bucket_lengths.append(bucket_length)
            self.bucket_temporal_stride_indices.append(temporal_stride_indices)
            self.bucket_specs.append(
                SourceBucketSpec(
                    name=name,
                    views_per_timestep=views,
                    history_timesteps=history_timesteps,
                    future_timesteps=future_timesteps,
                    total_slots=total_slots,
                    sampling_weight=sampling_weight,
                    cost_per_sample=cost_per_sample,
                    aspect_min=aspect_min,
                    aspect_max=aspect_max,
                    batch_size_override=batch_size_override,
                    max_batch_size=max_batch_size,
                    temporal_strides=temporal_strides,
                )
            )

    def __len__(self):
        return sum(self.bucket_lengths)

    # 本次修改：消费 (bucket_id, sample_id, S, aspect) 并在默认 collate 前完成跨模态槽位检查。
    def __getitem__(self, index):
        """Consume ``(bucket_id, local_index, image_num, aspect_ratio)``."""
        if not isinstance(index, tuple) or len(index) != 4:
            raise ValueError(
                "SourceBucketDataset expects "
                "(bucket_id, sample_index, image_num, aspect_ratio)."
            )
        bucket_id, sample_index, image_num, aspect_ratio = index
        bucket_id = int(bucket_id)
        spec = self.bucket_specs[bucket_id]
        if int(image_num) != spec.total_slots:
            raise ValueError(
                f"Sampler/layout mismatch for {spec.name!r}: expected "
                f"{spec.total_slots} slots, got {image_num}."
            )

        sample = self.datasets[bucket_id][
            (int(sample_index), spec.total_slots, float(aspect_ratio))
        ]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"Bucket {spec.name!r} dataset must return a mapping, "
                f"got {type(sample).__name__}."
            )
        sample = dict(sample)

        # 本次修改：在 collate 前校验所有几何模态的槽位数，尽早发现相机/时间顺序 adapter 错误。
        images = sample.get("images")
        if (
            images is None
            or not hasattr(images, "shape")
            or len(images.shape) != 4
            or int(images.shape[1]) != 3
        ):
            raise TypeError(
                f"Bucket {spec.name!r} must return stacked images [S,3,H,W] "
                f"before default collation, got {getattr(images, 'shape', None)}."
            )
        for key in self._TEMPORAL_KEYS:
            value = sample.get(key)
            if value is None or not hasattr(value, "shape") or len(value.shape) == 0:
                continue
            if int(value.shape[0]) != spec.total_slots:
                raise ValueError(
                    f"Bucket {spec.name!r} returned {key}.shape[0]="
                    f"{value.shape[0]}, expected {spec.total_slots}. The adapter "
                    "must return time-major tensors with a shared slot order."
                )

        # 本次修改：若 adapter 主动声明 layout，则与 bucket 交叉校验，避免把 8 个单目时刻误标成双目 4 时刻。
        adapter_layout = {
            "views_per_timestep": spec.views_per_timestep,
            "history_timesteps": spec.history_timesteps,
            "future_timesteps": spec.future_timesteps,
        }
        for key, expected in adapter_layout.items():
            if key not in sample:
                continue
            value = sample[key]
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(
                        f"Adapter metadata {key!r} must be scalar, got "
                        f"shape={tuple(value.shape)}."
                    )
                value = value.item()
            adapter_value = _strict_positive_int(value, f"adapter.{key}")
            if adapter_value != expected:
                raise ValueError(
                    f"Adapter/bucket layout mismatch for {spec.name!r}: "
                    f"adapter {key}={value}, bucket expects {expected}."
                )

        # 本次修改：把每批统一的非学习 layout 元数据交给 Trainer/VGGT/FM 动态切片和构造 RoPE。
        sample.update(
            {
                "source_bucket_id": bucket_id,
                "source_bucket_name": spec.name,
                "views_per_timestep": spec.views_per_timestep,
                "history_timesteps": spec.history_timesteps,
                "future_timesteps": spec.future_timesteps,
                "history_slots": spec.history_slots,
                "future_slots": spec.future_slots,
                "num_image_slots": spec.total_slots,
            }
        )
        return sample

    # 本次修改：把统一 epoch 继续传到底层各 dataset，使窗口/增强能够按 epoch 确定性更新。
    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


# 本次修改：生成跨 rank 同步的 source/layout 日程，并只在样本索引层做 DDP 分片。
class SourceAwareBatchSampler(Sampler[List[tuple]]):
    """Generate a rank-synchronized layout schedule and rank-sharded indices."""

    def __init__(
        self,
        dataset: SourceBucketDataset,
        max_img_per_gpu: int,
        batches_per_epoch: Optional[int],
        shuffle: bool,
        seed: int,
        bucket_sampling: str = "weighted_random",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.dataset = dataset
        self.max_img_per_gpu = int(max_img_per_gpu)
        if self.max_img_per_gpu <= 0:
            raise ValueError("max_img_per_gpu must be positive.")
        # 本次修改：从已初始化的分布式环境读取 rank/world size，也允许单元测试显式注入。
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank() if rank is None else int(rank)
            self.world_size = (
                dist.get_world_size() if world_size is None else int(world_size)
            )
        else:
            self.rank = 0 if rank is None else int(rank)
            self.world_size = 1 if world_size is None else int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank={self.rank} must be in [0, {self.world_size - 1}]."
            )

        # 本次修改：初始化采样模式、确定性随机种子以及精确的每 epoch batch 数。
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_sampling = str(bucket_sampling)
        if self.bucket_sampling not in ("weighted_random", "round_robin"):
            raise ValueError(
                "bucket_sampling must be 'weighted_random' or 'round_robin', "
                f"got {self.bucket_sampling!r}."
            )
        self.epoch = 0
        if batches_per_epoch is None:
            total_cost = sum(
                length * spec.cost_per_sample
                for length, spec in zip(
                    dataset.bucket_lengths, dataset.bucket_specs
                )
            )
            batches_per_epoch = math.ceil(
                total_cost / (self.max_img_per_gpu * self.world_size)
            )
        self.batches_per_epoch = int(batches_per_epoch)
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive.")
        if (
            self.bucket_sampling == "round_robin"
            and self.batches_per_epoch < len(self.dataset.bucket_specs)
        ):
            raise ValueError(
                "round_robin validation needs at least one batch per source: "
                f"batches_per_epoch={self.batches_per_epoch}, "
                f"buckets={len(self.dataset.bucket_specs)}."
            )

    # 本次修改：epoch 参与所有私有 RNG 的种子，但不引入 rank，从而保持各 GPU 布局一致。
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    # 本次修改：按图像/显存成本预算计算当前 bucket 的单卡 B，并应用显式上下限覆盖。
    def _local_batch_size(self, spec: SourceBucketSpec) -> int:
        if spec.batch_size_override is not None:
            batch_size = spec.batch_size_override
        else:
            batch_size = max(
                1, int(math.floor(self.max_img_per_gpu / spec.cost_per_sample))
            )
        if spec.max_batch_size is not None:
            batch_size = min(batch_size, spec.max_batch_size)
        return batch_size

    # 本次修改：为每个 source/cycle 构造所有 rank 完全一致的确定性样本排列。
    def _global_pool(
        self,
        bucket_id: int,
        cycle: int,
        temporal_stride: Optional[int] = None,
    ) -> List[int]:
        """Build the same shuffled source/stride index pool on every rank."""
        # 本次修改：有 batch 级 stride 时只打乱对应窗口池；否则保持原来的完整 dataset 索引池。
        grouped_indices = getattr(
            self.dataset, "bucket_temporal_stride_indices", None
        )
        if temporal_stride is None or grouped_indices is None:
            order = list(range(self.dataset.bucket_lengths[bucket_id]))
        else:
            order = list(grouped_indices[bucket_id][temporal_stride])
        if self.shuffle:
            pool_seed = (
                self.seed
                + self.epoch * 1_000_003
                + bucket_id * 10_007
                + cycle * 101
                + (0 if temporal_stride is None else temporal_stride * 1_009)
            )
            random.Random(pool_seed).shuffle(order)
        return order

    # 本次修改：逐 step 选择一个同质 bucket，生成统一 V/S/aspect/B 后再切出当前 rank 的索引。
    def __iter__(self):
        # 本次修改：bucket/layout RNG 不含 rank，保证所有 GPU 每一步使用相同 source、S 和 local B。
        schedule_rng = random.Random(self.seed + self.epoch * 1_000_003)
        # An explicit sampling_weight is a per-optimizer-step weight. Without
        # one, N_i/B_i approximates one pass over each bucket's samples.
        weights = [
            spec.sampling_weight
            if spec.sampling_weight is not None
            else length / self._local_batch_size(spec)
            for length, spec in zip(
                self.dataset.bucket_lengths, self.dataset.bucket_specs
            )
        ]
        # 本次修改：每个 (source bucket, temporal stride) 拥有独立游标，避免不同 stride 互相消耗索引周期。
        states: Dict[tuple, Dict[str, Any]] = {}

        # 本次修改：一次取完整 global batch；数据足够时保证 rank 间无重复，过小时确定性循环。
        def take_global_indices(
            bucket_id: int,
            count: int,
            temporal_stride: Optional[int] = None,
        ) -> List[int]:
            """Take one global batch, avoiding rank overlap whenever N >= global_B."""
            state_key = (bucket_id, temporal_stride)
            state = states.setdefault(
                state_key, {"cycle": 0, "pool": [], "cursor": 0}
            )
            grouped_indices = getattr(
                self.dataset, "bucket_temporal_stride_indices", None
            )
            if temporal_stride is None or grouped_indices is None:
                source_length = self.dataset.bucket_lengths[bucket_id]
            else:
                source_length = len(grouped_indices[bucket_id][temporal_stride])

            if source_length >= count:
                # Do not straddle shuffle cycles: discard a short tail and start
                # a fresh permutation so one global batch contains no duplicates.
                if not state["pool"] or state["cursor"] + count > len(state["pool"]):
                    state["pool"] = self._global_pool(
                        bucket_id, state["cycle"], temporal_stride
                    )
                    state["cursor"] = 0
                    state["cycle"] += 1
                start = state["cursor"]
                state["cursor"] += count
                return state["pool"][start:start + count]

            # A source smaller than world_size*local_B cannot be disjoint across
            # all ranks. Cycle deterministically and make the unavoidable repeat explicit.
            result = []
            while len(result) < count:
                if state["cursor"] >= len(state["pool"]):
                    state["pool"] = self._global_pool(
                        bucket_id, state["cycle"], temporal_stride
                    )
                    state["cursor"] = 0
                    state["cycle"] += 1
                available = min(
                    count - len(result), len(state["pool"]) - state["cursor"]
                )
                start = state["cursor"]
                result.extend(state["pool"][start:start + available])
                state["cursor"] += available
            return result

        for step in range(self.batches_per_epoch):
            # 本次修改：训练可按权重随机选 source；验证用 round_robin 保证稳定覆盖每个 bucket。
            if self.bucket_sampling == "round_robin":
                bucket_id = step % len(self.dataset.bucket_specs)
            else:
                bucket_id = schedule_rng.choices(
                    range(len(self.dataset.bucket_specs)),
                    weights=weights,
                    k=1,
                )[0]
            spec = self.dataset.bucket_specs[bucket_id]
            batch_size = self._local_batch_size(spec)
            # 本次修改：训练对 1..10 等配置做均匀随机选择；验证按 stride 轮询，且所有 rank/batch 样本一致。
            temporal_stride = None
            if spec.temporal_strides:
                if self.bucket_sampling == "round_robin":
                    # Continue the deterministic stride cycle across epochs; with
                    # a short validation limit (for example two batches), later
                    # epochs still reach every configured stride instead of
                    # repeatedly evaluating only stride 1 and 2.
                    global_step = self.epoch * self.batches_per_epoch + step
                    bucket_visit = global_step // len(self.dataset.bucket_specs)
                    temporal_stride = spec.temporal_strides[
                        bucket_visit % len(spec.temporal_strides)
                    ]
                else:
                    temporal_stride = schedule_rng.choice(spec.temporal_strides)
            aspect_ratio = (
                spec.aspect_min
                if self.bucket_sampling == "round_robin"
                else round(
                    schedule_rng.uniform(spec.aspect_min, spec.aspect_max), 2
                )
            )
            global_indices = take_global_indices(
                bucket_id,
                batch_size * self.world_size,
                temporal_stride=temporal_stride,
            )
            rank_start = self.rank * batch_size
            sample_indices = global_indices[rank_start:rank_start + batch_size]
            yield [
                (bucket_id, sample_index, spec.total_slots, aspect_ratio)
                for sample_index in sample_indices
            ]

    def __len__(self):
        return self.batches_per_epoch


# 本次修改：worker 随机种子同时纳入全局 seed、epoch、rank 和 worker id，避免重复增强序列。
def _seed_worker(
    worker_id: int,
    *,
    seed: int,
    epoch: int,
    rank: int,
    user_worker_init_fn: Optional[Callable],
):
    worker_seed = (seed + epoch * 1_000_003 + rank * 10_007 + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if user_worker_init_fn is not None:
        user_worker_init_fn(worker_id)


# 本次修改：提供与现有 Trainer/Hydra 兼容的 DataLoader factory，封装 bucket dataset 和 batch sampler。
class SourceAwareDynamicTorchDataset:
    """Hydra-compatible DataLoader factory for source-aware variable-length mixing."""

    # Trainer checks this flag so every rank passes the same epoch to get_loader.
    synchronize_layout_across_ranks = True

    def __init__(
        self,
        dataset: Any,
        common_config: Any,
        source_buckets: Sequence[Any],
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        max_img_per_gpu: int,
        batches_per_epoch: Optional[int] = None,
        bucket_sampling: str = "weighted_random",
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        # 本次修改：构造 source 路由 dataset、动态 sampler，并记录每个 bucket 的布局/预算诊断信息。
        del drop_last, kwargs  # batches are always complete by deterministic cycling
        self._seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.dataset = SourceBucketDataset(
            dataset=dataset,
            common_config=common_config,
            source_buckets=source_buckets,
        )
        self.batch_sampler = SourceAwareBatchSampler(
            dataset=self.dataset,
            max_img_per_gpu=max_img_per_gpu,
            batches_per_epoch=batches_per_epoch,
            shuffle=shuffle,
            seed=self._seed,
            bucket_sampling=bucket_sampling,
        )

        if self.batch_sampler.rank == 0:
            for bucket_length, spec in zip(
                self.dataset.bucket_lengths, self.dataset.bucket_specs
            ):
                local_batch_size = self.batch_sampler._local_batch_size(spec)
                if (
                    spec.batch_size_override is None
                    and spec.cost_per_sample > self.batch_sampler.max_img_per_gpu
                ):
                    logging.warning(
                        "Source bucket %s has cost %.2f above max_img_per_gpu=%d; "
                        "the minimum local batch size is still 1. Profile this "
                        "layout explicitly because the image budget cannot reduce "
                        "single-sample memory further.",
                        spec.name,
                        spec.cost_per_sample,
                        self.batch_sampler.max_img_per_gpu,
                    )
                if bucket_length < local_batch_size * self.batch_sampler.world_size:
                    logging.warning(
                        "Source bucket %s has only %d samples but one global batch "
                        "needs %d; cross-rank sample repetition is unavoidable.",
                        spec.name,
                        bucket_length,
                        local_batch_size * self.batch_sampler.world_size,
                    )
                logging.info(
                    "Source bucket %s: V=%d, history=%d, future=%d, S=%d, "
                    "local_B=%d, step_weight=%s, cost=%.2f, temporal_strides=%s",
                    spec.name,
                    spec.views_per_timestep,
                    spec.history_slots,
                    spec.future_slots,
                    spec.total_slots,
                    local_batch_size,
                    (
                        f"{spec.sampling_weight:.3f}"
                        if spec.sampling_weight is not None
                        else "auto(N/B)"
                    ),
                    spec.cost_per_sample,
                    spec.temporal_strides or "per-sample/default",
                )

    # 本次修改：Trainer 在 Hydra 实例化后覆盖 seed 时，同步更新 batch 日程 RNG，而不只改 wrapper 属性。
    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = int(value)
        if hasattr(self, "batch_sampler"):
            self.batch_sampler.seed = self._seed

    # 本次修改：按 epoch 重建 DataLoader，安装确定性 worker seed，并沿用 batch_sampler 的完整 batch。
    def get_loader(self, epoch: int):
        epoch = int(epoch)
        self.batch_sampler.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=partial(
                _seed_worker,
                seed=self._seed,
                epoch=epoch,
                rank=self.batch_sampler.rank,
                user_worker_init_fn=self.worker_init_fn,
            ),
        )
