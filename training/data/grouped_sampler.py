# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""用于多数据集混训的“按数据集分组”动态批采样器。

原始 VGGT 会先拼接全部数据集，再从整个索引空间中采样。本采样器仍然使用
虚拟数据集长度作为权重，但会先选中一个数据集，再完全从该数据集中构造
一个 batch。这样与 DVGT 的混训方式一致，可以避免不同数据集的样本形状
或元数据不兼容，却被 DataLoader 拼进同一个 batch 的问题。
"""

import math
from typing import Iterator, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Sampler


class GroupedDynamicBatchSampler(Sampler):
    """按照虚拟长度混合数据集，并保证每个 batch 内的数据集一致。

    每个索引项的格式为 ``(global_index, image_num, aspect_ratio)``，与
    VGGT 的 :class:`TupleConcatDataset` 接口保持一致。

    数据集选择权重为 ``len(sub_dataset)``。因此，现有数据集的
    ``len_train``，或者后续数据集通过 ``__len__`` 返回的虚拟长度，都会
    直接控制多数据集采样比例。
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        aspect_ratio_range: List[float],
        image_num_range: List[int],
        max_img_per_gpu: int,
        shuffle: bool = False,
        seed: int = 42,
        drop_last: bool = True,
        fixed_batch_size: int = -1,
        fixed_num_images: int = 0,
        fixed_aspect_ratio: float = 0.0,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("GroupedDynamicBatchSampler requires a ConcatDataset")
        if not dataset.datasets:
            raise ValueError("At least one sub-dataset is required")
        if len(aspect_ratio_range) != 2 or aspect_ratio_range[0] > aspect_ratio_range[1]:
            raise ValueError(
                "aspect_ratio_range must be [min, max] with min <= max, "
                f"got {aspect_ratio_range}"
            )
        if (
            len(image_num_range) != 2
            or image_num_range[0] < 1
            or image_num_range[0] > image_num_range[1]
        ):
            raise ValueError(
                "image_num_range must be [min, max] with 1 <= min <= max, "
                f"got {image_num_range}"
            )
        if max_img_per_gpu < image_num_range[0]:
            raise ValueError(
                "max_img_per_gpu must fit at least the minimum number of images "
                f"({image_num_range[0]}), got {max_img_per_gpu}"
            )

        # 保存动态 batch 的公共配置。image_num 表示单个样本中的图像数量，
        # max_img_per_gpu / image_num 决定当前 step 的实际 batch size。
        self.dataset = dataset
        self.aspect_ratio_range = list(aspect_ratio_range)
        self.image_num_range = list(image_num_range)
        self.max_img_per_gpu = max_img_per_gpu
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.fixed_batch_size = fixed_batch_size
        self.fixed_num_images = fixed_num_images
        self.fixed_aspect_ratio = fixed_aspect_ratio
        self.epoch = 0

        # 在 DDP 环境中，每个 rank 只负责各子数据集的一部分索引；本地测试
        # 或单卡训练时退化为 num_replicas=1、rank=0。
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        # 这里读取的是“虚拟长度”。对 VGGT 数据集而言通常就是 len_train，
        # 不要求磁盘中真的复制出这么多份场景。
        self.dataset_lengths = [len(sub_dataset) for sub_dataset in dataset.datasets]
        if any(length <= 0 for length in self.dataset_lengths):
            raise ValueError(
                "All sub-datasets must have a positive virtual length, got "
                f"{self.dataset_lengths}"
            )

        # 记录每个子数据集在 ConcatDataset 全局索引空间中的起始偏移。
        self.dataset_offsets = [0]
        for length in self.dataset_lengths[:-1]:
            self.dataset_offsets.append(self.dataset_offsets[-1] + length)

        # 多数据集混训的核心策略：虚拟长度控制下一个同源 batch 选择各
        # 数据集的概率。后续调整 len_train 即可调整采样比例。
        self.dataset_weights = torch.tensor(self.dataset_lengths, dtype=torch.float64)

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，使每个 epoch 的随机 batch 顺序可复现。"""
        self.epoch = epoch

    def _rank_indices(self, dataset_idx: int, generator: torch.Generator) -> torch.Tensor:
        """为当前 DDP rank 生成某个子数据集对应的全局索引。"""
        dataset_len = self.dataset_lengths[dataset_idx]
        offset = self.dataset_offsets[dataset_idx]

        # 先把每个子数据集分别切分给所有 rank，确保各 rank 获得相同数量
        # 的样本，从而能够同步执行相同数量的分布式训练 step。
        if self.drop_last:
            total_size = dataset_len // self.num_replicas * self.num_replicas
        else:
            total_size = math.ceil(dataset_len / self.num_replicas) * self.num_replicas

        # shuffle=True 时先在子数据集内部打乱；默认关闭，以避免大规模
        # randperm 带来的额外 CPU 内存，并由数据集内部 inside_random 随机。
        if self.shuffle:
            local_indices = torch.randperm(dataset_len, generator=generator)
            if total_size > dataset_len:
                padding_size = total_size - dataset_len
                repeats = math.ceil(padding_size / dataset_len)
                padding = local_indices.repeat(repeats)[:padding_size]
                local_indices = torch.cat((local_indices, padding))
            else:
                local_indices = local_indices[:total_size]
            local_indices = local_indices[self.rank:total_size:self.num_replicas]
        else:
            # 用取模实现与 DistributedSampler 相同的补齐效果，同时不创建
            # 包含全部全局索引的 Python list，降低 CPU 内存开销。
            local_indices = torch.arange(
                self.rank, total_size, self.num_replicas, dtype=torch.int64
            )
            local_indices.remainder_(dataset_len)

        return local_indices + offset

    def _generate_batch_config(
        self, generator: torch.Generator
    ) -> Tuple[int, int, float]:
        """为一个 batch 生成 ``(batch_size, image_num, aspect_ratio)``。"""
        # 调试时可以固定图像数；正常训练则在配置范围内随机采样序列长度。
        if self.fixed_num_images > 0:
            image_num = self.fixed_num_images
        else:
            image_num = torch.randint(
                self.image_num_range[0],
                self.image_num_range[1] + 1,
                (1,),
                generator=generator,
            ).item()

        if image_num > self.max_img_per_gpu and self.fixed_batch_size <= 0:
            raise ValueError(
                f"image_num={image_num} exceeds max_img_per_gpu={self.max_img_per_gpu}"
            )

        # 一个 batch 内所有样本使用相同宽高比，保证 default_collate 可以
        # 直接堆叠图像张量。
        if self.fixed_aspect_ratio > 0:
            aspect_ratio = self.fixed_aspect_ratio
        else:
            low, high = self.aspect_ratio_range
            aspect_ratio = low + (high - low) * torch.rand(1, generator=generator).item()
            aspect_ratio = round(aspect_ratio, 2)

        # 用“每卡最多图像数”限制显存：序列越长，样本 batch size 越小。
        if self.fixed_batch_size > 0:
            batch_size = self.fixed_batch_size
        else:
            batch_size = max(1, self.max_img_per_gpu // image_num)

        return batch_size, image_num, aspect_ratio

    def __iter__(self) -> Iterator[List[Tuple[int, int, float]]]:
        # 所有 rank 必须使用相同随机种子，才能在同一个 DDP step 选择相同
        # 数据集、图像数和宽高比，避免 collective 时出现张量形状不一致。
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # 每个子数据集独立完成 DDP 切分，后续 batch 不会跨数据集取样。
        indices_by_dataset = [
            self._rank_indices(dataset_idx, generator)
            for dataset_idx in range(len(self.dataset_lengths))
        ]
        current_positions = [0] * len(indices_by_dataset)
        rank_lengths = [len(indices) for indices in indices_by_dataset]

        while True:
            # 只从尚未耗尽的子数据集中选择下一批数据。
            available = [
                dataset_idx
                for dataset_idx, position in enumerate(current_positions)
                if position < rank_lengths[dataset_idx]
            ]
            if not available:
                break

            # 根据虚拟长度进行加权随机选择；选择结果在全部 rank 上一致。
            available_weights = self.dataset_weights[available]
            selected = torch.multinomial(
                available_weights, 1, generator=generator
            ).item()
            dataset_idx = available[selected]

            # 当前 batch 的全部索引都从刚刚选中的 dataset_idx 中截取。
            batch_size, image_num, aspect_ratio = self._generate_batch_config(generator)
            start = current_positions[dataset_idx]
            end = min(start + batch_size, rank_lengths[dataset_idx])
            batch_indices = indices_by_dataset[dataset_idx][start:end]
            current_positions[dataset_idx] = end

            # 子数据集末尾不足一个完整 batch 时，按配置决定是否丢弃。
            if len(batch_indices) < batch_size and self.drop_last:
                continue

            yield [
                (int(index), image_num, aspect_ratio)
                for index in batch_indices.tolist()
            ]

    def __len__(self) -> int:
        # batch size 会动态变化，精确计算 batch 数需要重放完整随机序列。
        # 训练配置通常通过 limit_train_batches 明确每个 epoch 的 step 数；
        # 这里返回每个 rank 的样本数上界，保证 DataLoader 的 len() 合法。
        if self.drop_last:
            rank_samples = sum(
                length // self.num_replicas for length in self.dataset_lengths
            )
        else:
            rank_samples = sum(
                math.ceil(length / self.num_replicas)
                for length in self.dataset_lengths
            )
        return rank_samples
