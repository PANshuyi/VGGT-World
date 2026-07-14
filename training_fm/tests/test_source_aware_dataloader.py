# 本次修改：验证动态 B、DDP 同布局异样本、epoch 可复现、验证轮询及跨周期不重叠等采样不变量。
import unittest

from training_fm.source_aware_dataloader import (
    SourceAwareBatchSampler,
    SourceBucketSpec,
)


# 本次修改：集中构造不同相机路数的最小 bucket spec，避免各测试重复布局配置。
def _spec(name, views, total_slots, weight=1.0, temporal_strides=()):
    return SourceBucketSpec(
        name=name,
        views_per_timestep=views,
        history_timesteps=2,
        future_timesteps=2,
        total_slots=total_slots,
        sampling_weight=weight,
        cost_per_sample=float(total_slots),
        aspect_min=1.0,
        aspect_max=1.0,
        batch_size_override=None,
        max_batch_size=None,
        temporal_strides=tuple(temporal_strides),
    )


# 本次修改：提供常规三布局及非整除小数据源 fixture，只测试 sampler 而不依赖真实图像。
class _FakeBucketDataset:
    bucket_lengths = [120, 120, 120]
    bucket_specs = [
        _spec("mono", views=1, total_slots=4),
        _spec("stereo", views=2, total_slots=8),
        _spec("six_camera", views=6, total_slots=24),
    ]


class _NonDivisibleDataset:
    bucket_lengths = [5]
    bucket_specs = [_spec("small", views=1, total_slots=4)]


# 本次修改：构造三个互不重叠的 stride 索引池，验证 batch 内时间间隔统一且 DDP 同步。
class _TemporalGroupedDataset:
    bucket_lengths = [120]
    bucket_specs = [
        _spec(
            "stereo_temporal",
            views=2,
            total_slots=8,
            temporal_strides=(1, 2, 3),
        )
    ]
    bucket_temporal_stride_indices = [
        {
            1: tuple(range(0, 40)),
            2: tuple(range(40, 80)),
            3: tuple(range(80, 120)),
        }
    ]


def _stride_from_grouped_index(index):
    return index // 40 + 1


# 本次修改：覆盖 source-aware sampler 对显存预算、分布式同步和确定性验证的核心行为。
class SourceAwareBatchSamplerTest(unittest.TestCase):
    def _sampler(self, rank=0, world_size=1, epoch=3):
        sampler = SourceAwareBatchSampler(
            dataset=_FakeBucketDataset(),
            max_img_per_gpu=24,
            batches_per_epoch=200,
            shuffle=True,
            seed=17,
            rank=rank,
            world_size=world_size,
        )
        sampler.set_epoch(epoch)
        return sampler

    def test_dynamic_batch_size_follows_image_budget(self):
        observed = {}
        for batch in self._sampler():
            bucket_id = batch[0][0]
            observed[bucket_id] = len(batch)
        self.assertEqual(observed, {0: 6, 1: 3, 2: 1})

    def test_all_ranks_share_layout_but_read_different_indices(self):
        rank0 = list(self._sampler(rank=0, world_size=2))
        rank1 = list(self._sampler(rank=1, world_size=2))
        self.assertEqual(len(rank0), len(rank1))

        for batch0, batch1 in zip(rank0, rank1):
            layout0 = (batch0[0][0], batch0[0][2], batch0[0][3], len(batch0))
            layout1 = (batch1[0][0], batch1[0][2], batch1[0][3], len(batch1))
            self.assertEqual(layout0, layout1)
            self.assertTrue(set(item[1] for item in batch0).isdisjoint(
                item[1] for item in batch1
            ))

    def test_epoch_is_reproducible_and_changes_schedule(self):
        first = list(self._sampler(epoch=5))
        repeated = list(self._sampler(epoch=5))
        next_epoch = list(self._sampler(epoch=6))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)

    def test_round_robin_validation_covers_every_bucket(self):
        sampler = SourceAwareBatchSampler(
            dataset=_FakeBucketDataset(),
            max_img_per_gpu=24,
            batches_per_epoch=6,
            shuffle=False,
            seed=17,
            bucket_sampling="round_robin",
        )
        bucket_ids = [batch[0][0] for batch in sampler]
        self.assertEqual(bucket_ids, [0, 1, 2, 0, 1, 2])

    def test_global_batch_stays_disjoint_across_cycle_boundary(self):
        samplers = []
        for rank in (0, 1):
            sampler = SourceAwareBatchSampler(
                dataset=_NonDivisibleDataset(),
                max_img_per_gpu=8,  # local B=2, global B=4 <= N=5
                batches_per_epoch=20,
                shuffle=True,
                seed=9,
                rank=rank,
                world_size=2,
            )
            samplers.append(list(sampler))
        for batch0, batch1 in zip(*samplers):
            self.assertTrue(set(item[1] for item in batch0).isdisjoint(
                item[1] for item in batch1
            ))

    # 本次修改：48 图预算下双目 8 槽样本得到 B=6，并且一次 batch 只来自一个随机 stride 池。
    def test_batch_uniform_temporal_stride_with_48_image_budget(self):
        sampler = SourceAwareBatchSampler(
            dataset=_TemporalGroupedDataset(),
            max_img_per_gpu=48,
            batches_per_epoch=30,
            shuffle=True,
            seed=23,
        )
        observed_strides = set()
        for batch in sampler:
            self.assertEqual(len(batch), 6)
            batch_strides = {
                _stride_from_grouped_index(item[1]) for item in batch
            }
            self.assertEqual(len(batch_strides), 1)
            observed_strides.update(batch_strides)
        self.assertEqual(observed_strides, {1, 2, 3})

    # 本次修改：验证 round-robin 模式按声明顺序覆盖 stride，便于稳定比较不同时间跨度。
    def test_validation_round_robin_covers_temporal_strides(self):
        sampler = SourceAwareBatchSampler(
            dataset=_TemporalGroupedDataset(),
            max_img_per_gpu=48,
            batches_per_epoch=6,
            shuffle=False,
            seed=23,
            bucket_sampling="round_robin",
        )
        observed = [
            _stride_from_grouped_index(batch[0][1]) for batch in sampler
        ]
        self.assertEqual(observed, [1, 2, 3, 1, 2, 3])

    # 本次修改：DDP 各 rank 必须选择同一 stride，但在该 stride 池内取得互不重复的本地样本。
    def test_temporal_stride_is_synchronized_across_ranks(self):
        rank_batches = []
        for rank in (0, 1):
            sampler = SourceAwareBatchSampler(
                dataset=_TemporalGroupedDataset(),
                max_img_per_gpu=48,
                batches_per_epoch=20,
                shuffle=True,
                seed=23,
                rank=rank,
                world_size=2,
            )
            rank_batches.append(list(sampler))

        for batch0, batch1 in zip(*rank_batches):
            stride0 = {_stride_from_grouped_index(item[1]) for item in batch0}
            stride1 = {_stride_from_grouped_index(item[1]) for item in batch1}
            self.assertEqual(stride0, stride1)
            self.assertEqual(len(stride0), 1)
            self.assertTrue(
                set(item[1] for item in batch0).isdisjoint(
                    item[1] for item in batch1
                )
            )


if __name__ == "__main__":
    unittest.main()
