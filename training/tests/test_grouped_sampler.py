import unittest

from torch.utils.data import ConcatDataset, Dataset

from data.composed_dataset import TupleConcatDataset
from data.grouped_sampler import GroupedDynamicBatchSampler


class _DummyDataset(Dataset):
    """用于测试采样逻辑的最小虚拟数据集。"""

    def __init__(self, length, name="dummy"):
        self.length = length
        self.name = name

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        return self.name, index


class _CommonConfig(dict):
    """同时支持字典和属性访问，模拟 OmegaConf 配置对象。"""

    __getattr__ = dict.__getitem__


class GroupedDynamicBatchSamplerTest(unittest.TestCase):
    def _make_sampler(self):
        # 两个数据集的虚拟长度为 2:1，用于验证分组与长度兑现行为。
        dataset = ConcatDataset([_DummyDataset(40), _DummyDataset(20)])
        return GroupedDynamicBatchSampler(
            dataset=dataset,
            aspect_ratio_range=[0.5, 1.0],
            image_num_range=[2, 2],
            max_img_per_gpu=8,
            shuffle=False,
            seed=7,
            drop_last=True,
        )

    def test_batches_are_dataset_homogeneous_and_exhaust_virtual_lengths(self):
        """验证一个 batch 不跨数据集，且各虚拟长度均被完整遍历。"""
        sampler = self._make_sampler()
        batches = list(sampler)

        first_dataset_samples = 0
        second_dataset_samples = 0
        for batch in batches:
            global_indices = [item[0] for item in batch]
            from_first = [index < 40 for index in global_indices]
            self.assertTrue(all(from_first) or not any(from_first))
            self.assertTrue(all(item[1] == 2 for item in batch))
            if all(from_first):
                first_dataset_samples += len(batch)
            else:
                second_dataset_samples += len(batch)

        self.assertEqual(first_dataset_samples, 40)
        self.assertEqual(second_dataset_samples, 20)

    def test_epoch_schedule_is_reproducible(self):
        """验证相同 seed 和 epoch 会产生完全一致的 batch 顺序。"""
        first = self._make_sampler()
        second = self._make_sampler()
        first.set_epoch(3)
        second.set_epoch(3)
        self.assertEqual(list(first), list(second))

    def test_tuple_concat_preserves_sampler_selected_dataset(self):
        """验证 ConcatDataset 不会覆盖分组采样器选定的数据集。"""
        common_config = _CommonConfig(inside_random=True, grouped_sampling=True)
        dataset = TupleConcatDataset(
            [_DummyDataset(8, "first"), _DummyDataset(4, "second")],
            common_config,
        )
        sampler = GroupedDynamicBatchSampler(
            dataset=dataset,
            aspect_ratio_range=[1.0, 1.0],
            image_num_range=[2, 2],
            max_img_per_gpu=4,
            shuffle=False,
            seed=11,
            drop_last=True,
        )

        for batch in sampler:
            dataset_names = {dataset[item][0] for item in batch}
            self.assertEqual(len(dataset_names), 1)


if __name__ == "__main__":
    unittest.main()
