"""Waymo 单物理相机序列 loader。"""

from data.datasets.driving_parquet import DrivingParquetDataset


class WaymoDataset(DrivingParquetDataset):
    def __init__(self, common_conf, **kwargs):
        super().__init__(common_conf, dataset_name="waymo", original_fps=10, **kwargs)
