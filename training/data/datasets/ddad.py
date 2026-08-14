"""DDAD 单物理相机序列 loader。"""

from data.datasets.driving_parquet import DrivingParquetDataset


class DDADDataset(DrivingParquetDataset):
    def __init__(self, common_conf, **kwargs):
        super().__init__(common_conf, dataset_name="ddad", original_fps=10, **kwargs)
