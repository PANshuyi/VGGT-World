"""OpenScene 单物理相机序列 loader。"""

from data.datasets.driving_parquet import DrivingParquetDataset


class OpenSceneDataset(DrivingParquetDataset):
    def __init__(self, common_conf, **kwargs):
        # OpenScene 预处理元数据为 2 FPS。
        super().__init__(common_conf, dataset_name="openscene", original_fps=2, **kwargs)
