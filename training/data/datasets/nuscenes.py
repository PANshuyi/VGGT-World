"""nuScenes 单物理相机序列 loader。"""

from data.datasets.driving_parquet import DrivingParquetDataset


class NuScenesDataset(DrivingParquetDataset):
    def __init__(self, common_conf, **kwargs):
        # DVGT 产物中的目录/文件前缀是 nuscene（无结尾 s）。
        super().__init__(common_conf, dataset_name="nuscene", original_fps=2, **kwargs)
