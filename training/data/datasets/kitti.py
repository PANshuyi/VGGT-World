"""KITTI RAW 单物理相机序列 loader。"""

from data.datasets.driving_parquet import DrivingParquetDataset


class KITTIDataset(DrivingParquetDataset):
    def __init__(self, common_conf, **kwargs):
        # 这里读的是 DVGT 已预处理的 KITTI RAW，不是 Odometry zip。
        super().__init__(common_conf, dataset_name="kitti", original_fps=10, **kwargs)
