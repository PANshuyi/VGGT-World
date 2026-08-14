"""MVS-Synth 米制转换与远端 depth 过滤的回归测试。"""

import json

import numpy as np

from data.datasets.mvs_synth import MVSSynthDataset


def test_depth_divide_by_ten_and_positive_p95_filter():
    # 0/负值/NaN/Inf 都不得参与 P95；20 是唯一被 P95 滤掉的最远值。
    raw = np.array([0.0, -1.0, np.nan, np.inf, *range(1, 21)], dtype=np.float32)
    depth = MVSSynthDataset._convert_depth_to_metric(raw, 10.0, 95.0)

    assert np.isfinite(depth).all()
    assert np.count_nonzero(depth) == 19
    assert np.isclose(depth[4], 0.1)
    assert np.isclose(depth[-2], 1.9)
    assert depth[-1] == 0.0


def test_camera_translation_uses_same_divide_by_ten(tmp_path):
    raw_w2c = np.eye(4, dtype=np.float64)
    raw_w2c[:3, 3] = [10.0, 20.0, 30.0]
    camera_path = tmp_path / "0000.json"
    camera_path.write_text(
        json.dumps(
            {
                "f_x": 100.0,
                "f_y": 100.0,
                "c_x": 50.0,
                "c_y": 50.0,
                "extrinsic": raw_w2c.tolist(),
            }
        ),
        encoding="utf-8",
    )

    # 避免构造完整 dataset；该方法只依赖尺度参数。
    dataset = object.__new__(MVSSynthDataset)
    dataset.game_units_per_meter = 10.0
    _, w2c_metric_3x4 = dataset._load_camera(camera_path)
    w2c_metric = np.eye(4, dtype=np.float64)
    w2c_metric[:3] = w2c_metric_3x4
    c2w_metric = np.linalg.inv(w2c_metric)

    flip_y = np.diag([1.0, -1.0, 1.0, 1.0])
    expected_c2w = flip_y @ np.linalg.inv(raw_w2c)
    expected_c2w[:3, 3] /= 10.0
    np.testing.assert_allclose(c2w_metric, expected_c2w, atol=1e-7)
