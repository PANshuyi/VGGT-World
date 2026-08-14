"""MVS-Synth 和 VKITTI 场景级划分的回归测试。"""

from pathlib import Path

from data.datasets.mvs_synth import split_mvs_synth_scenes
from data.datasets.vkitti import split_vkitti_camera_sequences
from data.datasets.driving_parquet import (
    sample_metadata_scenes,
    split_nonoverlapping_positions,
)

import pandas as pd


def test_mvs_synth_uses_mapanything_95_5_scene_split():
    scenes = [Path(f"{index:04d}") for index in range(120)]
    train, val = split_mvs_synth_scenes(scenes, val_ratio=0.05, split_seed=42)

    assert len(train) == 114
    assert len(val) == 6
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(scenes)
    # 独立 RNG 必须让划分不受外部训练随机状态影响。
    assert (train, val) == split_mvs_synth_scenes(
        reversed(scenes), val_ratio=0.05, split_seed=42
    )


def test_vkitti_holds_out_all_of_scene20_without_leakage():
    root = Path("/dataset/vkitti")
    sequences = [
        root / scene / condition / "frames" / "rgb" / camera
        for scene in ("Scene01", "Scene20")
        for condition in ("clone", "rain")
        for camera in ("Camera_0", "Camera_1")
    ]

    train = split_vkitti_camera_sequences(sequences, root, "train")
    val = split_vkitti_camera_sequences(sequences, root, "val")

    assert len(train) == 4
    assert len(val) == 4
    assert all(path.relative_to(root).parts[0] != "Scene20" for path in train)
    assert all(path.relative_to(root).parts[0] == "Scene20" for path in val)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(sequences)


def test_openscene_scene_sampling_keeps_complete_scenes_and_is_reproducible():
    metadata = pd.DataFrame(
        [
            {"scene_id": f"scene-{scene:03d}", "cam_type": camera, "frame_idx": frame}
            for scene in range(205)
            for camera in ("CAM_F0", "CAM_B0")
            for frame in range(40)
        ]
        + [
            {"scene_id": "too-short", "cam_type": camera, "frame_idx": frame}
            for camera in ("CAM_F0", "CAM_B0")
            for frame in range(39)
        ]
    )

    sampled = sample_metadata_scenes(
        metadata, max_scenes=200, seed=42, min_unique_frames=40
    )
    sampled_again = sample_metadata_scenes(
        metadata, max_scenes=200, seed=42, min_unique_frames=40
    )

    assert sampled["scene_id"].nunique() == 200
    assert "too-short" not in set(sampled["scene_id"])
    # 入选 scene 必须保留全部 2 路 x 40 帧，不能帧级抽样。
    assert sampled.groupby("scene_id").size().eq(80).all()
    assert sampled.index.tolist() == sampled_again.index.tolist()


def test_validation_clips_cover_sequence_without_overlap():
    clips = split_nonoverlapping_positions(41, 12, stride=2)

    # 41 帧按 stride=2 得到 21 帧，再切为 12+9；尾段不会重复补齐。
    assert [len(clip) for clip in clips] == [12, 9]
    assert [value for clip in clips for value in clip] == list(range(0, 41, 2))


def test_synthetic_validation_uses_fixed_stride_five_and_stable_clips():
    """MVS/VKITTI 验证应与真实数据一样保存确定 clip。"""
    clips = split_nonoverlapping_positions(121, 12, stride=5)

    # 0..120 每 5 帧取一帧，得到 25 帧，切成 12+12；
    # 最后单帧 120 不能构成 VGGT 验证样本，因此丢弃。
    assert [len(clip) for clip in clips] == [12, 12]
    assert clips[0].tolist() == list(range(0, 60, 5))
    assert clips[1].tolist() == list(range(60, 120, 5))
