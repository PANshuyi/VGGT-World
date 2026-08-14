"""VKITTI/MVS-Synth 固定随机 stride 采样的边界回归测试。"""

import random

import numpy as np
import pytest

from data.datasets.driving_parquet import sample_fixed_stride_positions


def test_training_uses_one_random_stride_between_two_and_five():
    for seed in range(32):
        random.seed(seed)
        positions = sample_fixed_stride_positions(
            100, 10, min_stride=2, max_stride=5, training=True, allow_duplicates=False
        )
        gaps = np.diff(positions)
        assert np.all(gaps == gaps[0])
        assert 2 <= gaps[0] <= 5
        assert positions[0] >= 0
        assert positions[-1] < 100


def test_near_boundary_restricts_stride_instead_of_overflowing():
    random.seed(7)
    positions = sample_fixed_stride_positions(
        25, 10, min_stride=2, max_stride=5, training=True, allow_duplicates=False
    )
    np.testing.assert_array_equal(np.diff(positions), np.full(9, 2))
    assert positions[-1] < 25


def test_validation_can_randomize_stride_between_two_and_five():
    observed_strides = set()
    for seed in range(32):
        random.seed(seed)
        positions = sample_fixed_stride_positions(
            100,
            10,
            min_stride=2,
            max_stride=5,
            training=False,
            allow_duplicates=False,
            randomize=True,
        )
        gaps = np.diff(positions)
        assert np.all(gaps == gaps[0])
        assert 2 <= gaps[0] <= 5
        assert positions[0] >= 0
        assert positions[-1] < 100
        observed_strides.add(int(gaps[0]))
    assert observed_strides == {2, 3, 4, 5}


def test_validation_can_still_be_made_deterministic_explicitly():
    positions = sample_fixed_stride_positions(
        100,
        10,
        min_stride=2,
        max_stride=5,
        training=False,
        allow_duplicates=False,
        randomize=False,
    )
    np.testing.assert_array_equal(positions, np.arange(0, 20, 2))


def test_short_sequence_degrades_to_consecutive_frames():
    positions = sample_fixed_stride_positions(
        15, 10, min_stride=2, max_stride=5, training=False, allow_duplicates=False
    )
    np.testing.assert_array_equal(positions, np.arange(10))


def test_too_short_sequence_requires_duplicate_permission():
    with pytest.raises(ValueError):
        sample_fixed_stride_positions(
            5, 10, min_stride=2, max_stride=5, training=False, allow_duplicates=False
        )

    positions = sample_fixed_stride_positions(
        5, 10, min_stride=2, max_stride=5, training=False, allow_duplicates=True
    )
    assert len(positions) == 10
    assert np.all(np.diff(positions) >= 0)
    assert positions[0] == 0 and positions[-1] == 4
