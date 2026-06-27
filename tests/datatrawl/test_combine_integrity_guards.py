# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datatrawl")

from pilot_proxy.datatrawl_plugins.combine import (
    _check_frames,
    _common_sample_rate_hz,
)


def _product(unit_index, frame_in_unit, *, delta=1.0 / 390_625.0):
    n = len(unit_index)
    return {
        "frame_index": np.arange(n, dtype=np.int64),
        "source_event_keys": np.asarray(["event-a", "event-b"]),
        "frame_unit_index": np.asarray(unit_index, dtype=np.int32),
        "frame_in_unit": np.asarray(frame_in_unit, dtype=np.int32),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "unit_delta_time": np.asarray([delta, delta], dtype=np.float64),
    }


def test_check_frames_rejects_equal_total_with_shifted_unit_boundary():
    first = _product([0, 0, 0, 1, 1], [0, 1, 2, 0, 1])
    second = _product([0, 0, 1, 1, 1], [0, 1, 0, 1, 2])
    second["physical_channel"] = np.asarray([15], dtype=np.int32)
    second["freq_id"] = np.asarray([829], dtype=np.int64)
    with pytest.raises(ValueError, match="different per-frame unit positions"):
        _check_frames([first, second])


def test_common_sample_rate_rejects_mixed_delta_time():
    first = _product([0, 1], [0, 0])
    second = _product([0, 1], [0, 0], delta=1.01 / 390_625.0)
    with pytest.raises(ValueError, match="unit_delta_time"):
        _common_sample_rate_hz([first, second])


def test_common_sample_rate_accepts_consistent_timing():
    first = _product([0, 1], [0, 0])
    second = _product([0, 1], [0, 0])
    assert _common_sample_rate_hz([first, second]) == pytest.approx(390_625.0)
