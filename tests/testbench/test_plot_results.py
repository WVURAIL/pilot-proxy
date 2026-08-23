# coding=utf-8
from __future__ import annotations

import math

import pytest

# noinspection PyProtectedMember
from pilot_proxy.testbench.plot_results import (
    _centered_moving_average,
    _normalized_coarse_power_ratio_db_to_estimated_data_shelf_snr_db,
    _estimated_data_shelf_snr_db_to_normalized_coarse_power_ratio_db,
)

RAW_VALUES = [1.0, 4.0, 7.0, 10.0]
EXPECTED_WINDOW_1 = RAW_VALUES
EXPECTED_WINDOW_3 = [2.5, 4.0, 7.0, 8.5]
VALUES_WITH_NAN = [1.0, math.nan, 7.0]
EXPECTED_NAN_WINDOW_3 = [1.0, 4.0, 7.0]


def test_centered_moving_average_leaves_window_one_unchanged() -> None:
    assert _centered_moving_average(RAW_VALUES, 1) == EXPECTED_WINDOW_1


def test_centered_moving_average_smooths_with_clipped_edges() -> None:
    assert _centered_moving_average(RAW_VALUES, 3) == EXPECTED_WINDOW_3


def test_centered_moving_average_ignores_nan_values() -> None:
    assert _centered_moving_average(VALUES_WITH_NAN, 3) == EXPECTED_NAN_WINDOW_3


def test_snr_shelf_fstat_axis_transform_round_trips() -> None:
    snr_values = [-60.0, -42.0, -26.0, 0.0]

    fstat_values = _estimated_data_shelf_snr_db_to_normalized_coarse_power_ratio_db(snr_values)
    recovered = _normalized_coarse_power_ratio_db_to_estimated_data_shelf_snr_db(fstat_values)

    assert list(recovered) == pytest.approx(snr_values)
