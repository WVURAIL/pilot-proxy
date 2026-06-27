# coding=utf-8
from __future__ import annotations

import math

import numpy as np
import pytest

from pilot_proxy.masking import masked_mean_excluding
from pilot_proxy.result_schema import (
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    MASK_VALUE_EXCLUDED,
    RESULT_SCHEMA_VERSION,
    fixed_point_contract,
    result_layout,
    result_schema_object,
)

FRAME_SIZE_SAMPLES = 16_384
DETECTOR_WINDOW_SAMPLES = 128
NUM_INPUT_STREAMS = 2
WINDOWS_PER_STREAM = 128
DETECTOR_ROWS = 256
DTV_BANDWIDTH_HZ = 6_000_000.0
BIN_ENBW_HZ = 3_051.7578125
PILOT_BELOW_DATA_DB = 11.3
PILOT_CAPTURE_EFFICIENCY = 1.0
THRESHOLD = {
    "threshold_snr_shelf_db": -26.0,
    "threshold_pnr_bin_db": -4.364,
    "threshold_fstat_raw": 1.366,
    "threshold_half_num": 123,
    "threshold_half_den": 180,
}


def test_result_layout_uses_all_rows_summed_convention() -> None:
    layout = result_layout(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
    )

    assert layout["windows_per_stream"] == WINDOWS_PER_STREAM
    assert layout["detector_rows_per_frame"] == DETECTOR_ROWS
    assert layout["combine_mode"] == COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO


def test_result_schema_object_contains_fixed_contract_and_threshold() -> None:
    schema = result_schema_object(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
        dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
        bin_enbw_hz=BIN_ENBW_HZ,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
        threshold=THRESHOLD,
    )

    assert schema["schema_version"] == RESULT_SCHEMA_VERSION
    assert schema["layout"]["num_input_streams"] == NUM_INPUT_STREAMS
    assert schema["calibration"]["pilot_below_data_db"] == PILOT_BELOW_DATA_DB
    assert schema["threshold"]["threshold_half_den"] == THRESHOLD[
        "threshold_half_den"
    ]
    assert schema["fixed_point_contract"]["detector_window_samples"] == (
        DETECTOR_WINDOW_SAMPLES
    )
    assert schema["fixed_point_contract"]["power_accumulator"] == "uint64"


def test_fixed_point_contract_states_k128_reason() -> None:
    contract = fixed_point_contract()

    assert "K=128" in contract["k128_lock_reason"]
    assert contract["packed_complex_bits"] == 8
    assert contract["sample_bits_per_component"] == 4


def test_masked_mean_excludes_samples_instead_of_zero_filling() -> None:
    values = np.asarray([10.0, 1000.0, 30.0])
    mask = np.asarray([0, MASK_VALUE_EXCLUDED, 0])

    assert masked_mean_excluding(values, mask) == pytest.approx(20.0)
    assert not math.isclose(float(np.mean(values * (1 - mask))), 20.0)
