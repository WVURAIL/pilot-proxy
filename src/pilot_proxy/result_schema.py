# coding=utf-8
"""Stable public result-schema helpers."""

from __future__ import annotations

from typing import Any

RESULT_SCHEMA_VERSION = "pilot_proxy_result_v2"
COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO = "all_rows_summed_before_ratio"
MASK_CONVENTION_VERSION = "fstat_mask_convention_v1"
MASK_VALUE_EXCLUDED = 1
MASK_VALUE_INCLUDED = 0
PERCENT = 100.0
LOCKED_DETECTOR_WINDOW_SAMPLES = 128
LOCKED_NUM_WEIGHT_TERMS = 3
LOCKED_SKIPPED_GUARD_BINS = 1
LOCKED_REFERENCE_OFFSET_BINS = LOCKED_SKIPPED_GUARD_BINS + 1
LOCKED_PACKED_COMPLEX_BITS = 8
LOCKED_SAMPLE_BITS_PER_COMPONENT = 4
POWER_SUM_ACCUMULATOR_BITS = 64

FSTAT_DEFINITION = "F = 2 * P_target / (P_ref_lower + P_ref_upper)"
ALL_ROWS_FSTAT_DEFINITION = (
    "F = 2 * sum_r(P_target,r) / "
    "(sum_r(P_ref_lower,r) + sum_r(P_ref_upper,r))"
)
PILOT_EXCESS_DEFINITION = "rho = F - 1"
SNR_SHELF_DEFINITION = "DTV data-shelf PSD relative to non-DTV noise floor"
MASKED_AVERAGE_DEFINITION = (
    "masked samples are excluded from averages; they are not zero-filled "
    "before averaging"
)
MASKED_AVERAGE_FORMULA = "sum_b(P_b * (1 - M_b)) / sum_b(1 - M_b)"
K128_LOCK_REASON = (
    "With signed 4-bit real/imaginary samples and weights, each real/imaginary "
    "dot-product component grows as 9 + log2(K) bits; K=128 is the largest "
    "power-of-two detector window that keeps the component accumulator inside "
    "signed int16."
)
REFERENCE_OFFSET_LOCK_REASON = (
    "skipped_guard_bins=1 leaves one fine DFT bin between the target bin and "
    "nearest reference bin. This corresponds to reference_offset_bins=2."
)


def statistic_contract() -> dict[str, str]:
    """Return stable definitions for public detector outputs."""
    return {
        "fstat_definition": FSTAT_DEFINITION,
        "all_rows_fstat_definition": ALL_ROWS_FSTAT_DEFINITION,
        "power_sum_rule": "sum powers first, then form F; do not average F-statistics",
        "pilot_excess_definition": PILOT_EXCESS_DEFINITION,
        "snr_shelf_definition": SNR_SHELF_DEFINITION,
    }


def fixed_point_contract(
    *,
    reference_offset_bins: int = LOCKED_REFERENCE_OFFSET_BINS,
) -> dict[str, Any]:
    """Return the locked fixed-point detector contract."""
    return {
        "detector_window_samples": int(LOCKED_DETECTOR_WINDOW_SAMPLES),
        "num_weight_terms": int(LOCKED_NUM_WEIGHT_TERMS),
        "skipped_guard_bins": int(max(0, int(reference_offset_bins) - 1)),
        "reference_offset_bins": int(reference_offset_bins),
        "locked_skipped_guard_bins": int(LOCKED_SKIPPED_GUARD_BINS),
        "locked_reference_offset_bins": int(LOCKED_REFERENCE_OFFSET_BINS),
        "packed_complex_bits": int(LOCKED_PACKED_COMPLEX_BITS),
        "sample_bits_per_component": int(LOCKED_SAMPLE_BITS_PER_COMPONENT),
        "input_format": "complex_int4_packed_int8",
        "power_accumulator": "uint64",
        "power_accumulator_bits": int(POWER_SUM_ACCUMULATOR_BITS),
        "host_masking_policy": "positive_excess_from_uint64_powers",
        "per_frequency_threshold": False,
        "k128_lock_reason": K128_LOCK_REASON,
        "reference_offset_lock_reason": REFERENCE_OFFSET_LOCK_REASON,
    }


def result_layout(
    *,
    frame_size_samples: int,
    num_input_streams: int,
    detector_window_samples: int = LOCKED_DETECTOR_WINDOW_SAMPLES,
    num_selected_channels: int = 1,
) -> dict[str, Any]:
    """Return stable public layout metadata for one combined detector frame."""
    frame = int(frame_size_samples)
    streams = int(num_input_streams)
    window = int(detector_window_samples)
    channels = int(num_selected_channels)
    if frame <= 0 or streams <= 0 or window <= 0 or channels <= 0:
        raise ValueError("frame size, streams, window, and channels must be positive.")
    if frame % window != 0:
        raise ValueError("frame_size_samples must be a multiple of detector window.")
    windows_per_stream = frame // window
    rows = streams * channels * windows_per_stream
    return {
        "frame_size_samples": frame,
        "num_input_streams": streams,
        "num_selected_channels": channels,
        "detector_window_samples": window,
        "windows_per_stream": int(windows_per_stream),
        "detector_rows_per_frame": int(rows),
        "detector_rows_per_block": int(rows),
        "combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        "row_index_rule": (
            "row = ((input_stream_index * num_selected_channels + "
            "selected_channel_index) * windows_per_stream) + window_index"
        ),
    }


def calibration_contract(
    *,
    dtv_bandwidth_hz: float,
    bin_enbw_hz: float,
    pilot_below_data_db: float,
    pilot_capture_efficiency: float,
) -> dict[str, float]:
    """Return SNR/PNR conversion constants used by a run."""
    return {
        "dtv_bandwidth_hz": float(dtv_bandwidth_hz),
        "bin_enbw_hz": float(bin_enbw_hz),
        "pilot_below_data_db": float(pilot_below_data_db),
        "pilot_capture_efficiency": float(pilot_capture_efficiency),
    }


def threshold_contract(threshold: dict[str, Any] | None) -> dict[str, Any]:
    """Return stable threshold fields, using null values when unset."""
    if threshold is None:
        return {
            "threshold_snr_shelf_db": None,
            "threshold_pnr_bin_db": None,
            "threshold_fstat_raw": None,
            "threshold_half_num": None,
            "threshold_half_den": None,
        }
    return {
        "threshold_snr_shelf_db": threshold.get("threshold_snr_shelf_db"),
        "threshold_pnr_bin_db": threshold.get("threshold_pnr_bin_db"),
        "threshold_fstat_raw": threshold.get("threshold_fstat_raw"),
        "threshold_half_num": threshold.get("threshold_half_num"),
        "threshold_half_den": threshold.get("threshold_half_den"),
    }


def mask_convention() -> dict[str, Any]:
    """Return the generic mask convention for before/after averages."""
    return {
        "schema_version": MASK_CONVENTION_VERSION,
        "mask_value_excluded": int(MASK_VALUE_EXCLUDED),
        "mask_value_included": int(MASK_VALUE_INCLUDED),
        "definition": MASKED_AVERAGE_DEFINITION,
        "masked_average_formula": MASKED_AVERAGE_FORMULA,
    }


def result_schema_object(
    *,
    frame_size_samples: int,
    num_input_streams: int,
    detector_window_samples: int,
    dtv_bandwidth_hz: float,
    bin_enbw_hz: float,
    pilot_below_data_db: float,
    pilot_capture_efficiency: float,
    threshold: dict[str, Any] | None,
    reference_offset_bins: int = LOCKED_REFERENCE_OFFSET_BINS,
    num_selected_channels: int = 1,
) -> dict[str, Any]:
    """Build the stable public result-schema object for a run."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "statistic": statistic_contract(),
        "layout": result_layout(
            frame_size_samples=int(frame_size_samples),
            num_input_streams=int(num_input_streams),
            detector_window_samples=int(detector_window_samples),
            num_selected_channels=int(num_selected_channels),
        ),
        "calibration": calibration_contract(
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
            bin_enbw_hz=float(bin_enbw_hz),
            pilot_below_data_db=float(pilot_below_data_db),
            pilot_capture_efficiency=float(pilot_capture_efficiency),
        ),
        "threshold": threshold_contract(threshold),
        "fixed_point_contract": fixed_point_contract(
            reference_offset_bins=int(reference_offset_bins),
        ),
        "mask_convention": mask_convention(),
    }
