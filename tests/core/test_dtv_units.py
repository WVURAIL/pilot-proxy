# coding=utf-8
from __future__ import annotations

import math

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    EFFECTIVE_BIN_BW_HZ,
    SPREADING_LOSS_DB,
    fstat_num_den_to_pilot_excess_linear,
    fstat_num_den_to_pnr_bin_db,
    fstat_num_den_to_raw,
    fstat_raw_to_pnr_bin_db,
    pnr_bin_db_to_snr_shelf_db,
    pnr_bin_to_snr_shelf_metadata,
    snr_shelf_db_to_fstat_raw_threshold,
    snr_shelf_db_to_half_threshold_rational,
    snr_shelf_db_to_pnr_bin_db,
    snr_shelf_threshold_fields,
)

REFERENCE_EFFECTIVE_BIN_BW_HZ = 3051.7578125
REFERENCE_SPREADING_LOSS_DB = 32.936012
RAW_FSTAT_LEVEL_EXAMPLE_DB = 1.0
EXPECTED_PNR_BIN_DB_FROM_1DB_RAW_F = -5.868253
EXPECTED_SHELF_SNR_DB_FROM_1DB_RAW_F = -27.504265
REFERENCE_CHANNEL_WIDTH_HZ = 390625.0
REFERENCE_DETECTOR_WINDOW_SAMPLES = 128

THRESHOLD_SNR_SHELF_DB = -26.0
EXPECTED_THRESHOLD_PNR_BIN_DB = -4.363988
EXPECTED_THRESHOLD_RAW_F = 1.366101
EXPECTED_THRESHOLD_HALF = 0.683050614
ABS_TOL_DB = 1e-6
ABS_TOL_HALF_THRESHOLD = 1e-10
NUMDEN_ZERO_REFERENCE_NUM = 1
NUMDEN_ZERO_REFERENCE_DEN = 0


def test_reference_geometry_pnr_to_shelf_example_is_explicit() -> None:
    assert math.isclose(EFFECTIVE_BIN_BW_HZ, REFERENCE_EFFECTIVE_BIN_BW_HZ)
    assert math.isclose(
        SPREADING_LOSS_DB,
        REFERENCE_SPREADING_LOSS_DB,
        abs_tol=ABS_TOL_DB,
    )

    f_raw = DB_LINEAR_BASE ** (RAW_FSTAT_LEVEL_EXAMPLE_DB / DB_POWER_FACTOR)
    pnr_bin_db = fstat_raw_to_pnr_bin_db(f_raw)
    shelf_snr_db = pnr_bin_db_to_snr_shelf_db(pnr_bin_db)

    assert math.isclose(
        pnr_bin_db,
        EXPECTED_PNR_BIN_DB_FROM_1DB_RAW_F,
        abs_tol=ABS_TOL_DB,
    )
    assert math.isclose(
        float(shelf_snr_db),
        EXPECTED_SHELF_SNR_DB_FROM_1DB_RAW_F,
        abs_tol=ABS_TOL_DB,
    )

    metadata = pnr_bin_to_snr_shelf_metadata()
    assert metadata["channel_width_hz"] == REFERENCE_CHANNEL_WIDTH_HZ
    assert metadata["detector_window_samples"] == REFERENCE_DETECTOR_WINDOW_SAMPLES
    assert metadata["bin_enbw_hz"] == EFFECTIVE_BIN_BW_HZ


def test_snr_shelf_threshold_converts_to_kernel_half_threshold() -> None:
    pnr_bin_db = float(snr_shelf_db_to_pnr_bin_db(THRESHOLD_SNR_SHELF_DB))
    raw = snr_shelf_db_to_fstat_raw_threshold(THRESHOLD_SNR_SHELF_DB)
    half_num, half_den = snr_shelf_db_to_half_threshold_rational(
        THRESHOLD_SNR_SHELF_DB
    )
    fields = snr_shelf_threshold_fields(THRESHOLD_SNR_SHELF_DB)

    assert math.isclose(pnr_bin_db, EXPECTED_THRESHOLD_PNR_BIN_DB, abs_tol=ABS_TOL_DB)
    assert math.isclose(raw, EXPECTED_THRESHOLD_RAW_F, abs_tol=ABS_TOL_DB)
    assert math.isclose(
        half_num / half_den,
        EXPECTED_THRESHOLD_HALF,
        abs_tol=ABS_TOL_HALF_THRESHOLD,
    )
    assert fields["threshold_half_num"] == half_num
    assert fields["threshold_half_den"] == half_den
    assert fields["threshold_snr_shelf_db"] == THRESHOLD_SNR_SHELF_DB


def test_numden_zero_denominator_policy_matches_c_helper() -> None:
    assert fstat_num_den_to_raw(
        NUMDEN_ZERO_REFERENCE_NUM,
        NUMDEN_ZERO_REFERENCE_DEN,
    ) == 0.0
    assert fstat_num_den_to_pilot_excess_linear(
        NUMDEN_ZERO_REFERENCE_NUM,
        NUMDEN_ZERO_REFERENCE_DEN,
    ) == 0.0
    assert math.isnan(
        fstat_num_den_to_pnr_bin_db(
            NUMDEN_ZERO_REFERENCE_NUM,
            NUMDEN_ZERO_REFERENCE_DEN,
        )
    )
