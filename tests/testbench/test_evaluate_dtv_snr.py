# coding=utf-8
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pytest

from pilot_proxy.dtv_units import DB_POWER_FACTOR
# noinspection PyProtectedMember
from pilot_proxy.testbench.evaluate_snr import (
    DEFAULT_SNR_SWEEP_MAX_DB,
    DEFAULT_SNR_SWEEP_MIN_DB,
    DEFAULT_SNR_SWEEP_STEP_DB,
    STANDARD_FREQUENCY_OFFSET_SWEEP_HZ,
    _positive_to_db,
    _frequency_offset_values,
    _requested_snr_shelf_values,
    apply_channel_impairments,
    add_complex_awgn_for_snr,
    add_gnuradio_awgn_for_snr,
    required_iq_samples,
)

IQ_SAMPLE_RATE_HZ = 10.0
ADC_SAMPLE_RATE_HZ = 100.0
OUTPUT_SAMPLES = 5
PFB_TAPS = 4
PFB_FFT_SIZE = 20
REQUIRED_IQ_SAMPLES = 17

RNG_SEED = 1234
AWGN_TEST_SAMPLES = 200_000
BANDWIDTH_AWGN_TEST_SAMPLES = 20_000
GNURADIO_AWGN_TEST_SAMPLES = 50_000
NEGATIVE_SNR_DB = -6.0
ZERO_SNR_DB = 0.0
UNIT_SIGNAL_POWER = 1.0
BANDWIDTH_NOISE_POWER = 2.0
FULL_SAMPLE_RATE_HZ = 10.0
HALF_BANDWIDTH_HZ = 5.0
HALF_BANDWIDTH_RATIO = 0.5
ACTUAL_SNR_TOLERANCE_DB = 0.04
GNURADIO_SNR_TOLERANCE_DB = 0.08
POSITIVE_TO_DB_INPUT = 100.0
POSITIVE_TO_DB_OUTPUT = 20.0

SNR_EXPLICIT_DB = -30.0
SNR_RANGE_START_DB = -20.0
SNR_RANGE_STOP_DB = -10.0
SNR_RANGE_STEP_DB = 5.0
SNR_RANGE_EXPECTED = [-30.0, -20.0, -15.0, -10.0]
DEFAULT_SNR_SWEEP_EXPECTED_COUNT = 21
FREQUENCY_OFFSET_HZ = 1_000.0
CHANNEL_GAIN_DB = 6.0
CHANNEL_PHASE_DEG = 90.0
CHANNEL_EFFECT_SAMPLE_RATE_HZ = 4_000.0
DB_AMPLITUDE_FACTOR = 20.0


def test_required_iq_samples_matches_adc_span() -> None:
    required = required_iq_samples(
        iq_sample_rate_hz=IQ_SAMPLE_RATE_HZ,
        adc_sample_rate_hz=ADC_SAMPLE_RATE_HZ,
        num_output_samples=OUTPUT_SAMPLES,
        pfb_taps=PFB_TAPS,
        pfb_fft_size=PFB_FFT_SIZE,
    )

    # (5 + 4 - 1) * 20 ADC samples, last index 159, at 0.1 IQ samples/ADC.
    assert required == REQUIRED_IQ_SAMPLES


def test_add_complex_awgn_for_snr_hits_requested_power_ratio() -> None:
    rng = np.random.default_rng(RNG_SEED)
    signal = np.ones(AWGN_TEST_SAMPLES, dtype=np.complex64)
    noisy, signal_power, noise_power = add_complex_awgn_for_snr(
        signal,
        snr_db=NEGATIVE_SNR_DB,
        rng=rng,
    )

    actual_noise = noisy - signal
    actual_snr = DB_POWER_FACTOR * math.log10(
        float(np.mean(np.abs(signal) ** 2))
        / float(np.mean(np.abs(actual_noise) ** 2))
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(
        DB_POWER_FACTOR * math.log10(signal_power / noise_power),
        NEGATIVE_SNR_DB,
    )
    assert math.isclose(actual_snr, NEGATIVE_SNR_DB, abs_tol=ACTUAL_SNR_TOLERANCE_DB)


def test_add_complex_awgn_for_snr_accounts_for_noise_bandwidth() -> None:
    rng = np.random.default_rng(RNG_SEED)
    signal = np.ones(BANDWIDTH_AWGN_TEST_SAMPLES, dtype=np.complex64)
    _, signal_power, noise_power = add_complex_awgn_for_snr(
        signal,
        snr_db=ZERO_SNR_DB,
        rng=rng,
        sample_rate_hz=FULL_SAMPLE_RATE_HZ,
        snr_bandwidth_hz=HALF_BANDWIDTH_HZ,
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(noise_power, BANDWIDTH_NOISE_POWER, rel_tol=1e-6)
    in_band_noise_power = noise_power * HALF_BANDWIDTH_RATIO
    assert math.isclose(
        DB_POWER_FACTOR * math.log10(signal_power / in_band_noise_power),
        ZERO_SNR_DB,
    )


def test_target_snr_values_accepts_explicit_values_and_range() -> None:
    args = argparse.Namespace(
        requested_snr_shelf_db=[SNR_EXPLICIT_DB],
        snr_start_db=SNR_RANGE_START_DB,
        snr_stop_db=SNR_RANGE_STOP_DB,
        snr_step_db=SNR_RANGE_STEP_DB,
    )

    assert _requested_snr_shelf_values(args) == SNR_RANGE_EXPECTED


def test_target_snr_values_default_to_public_sweep() -> None:
    args = argparse.Namespace(
        requested_snr_shelf_db=None,
        snr_start_db=None,
        snr_stop_db=None,
        snr_step_db=None,
    )

    values = _requested_snr_shelf_values(args)

    assert values[0] == DEFAULT_SNR_SWEEP_MIN_DB
    assert values[-1] == DEFAULT_SNR_SWEEP_MAX_DB
    assert len(values) == DEFAULT_SNR_SWEEP_EXPECTED_COUNT
    assert values[1] - values[0] == DEFAULT_SNR_SWEEP_STEP_DB


def test_frequency_offsets_support_standard_sweep_and_deduplication() -> None:
    args = argparse.Namespace(
        frequency_offset_hz=[0.0, FREQUENCY_OFFSET_HZ],
        standard_frequency_offset_sweep=True,
    )

    assert _frequency_offset_values(args) == [
        0.0,
        FREQUENCY_OFFSET_HZ,
        STANDARD_FREQUENCY_OFFSET_SWEEP_HZ[0],
    ]


def test_apply_channel_impairments_applies_gain_phase_and_frequency() -> None:
    signal = np.ones(4, dtype=np.complex64)
    shifted = apply_channel_impairments(
        signal,
        sample_rate_hz=CHANNEL_EFFECT_SAMPLE_RATE_HZ,
        frequency_offset_hz=FREQUENCY_OFFSET_HZ,
        gain_db=CHANNEL_GAIN_DB,
        phase_deg=CHANNEL_PHASE_DEG,
    )

    assert shifted.shape == signal.shape
    assert np.abs(shifted[0]) == pytest.approx(
        10.0 ** (CHANNEL_GAIN_DB / DB_AMPLITUDE_FACTOR)
    )
    assert shifted[0].real == pytest.approx(0.0, abs=1e-6)
    assert shifted[0].imag > 0.0
    assert shifted[1].real < 0.0


def test_positive_to_db_converts_float_audit_value() -> None:
    assert math.isclose(_positive_to_db(POSITIVE_TO_DB_INPUT), POSITIVE_TO_DB_OUTPUT)
    assert _positive_to_db(ZERO_SNR_DB) == float("-inf")


def test_gnuradio_awgn_helper_hits_requested_band_snr(tmp_path) -> None:
    pytest.importorskip("gnuradio")
    signal = np.ones(GNURADIO_AWGN_TEST_SAMPLES, dtype=np.complex64)
    input_path = tmp_path / "clean.cfile"
    output_path = tmp_path / "noisy.cfile"
    signal.tofile(input_path)

    noisy, signal_power, noise_power, metadata = add_gnuradio_awgn_for_snr(
        signal,
        input_iq_path=input_path,
        output_iq_path=output_path,
        snr_db=ZERO_SNR_DB,
        seed=RNG_SEED,
        gnuradio_python=sys.executable,
        sample_rate_hz=FULL_SAMPLE_RATE_HZ,
        snr_bandwidth_hz=HALF_BANDWIDTH_HZ,
    )

    realized_noise = noisy - signal
    realized_noise_power = float(np.mean(np.abs(realized_noise) ** 2))
    realized_in_band_noise_power = realized_noise_power * HALF_BANDWIDTH_RATIO
    realized_snr_db = DB_POWER_FACTOR * math.log10(
        float(signal_power) / realized_in_band_noise_power
    )

    assert math.isclose(signal_power, UNIT_SIGNAL_POWER, rel_tol=1e-6)
    assert math.isclose(noise_power, BANDWIDTH_NOISE_POWER, rel_tol=1e-6)
    assert metadata["gnuradio_block"] == "analog.noise_source_c"
    assert math.isclose(realized_snr_db, ZERO_SNR_DB, abs_tol=GNURADIO_SNR_TOLERANCE_DB)


def test_wilson_interval_closed_form() -> None:
    from pilot_proxy.testbench.evaluate_snr import wilson_interval

    # n=0 -> undefined
    lo, hi = wilson_interval(0, 0)
    assert lo != lo and hi != hi  # NaN
    # symmetric midpoint case, n=100, k=50: lo/hi ~ 0.404 / 0.596
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.40383, abs=2e-4)
    assert hi == pytest.approx(0.59617, abs=2e-4)
    # edge rates stay inside [0, 1] and are non-degenerate
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    lo, hi = wilson_interval(10, 10)
    assert 0.65 < lo < 1.0 and hi == 1.0


def test_summary_rows_report_detection_rates_with_wilson_bounds() -> None:
    from pilot_proxy.testbench.evaluate_snr import (
        _summarize_rows,
        wilson_interval,
    )

    def _trial(pe: int, mask: int) -> dict:
        return {
            "requested_snr_shelf_db": -30.0,
            "frequency_offset_hz": 0.0,
            "channel_gain_db": 0.0,
            "channel_phase_deg": 0.0,
            "measured_truth_snr_shelf_db": -30.0,
            "measured_truth_composite_atsc_snr_db": -20.0,
            "estimated_snr_shelf_db": -30.0,
            "snr_error_db": 0.0,
            "fstat_raw": 1.0,
            "fstat_level_db": 0.0,
            "pnr_bin_db": 0.0,
            "cpu_float_estimated_snr_shelf_db": -30.0,
            "cpu_float_snr_error_db": 0.0,
            "cpu_float_fstat_raw": 1.0,
            "cpu_gpu_abs_diff": 0.0,
            "cpu_float_gpu_snr_diff_db": 0.0,
            "num_input_streams": 4,
            "positive_excess": pe,
            "mask": mask,
        }

    rows = [_trial(1, 1), _trial(1, 0), _trial(0, 0), _trial(1, 1)]
    summary = _summarize_rows(
        rows,
        requested_values=[-30.0],
        frequency_offset_values=[0.0],
        composite_to_shelf_db=10.0,
        num_input_streams=4,
    )
    assert len(summary) == 1
    row = summary[0]
    assert row["trials"] == 4
    assert row["positive_excess_detection_rate"] == pytest.approx(0.75)
    lo, hi = wilson_interval(3, 4)
    assert row["positive_excess_detection_rate_wilson95_lo"] == pytest.approx(lo)
    assert row["positive_excess_detection_rate_wilson95_hi"] == pytest.approx(hi)
    assert row["threshold_detection_rate"] == pytest.approx(0.5)
    lo, hi = wilson_interval(2, 4)
    assert row["threshold_detection_rate_wilson95_lo"] == pytest.approx(lo)
    assert row["threshold_detection_rate_wilson95_hi"] == pytest.approx(hi)


def test_summary_rows_omit_detection_rates_for_legacy_trials() -> None:
    from pilot_proxy.testbench.evaluate_snr import _summarize_rows

    legacy = {
        "requested_snr_shelf_db": -30.0,
        "frequency_offset_hz": 0.0,
        "channel_gain_db": 0.0,
        "channel_phase_deg": 0.0,
        "measured_truth_snr_shelf_db": -30.0,
        "measured_truth_composite_atsc_snr_db": -20.0,
        "estimated_snr_shelf_db": -30.0,
        "snr_error_db": 0.0,
        "fstat_raw": 1.0,
        "fstat_level_db": 0.0,
        "pnr_bin_db": 0.0,
        "cpu_float_estimated_snr_shelf_db": -30.0,
        "cpu_float_snr_error_db": 0.0,
        "cpu_float_fstat_raw": 1.0,
        "cpu_gpu_abs_diff": 0.0,
        "cpu_float_gpu_snr_diff_db": 0.0,
        "num_input_streams": 4,
    }
    summary = _summarize_rows(
        [legacy],
        requested_values=[-30.0],
        frequency_offset_values=[0.0],
        composite_to_shelf_db=10.0,
        num_input_streams=4,
    )
    assert "positive_excess_detection_rate" not in summary[0]
    assert "threshold_detection_rate" not in summary[0]


def test_cpu_reference_measurements_match_exact_integers() -> None:
    import numpy as np

    from pilot_proxy.detector_contract import (
        norm_corrected_positive_excess,
        weight_term_norms_sq,
    )
    from pilot_proxy.detector_reference import fstat_cpu_reference_packed
    from pilot_proxy.testbench.evaluate_snr import _cpu_reference_measurements

    rng = np.random.default_rng(42)
    packed = rng.integers(-128, 128, size=(64, 128), dtype=np.int16).astype(np.int8)
    weights = rng.integers(-128, 128, size=(3, 128), dtype=np.int16).astype(np.int8)

    calib = dict(pilot_below_data_db=11.3, bin_enbw_hz=3051.7578125,
                 pilot_capture_efficiency=1.0, dtv_bandwidth_hz=6.0e6)
    out = _cpu_reference_measurements(packed=packed, weights=weights, bits=4, **calib)

    fstat, sums = fstat_cpu_reference_packed(packed, weights, 4)
    assert out["p_target_u64"] == int(round(float(sums[0])))
    assert out["p_ref_sum_u64"] == int(round(float(sums[1] + sums[2])))
    assert out["fstat_raw"] == pytest.approx(
        2.0 * out["p_target_u64"] / out["p_ref_sum_u64"]
    )
    assert out["diagnostic_raw_float32"] == pytest.approx(fstat, rel=1e-6)
    nt, nl, nu = weight_term_norms_sq(weights)
    assert out["positive_excess"] == norm_corrected_positive_excess(
        out["p_target_u64"], out["p_ref_sum_u64"],
        target_norm_sq=nt, ref_norm_sum_sq=nl + nu,
    )
    assert "mask" not in out  # no threshold requested


def test_cpu_reference_threshold_mask_is_exact_at_the_boundary() -> None:
    import numpy as np

    from pilot_proxy.testbench.evaluate_snr import (
        _cpu_reference_measurements,
        _measurements_from_powers,
    )

    calib = dict(pilot_below_data_db=11.3, bin_enbw_hz=3051.7578125,
                 pilot_capture_efficiency=1.0, dtv_bandwidth_hz=6.0e6)
    weights = np.ones((3, 4), dtype=np.int8)

    # Direct rational-half rule: mask iff p_t * den > num * p_ref, strictly.
    def _mask(p_t, p_ref, num, den):
        out = _measurements_from_powers(
            diagnostic_float=1.0, p_target=p_t,
            p_ref_lower=p_ref // 2, p_ref_upper=p_ref - p_ref // 2,
            weights=weights, threshold={"threshold_half_num": num,
                                        "threshold_half_den": den},
            mask=int(p_ref != 0 and p_t * den > num * p_ref),
            overflow=0, **calib)
        return out["mask"]

    assert _mask(50, 90, 5, 9) == 0   # 50*9 == 5*90: equality is no excess
    assert _mask(51, 90, 5, 9) == 1
    assert _mask(10, 0, 5, 9) == 0    # invalid reference floor

    # End-to-end through the CPU backend with a crafted threshold.
    rng = np.random.default_rng(7)
    packed = rng.integers(-128, 128, size=(16, 4), dtype=np.int16).astype(np.int8)
    out = _cpu_reference_measurements(
        packed=packed, weights=weights, bits=4,
        threshold={"threshold_half_num": 1, "threshold_half_den": 2}, **calib)
    expected = int(out["p_ref_sum_u64"] != 0
                   and 2 * out["p_target_u64"] > out["p_ref_sum_u64"])
    assert out["mask"] == expected
    assert out["rational_overflow_count"] == 0


def test_detector_backend_flag_parses() -> None:
    from pilot_proxy.testbench.evaluate_snr import build_parser

    args = build_parser().parse_args(
        ["--input-iq", "x.cfile", "--detector-backend", "cpu-reference"]
    )
    assert args.detector_backend == "cpu-reference"
    assert build_parser().parse_args(["--input-iq", "x.cfile"]).detector_backend == "cuda"
