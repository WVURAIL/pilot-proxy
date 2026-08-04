# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_reduction import (
    CFAR_MODE_MEDIAN_LEFT,
    CFAR_MODE_QUANTILE_FALLBACK,
    FINE_PAD_FACTOR,
    calibrate_cfar,
    check_v1_marginal_identity,
    exact_marginal_powers,
    fine_bin_count,
    fine_bin_frequencies_hz,
    fine_reduce,
    independent_bin_mask,
    p_fa_to_threshold_k,
    reduce_and_detect,
    v1_fstat_from_powers,
)

TERMS = 3
STREAMS = 8
WINDOWS = 32
ROWS = STREAMS * WINDOWS
RNG_SEED = 0xC0FFEE
MAX_COMPONENT = 14336  # kernel bound for the locked 4-bit path


def _random_row_sums(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(
        -MAX_COMPONENT, MAX_COMPONENT + 1, size=(TERMS, ROWS, 2), dtype=np.int32
    )


def test_exact_marginal_matches_brute_force_and_kernel_convention() -> None:
    rng = np.random.default_rng(RNG_SEED)
    row_sums = _random_row_sums(rng)

    marginal = exact_marginal_powers(row_sums, num_weight_terms=TERMS)

    brute = np.zeros(TERMS, dtype=np.int64)
    for n in range(TERMS):
        for m in range(ROWS):
            zr = int(row_sums[n, m, 0])
            zi = int(row_sums[n, m, 1])
            brute[n] += zr * zr + zi * zi
    np.testing.assert_array_equal(marginal, brute)

    check_v1_marginal_identity(marginal, brute)
    with pytest.raises(AssertionError, match="v1 marginal identity"):
        check_v1_marginal_identity(marginal, brute + 1)


def test_exact_marginal_rejects_complex_input() -> None:
    z = np.zeros((TERMS, ROWS), dtype=np.complex128)
    with pytest.raises(TypeError, match="integer row sums"):
        exact_marginal_powers(z, num_weight_terms=TERMS)


def test_fine_reduce_parseval_and_identity() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    row_sums = _random_row_sums(rng)

    result = fine_reduce(
        row_sums, num_streams=STREAMS, windows_per_stream=WINDOWS
    )
    p2 = fine_bin_count(WINDOWS)
    assert result.fine_power.shape == (TERMS, p2)
    assert result.fstat_fine.shape == (p2,)

    # Parseval with zero padding: sum_b S[n, b] == p2 * P[n]
    sums = result.fine_power.astype(np.float64).sum(axis=-1)
    expected = float(p2) * result.marginal_powers.astype(np.float64)
    np.testing.assert_allclose(sums, expected, rtol=1e-6)

    assert result.v1_fstat == pytest.approx(
        v1_fstat_from_powers(result.marginal_powers)
    )


def test_synthetic_tone_lands_in_correct_padded_bin() -> None:
    # A pure envelope tone at unpadded bin q must peak at padded bin
    # q * FINE_PAD_FACTOR in the target spectrum, coherently across streams.
    q = 5
    n = np.arange(WINDOWS)
    tone = np.exp(2j * np.pi * q * n / WINDOWS)
    amplitude = 100.0
    rng = np.random.default_rng(RNG_SEED + 9)
    z = np.zeros((TERMS, STREAMS, WINDOWS), dtype=np.complex128)
    z[0] = amplitude * tone[None, :]
    # broadband references so the denominator is positive in every padded
    # bin (a constant reference has zero power at even padded bins, and the
    # zero-denominator rule correctly maps those to F2 = 0)
    z[1] = rng.standard_normal((STREAMS, WINDOWS)) + 1j * rng.standard_normal(
        (STREAMS, WINDOWS)
    )
    z[2] = rng.standard_normal((STREAMS, WINDOWS)) + 1j * rng.standard_normal(
        (STREAMS, WINDOWS)
    )
    result = fine_reduce(
        z.reshape(TERMS, ROWS),
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
    )
    peak = int(np.argmax(result.fine_power[0]))
    assert peak == q * FINE_PAD_FACTOR
    assert int(np.argmax(result.fstat_fine)) == peak

    freqs = fine_bin_frequencies_hz(WINDOWS, envelope_rate_hz=3051.7578125)
    assert freqs[peak] == pytest.approx(q * 3051.7578125 / WINDOWS)


def test_independent_bin_mask_guard_and_census() -> None:
    p2 = fine_bin_count(WINDOWS)
    designated = [10]
    census = [40]
    mask = independent_bin_mask(
        p2,
        designated_bins=designated,
        census_excluded_bins=census,
        guard_fine_bins=1,
    )
    assert mask[0]
    assert not mask[1]  # padded neighbor: not independent
    guard = FINE_PAD_FACTOR
    for b in range(10 - guard, 10 + guard + 1):
        assert not mask[b % p2]
    assert not mask[40]
    assert mask[(10 + guard + FINE_PAD_FACTOR) % p2]


def test_cfar_median_left_mode_and_detection() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    p2 = fine_bin_count(WINDOWS)
    f = 1.0 + 0.05 * rng.standard_normal(p2)
    hot = 7 * FINE_PAD_FACTOR
    f[hot] = 5.0
    calibration = calibrate_cfar(f, designated_bins=[], p_fa=1.0e-3)
    assert calibration.mode == CFAR_MODE_MEDIAN_LEFT
    assert calibration.location == pytest.approx(1.0, abs=0.05)
    assert 0.0 < calibration.scale < 0.2
    assert calibration.nd_flag_rate < 0.05

    result = reduce_and_detect(
        _random_row_sums(rng),
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
    )
    assert result.cfar is not None
    assert result.detected_bins.ndim == 1


def test_cfar_quantile_fallback_triggers_on_contaminated_bulk() -> None:
    rng = np.random.default_rng(RNG_SEED + 3)
    p2 = fine_bin_count(WINDOWS)
    f = 1.0 + 0.02 * rng.standard_normal(p2)
    # contaminate a third of the independent bins with a hot shelf
    idx = np.flatnonzero(independent_bin_mask(p2))
    f[idx[:: 3]] = 3.0
    calibration = calibrate_cfar(f, p_fa=1.0e-3, fallback_flag_fraction=0.05)
    assert calibration.mode == CFAR_MODE_QUANTILE_FALLBACK


def test_threshold_k_matches_gaussian_tail() -> None:
    assert p_fa_to_threshold_k(0.1587) == pytest.approx(1.0, abs=2e-3)
    assert p_fa_to_threshold_k(1.0e-3) == pytest.approx(3.0902, abs=1e-3)
    with pytest.raises(ValueError):
        p_fa_to_threshold_k(0.9)


def test_kernel_powers_identity_enforced_in_reduce_and_detect() -> None:
    rng = np.random.default_rng(RNG_SEED + 4)
    row_sums = _random_row_sums(rng)
    powers = exact_marginal_powers(row_sums, num_weight_terms=TERMS)
    result = reduce_and_detect(
        row_sums,
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
        kernel_powers=powers,
    )
    assert result.v1_fstat == pytest.approx(v1_fstat_from_powers(powers))
    with pytest.raises(AssertionError):
        reduce_and_detect(
            row_sums,
            num_streams=STREAMS,
            windows_per_stream=WINDOWS,
            kernel_powers=powers + 1,
        )
