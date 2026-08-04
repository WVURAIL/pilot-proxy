# coding=utf-8
"""Gates for the frozen fine decision reference (fine decision v1).

The reference is exact integer arithmetic; the oracle here is
``fractions.Fraction`` (also exact), applied naively: sort the bulk F2
values, take the requested rank, compare each designated bin against
multiplier * rank value. Agreement is required exactly --- including
tie, wrap, and degenerate-denominator cases --- because the CUDA
epilogue is gated bit-for-bit against this module.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pilot_proxy.fine_decision import (
    FINE_BINS,
    MULTIPLIER_ONE,
    designated_bins,
    fine_mask_decision,
    pack_bulk_mask,
    unpack_bulk_mask,
)
from pilot_proxy.fine_reduction import independent_bin_mask

RNG_SEED = 20260805


def _oracle(S, anchor, half_width, mask, rank, mult_q16):
    """Naive exact oracle with Fractions."""
    num = 2 * S[0].astype(object)
    den = S[1].astype(object) + S[2].astype(object)
    bulk = [
        Fraction(int(num[b]), int(den[b]))
        for b in range(FINE_BINS)
        if mask[b] and int(den[b]) > 0
    ]
    if rank >= len(bulk):
        return 0
    thr = (
        Fraction(int(mult_q16), MULTIPLIER_ONE) * sorted(bulk)[rank]
    )
    for b in designated_bins(anchor, half_width):
        if int(den[b]) > 0 and Fraction(int(num[b]), int(den[b])) > thr:
            return 1
    return 0


def _random_powers(rng, scale=1 << 40):
    return rng.integers(0, scale, size=(3, FINE_BINS), dtype=np.uint64)


def test_designated_window_wraps_like_survey():
    np.testing.assert_array_equal(
        designated_bins(254, 2), np.array([252, 253, 254, 255, 0])
    )
    np.testing.assert_array_equal(
        designated_bins(1, 2), np.array([255, 0, 1, 2, 3])
    )
    np.testing.assert_array_equal(designated_bins(62, 0), np.array([62]))


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(20):
        m = rng.random(FINE_BINS) < 0.5
        np.testing.assert_array_equal(unpack_bulk_mask(pack_bulk_mask(m)), m)


def test_matches_fraction_oracle_randomized():
    rng = np.random.default_rng(RNG_SEED + 1)
    mask = independent_bin_mask(FINE_BINS, designated_bins=[62])
    for trial in range(200):
        S = _random_powers(rng)
        rank = int(rng.integers(0, 128))
        mult = int(rng.integers(1, 4 * MULTIPLIER_ONE))
        got = fine_mask_decision(
            S,
            anchor_bin=62,
            designated_half_width=2,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=mult,
        )
        assert got.mask == _oracle(S, 62, 2, mask, rank, mult), trial


def test_tied_rank_values_are_representative_independent():
    # Bins 0 and 2 carry the same rational with different representations
    # (2/4 == 1/2); either representative must give the same decision.
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    for b, (t, lo, up) in {0: (1, 2, 2), 2: (2, 4, 4), 4: (5, 1, 1)}.items():
        S[0, b], S[1, b], S[2, b] = t, lo, up
        mask[b] = True
    S[0, 100], S[1, 100], S[2, 100] = 3, 2, 2  # designated bin: F2 = 3/2
    for rank in (0, 1):  # both ranks land on the tied value 1/2
        d = fine_mask_decision(
            S,
            anchor_bin=100,
            designated_half_width=0,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=2 * MULTIPLIER_ONE,  # threshold = 2 * 1/2 = 1
        )
        assert d.valid and d.mask == 1  # 3/2 > 1
        assert d.rank_bin == 0  # lowest qualifying representative
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=2,  # rank value now 5/2; threshold 5 > 3/2
        multiplier_q16=2 * MULTIPLIER_ONE,
    )
    assert d.valid and d.mask == 0


def test_degenerate_rules():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.zeros(FINE_BINS, dtype=bool)
    # Bulk bin with zero denominator is excluded before ranking.
    S[0, 0] = 7  # den == 0
    mask[0] = True
    S[0, 2], S[1, 2], S[2, 2] = 1, 1, 1  # F2 = 1
    mask[2] = True
    S[0, 100], S[1, 100], S[2, 100] = 100, 1, 1
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=0,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.n_bulk == 1 and d.rank_bin == 2 and d.mask == 1
    # Rank beyond the usable bulk: invalid frame, mask forced 0.
    d = fine_mask_decision(
        S,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=mask,
        cfar_rank=1,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.mask == 0 and not d.valid
    # Designated bin with zero denominator can never fire.
    S2 = np.zeros((3, FINE_BINS), dtype=np.uint64)
    S2[0, 100] = 1 << 50  # den == 0 at the anchor
    S2[0, 2], S2[1, 2], S2[2, 2] = 1, 1, 1
    m2 = np.zeros(FINE_BINS, dtype=bool)
    m2[2] = True
    d = fine_mask_decision(
        S2,
        anchor_bin=100,
        designated_half_width=0,
        bulk_mask=m2,
        cfar_rank=0,
        multiplier_q16=MULTIPLIER_ONE,
    )
    assert d.mask == 0 and d.valid


def test_large_power_magnitudes_stay_exact():
    # Near the Parseval-corner scale (~2^55): cross products reach
    # ~2^110 and the threshold triple ~2^127 --- the regime the kernel
    # must cover with its fixed-width helpers.
    rng = np.random.default_rng(RNG_SEED + 2)
    mask = independent_bin_mask(FINE_BINS, designated_bins=[0])
    for _ in range(50):
        S = rng.integers(
            (1 << 54), (1 << 55), size=(3, FINE_BINS), dtype=np.uint64
        )
        rank = int(rng.integers(0, 64))
        mult = int(rng.integers(MULTIPLIER_ONE, 2 * MULTIPLIER_ONE))
        got = fine_mask_decision(
            S,
            anchor_bin=0,
            designated_half_width=2,
            bulk_mask=mask,
            cfar_rank=rank,
            multiplier_q16=mult,
        )
        assert got.mask == _oracle(S, 0, 2, mask, rank, mult)


def test_injected_line_fires_and_null_does_not():
    # Synthetic spectrum: flat references, flat target except an
    # injected excess at the anchor. The designated set fires; with the
    # line removed the same calibration keeps the mask at 0.
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    S[1, :] = 1000
    S[2, :] = 1000
    S[0, :] = 1000  # F2 = 1 everywhere
    mask = independent_bin_mask(FINE_BINS, designated_bins=[62])
    rank = int(np.count_nonzero(mask) // 2)
    mult = int(1.5 * MULTIPLIER_ONE)
    base = fine_mask_decision(
        S, anchor_bin=62, designated_half_width=2, bulk_mask=mask,
        cfar_rank=rank, multiplier_q16=mult,
    )
    assert base.mask == 0 and base.valid
    S[0, 62] = 10000  # injected pilot: F2 = 10 at the anchor
    fired = fine_mask_decision(
        S, anchor_bin=62, designated_half_width=2, bulk_mask=mask,
        cfar_rank=rank, multiplier_q16=mult,
    )
    assert fired.mask == 1 and fired.fired_bin == 62


def test_input_validation():
    S = np.zeros((3, FINE_BINS), dtype=np.uint64)
    mask = np.ones(FINE_BINS, dtype=bool)
    with pytest.raises(TypeError):
        fine_mask_decision(
            S.astype(np.float64), anchor_bin=0, designated_half_width=0,
            bulk_mask=mask, cfar_rank=0, multiplier_q16=MULTIPLIER_ONE,
        )
    with pytest.raises(ValueError):
        fine_mask_decision(
            S, anchor_bin=256, designated_half_width=0, bulk_mask=mask,
            cfar_rank=0, multiplier_q16=MULTIPLIER_ONE,
        )
    with pytest.raises(ValueError):
        fine_mask_decision(
            S, anchor_bin=0, designated_half_width=0, bulk_mask=mask,
            cfar_rank=-1, multiplier_q16=MULTIPLIER_ONE,
        )
