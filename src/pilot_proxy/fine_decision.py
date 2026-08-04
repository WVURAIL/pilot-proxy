# coding=utf-8
"""Frozen fine designated-set decision reference (fine decision v1).

This module is the bit-exact Python reference for the deployed mask
decision computed by the fused kernel epilogue (kernel core 2.3.0,
``FStat_Compute_FusedFineMask_U64``). The CUDA implementation must
reproduce ``fine_mask_decision`` exactly --- same mask bit for the same
exact fine powers and the same calibration constants --- so every value
here is arithmetic over exact integers (Python's arbitrary-precision
ints mirror the kernel's 128/192-bit fixed-width products).

Decision statement
------------------

Inputs: exact uint64 fine power sums ``S[term, bin]`` (the fused
kernel's ``FinePowers`` output; frozen fxfft256 v1 magnitudes summed
over feeds) and per-channel calibration data from the runtime bundle:

* ``anchor_bin``            measured pilot line bin (0..255);
* ``designated_half_width`` designated set = anchor +/- w, modulo 256
                            (the survey's window convention);
* ``bulk_mask``             256-bit mask of null-bulk bins: independent
                            (every ``pad_factor``-th) bins, guard- and
                            census-excluded --- built by
                            ``fine_reduction.independent_bin_mask`` at
                            bundle export time;
* ``cfar_rank``             0-indexed rank into the ascending bulk F2
                            values (the order-statistic CFAR estimate);
* ``multiplier_q16``        threshold multiplier in Q16 fixed point.

Per-bin statistic (a rational, never divided): ``F2[b] = num[b]/den[b]``
with ``num[b] = 2 * S[target, b]`` and
``den[b] = S[ref_lower, b] + S[ref_upper, b]``.

Rule: let ``bulk = {b : bulk_mask[b] and den[b] > 0}``. The frame is
*invalid* (mask = 0) when ``cfar_rank >= |bulk|``. Otherwise let
``F2_r`` be the value of rank ``cfar_rank`` in the ascending exact
ordering of ``{F2[b] : b in bulk}`` (rank selection by counting, so the
selected *value* is unique even under ties). The mask bit is 1 iff some
designated bin ``b`` with ``den[b] > 0`` satisfies::

    num[b] * 2**16 * den_r  >  multiplier_q16 * num_r * den[b]

i.e. ``F2[b] > (multiplier_q16 / 2**16) * F2_r``. Degenerate
denominators can never fire (the coarse rule's "zero-reference forced
0", applied per bin), and a frame with a degenerate bulk is forced 0.

Why rank-based (order-statistic) CFAR: the survey's recorded per-frame
calibration (``fine_reduction.calibrate_cfar``) uses float medians and
interpolated quantiles, which cannot be made bit-exact across
numpy/nvcc/compiler versions --- the same reason the deployed FFT is
fixed point. A rank threshold with a Q16 multiplier is exact, needs no
contamination fallback mode (order statistics reject the measured 0.29%
bulk tail by construction), and is calibrated by the same null-quantile
program: the campaign picks (rank, multiplier) from measured off-epoch
nulls to hit the target false-alarm rate. Rank and multiplier are
bundle *data*; this module and the kernel freeze only the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)

FINE_DECISION_VERSION = "fine_decision_v1"
FINE_BINS = 256
MULTIPLIER_Q = 16
MULTIPLIER_ONE = 1 << MULTIPLIER_Q


def designated_bins(anchor_bin: int, half_width: int) -> np.ndarray:
    """Designated set: anchor +/- half_width, modulo 256 (survey rule)."""
    a = int(anchor_bin)
    w = int(half_width)
    if not 0 <= a < FINE_BINS:
        raise ValueError("anchor_bin must be in [0, 256).")
    if not 0 <= w < FINE_BINS // 2:
        raise ValueError("designated_half_width must be in [0, 128).")
    return np.array(
        [(a + k) % FINE_BINS for k in range(-w, w + 1)], dtype=np.int64
    )


def pack_bulk_mask(mask: Sequence[bool]) -> tuple[int, int, int, int]:
    """Pack a 256-entry boolean mask into 4 uint64 words (bin b ->
    word b // 64, bit b % 64)."""
    m = np.asarray(mask, dtype=bool)
    if m.shape != (FINE_BINS,):
        raise ValueError("bulk mask must have exactly 256 entries.")
    words = [0, 0, 0, 0]
    for b in np.flatnonzero(m):
        words[int(b) >> 6] |= 1 << (int(b) & 63)
    return tuple(words)  # type: ignore[return-value]


def unpack_bulk_mask(words: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`pack_bulk_mask`."""
    w = [int(x) for x in words]
    if len(w) != 4 or any(not 0 <= x < (1 << 64) for x in w):
        raise ValueError("bulk mask words must be 4 uint64 values.")
    return np.array(
        [bool((w[b >> 6] >> (b & 63)) & 1) for b in range(FINE_BINS)],
        dtype=bool,
    )


@dataclass(frozen=True)
class FineDecision:
    """Decision outcome plus the diagnostics the tests pin down."""

    mask: int          # 1 = reject (pilot present), 0 = keep
    valid: bool        # False when the bulk was degenerate for the rank
    n_bulk: int        # usable bulk bins (mask bit set and den > 0)
    rank_bin: int      # representative bin of the rank value (-1 invalid)
    fired_bin: int     # first designated bin that fired (-1 when mask=0)


def _f2_less(num_i: int, den_i: int, num_j: int, den_j: int) -> bool:
    """Exact ``F2_i < F2_j`` by cross multiplication (dens > 0)."""
    return num_i * den_j < num_j * den_i


def fine_mask_decision(
    fine_powers: Any,
    *,
    anchor_bin: int,
    designated_half_width: int,
    bulk_mask: Sequence[bool] | Sequence[int],
    cfar_rank: int,
    multiplier_q16: int,
) -> FineDecision:
    """Frozen fine decision v1 over one frame's exact fine powers.

    ``fine_powers``: integer array ``[3, 256]`` (target, lower ref,
    upper ref) --- the fused kernel's exact uint64 ``FinePowers`` for
    one batch entry. ``bulk_mask`` accepts either 256 booleans or the
    4-word packed form. Returns a :class:`FineDecision`.
    """
    S = np.asarray(fine_powers)
    if S.shape != (3, FINE_BINS):
        raise ValueError("fine_powers must have shape [3, 256].")
    if S.dtype.kind not in "ui":
        raise TypeError("fine_powers must be exact integers.")
    rank = int(cfar_rank)
    if rank < 0:
        raise ValueError("cfar_rank must be non-negative.")
    mult = int(multiplier_q16)
    if mult < 0:
        raise ValueError("multiplier_q16 must be non-negative.")
    designated = designated_bins(anchor_bin, designated_half_width)

    mask_arr = (
        unpack_bulk_mask(bulk_mask)
        if len(bulk_mask) == 4
        else np.asarray(bulk_mask, dtype=bool)
    )
    if mask_arr.shape != (FINE_BINS,):
        raise ValueError("bulk mask must describe exactly 256 bins.")

    num = [2 * int(S[WEIGHT_TERM_TARGET, b]) for b in range(FINE_BINS)]
    den = [
        int(S[WEIGHT_TERM_REF_LOWER, b]) + int(S[WEIGHT_TERM_REF_UPPER, b])
        for b in range(FINE_BINS)
    ]

    bulk = [b for b in range(FINE_BINS) if mask_arr[b] and den[b] > 0]
    n_bulk = len(bulk)
    if rank >= n_bulk:
        return FineDecision(
            mask=0, valid=False, n_bulk=n_bulk, rank_bin=-1, fired_bin=-1
        )

    # Rank selection by counting: bin i holds the rank value iff
    # (# strictly below) <= rank < (# strictly below) + (# equal).
    # Every qualifying bin carries the same rational value, so the
    # decision is representative-independent; the reference reports the
    # lowest qualifying bin as the representative.
    rank_bin = -1
    for i in bulk:
        c_lt = 0
        c_eq = 0
        for j in bulk:
            if _f2_less(num[j], den[j], num[i], den[i]):
                c_lt += 1
            elif not _f2_less(num[i], den[i], num[j], den[j]):
                c_eq += 1
        if c_lt <= rank < c_lt + c_eq:
            rank_bin = i
            break
    assert rank_bin >= 0  # counting selection always finds the rank value
    num_r = num[rank_bin]
    den_r = den[rank_bin]

    fired_bin = -1
    for b in designated:
        bi = int(b)
        if den[bi] <= 0:
            continue  # zero-reference forced 0, per bin
        if num[bi] * MULTIPLIER_ONE * den_r > mult * num_r * den[bi]:
            fired_bin = bi
            break
    return FineDecision(
        mask=1 if fired_bin >= 0 else 0,
        valid=True,
        n_bulk=n_bulk,
        rank_bin=rank_bin,
        fired_bin=fired_bin,
    )
