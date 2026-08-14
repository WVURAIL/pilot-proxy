# coding=utf-8
"""v2 time-coherent fine reduction over exact kernel row sums.

Stage 1 (CUDA or reference) emits exact int32 complex row sums
``z[n, m]`` for every weight term ``n`` and detector row ``m``
(term-major, stream-major rows, interleaved re/im — see
``FStat_Compute_RowSums_I32`` in ``cuda/f_statistic.h``). This module is
stage 2:

1. reshape each term's rows to ``[num_streams, windows_per_stream]``
   (no gather: rows are stream-major),
2. FFT along the window axis with ``FINE_PAD_FACTOR``x zero padding,
3. square and sum incoherently over streams -> ``S[n, b]``,
4. form the fine-bin statistic ``F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])``.

Invariants (enforced, not assumed):

* **Exact v1 marginal identity.** The int64 sum over rows of ``|z[n, m]|^2``
  reproduces the deployed ``FStat_Compute_Powers_U64`` output bit-for-bit;
  ``fine_reduce`` computes it in the integer domain and
  ``check_v1_marginal_identity`` asserts it against kernel powers at
  runtime. The float Parseval identity
  ``sum_b S[n, b] == pad * windows * P[n]`` is the ULP-level gate on the
  FFT path itself.
* **Null-bulk CFAR calibration** (pre-registered): location = median and
  scale = left-side spread (median minus the 15.87th percentile) of the
  *independent, non-designated* fine bins — every ``FINE_PAD_FACTOR``-th
  bin, excluding designated transmitter bins with a guard margin and any
  census-excluded bins. Never the mean of the full mixture. When the null
  bulk itself is contaminated (flag fraction above
  ``fallback_flag_fraction``), fall back to lower-quantile calibration
  (location = P25, scale = (P25 - P2.275) / 2) and record the mode.
* **Any-bin detection.** Detections are declared on all bins against the
  null calibration, removing the census dependency of designated-only
  rules. The fraction of those same null-bulk bins that exceed the fitted
  threshold is reported as ``null_bulk_exceedance_fraction``. It is an
  in-sample threshold diagnostic, not an independent false-alarm-rate estimate.

The module is backend-agnostic: pass ``xp=numpy`` (default) or
``xp=cupy``. Integer marginals are computed with integer dtypes on either
backend; FFTs use complex128 under numpy (prototype/parity reference) and
complex64 under cupy (production), with the GPU-vs-prototype comparison
gated at ULP tolerance in ``tests/kernel/test_row_sums_gpu.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

FINE_PAD_FACTOR = 2
CFAR_LEFT_QUANTILE = 0.1587  # one-sided 1-sigma equivalent
CFAR_FALLBACK_LOCATION_QUANTILE = 0.25
CFAR_FALLBACK_LEFT_QUANTILE = 0.02275  # two-sided 2-sigma equivalent
CFAR_DEFAULT_P_FA = 1.0e-3
CFAR_DEFAULT_GUARD_FINE_BINS = 1
CFAR_DEFAULT_FALLBACK_FLAG_FRACTION = 0.2
CFAR_MODE_MEDIAN_LEFT = "median_left_side_scale"
CFAR_MODE_QUANTILE_FALLBACK = "quantile_fallback"

WEIGHT_TERM_TARGET = 0
WEIGHT_TERM_REF_LOWER = 1
WEIGHT_TERM_REF_UPPER = 2


def p_fa_to_threshold_k(p_fa: float) -> float:
    """Gaussian-consistent one-sided threshold multiplier for a target P_FA."""
    p = float(p_fa)
    if not 0.0 < p < 0.5:
        raise ValueError("p_fa must be in (0, 0.5).")
    return float(NormalDist().inv_cdf(1.0 - p))


def fine_bin_count(windows_per_stream: int, pad_factor: int = FINE_PAD_FACTOR) -> int:
    return int(pad_factor) * int(windows_per_stream)


def fine_bin_frequencies_hz(
    windows_per_stream: int,
    envelope_rate_hz: float,
    pad_factor: int = FINE_PAD_FACTOR,
) -> np.ndarray:
    """Centered fine-bin envelope frequencies (Hz) for the padded FFT."""
    p2 = fine_bin_count(windows_per_stream, pad_factor)
    bins = np.arange(p2)
    centered = ((bins + p2 // 2) % p2) - p2 // 2
    return centered * (float(envelope_rate_hz) / p2)


def independent_bin_mask(
    num_bins: int,
    *,
    pad_factor: int = FINE_PAD_FACTOR,
    designated_bins: Sequence[int] = (),
    guard_fine_bins: int = CFAR_DEFAULT_GUARD_FINE_BINS,
    census_excluded_bins: Sequence[int] = (),
) -> np.ndarray:
    """Boolean mask of independent, non-designated, non-excluded fine bins.

    Zero padding correlates adjacent bins; every ``pad_factor``-th bin is
    an independent sample of the underlying ``windows``-point spectrum.
    Designated bins are removed together with ``guard_fine_bins`` unpadded
    fine bins (``guard_fine_bins * pad_factor`` padded bins) on each side.
    """
    n = int(num_bins)
    mask = np.zeros(n, dtype=bool)
    mask[:: int(pad_factor)] = True
    guard = int(guard_fine_bins) * int(pad_factor)
    for b in designated_bins:
        lo = int(b) - guard
        hi = int(b) + guard
        idx = np.arange(lo, hi + 1) % n
        mask[idx] = False
    for b in census_excluded_bins:
        mask[int(b) % n] = False
    return mask


@dataclass
class CfarCalibration:
    location: float
    scale: float
    threshold: float
    threshold_k: float
    p_fa: float
    mode: str
    num_null_bins: int
    null_bulk_exceedance_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": float(self.location),
            "scale": float(self.scale),
            "threshold": float(self.threshold),
            "threshold_k": float(self.threshold_k),
            "p_fa": float(self.p_fa),
            "mode": str(self.mode),
            "num_null_bins": int(self.num_null_bins),
            "null_bulk_exceedance_fraction": float(
                self.null_bulk_exceedance_fraction
            ),
        }


@dataclass
class FineReductionResult:
    """Per-frame v2 products."""

    fine_power: np.ndarray  # [num_weight_terms, num_bins] float
    fstat_fine: np.ndarray  # [num_bins] float
    marginal_powers: np.ndarray  # [num_weight_terms] int64 (exact)
    v1_fstat: float
    num_streams: int
    windows_per_stream: int
    pad_factor: int
    cfar: CfarCalibration | None = None
    detected_bins: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    @property
    def num_bins(self) -> int:
        return int(self.fstat_fine.shape[-1])


def _as_complex_rows(row_sums: Any, num_weight_terms: int, xp: Any) -> Any:
    """Normalize kernel/reference output to complex [terms, rows]."""
    arr = xp.asarray(row_sums)
    if arr.dtype.kind == "c":
        if arr.ndim != 2 or arr.shape[0] != int(num_weight_terms):
            raise ValueError(
                "complex row sums must have shape [num_weight_terms, rows]."
            )
        return arr
    if arr.ndim == 1:
        if arr.size % (2 * int(num_weight_terms)) != 0:
            raise ValueError("flat row sums size must be terms * rows * 2.")
        arr = arr.reshape(int(num_weight_terms), -1, 2)
    if arr.ndim != 3 or arr.shape[0] != int(num_weight_terms) or arr.shape[-1] != 2:
        raise ValueError(
            "integer row sums must have shape [num_weight_terms, rows, 2]."
        )
    return arr


def exact_marginal_powers(
    row_sums: Any,
    *,
    num_weight_terms: int = 3,
    xp: Any = np,
) -> np.ndarray:
    """Exact int64 v1 power terms from integer row sums.

    Computed entirely in the integer domain so the result is bit-identical
    to ``FStat_Compute_Powers_U64`` regardless of backend or ordering.
    Complex inputs are rejected: exactness requires the integer form.
    """
    arr = xp.asarray(row_sums)
    if arr.dtype.kind == "c":
        raise TypeError(
            "exact_marginal_powers requires integer row sums; complex "
            "floats cannot guarantee the bit-exact v1 identity."
        )
    arr = _as_complex_rows(arr, num_weight_terms, xp)  # [terms, rows, 2] ints
    wide = arr.astype(xp.int64)
    mags = wide[..., 0] * wide[..., 0] + wide[..., 1] * wide[..., 1]
    out = mags.sum(axis=1)
    if hasattr(out, "get"):
        out = out.get()
    return np.asarray(out, dtype=np.int64)


def v1_fstat_from_powers(powers: Sequence[int]) -> float:
    """F = 2 P_t / (P_l + P_u), zero when the denominator vanishes."""
    p = np.asarray(powers, dtype=np.float64)
    den = float(p[WEIGHT_TERM_REF_LOWER] + p[WEIGHT_TERM_REF_UPPER])
    if den <= 0.0:
        return 0.0
    return float(2.0 * p[WEIGHT_TERM_TARGET] / den)


def check_v1_marginal_identity(
    marginal_powers: Sequence[int],
    kernel_powers: Sequence[int],
) -> None:
    """Assert the exact all-bin marginal identity against kernel powers."""
    a = np.asarray(marginal_powers, dtype=np.uint64)
    b = np.asarray(kernel_powers, dtype=np.uint64)
    if a.shape != b.shape or not bool(np.all(a == b)):
        raise AssertionError(
            "v1 marginal identity violated: row-sum marginal "
            f"{a.tolist()} != kernel powers {b.tolist()}. Stage-1 output "
            "is corrupt or mismatched; do not trust downstream products."
        )


def fine_reduce(
    row_sums: Any,
    *,
    num_streams: int,
    windows_per_stream: int,
    num_weight_terms: int = 3,
    pad_factor: int = FINE_PAD_FACTOR,
    xp: Any = np,
) -> FineReductionResult:
    """Reduce exact row sums to fine-bin spectra and the fine statistic.

    ``row_sums`` accepts the kernel's flat int32 buffer, an
    ``[terms, rows, 2]`` integer array, or a complex ``[terms, rows]``
    array (integer forms preserve the exact marginal; complex input skips
    it and records ``marginal_powers`` from the float spectra Parseval
    sum, rounded — prefer integers).
    """
    terms = int(num_weight_terms)
    streams = int(num_streams)
    windows = int(windows_per_stream)
    rows = streams * windows
    p2 = fine_bin_count(windows, pad_factor)

    arr = xp.asarray(row_sums)
    integer_input = arr.dtype.kind != "c"
    norm = _as_complex_rows(arr, terms, xp)
    if integer_input:
        if norm.shape[1] != rows:
            raise ValueError(
                f"row count {norm.shape[1]} != num_streams * windows_per_stream = {rows}."
            )
        marginal = exact_marginal_powers(arr, num_weight_terms=terms, xp=xp)
        complex_dtype = xp.complex128 if xp is np else xp.complex64
        z = norm[..., 0].astype(complex_dtype) + 1j * norm[..., 1].astype(
            complex_dtype
        )
    else:
        if norm.shape[1] != rows:
            raise ValueError(
                f"row count {norm.shape[1]} != num_streams * windows_per_stream = {rows}."
            )
        z = norm
        marginal = None

    z = z.reshape(terms, streams, windows)
    spectra = xp.fft.fft(z, n=p2, axis=-1)
    power = (spectra.real * spectra.real + spectra.imag * spectra.imag).sum(
        axis=1
    )

    if marginal is None:
        approx = power.sum(axis=-1) / float(p2)
        if hasattr(approx, "get"):
            approx = approx.get()
        marginal = np.rint(np.asarray(approx, dtype=np.float64)).astype(np.int64)

    s_t = power[WEIGHT_TERM_TARGET]
    s_l = power[WEIGHT_TERM_REF_LOWER]
    s_u = power[WEIGHT_TERM_REF_UPPER]
    den = s_l + s_u
    fstat = xp.where(den > 0, 2.0 * s_t / xp.where(den > 0, den, 1.0), 0.0)

    power_host = power.get() if hasattr(power, "get") else power
    fstat_host = fstat.get() if hasattr(fstat, "get") else fstat

    return FineReductionResult(
        fine_power=np.asarray(power_host, dtype=np.float32),
        fstat_fine=np.asarray(fstat_host, dtype=np.float32),
        marginal_powers=np.asarray(marginal, dtype=np.int64),
        v1_fstat=v1_fstat_from_powers(marginal),
        num_streams=streams,
        windows_per_stream=windows,
        pad_factor=int(pad_factor),
    )


def calibrate_cfar(
    fstat_fine: np.ndarray,
    *,
    designated_bins: Sequence[int] = (),
    census_excluded_bins: Sequence[int] = (),
    guard_fine_bins: int = CFAR_DEFAULT_GUARD_FINE_BINS,
    pad_factor: int = FINE_PAD_FACTOR,
    p_fa: float = CFAR_DEFAULT_P_FA,
    fallback_flag_fraction: float = CFAR_DEFAULT_FALLBACK_FLAG_FRACTION,
) -> CfarCalibration:
    """Pre-registered null-bulk calibration of the fine statistic.

    Location and scale come from the independent non-designated bins
    (median + left-side spread), never from the mean of the full mixture.
    A contaminated null bulk (flag fraction above
    ``fallback_flag_fraction``) triggers the lower-quantile fallback.
    """
    f = np.asarray(fstat_fine, dtype=np.float64)
    mask = independent_bin_mask(
        f.shape[-1],
        pad_factor=pad_factor,
        designated_bins=designated_bins,
        guard_fine_bins=guard_fine_bins,
        census_excluded_bins=census_excluded_bins,
    )
    null_bins = f[mask]
    if null_bins.size < 8:
        raise ValueError(
            "too few independent null bins for CFAR calibration; check "
            "designated/census exclusions."
        )
    k = p_fa_to_threshold_k(p_fa)

    location = float(np.median(null_bins))
    scale = float(location - np.quantile(null_bins, CFAR_LEFT_QUANTILE))
    mode = CFAR_MODE_MEDIAN_LEFT
    if scale <= 0.0:
        scale = float(np.std(null_bins)) or 1.0
    threshold = location + k * scale
    null_bulk_exceedance_fraction = float(np.mean(null_bins > threshold))

    if null_bulk_exceedance_fraction > float(fallback_flag_fraction):
        location = float(np.quantile(null_bins, CFAR_FALLBACK_LOCATION_QUANTILE))
        left = float(np.quantile(null_bins, CFAR_FALLBACK_LEFT_QUANTILE))
        scale = max((location - left) / 2.0, 1.0e-12)
        threshold = location + k * scale
        null_bulk_exceedance_fraction = float(
            np.mean(null_bins > threshold)
        )
        mode = CFAR_MODE_QUANTILE_FALLBACK

    return CfarCalibration(
        location=location,
        scale=scale,
        threshold=threshold,
        threshold_k=float(k),
        p_fa=float(p_fa),
        mode=mode,
        num_null_bins=int(null_bins.size),
        null_bulk_exceedance_fraction=null_bulk_exceedance_fraction,
    )


def detect_any_bin(
    fstat_fine: np.ndarray,
    calibration: CfarCalibration,
) -> np.ndarray:
    """Any-bin detection rule: indices of all bins above threshold."""
    f = np.asarray(fstat_fine, dtype=np.float64)
    return np.flatnonzero(f > float(calibration.threshold)).astype(np.int64)


def reduce_and_detect(
    row_sums: Any,
    *,
    num_streams: int,
    windows_per_stream: int,
    designated_bins: Sequence[int] = (),
    census_excluded_bins: Sequence[int] = (),
    guard_fine_bins: int = CFAR_DEFAULT_GUARD_FINE_BINS,
    p_fa: float = CFAR_DEFAULT_P_FA,
    kernel_powers: Sequence[int] | None = None,
    num_weight_terms: int = 3,
    pad_factor: int = FINE_PAD_FACTOR,
    xp: Any = np,
) -> FineReductionResult:
    """One-call v2 reduction: fine spectra, identity check, CFAR, detection."""
    result = fine_reduce(
        row_sums,
        num_streams=num_streams,
        windows_per_stream=windows_per_stream,
        num_weight_terms=num_weight_terms,
        pad_factor=pad_factor,
        xp=xp,
    )
    if kernel_powers is not None:
        check_v1_marginal_identity(result.marginal_powers, kernel_powers)
    calibration = calibrate_cfar(
        result.fstat_fine,
        designated_bins=designated_bins,
        census_excluded_bins=census_excluded_bins,
        guard_fine_bins=guard_fine_bins,
        pad_factor=pad_factor,
        p_fa=p_fa,
    )
    result.cfar = calibration
    result.detected_bins = detect_any_bin(result.fstat_fine, calibration)
    return result
