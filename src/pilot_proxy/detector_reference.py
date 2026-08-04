#!/usr/bin/env python3
# coding=utf-8
"""
Pure-NumPy reference implementation for the F-statistic kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Any

INT4_COMPONENT_BITS = 4
INT8_COMPONENT_BITS = 8
REFERENCE_WEIGHT_TERMS = 3
REFERENCE_TARGET_TERM_INDEX = 0
REFERENCE_LOWER_TERM_INDEX = 1
REFERENCE_UPPER_TERM_INDEX = 2
RAW_FSTAT_REFERENCE_SCALE = 2.0


def packed_dtype_for_component_bits(bits: int) -> np.dtype:
    """Return a packed integer dtype for a per-component bit depth."""
    bits = int(bits)
    if bits == INT4_COMPONENT_BITS:
        return np.dtype(np.int8)
    if bits == INT8_COMPONENT_BITS:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported component bit depth: {bits}. Expected 4 or 8.")


def quantize_complex_numpy(
    data: np.ndarray,
    bits: int,
    scale: float,
) -> np.ndarray:
    """Quantize complex data to packed integer format (NumPy version)."""
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (M, K). Got shape {data.shape}.")
    packed_dtype = packed_dtype_for_component_bits(bits)
    max_int = (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1

    r = np.asarray(
        np.clip(np.round(data.real * scale), -max_int, max_int),
        dtype=np.int32,
    )
    i = np.asarray(
        np.clip(np.round(data.imag * scale), -max_int, max_int),
        dtype=np.int32,
    )
    packed = np.asarray((r << bits) | (i & mask), dtype=packed_dtype)
    return np.ascontiguousarray(packed)


def unpack_packed_complex(
    packed: np.ndarray,
    bits: int,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Unpack packed integer samples to a complex array."""
    packed_dtype = packed_dtype_for_component_bits(bits)
    p = np.asarray(packed, dtype=packed_dtype).astype(np.int32, copy=False)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    real = p >> bits
    imag_raw = p & mask
    imag = np.where(imag_raw & sign_bit, imag_raw - (1 << bits), imag_raw)

    return real.astype(dtype) + 1j * imag.astype(dtype)  # type: ignore[arg-type]


def fstat_cpu_reference(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute F-statistic using natural weights and x * conj(w) dot products."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (M, K). Got shape {samples.shape}.")
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D (N, K). Got shape {weights.shape}.")
    if weights.shape[0] != REFERENCE_WEIGHT_TERMS:
        raise ValueError("weights must have N=3 (target, ref+, ref-).")
    if samples.shape[1] != weights.shape[1]:
        raise ValueError("samples K dimension must match weights K dimension.")

    dots = samples @ np.conjugate(weights).T
    power = np.abs(dots) ** 2
    sums = np.sum(power, axis=0)
    denom = float(sums[REFERENCE_LOWER_TERM_INDEX] + sums[REFERENCE_UPPER_TERM_INDEX])
    eps = np.finfo(np.float64).tiny
    if denom <= eps:
        return 0.0, sums
    fstat = float(
        RAW_FSTAT_REFERENCE_SCALE * sums[REFERENCE_TARGET_TERM_INDEX] / denom
    )
    return fstat, sums


def fstat_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> tuple[float, np.ndarray]:
    """Compute F-statistic from packed integer samples and weights."""
    samples = unpack_packed_complex(packed_samples, bits)
    weights = unpack_packed_complex(packed_weights, bits)
    return fstat_cpu_reference(samples, weights)
