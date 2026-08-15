# coding=utf-8
"""Generic mask/average helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .result_schema import MASK_VALUE_EXCLUDED, mask_convention


def masked_mean_excluding(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    axis: int | None = None,
) -> np.ndarray | float:
    """Mean with masked samples excluded, not zero-filled.

    A mask value of one means excluded. The returned mean divides the included
    value sum by the included sample count along the requested axis.
    """
    arr = np.asarray(values, dtype=np.float64)
    excluded = np.asarray(mask) == MASK_VALUE_EXCLUDED
    if excluded.shape != arr.shape:
        raise ValueError(
            "mask must have the same shape as values: "
            f"values={arr.shape}, mask={excluded.shape}"
        )
    included = np.asarray(~excluded, dtype=np.float64)
    numerator = np.sum(arr * included, axis=axis)
    denominator = np.sum(included, axis=axis)
    out = np.full_like(np.asarray(numerator, dtype=np.float64), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    if np.isscalar(numerator):
        return float(np.asarray(out).reshape(()))
    return out


def mask_metadata() -> dict[str, Any]:
    """Return the public masking convention metadata."""
    return mask_convention()
