# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.products import spectrum_before_after


def test_before_after_spectrum_excludes_masked_frames_from_denominator() -> None:
    baseband = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
    )
    mask = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    before_db, after_db = spectrum_before_after(baseband, mask)

    np.testing.assert_allclose(before_db, 10.0 * np.log10([20.0, 200.0]))
    np.testing.assert_allclose(after_db, 10.0 * np.log10([20.0, 150.0]))


def test_after_spectrum_all_masked_returns_nan_not_zero() -> None:
    before_db, after_db = spectrum_before_after(
        np.asarray([[10.0], [20.0]]),
        np.asarray([[1], [1]], dtype=np.uint8),
    )

    assert np.isfinite(before_db[0])
    assert np.isnan(after_db[0])


def test_before_after_spectrum_uses_valid_frame_denominators() -> None:
    baseband = np.asarray([[10.0], [1000.0], [30.0], [40.0]])
    mask = np.asarray([[0], [1], [0], [0]], dtype=np.uint8)
    valid = np.asarray([[1], [0], [1], [0]], dtype=np.uint8)

    before_db, after_db = spectrum_before_after(baseband, mask, valid)

    np.testing.assert_allclose(before_db, 10.0 * np.log10([20.0]))
    np.testing.assert_allclose(after_db, 10.0 * np.log10([20.0]))


def test_before_after_spectrum_all_invalid_returns_nan() -> None:
    before_db, after_db = spectrum_before_after(
        np.asarray([[10.0], [20.0]]),
        np.asarray([[0], [0]], dtype=np.uint8),
        np.asarray([[0], [0]], dtype=np.uint8),
    )

    assert np.isnan(before_db[0])
    assert np.isnan(after_db[0])
