# coding=utf-8
"""The per-frame PSD accumulator must stay int16, and that is a memory contract.

The analyzer computes each frame's PSD in float64 but retains only an int16 dB
encoding (``psd_frame_db_i16``). That choice is what bounds resident memory on a
long channel: measured on 2026-08-23, 35,095 frames -- the count needed to reach
the 4.6 GB figure recorded against the float64 behaviour -- costs 1102 MB as
int16 against 3291 MB as float64. A regression to a wider dtype would not fail
any correctness test, so it is asserted here directly.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("datatrawl.interfaces")

from pilot_proxy.datatrawl_plugins.detector import PilotProxyDetectorAnalyzer

NFFT = 1024


def _bare_analyzer(nfft=NFFT):
    """An instance with only the state _append_psd_frame touches."""
    a = object.__new__(PilotProxyDetectorAnalyzer)
    a._nfft = nfft
    a._psd_codes = []
    a._psd_ref = []
    return a


def test_psd_frames_are_retained_as_int16():
    a = _bare_analyzer()
    rng = np.random.default_rng(11)
    for _ in range(4):
        a._append_psd_frame(rng.gamma(2.0, 1.0, NFFT).astype(np.float64))
    assert len(a._psd_codes) == 4
    for code in a._psd_codes:
        assert code.dtype == np.int16, "PSD retention widened; resident memory triples"
        assert code.shape == (NFFT,)


def test_invalid_frame_uses_the_sentinel_not_a_float():
    a = _bare_analyzer()
    a._append_psd_frame(None)
    assert a._psd_codes[0].dtype == np.int16
    assert np.isnan(a._psd_ref[0])


def test_resident_cost_per_frame_is_two_bytes_per_bin():
    """The memory contract, stated as arithmetic rather than a timing test."""
    a = _bare_analyzer()
    rng = np.random.default_rng(12)
    for _ in range(8):
        a._append_psd_frame(rng.gamma(2.0, 1.0, NFFT).astype(np.float64))
    resident = sum(c.nbytes for c in a._psd_codes)
    assert resident == 8 * NFFT * 2
    # what that implies at archive scale, for the record
    assert 35095 * NFFT * 2 < 35095 * NFFT * 8
