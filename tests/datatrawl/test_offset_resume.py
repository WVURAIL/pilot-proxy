# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datatrawl")

from datatrawl.interfaces import RunContext
from pilot_proxy.datatrawl_plugins.offset import PilotProxyOffsetAnalyzer

NFFT = 256
FS_HZ = 390_625.0
FREQ_ID = 844
CENTER_HZ = 470.3125e6
N_FEEDS = 3


def _instrument():
    return SimpleNamespace(
        nfft=NFFT,
        fs_hz=FS_HZ,
        nyquist_zone=2,
        f0_mhz=800.0,
    )


def _context(window="hann"):
    return RunContext(
        instrument=_instrument(),
        selection=[FREQ_ID],
        options={
            "window_name": window,
            "offset_backend": "numpy",
            "stream_batch_size": 16,
            "peak_search_half_width_hz": 20_000.0,
        },
    )


def _meta(key):
    return {"f_center_hz": CENTER_HZ, "nfft": NFFT, "unit_key": key}


def _chunk(seed):
    rng = np.random.default_rng(seed)
    n = np.arange(NFFT)
    tone = np.exp(2j * np.pi * 1500.0 * n / FS_HZ)[:, None]
    noise = 0.1 * (
        rng.standard_normal((NFFT, N_FEEDS))
        + 1j * rng.standard_normal((NFFT, N_FEEDS))
    )
    return np.asarray(tone + noise, dtype=np.complex64)


def _assert_products_equal(a, b):
    assert set(a.files) == set(b.files)
    for key in a.files:
        x = np.asarray(a[key])
        y = np.asarray(b[key])
        if x.dtype.kind in "fc":
            assert np.allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True), key
        else:
            assert np.array_equal(x, y), key


def test_offset_resume_matches_uninterrupted(tmp_path):
    ctx = _context()
    chunks = [_chunk(1), _chunk(2)]

    clean = PilotProxyOffsetAnalyzer()
    clean.begin(ctx, _meta("event-a"))
    clean.consume_file([chunks[0]], _meta("event-a"))
    clean.consume_file([chunks[1]], _meta("event-b"))
    clean_path = tmp_path / "clean.npz"
    clean.save(str(clean_path))

    first = PilotProxyOffsetAnalyzer()
    first.begin(ctx, _meta("event-a"))
    first.consume_file([chunks[0]], _meta("event-a"))
    resumed_path = tmp_path / "resumed.npz"
    first.save(str(resumed_path))

    second = PilotProxyOffsetAnalyzer()
    assert second.resume(str(resumed_path), ctx)
    second.begin(ctx, _meta("event-b"))
    second.consume_file([chunks[1]], _meta("event-b"))
    second.save(str(resumed_path))

    with np.load(clean_path) as expected, np.load(resumed_path) as actual:
        _assert_products_equal(expected, actual)


def test_offset_resume_rejects_changed_window(tmp_path):
    ctx = _context("hann")
    analyzer = PilotProxyOffsetAnalyzer()
    analyzer.begin(ctx, _meta("event-a"))
    analyzer.consume_file([_chunk(1)], _meta("event-a"))
    path = tmp_path / "offset.npz"
    analyzer.save(str(path))

    changed = _context("rectangular")
    resumed = PilotProxyOffsetAnalyzer()
    assert resumed.resume(str(path), changed)
    with pytest.raises(SystemExit, match="configuration does not match"):
        resumed.begin(changed, _meta("event-b"))
