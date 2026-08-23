# coding=utf-8
"""Smoke gate for the coherent-gain Monte Carlo tool.

The critical assertion is the verify stage: the tool's batched
reduction must equal the packaged pipeline (bit-exact integer marginals,
float32-identical fine spectra). The MC smoke then checks the machinery
end to end at a reduced geometry: sane H0 thresholds and Pd -> 1 at
strong injection.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "measure_fine_gain", REPO / "tools" / "measure_fine_gain.py")
mfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfg)


def test_batched_reduction_matches_packaged_pipeline():
    mfg.verify(seed=7, trials=2, streams=32)


def test_mc_smoke_thresholds_and_detection(tmp_path):
    out = str(tmp_path)
    mfg.run_stage(out, "h0", trials=64, seed=1, streams=32, batch=8)
    mfg.run_stage(out, "sweep", trials=32, seed=2, snr_db=-10.0,
                  b0=mfg.ANCHOR, streams=32, batch=8)
    c0, f0, _ = mfg.collect(out, "h0_s*.npz")
    assert c0.size == 64 and f0.size == 64
    # H0 statistics center near 1 (null_power_ratio = 1 by construction)
    assert 0.9 < np.median(c0) < 1.1
    assert 0.9 < np.median(f0) < 1.3
    c1, f1, _ = mfg.collect(out, "h1_*dB_s*.npz")
    # strong injection: fine statistic far above the entire H0 sample
    assert f1.min() > f0.max()
    assert c1.min() > np.quantile(c0, 0.99)


def test_snr_amplitude_roundtrip():
    # amplitude formula inverts the documented SNR definition
    snr_db = -25.0
    amp = float(np.sqrt(2.0 * mfg.SIGMA**2 * 10 ** (snr_db / 10.0)))
    back = 10.0 * np.log10(amp**2 / (2.0 * mfg.SIGMA**2))
    assert abs(back - snr_db) < 1e-9
