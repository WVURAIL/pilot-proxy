# coding=utf-8
"""Control-band analyzer: marginal math, identity, and resume discipline.

Covers, in order:
  * the Parseval tie ``marginal.sum() == K**2 * band_power`` per frame;
  * a synthetic tone landing in the bin `marginal_bin_for_rf_hz` predicts,
    in both Nyquist zones (CHIME's inverted sense included);
  * the deployed-geometry F from the marginal: large at the tone, ~1 in null;
  * consume/save/resume round trip with per-frame identity and the
    freq_id-stripped source event keys the combine step joins on;
  * resume refusals: wrong analysis signature, wrong freq_id, wrong frame cap;
  * feed-count changes within a product refuse rather than average.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datatrawl.interfaces")

from pilot_proxy.datatrawl_plugins.control import (      # noqa: E402
    ControlBandAnalyzer, DETECTOR_WINDOW_SAMPLES, band_power, coarse_marginal,
    f_statistic_from_marginal, marginal_bin_for_rf_hz)

K = DETECTOR_WINDOW_SAMPLES
NFFT = 1024          # small but K-divisible: 8 windows per frame
N_FEEDS = 6
FS = 390625.0
F0_MHZ = 800.0


def _instrument(freq_id_center_mhz=None):
    def freq_of_freq_id(fid):
        return F0_MHZ - fid * (FS / 1e6)
    return SimpleNamespace(
        fs_hz=FS, nfft=NFFT, nyquist_zone=2, n_channels=1024,
        freq_of_freq_id=(freq_of_freq_id if freq_id_center_mhz is None
                         else (lambda fid: freq_id_center_mhz)))


def _ctx(selection, options=None, instrument=None):
    return SimpleNamespace(selection=selection, options=dict(options or {}),
                           instrument=instrument or _instrument())


def _frames(n_frames, *, tone_bin=None, amp=40.0, seed=0, n_feeds=N_FEEDS):
    """Integer-valued complex frames like the canonical reader yields."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_frames):
        x = (rng.integers(-4, 5, size=(NFFT, n_feeds))
             + 1j * rng.integers(-4, 5, size=(NFFT, n_feeds))).astype(np.complex64)
        if tone_bin is not None:
            n = np.arange(NFFT)[:, None]
            phases = np.exp(2j * np.pi * rng.random(n_feeds))[None, :]
            x = x + (amp * np.exp(2j * np.pi * tone_bin * n / K)
                     * phases).astype(np.complex64)
        out.append(x)
    return out


def _meta(unit, freq_id=591, center_hz=None):
    center = (F0_MHZ * 1e6 - freq_id * FS) if center_hz is None else center_hz
    return {"unit_key": f"cadc:site/baseband_{unit}_{freq_id}.h5",
            "unit_name": f"baseband_{unit}_{freq_id}.h5",
            "f_center_hz": center}


# -- science helpers ---------------------------------------------------------

def test_marginal_parseval_ties_to_band_power():
    (frame,) = _frames(1, tone_bin=17)
    marginal = coarse_marginal(frame, K)
    assert marginal.shape == (K,)
    np.testing.assert_allclose(marginal.sum(), K ** 2 * band_power(frame),
                               rtol=1e-10)


def test_tone_lands_in_predicted_bin_both_zones():
    tone_bin = 21
    (frame,) = _frames(1, tone_bin=tone_bin, amp=60.0)
    marginal = coarse_marginal(frame, K)
    assert int(np.argmax(marginal)) == tone_bin
    center = 566_406_250.0
    bin_hz = FS / K
    # A baseband tone at +tone_bin*bin_hz is sky center - offset in zone 2
    # (inverted sense) and center + offset in zone 1.
    rf_zone2 = center - tone_bin * bin_hz
    rf_zone1 = center + tone_bin * bin_hz
    assert marginal_bin_for_rf_hz(rf_zone2, f_center_hz=center, fs_hz=FS,
                                  nyquist_zone=2, k=K) == tone_bin
    assert marginal_bin_for_rf_hz(rf_zone1, f_center_hz=center, fs_hz=FS,
                                  nyquist_zone=1, k=K) == tone_bin
    with pytest.raises(ValueError):
        marginal_bin_for_rf_hz(center + 0.6 * FS, f_center_hz=center,
                               fs_hz=FS, nyquist_zone=2, k=K)


def test_f_statistic_separates_tone_from_null():
    frames = _frames(8, tone_bin=40, amp=60.0, seed=3)
    marginal = np.stack([coarse_marginal(f, K) for f in frames])
    f_tone = f_statistic_from_marginal(marginal, 40)
    f_null = f_statistic_from_marginal(marginal, 90)
    assert np.all(f_tone > 10.0)
    assert abs(float(np.mean(f_null)) - 1.0) < 0.5
    # wraparound references: target bin 1 reaches bin K-1 without error
    assert np.isfinite(f_statistic_from_marginal(marginal, 1)).all()


# -- lifecycle ----------------------------------------------------------------

def test_consume_save_resume_roundtrip(tmp_path):
    path = str(tmp_path / "591.npz")
    a = ControlBandAnalyzer()
    a.begin(_ctx([591]), _meta("evtA"))
    n = a.consume_file(_frames(3, tone_bin=25, seed=1), _meta("evtA"))
    assert n == 3
    a.save(path)

    b = ControlBandAnalyzer()
    assert b.resume(path, _ctx([591]))
    assert b.processed_keys() == {_meta("evtA")["unit_key"]}
    b.begin(_ctx([591]), _meta("evtB"))
    assert b.consume_file(_frames(2, tone_bin=25, seed=2), _meta("evtB")) == 2
    b.save(path)

    z = np.load(path, allow_pickle=False)
    assert str(z["schema_version"]) == "pilotproxy_control_product_v1"
    assert int(z["n_frames"]) == 5
    assert z["coarse_marginal"].shape == (5, K)
    assert z["baseband_power_linear"].shape == (5,)
    np.testing.assert_allclose(z["coarse_marginal"].sum(axis=1),
                               K ** 2 * z["baseband_power_linear"], rtol=1e-10)
    assert list(z["frame_unit_index"]) == [0, 0, 0, 1, 1]
    assert list(z["frame_in_unit"]) == [0, 1, 2, 0, 1]
    assert int(z["integrated_spectrum_count"]) == 5
    assert z["integrated_spectrum_sum"].shape == (NFFT,)
    # event keys drop this product's freq_id token -> join key across freq_ids
    assert list(z["source_event_keys"]) == [
        "cadc:site/baseband_evtA.h5", "cadc:site/baseband_evtB.h5"]
    assert int(z["detector_window_samples"]) == K


def test_resume_refuses_other_analysis_and_wrong_invariants(tmp_path):
    path = str(tmp_path / "591.npz")
    a = ControlBandAnalyzer()
    a.begin(_ctx([591]), _meta("evtA"))
    a.consume_file(_frames(1), _meta("evtA"))
    a.save(path)

    with pytest.raises(SystemExit):     # wrong freq_id
        ControlBandAnalyzer().resume(path, _ctx([592]))
    with pytest.raises(SystemExit):     # capped vs uncapped product
        ControlBandAnalyzer().resume(
            path, _ctx([591], options={"max_frames_per_file": 4}))

    np.savez(path, analysis="spectrum", psd=np.zeros(4))
    with pytest.raises(SystemExit):     # foreign product file
        ControlBandAnalyzer().resume(path, _ctx([591]))


def test_begin_refuses_center_freq_id_disagreement():
    a = ControlBandAnalyzer()
    with pytest.raises(SystemExit):
        a.begin(_ctx([591]), _meta("evtA", center_hz=566_406_250.0 + 5e3))


def test_mismatched_file_center_is_skipped_not_folded(tmp_path):
    a = ControlBandAnalyzer()
    a.begin(_ctx([591]), _meta("evtA"))
    assert a.consume_file(_frames(2), _meta("evtA")) == 2
    other = _meta("evtB", center_hz=566_406_250.0 + FS)   # neighboring channel
    assert a.consume_file(_frames(2), other) == 0
    assert a.processed_keys() == {_meta("evtA")["unit_key"]}


def test_feed_count_change_refuses():
    a = ControlBandAnalyzer()
    a.begin(_ctx([591]), _meta("evtA"))
    a.consume_file(_frames(1), _meta("evtA"))
    with pytest.raises(ValueError):
        a.consume_file(_frames(1, n_feeds=N_FEEDS + 1), _meta("evtB"))
