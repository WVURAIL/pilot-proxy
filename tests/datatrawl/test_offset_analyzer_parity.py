# coding=utf-8
"""Parity test: the `pilot-proxy-offset` datatrawl analyzer reproduces PilotProxy's own
`run_frequency_offset_diagnostic` bit-for-bit on a synthetic CHIME baseband file.

Skipped automatically unless the optional datatrawl path is installed
(``pip install -e .[datatrawl]`` plus a datatrawl checkout). The analyzer wraps
fstat's DSP, so this guards against the wrapper drifting from the reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pandas")
datatrawl = pytest.importorskip("datatrawl")

from datatrawl.plugins.readers import _baseband_format as fmt
from datatrawl.plugins.readers.chime_baseband import ChimeBasebandReader
from datatrawl.instruments import load_instrument
from datatrawl.interfaces import RunContext

from pilot_proxy.chime.frequency_offset import (
    FrequencyOffsetConfig,
    run_frequency_offset_diagnostic,
)
from pilot_proxy.datatrawl_plugins.offset import PilotProxyOffsetAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[2]

NFFT = 16384
N_FRAMES = 3
N_FEEDS = 8
PHYS_CH = 14
F_CENTER_MHZ = 470.3125          # CHIME coarse-channel centre for DTV 14
FREQ_ID = 844                    # that centre as a CHIME freq_id
F_TONE_BB = 1500.0               # native tone; image stays inside the search window
HALF_WIDTH_HZ = 5000.0
STREAM_BATCH = 128

# arrays that must match, with (rtol, atol). DSP outputs are bit-identical in
# practice; tolerances guard only against cross-platform FFT last-bit drift.
_FLOAT_CHECKS = (
    ("peak_offset_hz", 1e-9, 1e-6),
    ("frequency_offset_hz", 1e-9, 1e-6),
    ("peak_power_linear", 1e-9, 1e-3),
    ("local_floor_power_linear", 1e-9, 1e-3),
    ("peak_prominence_db", 1e-9, 1e-9),
    ("relative_time_s", 1e-9, 1e-9),
    ("time_average_spectrum_power_linear", 1e-9, 1e-3),
    ("expected_pilot_offset_hz", 1e-12, 1e-6),
    ("coarse_channel_center_hz", 1e-12, 1e-6),
    ("pilot_frequency_hz", 1e-12, 1e-6),
)


def test_offset_analyzer_matches_fstat(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # default receiver-profile path resolves here

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    synth = data_dir / f"baseband_evt_{FREQ_ID}.h5"
    fmt.make_synth_file(
        str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
        f_center_mhz=F_CENTER_MHZ, f_tone_bb=F_TONE_BB, seed=7,
    )

    # reference: PilotProxy's own diagnostic
    ref_dir = tmp_path / "ref"
    cfg = FrequencyOffsetConfig(
        input_dir=data_dir, output_dir=ref_dir,
        physical_channels=[PHYS_CH], physical_channel_range=None,
        dataset_path="baseband", filename_pattern="*.h5",
        frame_size_samples=NFFT, frames_per_chunk=1, max_frames=None,
        fft_size=NFFT, stream_batch_size=STREAM_BATCH,
        peak_search_half_width_hz=HALF_WIDTH_HZ, window_name="hann",
        min_peak_prominence_db=None, backend="numpy", plot=False,
    )
    run_frequency_offset_diagnostic(cfg)
    ref = np.load(ref_dir / "frequency_offset_outputs.npz")

    # candidate: the datatrawl analyzer over the real chime-baseband reader
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "peak_search_half_width_hz": HALF_WIDTH_HZ,
        "stream_batch_size": STREAM_BATCH,
        "window_name": "hann",
        "offset_backend": "numpy",
    })
    reader = ChimeBasebandReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = "synth:dtv14"
    red = PilotProxyOffsetAnalyzer()
    red.begin(ctx, meta)
    red.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    out = tmp_path / "out" / "14.npz"
    red.save(str(out))
    got = np.load(out)

    assert int(got["physical_channel"][0]) == PHYS_CH
    assert int(got["freq_id"][0]) == FREQ_ID
    assert int(got["pilot_in_band"][0]) == 1  # ch14's pilot is in this coarse channel
    assert got["frequency_offset_hz"].shape == ref["frequency_offset_hz"].shape

    for name, rtol, atol in _FLOAT_CHECKS:
        a = np.asarray(ref[name], dtype=np.float64)
        b = np.asarray(got[name], dtype=np.float64)
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), (
            f"{name}: max|abs|={np.nanmax(np.abs(a - b)):.3e}"
        )

    assert np.array_equal(np.asarray(ref["valid"]), np.asarray(got["valid"]))
