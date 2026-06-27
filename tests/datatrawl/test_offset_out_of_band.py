# coding=utf-8
"""Graceful out-of-band handling for the `pilot-proxy-offset` analyzer.

Reproduces, self-contained, the real-data failure that surfaced on an uploaded
CHIME file: ``freq_id`` 400 (centre 643.75 MHz) is the nearest coarse channel to
ATSC ch43, but ch43's pilot sits 559 kHz away -- outside this channel's +/-fs/2
(195 kHz) Nyquist span. The peak search then had a sub-three-bin window and the
old analyzer raised an opaque error mid-run.

The analyzer must instead recognise the pilot is not in-band, emit an all-invalid
product (valid=0, NaN offsets) with ``pilot_in_band=0``, and never raise -- so a
mis-selected freq_id degrades to "no detection" rather than crashing an archive
scan. The contrasting in-band channel for ch43's pilot is freq_id 399.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
datatrawl = pytest.importorskip("datatrawl")

from datatrawl.plugins.readers import _baseband_format as fmt
from datatrawl.plugins.readers.chime_baseband import ChimeBasebandReader
from datatrawl.instruments import load_instrument
from datatrawl.interfaces import RunContext

from pilot_proxy.datatrawl_plugins.offset import PilotProxyOffsetAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[2]

NFFT = 16384
N_FRAMES = 3
N_FEEDS = 8
# freq_id 400: nearest ATSC channel is 43, whose pilot is 559 kHz off-centre
# (> fs/2), i.e. this coarse channel contains no in-band pilot.
FREQ_ID = 400
F_CENTER_MHZ = 643.75
NEAREST_ATSC = 43


def test_offset_analyzer_out_of_band_is_graceful(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # default receiver-profile path resolves here

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # filename carries the freq_id, exactly like a real CADC baseband file
    synth = data_dir / f"baseband_evt_{FREQ_ID}.h5"
    fmt.make_synth_file(
        str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=11,
    )

    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "stream_batch_size": 128,
        "window_name": "hann",
        "offset_backend": "numpy",
    })
    reader = ChimeBasebandReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = f"synth:freqid{FREQ_ID}"

    red = PilotProxyOffsetAnalyzer()
    # begin must warn that there is no in-band pilot here ...
    with pytest.warns(RuntimeWarning, match="does not contain"):
        red.begin(ctx, meta)
    # ... and consuming the file must NOT raise (the old failure mode)
    n = red.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    assert n == N_FRAMES

    out = tmp_path / "out" / f"{FREQ_ID}.npz"
    red.save(str(out))
    got = np.load(out)

    # identity: labelled with the nearest ATSC channel, freq_id recorded, flagged
    assert int(got["physical_channel"][0]) == NEAREST_ATSC
    assert int(got["freq_id"][0]) == FREQ_ID
    assert int(got["pilot_in_band"][0]) == 0

    # every frame present but invalid (no in-band pilot to measure)
    valid = np.asarray(got["valid"]).reshape(-1)
    assert valid.shape[0] == N_FRAMES
    assert int(valid.sum()) == 0
    for key in ("peak_offset_hz", "frequency_offset_hz", "peak_prominence_db"):
        col = np.asarray(got[key], dtype=np.float64)
        assert np.all(np.isnan(col)), f"{key} should be all-NaN out of band"

    # the time-average spectrum is still accumulated (diagnostic of what IS there)
    assert int(got["time_average_spectrum_count"][0]) == N_FRAMES
    assert np.isfinite(
        np.asarray(got["time_average_spectrum_power_linear"], dtype=np.float64)
    ).any()
