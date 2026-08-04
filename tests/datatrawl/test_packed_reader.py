# coding=utf-8
"""The native-packed reader yields the same samples as the bundled unpacking
reader, just in raw 4+4-bit form (so the kernel packing keeps the int4 grid)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
datatrawl = pytest.importorskip("datatrawl")

from datatrawl.plugins.readers import _baseband_format as fmt
from datatrawl.plugins.readers.chime_baseband import ChimeBasebandReader
from datatrawl.instruments import load_instrument
from datatrawl.interfaces import RunContext

from pilot_proxy.datatrawl_plugins.packed_reader import ChimeBasebandPackedReader

NFFT = 16384
N_FRAMES = 3
N_FEEDS = 8


def test_packed_reader_matches_unpacked(tmp_path):
    synth = tmp_path / "chime_synth.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=3)

    ctx = RunContext(instrument=load_instrument("chime"))
    packed_reader = ChimeBasebandPackedReader()
    unpacked_reader = ChimeBasebandReader()

    # The packed reader's probe is a superset of the bundled reader's: identical
    # channel/format keys, plus a per-unit absolute-time axis read from the file
    # root attrs (NaN/0/None/"" on a synth file that carries only `freq`).
    unpacked_probe = unpacked_reader.probe(str(synth))
    packed_probe = packed_reader.probe(str(synth))
    for k, v in unpacked_probe.items():
        assert packed_probe[k] == v, k
    for k in ("time0_ctime", "delta_time", "time0_fpga_count",
              "event_id", "archive_version"):
        assert k in packed_probe, k

    packed_chunks = list(packed_reader.iter_arrays(str(synth), ctx))
    unpacked_chunks = list(unpacked_reader.iter_arrays(str(synth), ctx))
    assert len(packed_chunks) == len(unpacked_chunks) == N_FRAMES

    for raw, cplx in zip(packed_chunks, unpacked_chunks):
        assert raw.dtype == np.uint8
        assert raw.shape == (NFFT, N_FEEDS)
        # unpacking the raw bytes must reproduce the bundled reader's complex chunk
        assert np.array_equal(fmt.unpack_4bit(raw), cplx)
