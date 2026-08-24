# coding=utf-8
"""The native-packed reader yields the same samples as the bundled unpacking
reader, just in raw 4+4-bit form (so the kernel packing keeps the int4 grid)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("datatrawl.interfaces")

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


def _ctx_with_nfft(nfft):
    """RunContext with a per-run nfft override, via the same attribute the CLI's
    --nfft mutates and iter_arrays reads (ctx.instrument.nfft)."""
    inst = load_instrument("chime")
    inst.nfft = nfft
    return RunContext(instrument=inst)


def test_probe_rejects_short_file_as_unreadable(tmp_path):
    """A file too short for one transform must be quarantinable, not run-fatal.

    Found by the 2026-08-23 pre-flight: a 3.9 MB stub in the CANFAR staging
    aborted a scan that had already completed eight channels, because the
    analyzer raised on zero frames and datatrawl treats analyzer exceptions as
    run-level errors by design. Classifying it at probe time hands the engine
    the one disposition that quarantines without interrupting the run at all;
    the iteration-time check below covers the configurations probe cannot see.
    """
    from datatrawl.interfaces import UnreadableUnitError

    short = tmp_path / "too_short.h5"
    fmt.make_synth_file(str(short), n_time=NFFT - 1, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=5)
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        ChimeBasebandPackedReader().probe(str(short))


def test_nfft_above_default_quarantines_at_iteration(tmp_path):
    """Instrument nfft above the format default: a file with one default-length
    frame but zero frames at the run's nfft must raise the quarantinable type
    from iter_arrays, not reach the analyzer's run-fatal zero-frame check."""
    from datatrawl.interfaces import UnreadableUnitError

    path = tmp_path / "one_default_frame.h5"
    fmt.make_synth_file(str(path), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=8)
    reader = ChimeBasebandPackedReader()
    reader.probe(str(path))
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        list(reader.iter_arrays(str(path), _ctx_with_nfft(2 * NFFT)))


def test_nfft_below_default_is_deliberately_conservative(tmp_path):
    """Instrument nfft below the format default: probe still quarantines.

    Probe never sees the configured nfft, so its gate uses the format default
    that every shipped instrument frames at. A file that could frame at a
    smaller configured nfft is therefore quarantined at probe -- the accepted
    cost of catching stubs without interrupting an unattended multi-week scan;
    no shipped instrument configures nfft below the default. This test pins
    that trade so a future below-default instrument revisits it deliberately.
    """
    from datatrawl.interfaces import UnreadableUnitError

    small = NFFT // 4
    path = tmp_path / "two_small_frames.h5"
    fmt.make_synth_file(str(path), n_time=2 * small, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=9)
    reader = ChimeBasebandPackedReader()
    with pytest.raises(UnreadableUnitError, match="shorter than one transform"):
        reader.probe(str(path))
    chunks = list(reader.iter_arrays(str(path), _ctx_with_nfft(small)))
    assert len(chunks) == 2
    assert all(c.shape == (small, N_FEEDS) for c in chunks)


def test_probe_accepts_exactly_one_frame(tmp_path):
    """The boundary is inclusive: exactly nfft samples is one usable frame."""
    exact = tmp_path / "exactly_one.h5"
    fmt.make_synth_file(str(exact), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1500.0, seed=6)
    probe = ChimeBasebandPackedReader().probe(str(exact))
    assert probe["num_input_streams"] == N_FEEDS
    ctx = RunContext(instrument=load_instrument("chime"))
    assert len(list(ChimeBasebandPackedReader().iter_arrays(str(exact), ctx))) == 1
