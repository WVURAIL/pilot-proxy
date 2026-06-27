# coding=utf-8
"""Guards for selection + per-file/per-product validation (review cluster A).

Covers the reviewer's requested cases:
  #1  an omitted selection fails cleanly instead of mixing channels;
  #2  a file whose channel disagrees with the product is a hard error;
and the combiner refusing to stack products with mismatched geometry.

All GPU-free: the channel guard fires before any detector work, so the detector
analyzer needs only a stub kernel / weights / dummy detector_fn to reach it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

datatrawl = pytest.importorskip("datatrawl")
pytest.importorskip("h5py")

from datatrawl.instruments import load_instrument
from datatrawl.interfaces import RunContext

from pilot_proxy.datatrawl_plugins.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.datatrawl_plugins.offset import PilotProxyOffsetAnalyzer
from pilot_proxy.datatrawl_plugins.scan import run_chime_scan
from pilot_proxy.datatrawl_plugins.combine import combine_detector_products

K = 128
CH14_HZ = 470.3125e6   # -> ATSC channel 14, CHIME freq_id 844
CH20_HZ = 506.3125e6   # -> a clearly different ATSC channel / freq_id
FREQ_ID14 = 844        # chime_freq_id_from_hz(CH14_HZ)


def _stub_kernel(k=K):
    specs = SimpleNamespace(K=k, N=3, bits=4, reference_offset_bins=2,
                            as_descriptive_dict=lambda: {})
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "t"))


def _detector_ctx():
    return RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID14], options={
        "detector_fn": lambda **kw: {"results": [], "rational_overflow_count": 0},
        "kernel": _stub_kernel(),
        "weights": np.zeros((3, K), dtype=np.int8),
    })


def _offset_ctx():
    return RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID14], options={})


# -- #1: omitted selection must fail, not silently mix -----------------------

@pytest.mark.parametrize("analyzer_cls", [PilotProxyDetectorAnalyzer, PilotProxyOffsetAnalyzer])
def test_plan_runs_rejects_empty_selection(analyzer_cls):
    red = analyzer_cls()
    ctx = RunContext(instrument=load_instrument("chime"))
    for empty in (None, "", "   ", []):
        with pytest.raises(ValueError, match="explicit freq_id selection"):
            red.plan_runs(ctx, empty)


def test_run_chime_scan_requires_select(tmp_path):
    for empty in (None, "", []):
        with pytest.raises(SystemExit, match="--select is required"):
            run_chime_scan(input_dir=tmp_path, output_dir=tmp_path / "o",
                           select=empty, analyzer="pilot-proxy-detector", verbose=False)


# -- #2: a file from the wrong channel is a hard error -----------------------

def test_detector_rejects_channel_mismatch():
    red = PilotProxyDetectorAnalyzer()
    red.begin(_detector_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    # a later file that belongs to a different coarse channel must not be absorbed
    with pytest.raises(ValueError, match="Refusing to mix coarse channels"):
        red.consume_file(iter(()), {"f_center_hz": CH20_HZ})


def test_offset_rejects_channel_mismatch():
    red = PilotProxyOffsetAnalyzer()
    red.begin(_offset_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    with pytest.raises(ValueError, match="Refusing to mix coarse channels"):
        red.consume_file(iter(()), {"f_center_hz": CH20_HZ})


def test_detector_rejects_nfft_mismatch():
    red = PilotProxyDetectorAnalyzer()
    red.begin(_detector_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    with pytest.raises(ValueError, match="nfft"):
        red.consume_file(iter(()), {"f_center_hz": CH14_HZ, "nfft": 8192})


# -- combiner refuses mismatched geometry (F10) ------------------------------

def _write_min_detector_product(path, *, channel, nfft, n_frames=2, unit_keys=(),
                                event_keys=None, contract=None):
    fields = dict(
        physical_channel=np.asarray([channel], dtype=np.int32),
        freq_id=np.asarray([800 - channel], dtype=np.int64),  # arbitrary but recorded
        schema_version=np.asarray("fstat_detector_v_test"),
        nfft=np.asarray(int(nfft), dtype=np.int64),
        detector_window_samples=np.asarray(K, dtype=np.int64),
        sense=np.asarray(-1, dtype=np.int64),
        frame_index=np.arange(int(n_frames), dtype=np.int64),
        unit_keys=np.asarray([str(k) for k in unit_keys]),
    )
    if event_keys is not None:
        fields["source_event_keys"] = np.asarray([str(k) for k in event_keys])
    if contract is not None:
        fields["detector_contract_json"] = np.asarray(str(contract))
    np.savez(path, **fields)


def test_combine_rejects_mismatched_nfft(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384)
    _write_min_detector_product(b, channel=20, nfft=8192)  # mismatched geometry
    with pytest.raises(ValueError, match="disagree on 'nfft'"):
        combine_detector_products([a, b], tmp_path / "out")


def test_combine_rejects_frame_grid_mismatch(tmp_path):
    # same geometry, but one channel processed fewer files -> shorter, mis-aligned
    # frame grid. Positional intersection would silently mis-time frames, so the
    # combiner must refuse with an actionable diagnostic instead (F4).
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384, n_frames=3,
                                unit_keys=("f1", "f2", "f3"))
    _write_min_detector_product(b, channel=20, nfft=16384, n_frames=2,
                                unit_keys=("f1", "f3"))  # f2 quarantined here
    with pytest.raises(ValueError, match="not time-aligned"):
        combine_detector_products([a, b], tmp_path / "out")


# -- review round 2 -----------------------------------------------------------

# #1: equal frame counts but different source events must NOT stack as aligned.
def test_combine_rejects_same_count_different_events(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384, n_frames=2,
                                event_keys=["baseband_eventA.h5", "baseband_eventB.h5"])
    _write_min_detector_product(b, channel=20, nfft=16384, n_frames=2,
                                event_keys=["baseband_eventA.h5", "baseband_eventC.h5"])
    with pytest.raises(ValueError, match="different source events"):
        combine_detector_products([a, b], tmp_path / "out")


# #9: two coarse channels that resolve to the same ATSC channel must be rejected.
def test_combine_rejects_duplicate_physical_channel(tmp_path):
    a = tmp_path / "399.npz"
    b = tmp_path / "400.npz"
    _write_min_detector_product(a, channel=43, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"])
    _write_min_detector_product(b, channel=43, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"])
    with pytest.raises(ValueError, match="appear in more than one"):
        combine_detector_products([a, b], tmp_path / "out")


# #9: a duplicate freq_id in --select must be rejected.
@pytest.mark.parametrize("analyzer_cls", [PilotProxyDetectorAnalyzer, PilotProxyOffsetAnalyzer])
def test_plan_runs_rejects_duplicate_freq_id(analyzer_cls):
    red = analyzer_cls()
    ctx = RunContext(instrument=load_instrument("chime"))
    with pytest.raises(ValueError, match="duplicate freq_id"):
        red.plan_runs(ctx, "844,844")


# #7: a callable combine must reject products with different detector contracts.
def test_combine_rejects_contract_mismatch(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"],
                                contract='{"detector_window_samples": 128}')
    _write_min_detector_product(b, channel=20, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"],
                                contract='{"detector_window_samples": 64}')  # differs
    with pytest.raises(ValueError, match="detector_contract_json"):
        combine_detector_products([a, b], tmp_path / "out")


# #2: the FIRST file must match the requested freq_id, not just later files.
def test_detector_rejects_first_file_freq_id_mismatch():
    red = PilotProxyDetectorAnalyzer()
    # ctx requests freq_id 844, but the first file's centre is a different channel
    with pytest.raises(ValueError, match="requested freq_id 844"):
        red.begin(_detector_ctx(), {"f_center_hz": CH20_HZ, "nfft": 16384})


def test_offset_rejects_first_file_freq_id_mismatch():
    red = PilotProxyOffsetAnalyzer()
    with pytest.raises(ValueError, match="requested freq_id 844"):
        red.begin(_offset_ctx(), {"f_center_hz": CH20_HZ, "nfft": 16384})


# #5: an out-of-band pilot must yield an explicitly invalid detector product and
# must NOT invoke the (GPU) kernel.
def test_detector_out_of_band_emits_invalid_without_kernel(tmp_path):
    def _boom(**kw):
        raise AssertionError("detector_fn must not run for an out-of-band pilot")

    ctx = RunContext(instrument=load_instrument("chime"), selection=[400], options={
        "detector_fn": _boom,
        "kernel": _stub_kernel(),
        "weights": np.zeros((3, K), dtype=np.int8),
    })
    red = PilotProxyDetectorAnalyzer()
    # freq_id 400 (643.75 MHz) -> nearest ATSC 43, whose pilot is 559 kHz off-centre
    with pytest.warns(RuntimeWarning, match="does not contain"):
        red.begin(ctx, {"f_center_hz": 643.75e6, "nfft": 16384})
    chunk = np.zeros((16384, 4), dtype=np.uint8)
    meta = {"f_center_hz": 643.75e6, "nfft": 16384, "unit_key": "baseband_e_400.h5"}
    n = red.consume_file([chunk, chunk], meta)  # must not raise (kernel skipped)
    assert n == 2
    out = tmp_path / "400.npz"
    red.save(str(out))
    got = np.load(out)
    assert int(got["pilot_in_band"][0]) == 0
    assert int(got["physical_channel"][0]) == 43
    assert int(got["freq_id"][0]) == 400
    assert int(np.asarray(got["reject_mask"]).sum()) == 0
    assert int(np.asarray(got["valid"]).sum()) == 0
