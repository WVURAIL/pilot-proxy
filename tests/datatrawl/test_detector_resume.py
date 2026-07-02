# coding=utf-8
"""Resume / relaunch safety for the ``pilot-proxy-detector`` analyzer.

A multi-day CANFAR detector run outlives a single interactive session, so
correctness on relaunch is load-bearing: a killed run must continue from its
checkpoint, not start over (scarce GPU) and not corrupt the product. These
tests drive the analyzer and the real scan entry point with an injected CPU
detector (no GPU) and assert:

  * a stream interrupted mid-way, resumed from its checkpoint, yields a product
    byte-identical to one consumed in a single pass (no reprocessing, no drift);
  * relaunching a scan whose channel is already complete is a no-op, not an
    error (the produced-check counts resumed units, not just new ones);
  * a product built with a per-file cap refuses to be silently "completed" by an
    uncapped run.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")
datatrawl = pytest.importorskip("datatrawl")

from datatrawl.plugins.readers import _baseband_format as fmt
from datatrawl.instruments import load_instrument
from datatrawl.interfaces import RunContext

from pilot_proxy.detector_contract import (
    norm_corrected_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    INT4_COMPONENT_BITS,
    fstat_cpu_reference,
    unpack_packed_complex,
)
from pilot_proxy.datatrawl_plugins.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.datatrawl_plugins.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.datatrawl_plugins.scan import run_chime_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
F_CENTER_MHZ = 470.3125
FREQ_ID = 844

# arrays that must match exactly / to fp tolerance between a clean and resumed run
_EXACT = ("p_target_u64", "p_ref_sum_u64", "reject_mask", "valid", "frame_index",
          "unit_keys", "unit_order", "frame_unit_index", "frame_in_unit",
          "unit_time0_fpga", "unit_event_id", "archive_version")
_CLOSE = ("fstat_raw", "fstat_level_db", "pnr_bin_db", "snr_shelf_db",
          "baseband_power_linear", "integrated_spectrum_before_mask",
          "integrated_spectrum_after_mask", "unit_time0_ctime", "unit_delta_time")


def _cpu_ref_detector_fn(*, packed, weights, kernel):
    """Input-dependent CPU reference standing in for the GPU kernel."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    w = unpack_packed_complex(np.asarray(weights, dtype=np.int8), INT4_COMPONENT_BITS)
    _nt, _nl, _nu = weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    _nrs = int(_nl + _nu)
    results = []
    for b in range(int(pk.shape[0])):
        samples = unpack_packed_complex(pk[b], INT4_COMPONENT_BITS)
        _fstat, sums = fstat_cpu_reference(samples, w)
        num = int(round(float(sums[0])))
        den = int(round(float(sums[1] + sums[2])))
        results.append({
            "block_index": b,
            "mask": norm_corrected_positive_excess(
                num, den, target_norm_sq=_nt, ref_norm_sum_sq=_nrs
            ),
            "p_target_u64": num,
            "p_ref_sum_u64": den,
        })
    return {
        "batch": int(pk.shape[0]),
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "results": results,
    }


def _stub_kernel(detector_window_samples: int, reference_offset_bins: int = 2):
    specs = SimpleNamespace(
        K=detector_window_samples, N=3, bits=4,
        reference_offset_bins=reference_offset_bins,
        as_descriptive_dict=lambda: {
            "detector_window_samples": detector_window_samples,
            "num_weight_terms": 3, "sample_bits_per_component": 4,
            "reference_offset_bins": reference_offset_bins,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _make_files(input_dir: Path, n_events: int) -> dict:
    input_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i in range(n_events):
        p = input_dir / f"baseband_evt{i}_{FREQ_ID}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                            f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=100 + i)
        files[(f"evt{i}", FREQ_ID)] = p
    return files


def _write_inventory(inv: Path, n_events: int) -> None:
    with open(inv, "w") as fh:
        for i in range(n_events):
            fh.write(json.dumps({"common_path": "cadc:TEST", "event": f"evt{i}",
                                 "freq_id": FREQ_ID, "size_bytes": 1}) + "\n")


def _assert_products_equal(ref, got):
    for name in _EXACT:
        assert np.array_equal(np.asarray(ref[name]), np.asarray(got[name])), name
    for name in _CLOSE:
        assert np.allclose(np.asarray(ref[name], dtype=np.float64),
                           np.asarray(got[name], dtype=np.float64),
                           rtol=1e-9, atol=1e-9, equal_nan=True), name


# -- analyzer level: a mid-stream checkpoint resumes to the same product ------

def test_analyzer_resume_matches_uninterrupted(tmp_path):
    rng = np.random.default_rng(7)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=3)
    paths = [files[(f"evt{i}", FREQ_ID)] for i in range(3)]

    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K), "weights": weights,
    })
    reader = ChimeBasebandPackedReader()

    def _meta(i):
        m = dict(reader.probe(str(paths[i])))
        m["unit_key"] = f"synth:{i}"
        return m

    def _consume(analyzer, idxs):
        for i in idxs:
            analyzer.consume_file(reader.iter_arrays(str(paths[i]), ctx), _meta(i))

    # uninterrupted: all three files through one analyzer
    a_clean = PilotProxyDetectorAnalyzer()
    a_clean.begin(ctx, _meta(0))
    _consume(a_clean, [0, 1, 2])
    clean_path = tmp_path / "clean.npz"
    a_clean.save(str(clean_path))

    # interrupted: files 0,1 -> checkpoint, then a fresh analyzer resumes file 2
    a1 = PilotProxyDetectorAnalyzer()
    a1.begin(ctx, _meta(0))
    _consume(a1, [0, 1])
    ckpt = tmp_path / "resumed.npz"
    a1.save(str(ckpt))

    a2 = PilotProxyDetectorAnalyzer()
    assert a2.resume(str(ckpt), ctx) is True
    assert a2.processed_keys() == {"synth:0", "synth:1"}
    a2.begin(ctx, _meta(2))            # first NEW file
    _consume(a2, [2])
    a2.save(str(ckpt))

    clean = np.load(clean_path)
    got = np.load(ckpt)
    assert got["frame_index"].shape[0] == clean["frame_index"].shape[0] == 3 * N_FRAMES
    _assert_products_equal(clean, got)


def test_analyzer_resume_absent_product_is_fresh(tmp_path):
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={})
    a = PilotProxyDetectorAnalyzer()
    assert a.resume(str(tmp_path / "nope.npz"), ctx) is False
    assert a.processed_keys() == set()


def test_analyzer_resume_rejects_changed_weights(tmp_path):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    base_options = {
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K),
        "weights": weights,
    }
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options=base_options)
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / "weights.npz"
    first.save(str(checkpoint))

    changed = weights.copy()
    changed[0, 0] = 2
    changed_ctx = RunContext(
        instrument=load_instrument("chime"), selection=[FREQ_ID],
        options={**base_options, "weights": changed},
    )
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="weights_hash"):
        resumed.begin(changed_ctx, meta)


def test_analyzer_resume_rejects_changed_detector_contract(tmp_path):
    files = _make_files(tmp_path / "data", n_events=1)
    path = files[("evt0", FREQ_ID)]
    reader = ChimeBasebandPackedReader()
    weights = np.ones((3, K), dtype=np.int8)
    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K), "weights": weights,
    })
    meta = dict(reader.probe(str(path)))
    meta["unit_key"] = "synth:0"
    first = PilotProxyDetectorAnalyzer()
    first.begin(ctx, meta)
    first.consume_file(reader.iter_arrays(str(path), ctx), meta)
    checkpoint = tmp_path / "contract.npz"
    first.save(str(checkpoint))

    changed_ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options={
        "detector_fn": _cpu_ref_detector_fn,
        "kernel": _stub_kernel(K, reference_offset_bins=3),
        "weights": weights,
    })
    resumed = PilotProxyDetectorAnalyzer()
    assert resumed.resume(str(checkpoint), changed_ctx)
    with pytest.raises(SystemExit, match="detector_contract"):
        resumed.begin(changed_ctx, meta)


# -- scan level: relaunch through the real entry point ------------------------

def _fake_fetch_factory(files):
    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(files[(str(unit.meta["event"]), int(unit.meta["freq_id"]))], dest)
        return True, ""
    return _fake_fetch


def test_scan_resume_and_noop_relaunch(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rng = np.random.default_rng(9)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=3)
    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, n_events=3)

    from datatrawl.plugins.sources.cadc_datatrail import CadcDatatrailSource
    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch_factory(files))

    inject = {"detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K),
              "weights": weights}

    def _scan(out_dir, **kw):
        run_chime_scan(output_dir=out_dir, source="cadc-datatrail", inventory=inv,
                       analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                       analyzer_options=inject, verbose=False, **kw)

    clean = tmp_path / "clean"
    _scan(clean)
    ref = np.load(clean / "_per_pilot" / f"{FREQ_ID}.npz")
    assert ref["frame_index"].shape[0] == 3 * N_FRAMES

    resumed = tmp_path / "resumed"
    _scan(resumed, max_files=2)                                   # process 2 of 3
    partial = np.load(resumed / "_per_pilot" / f"{FREQ_ID}.npz")
    assert partial["frame_index"].shape[0] == 2 * N_FRAMES        # checkpoint at 2
    _scan(resumed)                                                # resume -> 3rd file
    _scan(resumed)                                                # complete -> no-op (not an error)
    got = np.load(resumed / "_per_pilot" / f"{FREQ_ID}.npz")
    assert got["frame_index"].shape[0] == 3 * N_FRAMES
    _assert_products_equal(ref, got)


def test_scan_refuses_incompatible_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    rng = np.random.default_rng(8)
    weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)
    files = _make_files(tmp_path / "data", n_events=2)
    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, n_events=2)

    from datatrawl.plugins.sources.cadc_datatrail import CadcDatatrailSource
    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch_factory(files))

    inject = {"detector_fn": _cpu_ref_detector_fn, "kernel": _stub_kernel(K),
              "weights": weights}
    out = tmp_path / "capped"
    run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                   analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                   analyzer_options=inject, verbose=False, max_chunks_per_file=1)
    # The same cap is compatible and should be a no-op on a complete product.
    run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                   analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                   analyzer_options=inject, verbose=False, max_chunks_per_file=1)
    # relaunch WITHOUT the cap must refuse rather than complete the capped product
    with pytest.raises(SystemExit, match="capped product cannot be completed"):
        run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                       analyzer="pilot-proxy-detector", select=str(FREQ_ID),
                       analyzer_options=inject, verbose=False)
