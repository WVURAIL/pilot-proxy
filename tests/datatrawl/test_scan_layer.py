# coding=utf-8
"""Scan-orchestration guards (review round 2).

Covers the source/plumbing and failure-handling the second review asked for:
  #3  cadc-datatrail source plumbing (--inventory / --source-root) + freq_id
      enumeration, exercised offline with a fake inventory and a mocked fetch;
  #4  a GPU/cupy preflight for pilot-proxy-detector before any staging;
  #6  an all-units-failed / all-quarantined scan is surfaced, not silently turned
      into an absent/empty product fed to combine.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
datatrawl = pytest.importorskip("datatrawl")

from datatrawl.plugins.readers import _baseband_format as fmt

from pilot_proxy.datatrawl_plugins.scan import run_chime_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
N_FEEDS = 4
K = 128


def _stub_detector_fn(*, packed, weights, kernel):
    """Trivial CPU detector: valid-schema per-block sums (plumbing only, not parity)."""
    pk = np.asarray(packed)
    if pk.ndim == 2:
        pk = pk[None, ...]
    n = int(pk.shape[0])
    return {
        "batch": n,
        "detector_rows_per_block": int(pk.shape[1]),
        "rational_overflow_count": 0,
        "results": [
            {"block_index": b, "mask": 0, "p_target_u64": 10, "p_ref_sum_u64": 20}
            for b in range(n)
        ],
    }


def _stub_kernel(k):
    specs = SimpleNamespace(
        K=k, N=3, bits=4, reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": k, "num_weight_terms": 3,
            "sample_bits_per_component": 4, "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _cpu_detector_options():
    """CPU detector injection so scan plumbing runs GPU-free."""
    rng = np.random.default_rng(0)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8)
        for ch in range(10, 41)
    }
    return {
        "detector_fn": _stub_detector_fn,
        "kernel": _stub_kernel(K),
        "weights_by_channel": weights_by_channel,
    }
# freq_id -> coarse-channel centre (MHz)
CHAN_MHZ = {844: 470.3125, 829: 476.171875, 752: 506.171875}

_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _make_inventory(inv_path, *, common_path, event, channels):
    with open(inv_path, "w") as fh:
        for ch in channels:
            fh.write(json.dumps({"common_path": common_path, "event": event,
                                 "freq_id": int(ch), "size_bytes": 1}) + "\n")


# -- #3: CADC plumbing + freq_id enumeration (offline) -----------------------

def test_cadc_scan_enumerates_by_freq_id(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # receiver-profile default path resolves here
    data = tmp_path / "data"
    data.mkdir()
    synth = {}
    for ch, mhz in CHAN_MHZ.items():
        p = data / f"baseband_evt1_{ch}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * 2, n_feeds=N_FEEDS,
                            f_center_mhz=mhz, f_tone_bb=1300.0, seed=ch)
        synth[ch] = p
    inv = tmp_path / "inventory.jsonl"
    _make_inventory(inv, common_path="cadc:TEST", event="evt1",
                    channels=sorted(CHAN_MHZ))  # all three listed

    # offline fetch: copy the local synth matching the unit's channel
    from datatrawl.plugins.sources.cadc_datatrail import CadcDatatrailSource

    def _fake_fetch(self, unit, dest, *a, **k):
        shutil.copyfile(synth[int(unit.meta["freq_id"])], dest)
        return True, ""

    monkeypatch.setattr(CadcDatatrailSource, "fetch", _fake_fetch)

    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="cadc-datatrail", inventory=inv,
                   analyzer="pilot-proxy-detector", select="829,844",
                   analyzer_options=_cpu_detector_options(), verbose=False)

    work = out / "_per_pilot"
    assert (work / "829.npz").exists()
    assert (work / "844.npz").exists()
    assert not (work / "752.npz").exists()   # listed in inventory but not selected
    assert (out / "chime_detector_outputs.npz").exists()  # combined product


# -- #3: explicit per-source option validation -------------------------------

def test_local_requires_input_dir(tmp_path):
    with pytest.raises(SystemExit, match="--input-dir"):
        run_chime_scan(output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


def test_cadc_requires_inventory_or_root(tmp_path):
    with pytest.raises(SystemExit, match="--inventory"):
        run_chime_scan(output_dir=tmp_path / "o", source="cadc-datatrail",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


# -- #4: GPU preflight for the detector --------------------------------------

@pytest.mark.skipif(_HAS_CUPY, reason="cupy is installed; the preflight would pass")
def test_detector_preflight_requires_cupy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_844.h5"), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=470.3125, f_tone_bb=1300.0, seed=1)
    # no injected detector_fn -> the real CUDA kernel is required -> must preflight
    with pytest.raises(SystemExit, match="cupy"):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844", verbose=False)


# -- #6: all-units-failed/quarantined is surfaced ----------------------------

def test_all_units_failed_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    # a file the reader cannot open -> the only unit fails -> no product written
    (data / "baseband_e_844.h5").write_bytes(b"not a valid hdf5 file")
    with pytest.raises(SystemExit, match="no usable product"):
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o", source="local",
                       analyzer="pilot-proxy-detector", select="844",
                       analyzer_options=_cpu_detector_options(), verbose=False)


def test_checkpoint_every_reaches_pipeline(tmp_path, monkeypatch):
    # --checkpoint-every must thread CLI -> run_chime_scan -> pipeline.run, and an
    # unset value must fall back to the engine default (50). Spy on pipeline.run and
    # stop before the real work, so the guard needs no GPU and no processable data.
    monkeypatch.chdir(REPO_ROOT)
    import datatrawl.pipeline as _dpl
    seen: dict = {}

    class _Stop(Exception):
        pass

    def _spy(*a, checkpoint_every=None, **k):
        seen["ckpt"] = checkpoint_every
        raise _Stop

    monkeypatch.setattr(_dpl, "run", _spy)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "baseband_e_829.h5"), n_time=NFFT, n_feeds=N_FEEDS,
                        f_center_mhz=476.3125, f_tone_bb=1200.0, seed=1)

    with pytest.raises(_Stop):                       # explicit value threads through
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o1", source="local",
                       analyzer="pilot-proxy-detector", select="829", checkpoint_every=7,
                       analyzer_options=_cpu_detector_options(), verbose=False)
    assert seen["ckpt"] == 7

    with pytest.raises(_Stop):                       # unset -> engine default
        run_chime_scan(input_dir=data, output_dir=tmp_path / "o2", source="local",
                       analyzer="pilot-proxy-detector", select="829",
                       analyzer_options=_cpu_detector_options(), verbose=False)
    assert seen["ckpt"] == 50
