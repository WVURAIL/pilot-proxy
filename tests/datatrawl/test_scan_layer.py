# coding=utf-8
"""Scan-orchestration guards.

Covers source/plumbing and failure handling:
  #3  cadc-datatrail source plumbing (--inventory / --source-root) + freq_id
      enumeration, exercised offline with a fake inventory and a mocked fetch;
  #4  a GPU/cupy preflight for pilot-proxy-detector before any staging;
  #6  an all-units-failed / all-quarantined scan is surfaced rather than silently turned
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
pytest.importorskip("datatrawl.interfaces")

from datatrawl.plugins.readers import _baseband_format as fmt

from pilot_proxy.datatrawl_plugins.scan import run_chime_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
NFFT = 16384
N_FEEDS = 4
K = 128


def _stub_detector_fn(*, packed, weights, kernel):
    """Trivial CPU detector: valid-schema per-block sums (plumbing only; no parity claim)."""
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
# freq_id -> coarse-channel center (MHz)
CHAN_MHZ = {844: 470.3125, 829: 476.171875, 752: 506.171875}

_HAS_CUPY = importlib.util.find_spec("cupy") is not None


def _make_inventory(inv_path, *, common_path, event, channels):
    with open(inv_path, "w") as fh:
        for ch in channels:
            fh.write(json.dumps({"common_path": common_path, "event": event,
                                 "scope": "test.scope",
                                 "name": f"baseband_{event}_{int(ch)}.h5",
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
    # The injected backend always says mask=0. The canonical product instead
    # derives the policy it declares from exact powers and stored weight norms.
    with np.load(work / "829.npz", allow_pickle=False) as product:
        num = np.asarray(product["p_target_u64"]).reshape(-1)
        den = np.asarray(product["p_ref_sum_u64"]).reshape(-1)
        target_norm = int(np.asarray(product["target_norm_sq"]).reshape(-1)[0])
        reference_norm = int(
            np.asarray(product["reference_norm_sum_sq"]).reshape(-1)[0]
        )
        expected = [
            int(int(n) * reference_norm > target_norm * int(d)) if d else 0
            for n, d in zip(num, den)
        ]
        assert np.asarray(product["reject_mask"]).reshape(-1).tolist() == expected
    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["complete"] is True
    assert scope["totals"] == {
        "requested": 2,
        "enumerated": 2,
        "completed": 2,
        "capped": 0,
        "failed": 0,
        "quarantined": 0,
        "unprocessed": 0,
        "extra_completed": 0,
        "pilots_requested": 2,
    }


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
    scope = json.loads((tmp_path / "o" / "scan_scope.json").read_text())
    assert scope["pilots"][0]["enumerated"] == 1
    assert scope["pilots"][0]["completed"] == 0
    assert scope["pilots"][0]["quarantined"] == 1


def test_partial_scan_requires_explicit_allow_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in ("a", "b"):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT * 2,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=ord(event),
        )
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "max_files": 1,
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(SystemExit, match="--allow-partial"):
        run_chime_scan(**kwargs)

    scope = json.loads((out / "scan_scope.json").read_text())
    assert scope["pilots"][0]["enumerated"] == 2
    assert scope["pilots"][0]["completed"] == 1
    assert scope["pilots"][0]["unprocessed"] == 1
    assert scope["complete"] is False

    result = run_chime_scan(**kwargs, allow_partial=True)
    assert result["scan_scope"] == out / "scan_scope.json"
    assert (out / "chime_detector_outputs.npz").exists()


def test_chunk_cap_records_capped_units_and_requires_allow_partial(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(
        str(data / "baseband_a_844.h5"),
        n_time=NFFT * 2,
        n_feeds=N_FEEDS,
        f_center_mhz=CHAN_MHZ[844],
        f_tone_bb=1300.0,
        seed=1,
    )
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "max_chunks_per_file": 1,
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }

    with pytest.raises(SystemExit, match="--allow-partial"):
        run_chime_scan(**kwargs)

    assert not (out / "chime_detector_outputs.npz").exists()
    with np.load(out / "_per_pilot" / "844.npz", allow_pickle=False) as product:
        assert np.asarray(product["frame_index"]).shape == (1,)
    scope = json.loads((out / "scan_scope.json").read_text())
    pilot = scope["pilots"][0]
    assert scope["max_chunks_per_file"] == 1
    assert scope["complete"] is False
    assert pilot["status"] == "capped"
    assert pilot["completed"] == 0
    assert pilot["capped"] == 1
    assert len(pilot["capped_unit_keys"]) == 1
    assert pilot["unprocessed"] == 0

    result = run_chime_scan(**kwargs, allow_partial=True)
    assert result["scan_scope"] == out / "scan_scope.json"
    assert (out / "chime_detector_outputs.npz").exists()


def test_aborted_scan_counts_only_current_unit_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in (1, 2, 3):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )
    calls = 0

    def fail_second_unit(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-unit failure")
        return _stub_detector_fn(**kwargs)

    options = _cpu_detector_options()
    options["detector_fn"] = fail_second_unit
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="second-unit failure"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=options,
            verbose=False,
        )

    pilot = json.loads((out / "scan_scope.json").read_text())["pilots"][0]
    assert pilot["completed"] == 0
    assert pilot["non_durable_completed"] == 1
    assert pilot["failed"] == 1
    assert pilot["unprocessed"] == 2


def test_later_pilot_units_remain_accounted_when_earlier_pilot_aborts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for freq_id in (829, 844):
        fmt.make_synth_file(
            str(data / f"baseband_1_{freq_id}.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[freq_id],
            f_tone_bb=1300.0,
            seed=freq_id,
        )

    def fail_first_pilot(**kwargs):
        del kwargs
        raise RuntimeError("injected first-pilot failure")

    options = _cpu_detector_options()
    options["detector_fn"] = fail_first_pilot
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="first-pilot failure"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="829,844",
            analyzer_options=options,
            verbose=False,
        )

    entries = {
        tuple(entry["selection"]): entry
        for entry in json.loads((out / "scan_scope.json").read_text())["pilots"]
    }
    assert entries[(829,)]["failed"] == 1
    assert entries[(829,)]["unprocessed"] == 0
    assert entries[(844,)]["status"] == "pending"
    assert entries[(844,)]["unprocessed"] == 1


def test_shorter_than_nfft_unit_is_not_committed(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(
        str(data / "baseband_1_844.h5"),
        n_time=NFFT - 1,
        n_feeds=N_FEEDS,
        f_center_mhz=CHAN_MHZ[844],
        f_tone_bb=1300.0,
        seed=1,
    )
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="zero complete nfft frames"):
        run_chime_scan(
            input_dir=data,
            output_dir=out,
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=_cpu_detector_options(),
            verbose=False,
        )

    pilot = json.loads((out / "scan_scope.json").read_text())["pilots"][0]
    assert pilot["completed"] == 0
    assert pilot["failed"] == 1
    assert not (out / "_per_pilot" / "844.npz").exists()


def test_resume_rejects_saved_units_missing_from_current_enumeration(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    paths = []
    for event in (1, 2):
        path = data / f"baseband_{event}_844.h5"
        fmt.make_synth_file(
            str(path),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )
        paths.append(path)
    out = tmp_path / "out"
    kwargs = {
        "input_dir": data,
        "output_dir": out,
        "source": "local",
        "analyzer": "pilot-proxy-detector",
        "select": "844",
        "analyzer_options": _cpu_detector_options(),
        "verbose": False,
    }
    run_chime_scan(**kwargs)
    paths[1].unlink()

    with pytest.raises(SystemExit, match="absent|stale|current enumeration"):
        run_chime_scan(**kwargs, allow_partial=True)

    pilot = json.loads((out / "scan_scope.json").read_text())["pilots"][0]
    assert pilot["extra_completed"] == 1
    assert pilot["extra_unit_keys"] == [str(paths[1].resolve())]
    assert pilot["status"] == "stale"


def test_enumerated_scope_is_persisted_before_pipeline_runs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    for event in (1, 2):
        fmt.make_synth_file(
            str(data / f"baseband_{event}_844.h5"),
            n_time=NFFT,
            n_feeds=N_FEEDS,
            f_center_mhz=CHAN_MHZ[844],
            f_tone_bb=1300.0,
            seed=event,
        )

    import datatrawl.pipeline as _dpl

    class _Stop(Exception):
        pass

    def inspect_scope_before_work(*args, out_path, **kwargs):
        del args, kwargs
        scope_path = Path(out_path).parents[1] / "scan_scope.json"
        scope = json.loads(scope_path.read_text())
        pilot = scope["pilots"][0]
        assert pilot["requested"] == 2
        assert pilot["enumerated"] == 2
        assert pilot["unprocessed"] == 2
        assert scope["totals"]["requested"] == 2
        assert scope["totals"]["pilots_requested"] == 1
        raise _Stop

    monkeypatch.setattr(_dpl, "run", inspect_scope_before_work)

    with pytest.raises(_Stop):
        run_chime_scan(
            input_dir=data,
            output_dir=tmp_path / "out",
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options=_cpu_detector_options(),
            verbose=False,
        )


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


# -- --source-freq-id-regex reaches the datatrawl local source ----------------

def test_local_scan_freq_id_regex_reaches_source(tmp_path, monkeypatch):
    """The flag must populate source_freq_id_regex (what the paired
    LocalDirectorySource reads)."""
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    # Default parser expects _<freq_id>.h5; this layout needs the custom regex.
    fmt.make_synth_file(str(data / "evt1.freq844.h5"), n_time=NFFT * 2,
                        n_feeds=N_FEEDS, f_center_mhz=CHAN_MHZ[844],
                        f_tone_bb=1300.0, seed=11)
    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="local", input_dir=data,
                   analyzer="pilot-proxy-detector", select="844",
                   source_freq_id_regex=r"freq(\d+)\.h5$",
                   analyzer_options=_cpu_detector_options(), verbose=False)
    assert (out / "_per_pilot" / "844.npz").exists()


def test_local_scan_set_regex_overrides_flag(tmp_path, monkeypatch):
    """An explicit --set source_freq_id_regex=... takes precedence over the flag."""
    monkeypatch.chdir(REPO_ROOT)
    data = tmp_path / "data"
    data.mkdir()
    fmt.make_synth_file(str(data / "evt1.freq844.h5"), n_time=NFFT * 2,
                        n_feeds=N_FEEDS, f_center_mhz=CHAN_MHZ[844],
                        f_tone_bb=1300.0, seed=12)
    opts = _cpu_detector_options()
    opts["source_freq_id_regex"] = r"freq(\d+)\.h5$"     # the explicit --set
    out = tmp_path / "out"
    run_chime_scan(output_dir=out, source="local", input_dir=data,
                   analyzer="pilot-proxy-detector", select="844",
                   source_freq_id_regex=r"nomatch(\d+)$",  # flag must NOT win
                   analyzer_options=opts, verbose=False)
    assert (out / "_per_pilot" / "844.npz").exists()
