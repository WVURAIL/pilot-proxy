#!/usr/bin/env python3
"""Decisive pass/fail checks on pilot-proxy-detector chime-scan output.

Two modes:

  check   <run_dir>            invariants on one scan's products (after a pilot or
                              the real run): schema, array alignment, mask
                              discipline, spectra/time sanity, combine stacking.

  compare <clean> <resumed>   equivalence of two runs of the SAME selection -- the
                              kill/resume test. Confirms a mid-run kill re-consumed
                              rather than double-counted or dropped, and that the
                              spectra/time-axis survived the restart.

Reads only the product .npz files (no GPU, no deps beyond numpy). Exits non-zero
on any failure so it can gate a script. Per-pilot products live in
<run_dir>/_per_pilot/<freq_id>.npz; combined products in <run_dir>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCHEMA = "pilotproxy_detector_datatrawl_v2"
PER_FRAME = ("frame_index", "p_target_u64", "p_ref_sum_u64", "fstat_raw",
             "fstat_level_db", "pnr_bin_db", "snr_shelf_db", "valid",
             "reject_mask", "baseband_power_linear", "frame_unit_index",
             "frame_in_unit")
PER_UNIT = ("unit_order", "unit_time0_ctime", "unit_time0_fpga", "unit_event_id",
            "unit_delta_time", "archive_version")
COMPARE_EXACT = ("p_target_u64", "p_ref_sum_u64", "reject_mask", "valid",
                 "frame_index", "frame_unit_index", "frame_in_unit", "source_event_keys",
                 "unit_time0_ctime", "unit_time0_fpga", "unit_event_id",
                 "unit_delta_time", "archive_version")


class Report:
    def __init__(self):
        self.fail = 0
        self.warn = 0

    def ok(self, msg):
        print(f"  [ ok ] {msg}")

    def bad(self, msg):
        print(f"  [FAIL] {msg}")
        self.fail += 1

    def warning(self, msg):
        print(f"  [warn] {msg}")
        self.warn += 1

    def check(self, cond, msg):
        self.ok(msg) if cond else self.bad(msg)
        return bool(cond)


def _1d(z, k):
    return np.asarray(z[k]).reshape(-1)


def _scalar(z, k):
    return np.asarray(z[k]).reshape(-1)[0]


def _per_pilot_paths(run: Path):
    pp = run / "_per_pilot"
    if not pp.is_dir():
        return []
    return sorted(p for p in pp.glob("*.npz")
                  if p.stem not in ("quarantine",) and p.stem.isdigit())


def check_per_pilot(path: Path, r: Report):
    print(f"\n--- per-pilot {path.name} ---")
    z = np.load(path)
    if not r.check(str(_scalar(z, "schema_version")) == SCHEMA,
                   f"schema_version == {SCHEMA}"):
        return None
    missing = [k for k in PER_FRAME + PER_UNIT if k not in z.files]
    r.check(not missing, f"all v2 fields present"
            + (f" (missing {missing})" if missing else ""))

    # per-frame arrays all the same length
    lens = {k: _1d(z, k).size for k in PER_FRAME if k in z.files}
    n = lens.get("frame_index", 0)
    r.check(len(set(lens.values())) == 1,
            f"per-frame arrays aligned (n_frames={n}; {sorted(set(lens.values()))})")

    # per-unit arrays all the same length
    ulens = {k: _1d(z, k).size for k in PER_UNIT if k in z.files}
    u = ulens.get("unit_order", 0)
    r.check(len(set(ulens.values())) == 1,
            f"per-unit arrays aligned (n_units={u}; {sorted(set(ulens.values()))})")

    fui = _1d(z, "frame_unit_index")
    r.check(u > 0 and fui.min() >= 0 and fui.max() < u,
            f"frame_unit_index in [0,{u})")

    valid = _1d(z, "valid").astype(bool)
    reject = _1d(z, "reject_mask").astype(bool)
    r.check(not np.any(reject & ~valid),
            "reject_mask=1 only on valid frames (positive excess needs valid)")

    nfft = int(_scalar(z, "nfft"))
    before = _1d(z, "integrated_spectrum_before_mask")
    after = _1d(z, "integrated_spectrum_after_mask")
    r.check(before.size == after.size == nfft, f"spectra shape == ({nfft},)")
    peak = float(before.max()) if before.size else 0.0
    r.check(np.all(after <= before + 1e-6 * (peak or 1.0)),
            "integrated_spectrum_after <= before everywhere (subset of frames)")

    in_band = bool(_scalar(z, "pilot_in_band"))
    n_valid = int(valid.sum())
    if in_band and n_valid > 0:
        r.check(peak > 0.0,
                f"in-band channel with {n_valid} valid frames has non-zero spectrum")
    elif n_valid == 0:
        r.check(peak == 0.0, "no valid frames -> zero spectrum")

    # time axis: real data is timed; flag if not (no LST possible)
    t0 = _1d(z, "unit_time0_ctime")
    dt = _1d(z, "unit_delta_time")
    n_timed = int(np.isfinite(t0).sum())
    if n_timed == 0:
        r.warning(f"no unit has a finite time0_ctime -- LST/time folding unavailable")
    else:
        r.check(n_timed == u, f"all {u} units carry a finite time0_ctime")
        # per-frame time monotonic within each unit
        fin = _1d(z, "frame_in_unit")
        mono = all(
            np.all(np.diff(fin[fui == ui]) > 0) for ui in range(u) if np.any(fui == ui))
        r.check(mono, "frame_in_unit strictly increasing within each unit")

    overflow = int(_scalar(z, "rational_overflow_count"))
    if overflow == 0:
        r.ok("rational_overflow_count == 0")
    elif overflow < max(1, n // 1000):
        r.warning(f"rational_overflow_count == {overflow} (small; {n} frames)")
    else:
        r.bad(f"rational_overflow_count == {overflow} of {n} frames (high)")

    masked = float(reject.sum() / n_valid) if n_valid else float("nan")
    span = (float(np.nanmax(t0)) - float(np.nanmin(t0))) if n_timed else float("nan")
    print(f"   summary: freq_id={int(_scalar(z,'freq_id'))} "
          f"chan={int(_scalar(z,'physical_channel'))} frames={n} units={u} "
          f"valid={n_valid} masked_frac={masked:.4f} t_span={span:.3f}s "
          f"overflow={overflow}")
    return {"freq_id": int(_scalar(z, "freq_id")), "n": n, "u": u,
            "before": before, "path": path}


def check_combined(run: Path, per: list, r: Report):
    print("\n--- combined products ---")
    det = run / "chime_detector_outputs.npz"
    r.check(det.exists(), "chime_detector_outputs.npz present")
    spec_path = run / "chime_integrated_spectra.npz"
    if not r.check(spec_path.exists(), "chime_integrated_spectra.npz present"):
        return
    s = np.load(spec_path)
    n_ch = _1d(s, "freq_id").size
    r.check(n_ch == len(per), f"integrated_spectra has all {len(per)} channels")
    sb = np.asarray(s["integrated_spectrum_before_mask"])
    nfft = int(_scalar(s, "nfft"))
    r.check(sb.shape == (n_ch, nfft), f"stacked spectra shape == ({n_ch},{nfft})")
    sr = float(_scalar(s, "sample_rate_hz"))
    r.check(np.isfinite(sr) and sr > 0,
            f"sample_rate_hz finite ({sr:.1f} Hz)"
            if np.isfinite(sr) else "sample_rate_hz finite (got NaN -- untimed inputs)")
    mf = _1d(s, "masked_fraction_by_channel")
    r.check(np.all((mf >= 0) & (mf <= 1) | np.isnan(mf)),
            "masked_fraction_by_channel in [0,1] or NaN")
    # stacked spectra match the authoritative per-pilot copies
    by_fid = {p["freq_id"]: p["before"] for p in per}
    fids = _1d(s, "freq_id")
    mism = [int(f) for i, f in enumerate(fids)
            if int(f) in by_fid and not np.array_equal(sb[i], by_fid[int(f)])]
    r.check(not mism, "stacked spectra match per-pilot products"
            + (f" (mismatch {mism})" if mism else ""))


def cmd_check(run_dir: str) -> int:
    run = Path(run_dir)
    r = Report()
    paths = _per_pilot_paths(run)
    print(f"checking {run}  ({len(paths)} per-pilot product(s))")
    if not paths:
        r.bad(f"no per-pilot products under {run/'_per_pilot'}")
        return 1
    per = [p for p in (check_per_pilot(pp, r) for pp in paths) if p]
    if per:
        check_combined(run, per, r)
    print(f"\n=== {'PASS' if not r.fail else 'FAIL'}: "
          f"{r.fail} failure(s), {r.warn} warning(s) ===")
    return 1 if r.fail else 0


def cmd_compare(dir_a: str, dir_b: str) -> int:
    a, b = Path(dir_a), Path(dir_b)
    r = Report()
    pa = {p.stem: p for p in _per_pilot_paths(a)}
    pb = {p.stem: p for p in _per_pilot_paths(b)}
    print(f"comparing clean={a}  vs  resumed={b}")
    if not r.check(set(pa) == set(pb),
                   f"same freq_ids ({sorted(pa)} vs {sorted(pb)})"):
        return 1
    tot_a = tot_b = 0
    for fid in sorted(pa, key=int):
        print(f"\n--- freq_id {fid} ---")
        za, zb = np.load(pa[fid]), np.load(pb[fid])
        na, nb = _1d(za, "frame_index").size, _1d(zb, "frame_index").size
        ua, ub = _1d(za, "unit_order").size, _1d(zb, "unit_order").size
        tot_a += na
        tot_b += nb
        # the headline: no double-count, no loss
        r.check(na == nb, f"n_frames equal (clean={na}, resumed={nb})")
        r.check(ua == ub, f"n_units equal (clean={ua}, resumed={ub})")
        for k in COMPARE_EXACT:
            if k in za.files and k in zb.files:
                xa, xb = _1d(za, k), _1d(zb, k)
                same = xa.shape == xb.shape and np.array_equal(xa, xb)
                (r.ok if same else r.bad)(f"{k} identical")
        # spectra: same GPU backend + same frame order -> ~bit-identical
        ba = _1d(za, "integrated_spectrum_before_mask")
        bb = _1d(zb, "integrated_spectrum_before_mask")
        peak = float(ba.max()) if ba.size else 1.0
        r.check(ba.shape == bb.shape and np.allclose(ba, bb, rtol=0, atol=1e-6 * peak),
                "integrated_spectrum_before survived restart (matches clean)")
        aa = _1d(za, "integrated_spectrum_after_mask")
        ab = _1d(zb, "integrated_spectrum_after_mask")
        r.check(aa.shape == ab.shape and np.allclose(aa, ab, rtol=0, atol=1e-6 * peak),
                "integrated_spectrum_after survived restart (matches clean)")
    r.check(tot_a == tot_b,
            f"TOTAL frames equal across all channels (clean={tot_a}, resumed={tot_b})")
    print(f"\n=== {'PASS' if not r.fail else 'FAIL'}: "
          f"{r.fail} failure(s), {r.warn} warning(s) ===")
    return 1 if r.fail else 0


def main(argv) -> int:
    if len(argv) >= 3 and argv[1] == "check":
        return cmd_check(argv[2])
    if len(argv) >= 4 and argv[1] == "compare":
        return cmd_compare(argv[2], argv[3])
    print(__doc__)
    print("usage:\n  check_scan.py check <run_dir>\n"
          "  check_scan.py compare <clean_dir> <resumed_dir>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
