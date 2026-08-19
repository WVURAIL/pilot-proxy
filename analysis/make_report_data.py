#!/usr/bin/env python3
"""Build report_data.json, the data behind the pilot-proxy trawl report page.

One pass over the per-pilot survey products (*.npz under $PP_PER_PILOT or
--products) produces every per-channel field of the published report:
level/occupancy statistics, monthly aggregates, month-of-year and
hour-of-day profiles, the averaged fine spectrum, the off-nominal carrier
envelope measurement, level histograms, and the max-pooled integrated
spectra. A second stage runs the residual threshold sweeps for the eight
featured channels through the released ``baonoise`` package
(bao-noise-tolerance) --- the same cross-repository dependency
``tools/make_dissertation_tables.py`` and ``plot_channel_histograms.py``
already carry --- and merges them in, also writing threshold_sweeps.json
(kept as its own output: the bao two-walls figure regeneration reads it).

Render the page afterwards with ``render_artifacts.py``.

    python3 analysis/make_report_data.py --products DIR --out DIR
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import zoneinfo
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  -- puts <repo>/src on sys.path
from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO, ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB)
from baonoise import residual as res

import _products as P

TZ = zoneinfo.ZoneInfo("America/Vancouver")   # DRAO local time
HOT_THRESH = 5.0                              # F above which a unit is "hot"
# survey span 2018-12 .. 2026-07 (extend M1 when a newer snapshot adds months)
M0 = 2018 * 12 + 11
M1 = 2026 * 12 + 6
NMONTHS = M1 - M0 + 1

# fine-comb geometry for the envelope measurement
COARSE_HZ = 390625.0 / 128
FINE_HZ = COARSE_HZ / 256
# channels strong enough for a parabolic carrier-offset fit
ENV_STRONG = {798, 721, 690, 598, 583, 521}

# level-histogram edges (dB): 129 edges -> 128 bins, values clipped in
HIST_EDGES = np.arange(-4, 28.25, 0.25)
PSD_POOL = 16                                 # 16384 -> 1024 points, max-pooled

# integrated-spectrum demo channel (fid 598 = ch30, the Penticton carrier)
SPEC_DEMO_FID = 598

# threshold-sweep cases: (fid, label, off_through). ch35 excludes its
# pre-sign-on epoch (sign-on Nov 2021); the second pass substitutes the
# floor-provenance implied sigma where the measured floor refuses a sweep.
SWEEP_CASES = [
    (521, "ch35", "2021-08"),
    (537, "ch34", None),
    (506, "ch36", None),
    (721, "ch22", None),
    (583, "ch31", None),
    (798, "ch17", None),
]
FLOOR_CASES = [(506, "ch36"), (721, "ch22"), (798, "ch17"),
               (598, "ch30"), (690, "ch24")]
SWEEP_META = {name: (fid, off) for fid, name, off in SWEEP_CASES}
SWEEP_META.update({name: (fid, None) for fid, name in FLOOR_CASES
                   if name not in SWEEP_META})
DAY_CAP = 86164.0


def month_key(ts):
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    return dt.year * 12 + (dt.month - 1)


def pool_db(a):
    """Max-pooled dB-relative-to-median rendering of an integrated spectrum."""
    if not np.any(a > 0):
        return None
    a = np.fft.fftshift(a)
    m = a.reshape(-1, PSD_POOL).max(axis=1)
    m = np.maximum(m, np.max(m) * 1e-12)
    db = 10 * np.log10(m / np.median(m))
    return np.round(db, 2).tolist()


def channel_record(z):
    """Every per-channel report field from one product archive."""
    fid = int(z["freq_id"][0])
    mu0 = float(z["mu0"][0])
    fstat = z[ARCHIVED_COARSE_POWER_RATIO][:, 0]
    fui = z["frame_unit_index"]
    ev_id = z["unit_event_id"]
    t0 = z["unit_time0_ctime"]
    nunits = len(ev_id)

    sums = np.zeros(nunits); cnts = np.zeros(nunits, dtype=np.int64)
    np.add.at(sums, fui, fstat); np.add.at(cnts, fui, 1)
    maxs = np.full(nunits, -np.inf); np.maximum.at(maxs, fui, fstat)
    keep = cnts > 0
    mean_f = sums[keep] / cnts[keep]
    max_f = maxs[keep]
    t = t0[keep]; ev = ev_id[keep]
    lvl_db = 10 * np.log10(np.maximum(mean_f, 1e-6) / mu0)

    # monthly aggregates
    mk = np.array([month_key(x) for x in t]) - M0
    med = [None] * NMONTHS; p90 = [None] * NMONTHS
    nn = [0] * NMONTHS; hotfrac = [None] * NMONTHS
    for i in range(NMONTHS):
        sel = mk == i
        n = int(sel.sum()); nn[i] = n
        if n:
            med[i] = round(float(np.median(lvl_db[sel])), 2)
            p90[i] = round(float(np.percentile(lvl_db[sel], 90)), 2)
            hotfrac[i] = round(float((max_f[sel] > HOT_THRESH).mean()), 3)

    # month-of-year & hour-of-day profiles (local time)
    dts = [datetime.datetime.fromtimestamp(x, TZ) for x in t]
    months = np.array([x.month for x in dts])
    hours = np.array([x.hour for x in dts])
    hot = max_f > HOT_THRESH
    moy = [round(float(hot[months == m].mean()), 3)
           if (months == m).sum() >= 20 else None for m in range(1, 13)]
    hod = [round(float(hot[hours == h].mean()), 3)
           if (hours == h).sum() >= 20 else None for h in range(24)]
    moy_med = [round(float(np.median(lvl_db[months == m])), 3)
               if (months == m).sum() >= 20 else None for m in range(1, 13)]
    hod_med = [round(float(np.median(lvl_db[hours == h])), 3)
               if (hours == h).sum() >= 20 else None for h in range(24)]

    # averaged fine spectrum: natural order for the envelope fit,
    # fftshifted dB for display
    fine = np.nanmean(z[ARCHIVED_FINE_POWER_RATIO], axis=0)
    fine_db = (10 * np.log10(np.maximum(np.fft.fftshift(fine), 1e-3))).round(2)

    # off-nominal carrier envelope: nominal pilot offset within the comb,
    # measured position by parabolic interpolation where strong
    pilot = float(z["pilot_frequency_hz"][0])
    center = float(z["chime_frequency_hz"][0])
    eff = -(pilot - center)          # receiver frame: sense inverted vs RF
    k0 = round(eff / COARSE_HZ)
    nom = eff - k0 * COARSE_HZ
    if fid in ENV_STRONG:
        obs = int(np.argmax(fine))
        y0, y1, y2 = (np.log(fine[(obs - 1) % 256]), np.log(fine[obs]),
                      np.log(fine[(obs + 1) % 256]))
        delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        cent = ((obs + 128) % 256) - 128
        f_meas = (cent + float(delta)) * FINE_HZ
        env_meas = round(float(f_meas), 1)
        env_station = round(float(f_meas - nom), 1)
    else:
        env_meas = env_station = None

    valid = z["valid"][:, 0].astype(bool)
    lvl_all = z[ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB][:, 0]
    lva = lvl_all[np.isfinite(lvl_all) & valid]

    # per-frame level histogram, clipped into the fixed edges
    below = int((lva < HIST_EDGES[0]).sum())
    above = int((lva > HIST_EDGES[-1]).sum())
    h, _ = np.histogram(np.clip(lva, HIST_EDGES[0], HIST_EDGES[-1]),
                        bins=HIST_EDGES)

    rec = dict(
        freq_id=fid,
        phys=int(z["physical_channel"][0]),
        pilot_mhz=round(pilot / 1e6, 6),
        n_frames=int(len(fstat)), n_units=int(nunits),
        n_events=int(len(np.unique(ev_id))),
        t_first=datetime.datetime.fromtimestamp(
            float(np.min(t0)), datetime.timezone.utc).strftime("%Y-%m-%d"),
        t_last=datetime.datetime.fromtimestamp(
            float(np.max(t0)), datetime.timezone.utc).strftime("%Y-%m-%d"),
        mu0=round(mu0, 5),
        lvl_db_med=round(float(np.median(lva)), 2),
        lvl_db_p5=round(float(np.percentile(lva, 5)), 2),
        lvl_db_p95=round(float(np.percentile(lva, 95)), 2),
        lvl_db_p99=round(float(np.percentile(lva, 99)), 2),
        lvl_db_max=round(float(np.max(lva)), 2),
        frac_excess_pos=round(float(((fstat > mu0) & valid).mean()), 4),
        hot_frac=round(float(hot.mean()), 4),
        det_bins_per_frame=round(float(z["fine_detected_count"][valid, 0].mean()), 1),
        monthly_med=med, monthly_p90=p90, monthly_n=nn, monthly_hot=hotfrac,
        moy_hot=moy, hod_hot=hod,
        fine_db=fine_db.tolist(),
        moy_med=moy_med, hod_med=hod_med,
        env_nom_hz=round(float(nom), 1),
        env_meas_hz=env_meas, env_station_hz=env_station,
        hist=h.astype(int).tolist(), hist_clipped=below + above,
        psd_before=pool_db(z["integrated_spectrum_before_mask"][:]),
        psd_after=pool_db(z["integrated_spectrum_after_mask"][:]),
        pilot_rx_khz=round(eff / 1000, 3),
    )
    return rec, ev, np.unique(ev_id)


def spec_demo(path):
    """Before/after-mask integrated spectrum zoomed on the demo channel's peak."""
    with np.load(path, allow_pickle=False) as z:
        sb = z["integrated_spectrum_before_mask"][:]
        sa = z["integrated_spectrum_after_mask"][:]
    pk = int(np.argmax(sb))
    half = 120
    sl = slice(pk - half, pk + half + 1)
    binhz = 390625.0 / 16384
    ref = float(np.median(sb))
    return dict(
        peak_bin=pk, bin_hz=round(binhz, 3),
        offset_hz=[round((i - pk) * binhz, 1)
                   for i in range(pk - half, pk + half + 1)],
        before_db=(10 * np.log10(np.maximum(sb[sl], 1e-30) / ref)).round(2).tolist(),
        after_db=(10 * np.log10(np.maximum(sa[sl], 1e-30) / ref)).round(2).tolist(),
        before_med_rel_db=0.0,
        after_total_ratio_db=round(float(10 * np.log10(np.sum(sa) / np.sum(sb))), 2),
    )


def run_sweeps(by_fid):
    """Two-pass residual threshold sweeps for the featured channels.

    Pass 1 sweeps against each channel's measured floor; channels whose
    floor refuses a sweep drop out. Pass 2 re-runs the floor cases with the
    floor-provenance implied sigma substituted, replacing or adding entries
    (order of survivors is preserved, so the report's panel order is stable).
    """
    results = {}
    for fid, name, off in SWEEP_CASES:
        try:
            sweep = res.threshold_sweep(by_fid[fid], off_through=off)
            results[name] = (dict(sweep=sweep,
                                  best=res.best_operating_point(sweep))
                             if sweep else None)
        except Exception as e:
            print(f"  {name}: measured-floor sweep failed: {e}")
            results[name] = None
    results = {k: v for k, v in results.items() if v}

    for fid, name in FLOOR_CASES:
        try:
            fp = res.floor_provenance(by_fid[fid])
            floor_db = float(fp.sigma_implied_db)
            sweep = res.threshold_sweep(by_fid[fid], floor_db=floor_db)
            if sweep:
                results[name] = dict(sweep=sweep,
                                     best=res.best_operating_point(sweep),
                                     floor_db=floor_db, floor_substituted=True)
        except Exception as e:
            print(f"  {name}: floor-substituted sweep failed: {e}")
    return results


def merged_sweeps(results, by_fid):
    """The report's compact sweep block, with the coherence time attached."""
    out = {}
    for name, data in results.items():
        fid, off = SWEEP_META[name]
        corr = res.correlation_time(by_fid[fid], off_through=off)
        tau = float(corr.tau_for_budget)
        measured = tau < DAY_CAP * 0.99
        rows = [dict(eta=row["eta"], f=round(row["f"], 4),
                     net=round(row["net"], 4),
                     r_masked=float(f"{row['r_masked']:.4g}"))
                for row in data["sweep"]]
        out[name] = dict(
            fid=fid, rows=rows,
            best=dict(eta=data["best"]["eta"], f=round(data["best"]["f"], 4),
                      net=round(data["best"]["net"], 3)),
            r_unmasked=float(f"{data['sweep'][0]['r_unmasked']:.4g}"),
            tau_s=round(tau), tau_measured=measured,
            floor_substituted=bool(data.get("floor_substituted", False)),
            off_through=off,
        )
        print(name, "tau", round(tau),
              "measured" if measured else "BOUND(day cap)",
              "| best eta", out[name]["best"]["eta"],
              "f", out[name]["best"]["f"], "net", out[name]["best"]["net"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path, default=None,
                    help="directory of per-pilot survey products "
                         "(default: $PP_PER_PILOT)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    per_pilot = P.PER_PILOT if args.products is None else args.products
    files = sorted(glob.glob(str(per_pilot / "*.npz")),
                   key=lambda p: int(Path(p).stem))
    if not files:
        raise SystemExit(f"no per-pilot products (*.npz) under {per_pilot}; "
                         "set PP_PER_PILOT or pass --products")
    by_fid = {int(Path(p).stem): p for p in files}

    month_labels = [f"{(M0+i)//12}-{(M0+i)%12+1:02d}" for i in range(NMONTHS)]
    report = {"month_labels": month_labels, "channels": [],
              "events_union": None}
    usable_events = {}
    probed_events = set()
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            rec, ev, ev_all = channel_record(z)
        report["channels"].append(rec)
        for e in ev:
            usable_events[int(e)] = usable_events.get(int(e), 0) + 1
        probed_events.update(int(e) for e in ev_all)
        print("done", rec["freq_id"])

    # union / coverage: probed union counts every unit the archive holds,
    # the usable union only units that kept at least one frame
    cov = np.array(list(usable_events.values()))
    report["events_union"] = len(probed_events)
    report["coverage_hist"] = np.bincount(cov, minlength=21).tolist()
    report["total_units"] = int(sum(c["n_units"] for c in report["channels"]))
    report["total_frames"] = int(sum(c["n_frames"] for c in report["channels"]))
    report["spec598"] = spec_demo(by_fid[SPEC_DEMO_FID])
    report["events_union_usable"] = len(usable_events)

    results = run_sweeps(by_fid)
    with open(args.out / "threshold_sweeps.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    report["sweeps"] = merged_sweeps(results, by_fid)
    report["hist_edges"] = [round(float(x), 2) for x in HIST_EDGES.tolist()]

    out_path = args.out / "report_data.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    print("coverage hist (n channels -> events):", report["coverage_hist"])
    print("events union:", report["events_union"],
          "usable:", report["events_union_usable"])
    print("wrote", out_path, os.path.getsize(out_path), "bytes")
    return out_path


if __name__ == "__main__":
    main()
