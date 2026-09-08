#!/usr/bin/env python3
# coding=utf-8
"""Spatial-scaling Monte Carlo: null width and deflection against M.

Chapter 6 (``sec:detection:synthetic``, panel (a) of the four-panel
synthetic-verification figure) asks for the measured null width --- raw and
robust core --- and the deflection of the fine and coarse statistics against
the stream count M, on log axes, against the ``1/sqrt(beta M)`` and
``sqrt(M)`` predictions, with the replicated-capture case drawn as a
correlation stress case and not as a universal bound.

The populations are produced by ``tools/measure_fine_gain.py`` --- the same
row-sum Monte Carlo, the same integer marginals, the same deployed statistics,
the same shard format --- run once per stream count with ``--streams M``.
This module adds only the two things that tool does not carry:

  * the *replicated capture*: one stream's row sums tiled across all M
    inputs, the fully correlated stress case.  Both deployed statistics are
    ratios of sums over streams, so tiling multiplies numerator and
    denominator by the same M and the statistic is *identically* the M=1
    statistic; ``--stage verify-rep`` checks that on real trials rather than
    asserting it, and the measured flat curve is what the figure draws.
  * the report: widths, deflections, log-log fits and the figure.

Widths, per statistic and per M, on the H0 population:

  raw          sample standard deviation.  For the fine statistic this is a
               sample of a distribution whose second moment does not exist
               below M = 2: the designated-set ratio divides by a chi-square
               with 4M degrees of freedom, and E[X^-2] diverges for 4M <= 4.
               The raw width is reported at every M anyway, per seed, so the
               instability is visible rather than smoothed.
  robust core  normal-consistent inter-quartile scale, 0.7413 (q75 - q25).
  left scale   median - q15.87, the one-sided bulk scale the archive
               calibration uses, carried as a third column.

Deflection is the H1 mean excess over the H0 mean divided by the H0 width.
The excess is exactly linear in the injected per-row-sum SNR for the coarse
statistic and linear to within the measured residual for the fine one, so it
is measured with a strong probe (precise) at two probe levels (the linearity
check) and reported as excess per unit SNR; the deflection printed at a
reference SNR is that coefficient times the reference SNR, and is labelled as
such.

Stages::

  --stage verify-rep                       (tiling identity, on real trials)
  --stage null-rep  --streams M --trials N --seed K
  --stage defl-rep  --streams M --snr-db X --trials N --seed K
  --stage report    --root DIR
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

_TOOLS = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "measure_fine_gain", _TOOLS / "measure_fine_gain.py")
mfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mfg)

WINDOWS = mfg.WINDOWS
SIGMA = mfg.SIGMA
ANCHOR = mfg.ANCHOR
HALF_BIN = mfg.HALF_BIN
PFA_LIST = mfg.PFA_LIST

# 1/sqrt(beta M) reference values.  Coarse: F = 2A/(B+C) with A,B,C
# independent chi-square of 2ML degrees of freedom, so Var(F) ~ 3/(2ML) and
# beta = L/1.5.  Fine, single bin: the same algebra with 2M degrees of
# freedom, beta = 1/1.5.  Both are the analytic reference the fit is judged
# against, not a fitted curve.
L_WINDOWS = float(WINDOWS)
BETA_COARSE_PRED = L_WINDOWS / 1.5
BETA_FINE_PRED = 1.0 / 1.5


def amp_for_snr(snr_db: float) -> float:
    return float(np.sqrt(2.0 * SIGMA * SIGMA * 10.0 ** (snr_db / 10.0)))


def gen_rows_replicated(rng, batch, streams, amp=0.0, b0=0.0):
    """One captured stream tiled across ``streams`` inputs (fully correlated)."""
    one = mfg.gen_rows(rng, batch, 1, amp=amp, b0=b0)
    return np.repeat(one, streams, axis=2)


def verify_rep(seed=0, trials=3, streams=64):
    """The tiling identity, checked on real trials at the product's precision."""
    rng = np.random.default_rng(seed)
    one = mfg.gen_rows(rng, trials, 1, amp=amp_for_snr(-6.0), b0=ANCHOR)
    c1, f1 = mfg.reduce_batch(one)
    tiled = np.repeat(one, streams, axis=2)
    cm, fm = mfg.reduce_batch(tiled)
    dc = np.abs(cm - c1).max()
    df = np.abs(fm - f1).max()
    rel_c = float(dc / max(np.abs(c1).max(), 1e-300))
    rel_f = float(df / max(np.abs(f1).max(), 1e-300))
    assert rel_c < 1e-12 and rel_f < 1e-12, (
        f"tiling identity violated: rel coarse {rel_c:g}, rel fine {rel_f:g}")
    print(f"verify-rep: {trials} trials, M=1 vs M={streams} replicated -- "
          f"coarse max|d|={dc:.3e} (rel {rel_c:.2e}), "
          f"fine max|d|={df:.3e} (rel {rel_f:.2e})")
    return 0


def run_rep(out, kind, trials, seed, snr_db=None, streams=1, batch=8, b0=ANCHOR):
    rng = np.random.default_rng(seed)
    amp = 0.0 if kind == "null" else amp_for_snr(snr_db)
    co, fi, done = [], [], 0
    while done < trials:
        b = min(batch, trials - done)
        zi = gen_rows_replicated(rng, b, streams, amp=amp, b0=b0)
        c, f = mfg.reduce_batch(zi)
        co.append(c)
        fi.append(f)
        done += b
    co = np.concatenate(co)
    fi = np.concatenate(fi)
    name = (f"h0_s{seed}.npz" if kind == "null"
            else f"h1_{snr_db:+06.2f}dB_s{seed}.npz")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, name), coarse=co, fine=fi,
                        snr_db=(np.nan if snr_db is None else snr_db),
                        b0=b0, streams=streams, sigma=SIGMA, seed=seed,
                        replicated=True)
    print(f"wrote {out}/{name}: {trials} trials "
          f"(coarse med {np.median(co):.5f}, fine med {np.median(fi):.4f})")
    return 0


# ----------------------------------------------------------------- analysis

def _widths(x):
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    q1587 = float(np.quantile(x, 0.15865))
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(q50),
        "raw": float(x.std(ddof=1)),
        "robust_core": float(0.7413 * (q75 - q25)),
        "left_scale": float(q50 - q1587),
    }


def _load(dirpath, pattern):
    per_shard, pooled = [], []
    for p in sorted(glob.glob(os.path.join(dirpath, pattern))):
        z = np.load(p)
        per_shard.append((os.path.basename(p), z["coarse"], z["fine"],
                          float(z["snr_db"]), int(z["streams"])))
        pooled.append(p)
    return per_shard


def _null_row(dirpath, m):
    shards = _load(dirpath, "h0_s*.npz")
    if not shards:
        return None
    c = np.concatenate([s[1] for s in shards])
    f = np.concatenate([s[2] for s in shards])
    row = {"streams": m, "coarse": _widths(c), "fine": _widths(f),
           "shards": len(shards)}
    # per-shard raw widths: the stability check the heavy tail needs
    row["coarse_raw_by_shard"] = [float(s[1].std(ddof=1)) for s in shards]
    row["fine_raw_by_shard"] = [float(s[2].std(ddof=1)) for s in shards]
    row["coarse_core_by_shard"] = [
        float(0.7413 * np.subtract(*np.quantile(s[1], [0.75, 0.25])))
        for s in shards]
    row["fine_core_by_shard"] = [
        float(0.7413 * np.subtract(*np.quantile(s[2], [0.75, 0.25])))
        for s in shards]
    return row


def _defl_rows(dirpath, null_row):
    out = []
    shards = _load(dirpath, "h1_*dB_s*.npz")
    by_snr = {}
    for name, c, f, s, m in shards:
        by_snr.setdefault(s, [[], []])
        by_snr[s][0].append(c)
        by_snr[s][1].append(f)
    for s in sorted(by_snr):
        c = np.concatenate(by_snr[s][0])
        f = np.concatenate(by_snr[s][1])
        snr = 10.0 ** (s / 10.0)
        rec = {"snr_db": s, "n": int(c.size), "snr_linear": snr}
        for key, arr in (("coarse", c), ("fine", f)):
            exc = float(arr.mean() - null_row[key]["mean"])
            se = float(arr.std(ddof=1) / np.sqrt(arr.size))
            rec[key] = {
                "h1_mean": float(arr.mean()),
                "excess": exc,
                "excess_se": se,
                "excess_per_snr": exc / snr,
                "deflection_raw": exc / null_row[key]["raw"],
                "deflection_core": exc / null_row[key]["robust_core"],
            }
        out.append(rec)
    return out


def _loglog_fit(ms, ys):
    ms = np.asarray(ms, float)
    ys = np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log(ms[ok]), np.log(ys[ok]), 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "n_points": int(ok.sum())}


def _crossing(xs, ys, target=0.5):
    for i in range(1, len(xs)):
        if ys[i - 1] < target <= ys[i]:
            a, b = ys[i - 1], ys[i]
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (target - a) / (b - a)
    return float("nan")


BOOTSTRAP_REPLICATES = 400


def _gain_bootstrap(pops, thr_pair, seed, replicates):
    """Trial bootstrap of the three crossings and the two gains.

    Whole trials are resampled with replacement inside each SNR point, so
    every replicate keeps the sweep's own trial allocation, and the coarse
    and fine statistics of a resampled trial are carried together (they are
    the same trial).  The thresholds are held at their full-sample values,
    which is what a fixed-false-alarm threshold means.
    """
    thr_c, thr_f = thr_pair
    rng = np.random.default_rng(seed)
    keys_c = sorted(pops.get("centered", {}))
    keys_h = sorted(pops.get("half_bin", {}))
    if not keys_c:
        return None
    out = {"coarse_snr50_db": [], "fine_snr50_db_centered": [],
           "fine_snr50_db_half_bin": [], "gain_db_centered": [],
           "gain_db_half_bin": [], "scalloping_loss_db": []}
    for _ in range(replicates):
        pc, pf = [], []
        for k in keys_c:
            c, f = pops["centered"][k]
            idx = rng.integers(0, c.size, c.size)
            pc.append(float((c[idx] > thr_c).mean()))
            pf.append(float((f[idx] > thr_f).mean()))
        s_c = _crossing(keys_c, pc)
        s_f = _crossing(keys_c, pf)
        out["coarse_snr50_db"].append(s_c)
        out["fine_snr50_db_centered"].append(s_f)
        out["gain_db_centered"].append(s_c - s_f)
        if keys_h:
            ph = []
            for k in keys_h:
                c, f = pops["half_bin"][k]
                idx = rng.integers(0, f.size, f.size)
                ph.append(float((f[idx] > thr_f).mean()))
            s_h = _crossing(keys_h, ph)
            out["fine_snr50_db_half_bin"].append(s_h)
            out["gain_db_half_bin"].append(s_c - s_h)
            out["scalloping_loss_db"].append(s_h - s_f)
    res = {}
    for k, v in out.items():
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        if v.size:
            res[k] = {"n_replicates": int(v.size),
                      "ci95": [float(np.quantile(v, 0.025)),
                               float(np.quantile(v, 0.975))],
                      "sd": float(v.std(ddof=1))}
    return res


def gain_panel(gain_dir):
    """Panel (c): fine-axis gain against 5 log10 L, bin-centred and off-centre."""
    shards = _load(gain_dir, "h0_s*.npz")
    if not shards:
        return None
    c0 = np.concatenate([s[1] for s in shards])
    f0 = np.concatenate([s[2] for s in shards])
    thr = {pfa: (float(np.quantile(c0, 1 - pfa)), float(np.quantile(f0, 1 - pfa)))
           for pfa in PFA_LIST}
    curves = {}
    for tag, pat in (("centered", "h1_*dB_s*.npz"), ("half_bin", "h1_half_*dB_s*.npz")):
        pts = {}
        for name, c, f, s, m in _load(gain_dir, pat):
            if tag == "centered" and name.startswith("h1_half"):
                continue
            pts.setdefault(s, [[], []])
            pts[s][0].append(c)
            pts[s][1].append(f)
        rows = []
        for s in sorted(pts):
            c = np.concatenate(pts[s][0])
            f = np.concatenate(pts[s][1])
            row = {"snr_db": s, "n": int(c.size)}
            for pfa in PFA_LIST:
                row[f"pd_coarse_{pfa:g}"] = float((c > thr[pfa][0]).mean())
                row[f"pd_fine_{pfa:g}"] = float((f > thr[pfa][1]).mean())
            rows.append(row)
        if rows:
            curves[tag] = rows
    # per-trial populations, kept for the bootstrap
    pops = {}
    for tag, pat in (("centered", "h1_*dB_s*.npz"), ("half_bin", "h1_half_*dB_s*.npz")):
        d = {}
        for name, c, f, s_, m in _load(gain_dir, pat):
            if tag == "centered" and name.startswith("h1_half"):
                continue
            d.setdefault(s_, [[], []])
            d[s_][0].append(c)
            d[s_][1].append(f)
        pops[tag] = {k: (np.concatenate(v[0]), np.concatenate(v[1]))
                     for k, v in d.items()}

    gains = {}
    for pfa in PFA_LIST:
        rec = {}
        base = curves.get("centered")
        if base:
            xs = [r["snr_db"] for r in base]
            s_c = _crossing(xs, [r[f"pd_coarse_{pfa:g}"] for r in base])
            s_f = _crossing(xs, [r[f"pd_fine_{pfa:g}"] for r in base])
            rec["coarse_snr50_db"] = s_c
            rec["fine_snr50_db_centered"] = s_f
            rec["gain_db_centered"] = s_c - s_f
        hb = curves.get("half_bin")
        if hb and base:
            xs = [r["snr_db"] for r in hb]
            s_h = _crossing(xs, [r[f"pd_fine_{pfa:g}"] for r in hb])
            rec["fine_snr50_db_half_bin"] = s_h
            rec["gain_db_half_bin"] = rec["coarse_snr50_db"] - s_h
            rec["scalloping_loss_db"] = s_h - rec["fine_snr50_db_centered"]
        rec["bootstrap"] = _gain_bootstrap(pops, thr[pfa], seed=20260907,
                                           replicates=BOOTSTRAP_REPLICATES)
        gains[f"{pfa:g}"] = rec
    return {"h0_trials": int(c0.size),
            "thresholds": {f"{k:g}": {"coarse": v[0], "fine": v[1]}
                           for k, v in thr.items()},
            "curves": curves, "gains": gains,
            "benchmark_db": float(5.0 * np.log10(L_WINDOWS))}


def _write_csv(res, path):
    import csv as _csv
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["population", "streams", "null_trials", "statistic",
                    "null_median", "null_mean", "width_raw_sd",
                    "width_robust_core", "width_left_scale",
                    "width_pred_1_over_sqrt_beta_M", "beta_implied_core",
                    "probe_snr_db", "h1_trials", "excess_per_snr",
                    "deflection_per_snr_core"])
        for kind, rows in (("independent", res["independent"]),
                           ("replicated", res["replicated"])):
            for r in rows:
                m = r["streams"]
                for stat, beta in (("coarse", BETA_COARSE_PRED),
                                   ("fine", BETA_FINE_PRED)):
                    pred = 1.0 / np.sqrt(beta * m)
                    defl = r.get("deflection") or []
                    if not defl:
                        defl = [None]
                    for d in defl:
                        w.writerow([
                            kind, m, r[stat]["n"], stat,
                            f"{r[stat]['median']:.8g}",
                            f"{r[stat]['mean']:.8g}",
                            f"{r[stat]['raw']:.8g}",
                            f"{r[stat]['robust_core']:.8g}",
                            f"{r[stat]['left_scale']:.8g}",
                            f"{pred:.8g}",
                            f"{r[stat].get('beta_implied_robust_core', float('nan')):.8g}",
                            "" if d is None else f"{d['snr_db']:.8g}",
                            "" if d is None else d["n"],
                            "" if d is None else f"{d[stat]['excess_per_snr']:.8g}",
                            "" if d is None else
                            f"{d[stat]['deflection_core'] / d['snr_linear']:.8g}",
                        ])
    print(f"wrote {path}")


def report(root, out_json, make_figure=True):
    root = pathlib.Path(root)
    res = {"geometry": {"windows_per_stream": WINDOWS, "fine_bins": mfg.BINS,
                        "sigma": SIGMA, "anchor_fine_bin": ANCHOR,
                        "designated_half_width": 2},
           "beta_prediction": {"coarse": BETA_COARSE_PRED,
                               "fine_single_bin": BETA_FINE_PRED},
           "independent": [], "replicated": []}
    for kind, npat, dpat in (("independent", "null_m*", "defl_m*"),
                             ("replicated", "rep_null_m*", "rep_defl_m*")):
        rows = []
        for d in sorted(root.glob(npat), key=lambda p: int(p.name.split("_m")[-1])):
            m = int(d.name.split("_m")[-1])
            nr = _null_row(str(d), m)
            if nr is None:
                continue
            dd = root / dpat.replace("*", str(m))
            nr["deflection"] = _defl_rows(str(dd), nr) if dd.is_dir() else []
            rows.append(nr)
        res[kind] = rows
    for kind in ("independent", "replicated"):
        rows = res[kind]
        if not rows:
            continue
        ms = [r["streams"] for r in rows]
        fits = {}
        for stat in ("coarse", "fine"):
            for w in ("raw", "robust_core"):
                fits[f"{stat}_{w}_all"] = _loglog_fit(
                    ms, [r[stat][w] for r in rows])
                big = [(m, r) for m, r in zip(ms, rows) if m >= 64]
                fits[f"{stat}_{w}_m_ge_64"] = _loglog_fit(
                    [m for m, _ in big], [r[stat][w] for _, r in big])
        res[f"{kind}_fits"] = fits
        # implied beta from width = 1/sqrt(beta M)
        for r in rows:
            for stat in ("coarse", "fine"):
                for w in ("raw", "robust_core"):
                    ww = r[stat][w]
                    r[stat][f"beta_implied_{w}"] = (
                        float(1.0 / (r["streams"] * ww * ww)) if ww > 0 else None)
    gd = root / "null_m2048"
    g = gain_panel(str(gd)) if gd.is_dir() else None
    if g:
        res["gain_panel"] = g
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"wrote {out_json}")
    _write_csv(res, str(pathlib.Path(out_json).with_name("m_scaling_widths.csv")))
    _print_summary(res)
    if make_figure:
        make_fig(res, str(pathlib.Path(out_json).with_name("m_scaling.png")))
    return 0


def _print_summary(res):
    print("\n=== null width vs M (independent streams) ===")
    print(f"{'M':>6} {'shards':>6} | {'coarse raw':>11} {'coarse core':>11} "
          f"{'pred 1/sqrt(bM)':>15} | {'fine raw':>11} {'fine core':>11} "
          f"{'pred 1/sqrt(bM)':>15}")
    for r in res["independent"]:
        m = r["streams"]
        pc = 1.0 / np.sqrt(BETA_COARSE_PRED * m)
        pf = 1.0 / np.sqrt(BETA_FINE_PRED * m)
        print(f"{m:>6} {r['shards']:>6} | {r['coarse']['raw']:>11.5f} "
              f"{r['coarse']['robust_core']:>11.5f} {pc:>15.5f} | "
              f"{r['fine']['raw']:>11.4f} {r['fine']['robust_core']:>11.4f} "
              f"{pf:>15.4f}")
    for kind in ("independent", "replicated"):
        fits = res.get(f"{kind}_fits")
        if not fits:
            continue
        print(f"\nlog-log slopes ({kind}; -0.5 is the 1/sqrt(M) law):")
        for k in sorted(fits):
            v = fits[k]
            if v:
                print(f"  {k:<28} slope {v['slope']:+.3f}  (n={v['n_points']})")
    print("\n=== deflection per unit per-row-sum SNR ===")
    print(f"{'M':>6} | {'probe dB':>8} {'coarse exc/snr':>14} "
          f"{'coarse d/snr (core)':>20} {'fine exc/snr':>13} "
          f"{'fine d/snr (core)':>18}")
    for r in res["independent"]:
        for d in r["deflection"]:
            print(f"{r['streams']:>6} | {d['snr_db']:>8.1f} "
                  f"{d['coarse']['excess_per_snr']:>14.4f} "
                  f"{d['coarse']['deflection_core'] / d['snr_linear']:>20.3f} "
                  f"{d['fine']['excess_per_snr']:>13.3f} "
                  f"{d['fine']['deflection_core'] / d['snr_linear']:>18.3f}")
    if res.get("replicated"):
        print("\n=== replicated capture (correlation stress case) ===")
        for r in res["replicated"]:
            print(f"{r['streams']:>6} | coarse raw {r['coarse']['raw']:.5f} "
                  f"core {r['coarse']['robust_core']:.5f} | fine raw "
                  f"{r['fine']['raw']:.4f} core {r['fine']['robust_core']:.4f}")
    g = res.get("gain_panel")
    if g:
        print(f"\n=== fine-axis gain at M=2048 ({g['h0_trials']} H0 trials; "
              f"benchmark 5log10(L) = {g['benchmark_db']:.2f} dB) ===")
        for pfa, rec in sorted(g["gains"].items()):
            if not rec:
                continue
            print(f"  Pfa={pfa}: coarse {rec.get('coarse_snr50_db', float('nan')):+.2f} dB, "
                  f"fine centred {rec.get('fine_snr50_db_centered', float('nan')):+.2f} dB "
                  f"-> gain {rec.get('gain_db_centered', float('nan')):.2f} dB; "
                  f"fine half-bin {rec.get('fine_snr50_db_half_bin', float('nan')):+.2f} dB "
                  f"-> gain {rec.get('gain_db_half_bin', float('nan')):.2f} dB "
                  f"(scalloping {rec.get('scalloping_loss_db', float('nan')):+.2f} dB)")
            b = rec.get("bootstrap") or {}
            for k in ("gain_db_centered", "gain_db_half_bin", "scalloping_loss_db"):
                if k in b:
                    lo, hi = b[k]["ci95"]
                    print(f"      {k:<22} 95% CI [{lo:+.2f}, {hi:+.2f}] dB "
                          f"(sd {b[k]['sd']:.2f}, {b[k]['n_replicates']} replicates)")


def make_fig(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
    BLUE, RED, AMBER = "#2a78d6", "#c2453a", "#c98a1b"
    rows = res["independent"]
    if not rows:
        return
    ms = np.array([r["streams"] for r in rows], float)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=200, facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_xscale("log", base=2)
        ax.tick_params(colors=INK2, labelsize=8)

    ax = axes[0]
    for stat, col in (("coarse", INK2), ("fine", BLUE)):
        ax.plot(ms, [r[stat]["raw"] for r in rows], "o--", color=col, ms=4,
                lw=1.4, label=f"{stat} raw (sd)")
        ax.plot(ms, [r[stat]["robust_core"] for r in rows], "s-", color=col,
                ms=4, lw=2.0, label=f"{stat} robust core")
    ax.plot(ms, 1.0 / np.sqrt(BETA_COARSE_PRED * ms), ":", color=RED, lw=1.6,
            label=r"$1/\sqrt{\beta M}$, $\beta=L/1.5$")
    ax.plot(ms, 1.0 / np.sqrt(BETA_FINE_PRED * ms), ":", color=AMBER, lw=1.6,
            label=r"$1/\sqrt{\beta M}$, $\beta=1/1.5$")
    if res.get("replicated"):
        rm = np.array([r["streams"] for r in res["replicated"]], float)
        ax.plot(rm, [r["fine"]["robust_core"] for r in res["replicated"]],
                "^-", color=RED, ms=5, lw=1.6, alpha=0.85,
                label="fine core, replicated capture")
        ax.plot(rm, [r["coarse"]["robust_core"] for r in res["replicated"]],
                "^-", color="#8a8880", ms=5, lw=1.6, alpha=0.85,
                label="coarse core, replicated")
    ax.set_yscale("log")
    ax.set_xlabel("input streams $M$", fontsize=9, color=INK)
    ax.set_ylabel("null width", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False, labelcolor=INK, loc="lower left")
    ax.set_title("(a) measured null width", fontsize=9.5, color=INK, loc="left")

    ax = axes[1]
    for stat, col in (("coarse", INK2), ("fine", BLUE)):
        xs, ys = [], []
        for r in rows:
            if not r["deflection"]:
                continue
            d = r["deflection"][0]
            xs.append(r["streams"])
            ys.append(d[stat]["deflection_core"] / d["snr_linear"])
        if xs:
            ax.plot(xs, ys, "o-", color=col, ms=4, lw=2.0,
                    label=f"{stat} (robust core)")
            ref = ys[0] * np.sqrt(np.array(xs, float) / xs[0])
            ax.plot(xs, ref, ":", color=(RED if stat == "coarse" else AMBER),
                    lw=1.4,
                    label=r"$\sqrt{M}$ from $M=%d$ (%s)" % (xs[0], stat))
    if res.get("replicated"):
        xs, ys = [], []
        for r in res["replicated"]:
            if not r["deflection"]:
                continue
            d = r["deflection"][0]
            xs.append(r["streams"])
            ys.append(d["fine"]["deflection_core"] / d["snr_linear"])
        if xs:
            ax.plot(xs, ys, "^-", color=RED, ms=5, lw=1.6, alpha=0.85,
                    label="fine, replicated capture")
    ax.set_yscale("log")
    ax.set_xlabel("input streams $M$", fontsize=9, color=INK)
    ax.set_ylabel("deflection per unit SNR", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    ax.set_title("(b) measured deflection", fontsize=9.5, color=INK, loc="left")

    ax = axes[2]
    ax.set_xscale("linear")
    g = res.get("gain_panel")
    if g:
        pfa = "0.01"
        cur = g["curves"]
        for tag, col, ls, lab in (
                ("centered", INK2, "--", "coarse (v1)"),):
            rowsg = cur.get("centered", [])
            ax.plot([r["snr_db"] for r in rowsg],
                    [r[f"pd_coarse_{float(pfa):g}"] for r in rowsg], "--",
                    color=INK2, lw=1.8, label="coarse")
        rowsg = cur.get("centered", [])
        ax.plot([r["snr_db"] for r in rowsg],
                [r[f"pd_fine_{float(pfa):g}"] for r in rowsg], "-o", color=BLUE,
                ms=3.5, lw=2.0, label="fine, bin centre")
        hb = cur.get("half_bin", [])
        if hb:
            ax.plot([r["snr_db"] for r in hb],
                    [r[f"pd_fine_{float(pfa):g}"] for r in hb], ":s", color=AMBER,
                    ms=3.5, lw=1.8, label="fine, half-bin offset")
        rec = g["gains"].get(pfa, {})
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.set_xlabel("per-row-sum pilot SNR (dB)", fontsize=9, color=INK)
        ax.set_ylabel("detection probability", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
        ax.set_title(
            f"(c) fine-axis gain, $M$=2048, $P_{{fa}}$={float(pfa):g}\n"
            f"centred {rec.get('gain_db_centered', float('nan')):.2f} dB, "
            f"half-bin {rec.get('gain_db_half_bin', float('nan')):.2f} dB, "
            f"benchmark {g['benchmark_db']:.2f} dB",
            fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=["verify-rep", "null-rep", "defl-rep", "report"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--half-bin", action="store_true")
    args = ap.parse_args()
    if args.stage == "verify-rep":
        return verify_rep(seed=args.seed, trials=args.trials,
                          streams=args.streams)
    if args.stage == "report":
        if not args.root:
            ap.error("--root required for report")
        out = args.report_json or os.path.join(args.root, "m_scaling_report.json")
        return report(args.root, out)
    if not args.out:
        ap.error("--out required")
    if args.stage == "defl-rep" and args.snr_db is None:
        ap.error("--snr-db required for defl-rep")
    return run_rep(args.out, "null" if args.stage == "null-rep" else "h1",
                   args.trials, args.seed, snr_db=args.snr_db,
                   streams=args.streams, batch=args.batch,
                   b0=(HALF_BIN if args.half_bin else ANCHOR))


if __name__ == "__main__":
    sys.exit(main())
