#!/usr/bin/env python3
"""Decompose the F tails: is the low tail hot references or cold targets?

Per frame, normalize p_target and p_ref_sum by their unit's core-frame
baseline (median over frames with |F-mu_hat| < 6e-3*mu0 in the same unit;
fallback: channel median). Then compare the normalized target/ref power of
low-tail, core, and high-tail frames -- and test event-level coincidence and
diurnal structure of the two tails.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
DRAO_LON_H = -119.6175 / 15.0
C_LOW, C_CORE, C_HIGH = "#D55E00", "0.55", "#009E73"

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}

SHOW = [32, 31, 35, 21]


def analyze(ch):
    s = study[ch]
    mu0 = float(s["mu0_manifest"])
    mu_hat = float(s["mu0_empirical"])
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    fui = z[f"ch{ch}_frame_unit_index"].astype(int)
    t0 = z[f"ch{ch}_unit_time0_ctime"]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    ok = valid & np.isfinite(f)
    core_win = np.abs(f - mu_hat) <= 6e-3 * mu0
    low = ok & (f < mu_hat - 12e-3 * mu0)
    high = ok & (f > mu_hat + 12e-3 * mu0)
    core = ok & core_win

    # per-unit baselines from core frames only
    n_units = t0.size
    base_t = np.full(n_units, np.nan)
    base_r = np.full(n_units, np.nan)
    for u in np.unique(fui):
        sel = core & (fui == u)
        if sel.sum() >= 2:
            base_t[u] = np.median(pt[sel])
            base_r[u] = np.median(pr[sel])
    med_t, med_r = np.nanmedian(base_t), np.nanmedian(base_r)
    base_t = np.where(np.isfinite(base_t), base_t, med_t)
    base_r = np.where(np.isfinite(base_r), base_r, med_r)
    nt = pt / base_t[fui]
    nr = pr / base_r[fui]

    def stats(sel):
        return (float(np.median(nt[sel])), float(np.median(nr[sel])),
                int(sel.sum()))
    res = {"low": stats(low), "core": stats(core), "high": stats(high)}

    # event-level coincidence
    units_low = set(fui[low])
    units_high = set(fui[high])
    all_units = set(fui[ok])
    p_low = len(units_low) / len(all_units)
    p_low_given_high = (len(units_low & units_high) / len(units_high)
                        if units_high else float("nan"))
    # diurnal profiles of tail rates
    tf = t0[fui]
    hours = ((tf / 3600.0 + DRAO_LON_H) % 24.0).astype(int)
    prof_low = np.full(24, np.nan)
    prof_high = np.full(24, np.nan)
    for h in range(24):
        selh = ok & (hours == h)
        if selh.sum() > 100:
            prof_low[h] = (low & selh).sum() / selh.sum()
            prof_high[h] = (high & selh).sum() / selh.sum()
    return dict(res=res, nt=nt, nr=nr, low=low, core=core, high=high,
                p_low=p_low, p_low_given_high=p_low_given_high,
                prof_low=prof_low, prof_high=prof_high)


print(f"{'ch':>3} {'class':>5} {'n':>6} {'target/base':>12} {'refs/base':>10}")
results = {}
for ch in SHOW:
    a = analyze(ch)
    results[ch] = a
    for cls in ("low", "core", "high"):
        t, r, n = a["res"][cls]
        print(f"{ch:>3} {cls:>5} {n:>6} {t:>12.3f} {r:>10.3f}")
    print(f"     coincidence: P(low-tail unit) = {a['p_low']:.3f}; "
          f"P(low | unit has high) = {a['p_low_given_high']:.3f} "
          f"(enhancement x{a['p_low_given_high']/max(a['p_low'],1e-9):.1f})")

# ---- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
ax = axes[0]
a = results[32]
sub = {}
rng = np.random.default_rng(7)
for cls, c, m in (("core", C_CORE, "."), ("high", C_HIGH, "^"),
                  ("low", C_LOW, "v")):
    sel = a[cls]
    idx = np.flatnonzero(sel)
    if idx.size > 4000:
        idx = rng.choice(idx, 4000, replace=False)
    ax.plot(10 * np.log10(a["nt"][idx]), 10 * np.log10(a["nr"][idx]),
            m, ms=2.5 if cls == "core" else 3.5, color=c, alpha=0.35,
            mew=0, label=f"{cls} tail" if cls != "core" else "core")
ax.axhline(0, color="0.6", lw=0.7)
ax.axvline(0, color="0.6", lw=0.7)
ax.set_xlabel("target power vs unit baseline [dB]")
ax.set_ylabel("reference power vs unit baseline [dB]")
ax.set_title("(a) ch32: what moved, target or references?", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)
ax.set_xlim(-6, 14)
ax.set_ylim(-6, 14)

ax = axes[1]
for ch, c in ((32, "#D55E00"), (31, "#7B4FA6"), (35, "#00795A")):
    a = results[ch]
    ax.plot(range(24), 100 * a["prof_low"], "-", color=c, lw=1.5,
            label=f"ch{ch} low tail")
    ax.plot(range(24), 100 * a["prof_high"], "--", color=c, lw=1.1,
            label=f"ch{ch} high tail")
ax.set_xlabel("local solar hour at DRAO")
ax.set_ylabel("tail rate [% of frames]")
ax.set_title("(b) both tails share diurnal structure "
             "(solid: low; dashed: high)", fontsize=10)
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=7, ncol=3)
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_tail_decomposition.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_tail_decomposition.pdf", bbox_inches="tight")
print("wrote fig_tail_decomposition")
