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
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT
DRAO_LON_H = -119.6175 / 15.0
C_LOW, C_CORE, C_HIGH = "#D55E00", "0.55", "#009E73"

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}

SHOW = [32, 31, 35, 21]


def analyze(ch):
    s = study[ch]
    mu0 = float(s["mu0_analytic"])
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
    # diurnal profiles of tail rates, with hourly denominators and
    # per-event (unit) block-bootstrap 68% intervals: frames within a
    # capture unit are not independent, so resample whole units.
    n_u = np.bincount(fui[ok], minlength=t0.size)
    lo_u = np.bincount(fui[low], minlength=t0.size)
    hi_u = np.bincount(fui[high], minlength=t0.size)
    ok_u = np.isfinite(t0) & (t0 > 1e9) & (n_u > 0)
    hours_u = np.full(t0.size, -1)
    hours_u[ok_u] = ((t0[ok_u] / 3600.0 + DRAO_LON_H) % 24.0).astype(int)
    rng = np.random.default_rng(23)
    prof_low = np.full(24, np.nan)
    prof_high = np.full(24, np.nan)
    band_low = np.full((24, 2), np.nan)
    band_high = np.full((24, 2), np.nan)
    denom = np.zeros(24)
    for h in range(24):
        us = np.nonzero(hours_u == h)[0]
        nh = n_u[us].sum()
        denom[h] = nh
        if nh <= 100:
            continue
        prof_low[h] = lo_u[us].sum() / nh
        prof_high[h] = hi_u[us].sum() / nh
        U = us.size
        bl, bh = np.empty(300), np.empty(300)
        for b in range(300):
            pick = us[rng.integers(0, U, U)]
            nb = max(n_u[pick].sum(), 1)
            bl[b] = lo_u[pick].sum() / nb
            bh[b] = hi_u[pick].sum() / nb
        band_low[h] = np.percentile(bl, [16, 84])
        band_high[h] = np.percentile(bh, [16, 84])
    return dict(res=res, nt=nt, nr=nr, low=low, core=core, high=high,
                p_low=p_low, p_low_given_high=p_low_given_high,
                prof_low=prof_low, prof_high=prof_high,
                band_low=band_low, band_high=band_high, denom=denom)


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
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(11.2, 5.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08],
                      height_ratios=[2.6, 1.0], hspace=0.12, wspace=0.24)
ax = fig.add_subplot(gs[:, 0])
a = results[32]
sel = a["core"]
x_c = 10 * np.log10(a["nt"][sel])
y_c = 10 * np.log10(a["nr"][sel])
H, ex, ey = np.histogram2d(x_c, y_c, bins=140,
                           range=[[-6, 14], [-6, 14]])
ax.pcolormesh(ex, ey, H.T, norm=LogNorm(vmin=0.8), cmap="Greys",
              rasterized=True, zorder=0)
cx = 0.5 * (ex[:-1] + ex[1:])
cy = 0.5 * (ey[:-1] + ey[1:])
ax.contour(cx, cy, H.T, levels=[10, 100, 1000], colors="0.25",
           linewidths=0.6, zorder=4)
rng = np.random.default_rng(7)
for cls, c, m in (("high", C_HIGH, "^"), ("low", C_LOW, "v")):
    sel = a[cls]
    idx = np.flatnonzero(sel)
    if idx.size > 4000:
        idx = rng.choice(idx, 4000, replace=False)
    ax.plot(10 * np.log10(a["nt"][idx]), 10 * np.log10(a["nr"][idx]),
            m, ms=3.5, color=c, alpha=0.35, mew=0, rasterized=True)
ax.axhline(0, color="0.6", lw=0.7)
ax.axvline(0, color="0.6", lw=0.7)
ax.set_xlabel("target power vs unit baseline [dB]")
ax.set_ylabel("reference power vs unit baseline [dB]")
ax.set_title("(a) ch32: what moved, target or references?", fontsize=10)
ax.legend(handles=[
    Patch(fc="0.55", label="core (log density + contours)"),
    Line2D([], [], marker="^", ls="", color=C_HIGH, label="high tail"),
    Line2D([], [], marker="v", ls="", color=C_LOW, label="low tail")],
    fontsize=8, loc="upper left")
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)
ax.set_xlim(-6, 14)
ax.set_ylim(-6, 14)

axb = fig.add_subplot(gs[0, 1])
axd = fig.add_subplot(gs[1, 1], sharex=axb)
for ch, c in ((32, "#D55E00"), (31, "#7B4FA6"), (35, "#00795A")):
    a = results[ch]
    axb.plot(range(24), 100 * a["prof_low"], "-", color=c, lw=1.5,
             label=f"ch{ch} low tail")
    axb.fill_between(range(24), 100 * a["band_low"][:, 0],
                     100 * a["band_low"][:, 1], color=c, alpha=0.13, lw=0)
    axb.plot(range(24), 100 * a["prof_high"], "--", color=c, lw=1.1,
             label=f"ch{ch} high tail")
    axb.fill_between(range(24), 100 * a["band_high"][:, 0],
                     100 * a["band_high"][:, 1], color=c, alpha=0.13, lw=0)
    axd.plot(range(24), a["denom"] / 1e3, "-", color=c, lw=1.0)
axb.set_ylabel(f"tail rate [{PCT} of frames]")
axb.set_title("(b) both tails share diurnal structure\n"
              f"(solid: low; dashed: high; 68{PCT} per-event intervals)",
              fontsize=10)
axb.tick_params(labelbottom=False)
ylo, yhi = axb.get_ylim()
axb.set_ylim(ylo - 0.30 * (yhi - ylo), yhi)
axb.legend(fontsize=6.5, ncol=3, loc="lower center")
axb.grid(color="0.93", lw=0.5)
axb.set_axisbelow(True)
axd.set_xlabel("local solar hour at DRAO")
axd.set_ylabel(r"frames [$10^3$]")
axd.set_xticks(range(0, 24, 3))
axd.grid(color="0.93", lw=0.5)
axd.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_tail_decomposition.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_tail_decomposition.pdf", dpi=300, bbox_inches="tight")
print("wrote fig_tail_decomposition")
