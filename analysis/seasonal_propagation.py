#!/usr/bin/env python3
"""Propagation statistics from the F tails: seasonal cycle, secular drift,
and fade/enhancement depth spectra, from 7.6 years of frame times.

Channel classes (from empirical_zero_points.csv):
  episodic = trusted zero point AND high_tail_frac > 0.03  (propagation-driven)
  quiet    = trusted, high_tail_frac <= 0.03               (static core, ~1% tails)
  untrusted (ch24, ch30) excluded everywhere.
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})

trusted = {ch: study[ch]["zero_point_trusted"] == "1" for ch in chans}
episodic = {ch: trusted[ch] and float(study[ch]["high_tail_frac"]) > 0.03
            for ch in chans}

SHOW = {17: "#009E73", 31: "#7B4FA6", 32: "#D55E00", 33: "#E69F00",
        35: "#0072B2"}

month_hi = np.full((len(chans), 12), np.nan)
month_cnt = np.zeros((len(chans), 12))       # frames per month (all chans)
month_hit = np.zeros((len(chans), 12))       # hi frames per month
depth = {}
qtr = {}                                     # ch -> {(yr,q): [hi, n]}
qtr_units = {}                               # ch -> {(yr,q): [(hi_u, n_u)]}
ym = {}                                      # ch -> {(yr,m): [hi, n]}
years_seen = {}                              # yr -> [hi, n] episodic only
for i, ch in enumerate(chans):
    s = study[ch]
    mu0 = float(s["mu0_analytic"])
    mu_hat = float(s["mu0_empirical"])
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    fui = z[f"ch{ch}_frame_unit_index"].astype(int)
    t0 = np.asarray(z[f"ch{ch}_unit_time0_ctime"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    fin_u = np.isfinite(t0) & (t0 > 1e9)
    ok = valid & np.isfinite(f) & fin_u[fui]
    hi = ok & (f > mu_hat + 12e-3 * mu0)
    lo = ok & (f < mu_hat - 12e-3 * mu0)
    mo_u = np.full(t0.size, -1, dtype=int)
    yr_u = np.full(t0.size, -1, dtype=int)
    for j in np.nonzero(fin_u)[0]:
        d = datetime.fromtimestamp(float(t0[j]), tz=timezone.utc)
        mo_u[j] = d.month - 1
        yr_u[j] = d.year
    months = mo_u[fui]
    years = yr_u[fui]
    for m in range(12):
        sel = ok & (months == m)
        n = int(sel.sum())
        month_cnt[i, m] = n
        month_hit[i, m] = int((hi & sel).sum())
        if n > 300:
            month_hi[i, m] = month_hit[i, m] / n
    # quarterly + per-(year,month) time series
    qd, ymd = {}, {}
    q_of = months // 3
    for y in np.unique(years[years > 0]):
        ysel = ok & (years == y)
        for q in range(4):
            sel = ysel & (q_of == q)
            n = int(sel.sum())
            if n:
                qd[(int(y), q)] = [int((hi & sel).sum()), n]
        for m in range(12):
            sel = ysel & (months == m)
            n = int(sel.sum())
            if n:
                ymd[(int(y), m)] = [int((hi & sel).sum()), n]
    qtr[ch] = qd
    ym[ch] = ymd
    # per-unit (event) counts per quarter, for block-bootstrap intervals
    n_u = np.bincount(fui[ok], minlength=t0.size)
    h_u = np.bincount(fui[hi], minlength=t0.size)
    qu = {}
    for j in np.nonzero(fin_u)[0]:
        if n_u[j] == 0:
            continue
        qu.setdefault((int(yr_u[j]), int(mo_u[j]) // 3), []).append(
            (int(h_u[j]), int(n_u[j])))
    qtr_units[ch] = qu
    if episodic[ch]:
        for y in np.unique(years[years > 0]):
            sel = ok & (years == y)
            e = years_seen.setdefault(int(y), [0, 0])
            e[0] += int((hi & sel).sum())
            e[1] += int(sel.sum())
    # Excursion spectrum of the detection statistic itself (baseline-free:
    # propagation events outlast a capture unit, so per-unit power baselines
    # cannot see them; F/mu0 can, frame by frame).
    if ch in SHOW:
        dev = 1e3 * (f - mu_hat) / mu0
        depth[ch] = dict(hi=dev[hi], lo=-dev[lo])

epi_idx = [i for i, ch in enumerate(chans) if episodic[ch]]
qui_idx = [i for i, ch in enumerate(chans) if trusted[ch] and not episodic[ch]]
epi_chs = [chans[i] for i in epi_idx]


def agg_months(idx):
    hit = month_hit[idx].sum(axis=0)
    cnt = month_cnt[idx].sum(axis=0)
    return np.where(cnt > 500, hit / np.maximum(cnt, 1), np.nan)


tot_epi = agg_months(epi_idx)
tot_qui = agg_months(qui_idx)

# Year-detrended seasonal anomaly: rate(y,m)/rate(y), folded across years.
# Removes the transmitter-side secular changes that dominate the raw fold.
anom_num = {ch: np.zeros(12) for ch in epi_chs}
anom_den = {ch: np.zeros(12) for ch in epi_chs}
step_years = []
for ch in epi_chs:
    for y in sorted({k[0] for k in ym[ch]}):
        h_y = sum(v[0] for k, v in ym[ch].items() if k[0] == y)
        n_y = sum(v[1] for k, v in ym[ch].items() if k[0] == y)
        if n_y < 1200 or h_y / n_y < 0.02:
            continue
        r_y = h_y / n_y
        rates = [v[0] / v[1] for k, v in ym[ch].items()
                 if k[0] == y and v[1] >= 200]
        # a >4x swing within one year is a transmitter step, not seasonality
        if len(rates) >= 3 and max(rates) > 4 * max(min(rates), 0.005):
            step_years.append((ch, y))
            continue
        for m in range(12):
            h, n = ym[ch].get((y, m), (0, 0))
            if n >= 200:
                anom_num[ch][m] += n * (h / n) / r_y
                anom_den[ch][m] += n

A_ch = {ch: np.where(anom_den[ch] > 0, anom_num[ch] /
                     np.maximum(anom_den[ch], 1), np.nan) for ch in epi_chs}
A_all = (np.nansum([anom_num[ch] for ch in epi_chs], axis=0) /
         np.maximum(np.nansum([anom_den[ch] for ch in epi_chs], axis=0), 1))

# ---------------- figure 1: detrended seasonal anomaly + depth spectra -------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
mnames = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
for ch in epi_chs:
    if ch in SHOW:
        ax1.plot(range(12), A_ch[ch], "-o", ms=3, lw=1.1, color=SHOW[ch],
                 label=f"ch{ch}", alpha=0.85)
ax1.plot(range(12), A_all, "k-", lw=2.4, zorder=5, label="episodic channels")
ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax1.set_xticks(range(12))
ax1.set_xticklabels(mnames)
ax1.set_xlabel("month")
ax1.set_ylabel("detection-rate anomaly (month / same-year mean)")
ax1.set_title("(a) seasonal anomaly after removing year-to-year drift",
              fontsize=10)
ax1.legend(fontsize=7.5, ncol=2, loc="upper left")
ax1.grid(color="0.92", lw=0.6)
ax1.set_axisbelow(True)

for ch, c in SHOW.items():
    d = depth.get(ch, {}).get("hi", np.array([]))
    if d.size > 100:
        xs = np.sort(d)
        ccdf = 1.0 - np.arange(xs.size) / xs.size
        ax2.loglog(xs, ccdf, color=c, lw=1.4, label=f"ch{ch}")
    dl = depth.get(ch, {}).get("lo", np.array([]))
    if dl.size > 100:
        xs = np.sort(dl)
        ccdf = 1.0 - np.arange(xs.size) / xs.size
        ax2.loglog(xs, ccdf, color=c, lw=1.1, ls="--")
ax2.set_xlabel(r"$|F-\hat{\mu}_0|/\mu_0$ beyond the zero point $[10^{-3}]$"
               "\n(axis starts at the mask band edge, $12\\times10^{-3}$)")
ax2.set_ylabel("fraction of tail frames beyond")
ax2.set_title("(b) excursion spectra: detections (solid) / fades (dashed)",
              fontsize=10)
ax2.legend(fontsize=7.5, loc="lower left")
ax2.grid(color="0.92", lw=0.6, which="both")
ax2.set_axisbelow(True)
ax2.set_ylim(1e-3, 1.1)
ax2.set_xlim(12, None)
fig.tight_layout()
fig.savefig(OUT / "fig_seasonal_propagation.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_seasonal_propagation.pdf", bbox_inches="tight")

# ---------------- figure 2: secular quarterly time series --------------------
# Lines break across quarters with no (or <=200-frame) sampled exposure;
# intervals are 68% per-event block bootstraps; the strip shows exposure.
fig2 = plt.figure(figsize=(9.2, 5.6))
gs2 = fig2.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.12)
ax = fig2.add_subplot(gs2[0])
axe = fig2.add_subplot(gs2[1], sharex=ax)
all_q = sorted({k for ch in epi_chs for k in qtr[ch]})
y0, q0 = all_q[0]
y1, q1 = all_q[-1]
grid = []
gy, gq = y0, q0
while (gy, gq) <= (y1, q1):
    grid.append((gy, gq))
    gq += 1
    if gq == 4:
        gq, gy = 0, gy + 1


def qx(k):
    return k[0] + (k[1] + 0.5) / 4.0


xs_grid = np.array([qx(k) for k in grid])


def boot_ci(pairs, n_boot=300, seed=31):
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    h = np.array([u[0] for u in pairs], dtype=float)
    n = np.array([u[1] for u in pairs], dtype=float)
    rng2 = np.random.default_rng(seed)
    U = len(pairs)
    r = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng2.integers(0, U, U)
        r[b] = h[pick].sum() / max(n[pick].sum(), 1.0)
    lo, hi_ = np.percentile(r, [16, 84])
    return (float(lo), float(hi_))


def series(qd, qu, floor):
    ys = np.full(len(grid), np.nan)
    e_lo = np.zeros(len(grid))
    e_hi = np.zeros(len(grid))
    for i, k in enumerate(grid):
        hn = qd.get(k)
        if hn and hn[1] > floor:
            rate = hn[0] / hn[1]
            ys[i] = 100 * rate
            lo, hi_ = boot_ci(qu.get(k, []))
            if np.isfinite(lo):
                e_lo[i] = 100 * max(rate - lo, 0.0)
                e_hi[i] = 100 * max(hi_ - rate, 0.0)
    return ys, e_lo, e_hi


for ch in chans:
    if ch not in SHOW:
        continue
    ys, e_lo, e_hi = series(qtr[ch], qtr_units[ch], 200)
    ax.errorbar(xs_grid, ys, yerr=[e_lo, e_hi], fmt="-o", ms=3, lw=1.1,
                color=SHOW[ch], label=f"ch{ch}", alpha=0.85, capsize=1.5,
                elinewidth=0.8)
agg_qd, agg_qu = {}, {}
for ch in epi_chs:
    for k, v in qtr[ch].items():
        e = agg_qd.setdefault(k, [0, 0])
        e[0] += v[0]
        e[1] += v[1]
    for k, v in qtr_units[ch].items():
        agg_qu.setdefault(k, []).extend(v)
ys, e_lo, e_hi = series(agg_qd, agg_qu, 500)
ax.errorbar(xs_grid, ys, yerr=[e_lo, e_hi], fmt="-", color="k", lw=2.4,
            label="episodic channels", zorder=5, capsize=0, elinewidth=1.0)
ax.set_ylabel(f"high-tail (detection) rate [{PCT} of frames]")
ax.set_title("Quarterly per-frame detection rates within sampled exposure "
             f"(68{PCT} per-event intervals)", fontsize=10.5)
ax.tick_params(labelbottom=False)
ax.legend(fontsize=8, ncol=3)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
expo = np.array([agg_qd.get(k, [0, 0])[1] for k in grid], dtype=float)
axe.bar(xs_grid, np.maximum(expo, 0.5), width=0.22, color="0.8", zorder=2)
axe.set_yscale("log")
axe.set_ylim(bottom=100)
axe.set_ylabel("frames")
axe.set_xlabel("year")
axe.grid(color="0.92", lw=0.6, axis="y", which="both")
axe.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(OUT / "fig_secular_rates.png", dpi=300, bbox_inches="tight")
fig2.savefig(OUT / "fig_secular_rates.pdf", bbox_inches="tight")

# ---------------- printed summary ---------------------------------------------
mx, mn = int(np.nanargmax(tot_epi)), int(np.nanargmin(tot_epi))
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
print(f"episodic channels: {epi_chs}")
print(f"raw folded aggregate: max {100*tot_epi[mx]:.2f}% in {names[mx]}, "
      f"min {100*tot_epi[mn]:.2f}% in {names[mn]} "
      f"(ratio {tot_epi[mx]/max(tot_epi[mn],1e-9):.2f}x; "
      "aliases secular drift)")
amx, amn = int(np.nanargmax(A_all)), int(np.nanargmin(A_all))
print(f"DETRENDED anomaly:    max {A_all[amx]:.2f} in {names[amx]}, "
      f"min {A_all[amn]:.2f} in {names[amn]} "
      f"(ratio {A_all[amx]/max(A_all[amn],1e-9):.2f}x)")
print("\nper-channel detrended peak/trough (SHOW):")
for ch in epi_chs:
    if ch in SHOW and np.any(np.isfinite(A_ch[ch])):
        pk = int(np.nanargmax(A_ch[ch]))
        tr = int(np.nanargmin(A_ch[ch]))
        print(f"  ch{ch}: peak {names[pk]} x{A_ch[ch][pk]:.2f}  "
              f"trough {names[tr]} x{A_ch[ch][tr]:.2f}")
print(f"\nchannel-years excluded as transmitter steps: {step_years}")
print("\ndetection-margin spectrum (hi tail, 10^-3 above zero point):")
for ch in SHOW:
    d = depth.get(ch, {}).get("hi", np.array([]))
    if d.size:
        q = np.percentile(d, [50, 90, 99])
        print(f"  ch{ch}: median {q[0]:7.1f}   p90 {q[1]:8.1f}   "
              f"p99 {q[2]:9.1f}   (n={d.size})")
print("\nepisodic aggregate by year (secular check):")
for y in sorted(years_seen):
    h, n = years_seen[y]
    print(f"  {y}: {100*h/max(n,1):5.2f}%  ({n} frames)")
print("\nwrote fig_seasonal_propagation + fig_secular_rates")
