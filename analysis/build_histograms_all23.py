#!/usr/bin/env python3
"""Per-channel F-statistic histograms, all 23 channels, full depth.

x-axis: (F - mu0)/mu0 in 1e-3, log-y density. Marks the manifest zero point
(x=0), the measured zero point (mu_hat), and the +/-12e-3 tail boundaries
used for the low/high-tail fractions. ch30 (signal-dominated) gets its own
x-range.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
INK, C_SUP, C_HAT, C_TAIL = "0.3", "#D55E00", "#0072B2", "0.75"
SUPPRESSED = {14, 21, 25, 28, 36}
SPAN = 60.0                         # +/- 1e-3 units for the shared panels

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})

fig, axes = plt.subplots(6, 4, figsize=(11.5, 12.6))
for j, ch in enumerate(chans):
    ax = axes.flat[j]
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    fv = f[valid & np.isfinite(f)]
    s = study[ch]
    mu0 = float(s["mu0_manifest"])
    mu_hat = float(s["mu0_empirical"])
    gap = 1e3 * (mu_hat - mu0) / mu0
    trusted = s["zero_point_trusted"] == "1"
    x = 1e3 * (fv / mu0 - 1.0)
    if ch == 30:
        lo, hi = np.percentile(x, [0.2, 99.8])
        pad = 0.06 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 320)
    else:
        bins = np.linspace(-SPAN, SPAN, 600)
    cnt, edges = np.histogram(x, bins=bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ax.semilogy(mids, np.maximum(cnt, 1e-6), color=INK, lw=0.7,
                drawstyle="steps-mid")
    ax.axvline(0.0, color="0.55", lw=0.7)
    if trusted:
        ax.axvline(gap, color=C_HAT, ls="--", lw=0.9)
        for t in (gap - 12.0, gap + 12.0):
            if abs(t) < SPAN:
                ax.axvline(t, color=C_TAIL, ls=":", lw=0.7)
    low, high = float(s["low_tail_frac"]), float(s["high_tail_frac"])
    ax.text(0.03, 0.92,
            f"$\\Delta$={gap:+.1f}" + ("" if trusted else " (untrusted)"),
            transform=ax.transAxes, fontsize=6.5,
            color=C_SUP if ch in SUPPRESSED else "black")
    ax.text(0.03, 0.80, f"low {100*low:.1f}% / high {100*high:.1f}%",
            transform=ax.transAxes, fontsize=6)
    tcol = C_SUP if ch in SUPPRESSED else "black"
    ax.set_title(f"ch{ch} (fid {s['freq_id']})", fontsize=7.5, color=tcol,
                 pad=2)
    ax.tick_params(labelsize=5.5)
    ax.set_ylim(bottom=1e-5)
    ax.grid(color="0.94", lw=0.35)
    ax.set_axisbelow(True)
for j in range(len(chans), 24):
    axes.flat[j].axis("off")
for ax in axes[-1, :]:
    ax.set_xlabel(r"$(F-\mu_0)/\mu_0\ [10^{-3}]$", fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("density", fontsize=7)
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([], [], color="0.55", label=r"manifest $\mu_0$"),
    Line2D([], [], color=C_HAT, ls="--", label=r"measured $\hat{\mu}_0$"),
    Line2D([], [], color=C_TAIL, ls=":",
           label=r"$\pm12\times10^{-3}$ tail bounds"),
], loc="lower center", ncol=3, fontsize=8, frameon=False,
    bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Valid-frame F distributions, all 23 channels (full depth; "
             "log density; suppressed-family titles in orange)",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.99))
fig.savefig(OUT / "fig_f_histograms_all23.png", dpi=230, bbox_inches="tight")
fig.savefig(OUT / "fig_f_histograms_all23.pdf", bbox_inches="tight")
print("wrote fig_f_histograms_all23")
