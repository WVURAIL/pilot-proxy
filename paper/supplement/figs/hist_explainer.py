#!/usr/bin/env python3
"""Advisor-facing explainer: how to read the F histograms (ch35 example).
Panel (a): the same data as a plain counts histogram (what one expects).
Panel (b): the paper rendering (log density) with every element annotated."""
import csv, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from pilot_proxy.plot_style import setup_matplotlib
plt = setup_matplotlib()

OUT = Path(os.environ.get("PP_OUT", "~/paper/out")).expanduser()
DUMPS = Path(os.environ.get("PP_DUMPS", "~/paper/dumps")).expanduser()
CH = 35
z = np.load(os.environ.get("PP_PERFRAME", str(DUMPS / "perframe.npz")))
s = {int(r["atsc_channel"]): r for r in
     csv.DictReader(open(OUT / "empirical_zero_points.csv"))}[CH]
mu0, mh = float(s["mu0_analytic"]), float(s["mu0_empirical"])
pt = z[f"ch{CH}_p_target_u64"].astype(np.float64)
pr = z[f"ch{CH}_p_ref_sum_u64"].astype(np.float64)
ok = z[f"ch{CH}_valid"].astype(bool)
with np.errstate(divide="ignore", invalid="ignore"):
    f = 2.0 * pt / pr
fv = f[ok & np.isfinite(f)]
x = 1e3 * (fv / mu0 - 1.0)
c = 1e3 * (mh - mu0) / mu0
N = x.size
KEPT_C, MASK_C, INK = "#0072B2", "#D55E00", "0.3"
bins = np.arange(-60, 60.2, 0.2)
cnt, edges = np.histogram(x, bins=bins)
mids = 0.5 * (edges[:-1] + edges[1:])
peak = int(cnt.max())

fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.8))

# (a) plain counts, linear axes
a.bar(mids, cnt, width=0.2, color=INK)
a.set_xlim(-60, 60)
a.set_xlabel(r"pilot excess $F/\mu_0-1\ [10^{-3}]$")
a.set_ylabel("frames per bin (counts)")
a.set_title("(a) same data, plain counts on linear axes", fontsize=10)
a.annotate(f"null core: peak bin holds\n{peak:,} of {N:,} frames",
           xy=(1.5, peak*0.97), xytext=(18, peak*0.8), fontsize=8,
           arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
a.annotate("the ~30% of frames in the tails\nare squashed against the baseline",
           xy=(33, peak*0.02), xytext=(14, peak*0.42), fontsize=8,
           arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
a.grid(color="0.94", lw=0.4); a.set_axisbelow(True)

# (b) the paper rendering, annotated
dens = cnt / (N * 0.2)
floor = 0.5 / (N * 0.2)
b.semilogy(mids, np.where(dens > 0, dens, np.nan), color=INK, lw=0.8,
           drawstyle="steps-mid")
k = mids <= 0
b.fill_between(mids[k], floor, np.where(dens[k] > 0, dens[k], floor),
               step="mid", color=KEPT_C, alpha=0.18)
b.fill_between(mids[~k], floor, np.where(dens[~k] > 0, dens[~k], floor),
               step="mid", color=MASK_C, alpha=0.14)
b.axvline(0, color="0.15", lw=1.1)
b.axvline(c, color=KEPT_C, ls="--", lw=0.9)
for tb in (c-12, c+12):
    b.axvline(tb, color="0.75", ls=":", lw=0.8)
b.set_xlim(-60, 60); b.set_ylim(floor, dens.max()*3)
b.set_xlabel(r"pilot excess $F/\mu_0-1\ [10^{-3}]$")
b.set_ylabel("density (log)")
b.set_title("(b) the paper rendering, element by element", fontsize=10)
hi_frac = 100*float(s["high_tail_frac"]); lo_frac = 100*float(s["low_tail_frac"])
b.annotate("measured core centre $\\hat{\\mu}_0$: agrees with the\nweight-derived prediction to $+0.2\\times10^{-3}$",
           xy=(c, dens.max()*1.15), xytext=(-57, dens.max()*0.85), fontsize=7.5,
           color=KEPT_C, arrowprops=dict(arrowstyle="->", color=KEPT_C, lw=0.9))
b.annotate("noise-only frames: bell at the zero\npoint, width $2.4\\times10^{-3}$ =\nradiometer statistics",
           xy=(-1.5, dens.max()*0.55), xytext=(-57, dens.max()*0.06), fontsize=7.5,
           arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
b.annotate(f"the same transmitter leaking into\nthe reference slots pushes F down:\n{lo_frac:.0f}% displaced left",
           xy=(-30, dens[np.argmin(abs(mids+30))]*1.5), xytext=(-57, floor*18),
           fontsize=7.5, color=KEPT_C,
           arrowprops=dict(arrowstyle="->", color=KEPT_C, lw=0.9))
b.annotate("decision line: frames RIGHT of here\nare masked as DTV (= 'detected')",
           xy=(0.4, dens.max()*0.12), xytext=(9, dens.max()*0.35), fontsize=7.5,
           arrowprops=dict(arrowstyle="->", color="0.15", lw=0.9))
b.annotate(f"pilot power in the target slot\npushes F up: {hi_frac:.0f}% of frames\n(the detection count lives here)",
           xy=(30, dens[np.argmin(abs(mids-30))]*1.5), xytext=(13.5, dens.max()*0.028),
           fontsize=7.5, color=MASK_C,
           arrowprops=dict(arrowstyle="->", color=MASK_C, lw=0.9))
b.text(c-12-1.6, floor*1.7, "$\\hat{\\mu}_0\\pm12\\times10^{-3}$ tail bounds", fontsize=6.5,
       color="0.5", rotation=90, va="bottom")
b.grid(color="0.94", lw=0.4); b.set_axisbelow(True)

fig.suptitle(f"How to read the F histograms (ch{CH}, {N:,} valid frames; one entry per detector frame "
             r"$= R=262{,}144$ samples $\approx 0.7$ s)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(HERE / "hist_explainer_ch35.pdf", bbox_inches="tight")
print("peak bin", peak, "N", N, "core", round(c,2))
