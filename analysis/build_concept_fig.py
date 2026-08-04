#!/usr/bin/env python3
"""Pedagogical schematic: why deep one-sided cuts concentrate contamination
and why the core (mean) anchor is the efficient operating point."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT
C_H0, C_SIG, C_REF = "0.35", "#009E73", "#D55E00"
C_TAU, C_BAND = "#0072B2", "#7B4FA6"

x = np.linspace(-8, 6, 2000)          # units of sigma_core about the core
h0 = norm.pdf(x, 0, 1)
sig = 0.10 * norm.pdf(x, 0.8, 1)      # sub-detectable signal (shift ~0.8 sig)
det = 0.06 * norm.pdf(x, 4.0, 1.2)    # detectable signal (masked at core)
ref = 0.05 * norm.pdf(x, -5.0, 1.4)   # reference-contaminated (low tail)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
ax1.semilogy(x, h0, color=C_H0, lw=1.6, label="clean frames (H0 core)")
ax1.semilogy(x, sig, color=C_SIG, lw=1.4,
             label="sub-detectable signal (shifted $\\sim$1$\\sigma$)")
ax1.semilogy(x, det, color=C_SIG, lw=1.4, ls=":",
             label="detectable signal ($\\gtrsim$3$\\sigma$)")
ax1.semilogy(x, ref, color=C_REF, lw=1.4,
             label="reference-contaminated (low tail)")
ax1.axvline(0, color=C_TAU, lw=1.4)
ax1.text(0.12, 2.2e-4, r"$\tau$ at core", color=C_TAU, fontsize=8.5,
         rotation=90, va="bottom")
ax1.axvline(-2, color=C_TAU, lw=1.2, ls="--")
ax1.text(-1.88, 2.2e-4, r'"aggressive" $\tau$ $-2\sigma$', color=C_TAU,
         fontsize=8.5, rotation=90, va="bottom")
ax1.axvline(-3.5, color=C_BAND, lw=1.2, ls="-.")
ax1.text(-3.38, 2.2e-4, "band floor", color=C_BAND, fontsize=8.5,
         rotation=90, va="bottom")
ax1.annotate(f"kept at $-2\\sigma$:\nclean 2.3{PCT}\nref-leak $\\approx$ all of it",
             xy=(-4.6, 3e-3), fontsize=8, color=C_REF, ha="center")
ax1.set_xlabel(r"$(F-\hat{\mu}_0)/\sigma_{\rm core}$")
ax1.set_ylabel("probability density (log)")
ax1.set_ylim(1e-4, 0.6)
ax1.set_title("(a) the populations a threshold selects", fontsize=10)
ax1.legend(fontsize=7.2, loc="upper right")
ax1.grid(color="0.93", lw=0.5)
ax1.set_axisbelow(True)

ks = np.linspace(0, 3, 200)
clean = norm.cdf(-ks)                    # H0 survivors below tau
subsig = 0.10 * norm.cdf(-ks - 0.8)      # weak-signal survivors
refk = np.full_like(ks, 0.05)            # low tail: constant until ~5 sigma
tot = clean + subsig + refk
ax2.fill_between(ks, 0, clean / tot, color="0.75", label="clean")
ax2.fill_between(ks, clean / tot, (clean + subsig) / tot, color=C_SIG,
                 alpha=0.65, label="sub-detectable signal")
ax2.fill_between(ks, (clean + subsig) / tot, 1, color=C_REF, alpha=0.75,
                 label="reference-contaminated")
ax2.set_xlabel(r"one-sided threshold depth $k$ [$\sigma_{\rm core}$]")
ax2.set_ylabel("composition of KEPT data")
ax2.set_title("(b) deeper cuts do not purify -- they concentrate",
              fontsize=10)
ax2.legend(fontsize=8, loc="center left")
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 1)
ax2.grid(color="0.93", lw=0.5)
ax2.set_axisbelow(True)
ax2t = ax2.twiny()  # no second scale -- annotate bandwidth cost as ticks
ax2t.set_xlim(0, 3)
ax2t.set_xticks([0, 1, 1.65, 2.33, 3])
ax2t.set_xticklabels([f"keep 50{PCT}", f"16{PCT}", f"5{PCT}", f"1{PCT}",
                      f"0.1{PCT}"], fontsize=7)
ax2t.set_xlabel("bandwidth kept (pure H0)", fontsize=8)
fig.suptitle("Schematic: one-sided depth vs kept-data composition "
             "(populations illustrative, scaled from ch31/32-like tails)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_concept_threshold_depth.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_concept_threshold_depth.pdf", bbox_inches="tight")
print("wrote fig_concept_threshold_depth")
