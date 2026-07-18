#!/usr/bin/env python3
"""Aggressive-masking tradeoff: slide tau below the measured core.

For k in [0, 3]: tau_k = mu_hat - k*sigma_core. Compare one-sided keep
(F <= tau_k) against band-keep (mu_hat - 2*sigma - k*sigma <= ... no:
band-keep = tau_k - band <= F <= tau_k with band = 3*sigma_core), using a
Gaussian H0 core model (amplitude-free: mean/sigma from the core window) to
estimate how much of the kept data is contamination (excess over H0).
"""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib
from scipy.stats import norm  # noqa: E402

plt = setup_matplotlib()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT
COARSE_MHZ = 400.0 / 1024.0
C_A, C_B, C_C = "#0072B2", "#D55E00", "#009E73"

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})

# k < 0 places the ceiling ABOVE the measured core (the region the adopted
# operating point actually occupies); k > 0 is the aggressive one-sided cut.
KS = np.linspace(-1.5, 3.0, 91)
TRUNC = 0.8796  # std correction for a +/-2.5-sigma truncated Gaussian window

per = {}
for ch in chans:
    s = study[ch]
    if s["zero_point_trusted"] != "1":
        continue
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    fv = f[valid & np.isfinite(f)]
    mu0 = float(s["mu0_analytic"])
    c = float(s["mu0_empirical"])
    w = fv[np.abs(fv - c) <= 6e-3 * mu0]
    sigma = float(w.std(ddof=1)) / TRUNC
    n = fv.size
    kept_1s, cont_1s, kept_bk, cont_bk = [], [], [], []
    band = 3.0 * sigma
    for k in KS:
        tau = c - k * sigma
        sel1 = fv <= tau
        kept = sel1.mean()
        h0_exp = norm.cdf((tau - c) / sigma)          # H0 fraction below tau
        cont = max(kept - h0_exp, 0.0)                # excess over H0 = contamination
        kept_1s.append(kept)
        cont_1s.append(cont / kept if kept > 0 else 0.0)
        selb = (fv <= tau) & (fv >= tau - band)
        keptb = selb.mean()
        h0_b = norm.cdf((tau - c) / sigma) - norm.cdf((tau - band - c) / sigma)
        contb = max(keptb - h0_b, 0.0)
        kept_bk.append(keptb)
        cont_bk.append(contb / keptb if keptb > 0 else 0.0)
    per[ch] = dict(sigma=sigma, kept_1s=np.array(kept_1s),
                   cont_1s=np.array(cont_1s), kept_bk=np.array(kept_bk),
                   cont_bk=np.array(cont_bk))

# aggregate recovered bandwidth vs k (trusted channels only)
rec_1s = sum(p["kept_1s"] for p in per.values()) * COARSE_MHZ
rec_bk = sum(p["kept_bk"] for p in per.values()) * COARSE_MHZ
h0_curve = norm.cdf(-KS) * len(per) * COARSE_MHZ

fig, ax2 = plt.subplots(figsize=(7.4, 4.6))
# panel (a) content folded onto a twin axis: the aggregate kept
# bandwidth rides behind the per-channel contamination curves.
axk = ax2.twinx()
axk.plot(KS, rec_1s, color="0.45", lw=1.4, alpha=0.8)
axk.plot(KS, rec_bk, color="0.45", lw=1.1, ls="--", alpha=0.8)
axk.set_ylabel("kept pilot-channel bandwidth [MHz] (grey)", color="0.35")
axk.tick_params(axis="y", colors="0.35")
axk.set_ylim(bottom=0)
axk.set_zorder(1)
ax2.set_zorder(2)
ax2.patch.set_visible(False)

SHOW = [(32, "ch32 (heavy low tail)"), (31, "ch31"), (35, "ch35"),
        (21, "ch21 (corrected)"), (34, "ch34")]
for (ch, lbl), col in zip(SHOW, ("#D55E00", "#7B4FA6", "#00795A",
                                 "#0072B2", "0.35")):
    if ch not in per:
        continue
    ax2.plot(KS, 100 * per[ch]["cont_1s"], color=col, lw=1.5, label=lbl)
    ax2.plot(KS, 100 * per[ch]["cont_bk"], color=col, lw=1.2, ls="--")
ax2.axvline(0.0, color="0.55", lw=0.8, ls=":")
ax2.set_xlabel(r"threshold offset below measured core  $k$  "
               r"[$\sigma_{\rm core}$]  (negative: ceiling above core)")
ax2.set_ylabel(f"non-null excess among retained frames [{PCT}]")
ax2.axvline(0.0, color="0.55", lw=0.8, ls=":")
ax2.annotate("adopted ceiling ($k{=}0$)", xy=(0.0, 0.62),
             xycoords=("data", "axes fraction"), xytext=(10, 0),
             textcoords="offset points", fontsize=7.5, color="0.25",
             rotation=90, va="center")
ax2.text(0.02, 0.965, "ceiling above core ($k<0$; not adopted)",
         transform=ax2.transAxes, fontsize=7.5, color="0.35", va="top")
ax2.set_title("deeper one-sided cuts lose data exponentially while the "
              "kept fraction gets dirtier\n(colour: non-null excess, "
              "solid one-sided / dashed band-keep;  grey: kept bandwidth)",
              fontsize=9.5)
ax2.legend(fontsize=7.5)
ax2.grid(color="0.92", lw=0.5)
ax2.set_axisbelow(True)
ax2.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT / "fig_aggressive_masking_tradeoff.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_aggressive_masking_tradeoff.pdf", bbox_inches="tight")

with open(OUT / "aggressive_masking.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "sigma_core_1e3_of_mu0",
                "kept_km1", "contfrac_km1",
                "kept_k0", "contfrac_k0", "kept_k1", "contfrac_k1",
                "kept_k165", "contfrac_k165", "kept_k233", "contfrac_k233",
                "keptband_k0", "contfracband_k0",
                "keptband_k165", "contfracband_k165"])
    def kidx(kk):
        return int(np.argmin(np.abs(KS - kk)))
    for ch, p in sorted(per.items()):
        s = study[ch]
        mu0 = float(s["mu0_analytic"])
        row = [ch, f"{1e3*p['sigma']/mu0:.2f}"]
        for kk in (-1.0, 0.0, 1.0, 1.65, 2.33):
            i = kidx(kk)
            row += [f"{p['kept_1s'][i]:.4f}", f"{p['cont_1s'][i]:.4f}"]
        for kk in (0.0, 1.65):
            i = kidx(kk)
            row += [f"{p['kept_bk'][i]:.4f}", f"{p['cont_bk'][i]:.4f}"]
        w.writerow(row)

i0, i165 = int(np.argmin(np.abs(KS))), int(np.argmin(np.abs(KS - 1.65)))
print(f"trusted channels: {len(per)}")
print(f"k=0    : one-sided {rec_1s[i0]:.2f} MHz | band {rec_bk[i0]:.2f} MHz")
print(f"k=1.65 : one-sided {rec_1s[i165]:.2f} MHz | band {rec_bk[i165]:.2f} MHz")
worst = max(per, key=lambda c: per[c]["cont_1s"][i165])
print(f"worst one-sided contamination at k=1.65: ch{worst} "
      f"{100*per[worst]['cont_1s'][i165]:.1f}% of kept "
      f"(band-keep: {100*per[worst]['cont_bk'][i165]:.1f}%)")
print("wrote fig_aggressive_masking_tradeoff + aggressive_masking.csv")
