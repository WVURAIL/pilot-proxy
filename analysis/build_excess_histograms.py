#!/usr/bin/env python3
"""Valid-frame F distributions about the operating threshold, all 23
channels. This is the merged diagnostic (it absorbed the former
fig_f_histograms_all23, which plotted the identical variable).

x = pilot excess F/mu0 - 1 (the stored product variable, 1e-3 units); the
mask fires for x > 0. Kept region shaded blue, masked region orange; the
measured core mu_hat, the mu_hat +/- 12e-3 tail bounds, the gap Delta, and
the low/high tail fractions are annotated per panel. The sub-threshold
signal leak -- the part of the elevated tail the mask KEEPS -- is estimated
by mirroring the measured H0 core's lower half and subtracting:
leak = sum over (core, 0] of max(n(x) - n(2c - x), 0).
Writes the grid figure and subthreshold_leakage.csv (leak at tau = mu0 and
at tau = mu_hat; estimator grid fixed at BINW independent of display bins).
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
INK, C_SUP = "0.3", "#D55E00"
KEPT_C, MASK_C, HAT_C, C_TAIL = "#0072B2", "#D55E00", "#0072B2", "0.75"
SUPPRESSED = {14, 21, 25, 28, 36}
SPAN = 60.0
BINW = 0.4                      # 1e-3 units (leak-estimator grid)
DISPW = 0.2                     # display bin width

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})


def leak_estimate(x, c, tau):
    """Signal fraction kept below tau: mirror the sub-core half about c."""
    if tau <= c:
        return 0.0, 0.0
    edges = np.arange(-SPAN, SPAN + BINW, BINW)
    n, _ = np.histogram(x, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    # mirrored H0 model: value at x is n(2c - x), interpolated on the grid
    mirror = np.interp(2 * c - mids, mids, n, left=0, right=0)
    excess = np.clip(n - mirror, 0, None)
    sel_kept = (mids > c) & (mids <= tau)
    sel_all = mids > c
    leak = float(excess[sel_kept].sum())
    total = float(excess[sel_all].sum())
    return leak / max(x.size, 1), (leak / total if total > 0 else 0.0)


rows_out = []
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
    mu0 = float(s["mu0_analytic"])
    mu_hat = float(s["mu0_empirical"])
    trusted = s["zero_point_trusted"] == "1"
    x = 1e3 * (fv / mu0 - 1.0)          # pilot excess, tau at 0
    c = 1e3 * (mu_hat - mu0) / mu0      # measured core in excess units
    if ch == 30:
        lo, hi = np.percentile(x, [0.2, 99.8])
        pad = 0.06 * (hi - lo)
        bins = np.linspace(min(lo - pad, -20), hi + pad, 320)
    else:
        bins = np.arange(-SPAN, SPAN + DISPW, DISPW)
    cnt, edges = np.histogram(x, bins=bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    # no clip floor: empty bins break the trace / fills instead of a shelf
    floor1 = 1.0 / (max(x.size, 1) * float(edges[1] - edges[0]))
    bottom = 0.5 * floor1
    ax.semilogy(mids, np.where(cnt > 0, cnt, np.nan), color=INK, lw=0.7,
                drawstyle="steps-mid")
    kept_sel = mids <= 0
    ax.fill_between(mids[kept_sel], bottom,
                    np.where(cnt[kept_sel] > 0, cnt[kept_sel], bottom),
                    step="mid", color=KEPT_C, alpha=0.18)
    ax.fill_between(mids[~kept_sel], bottom,
                    np.where(cnt[~kept_sel] > 0, cnt[~kept_sel], bottom),
                    step="mid", color=MASK_C, alpha=0.14)
    ax.axvline(0.0, color="0.15", lw=1.0)
    kept_frac = float((x <= 0).mean())
    low, high = float(s["low_tail_frac"]), float(s["high_tail_frac"])
    if trusted:
        ax.axvline(c, color=HAT_C, ls="--", lw=0.9)
        for tb in (c - 12.0, c + 12.0):     # tail bounds about the core
            if abs(tb) < SPAN:
                ax.axvline(tb, color=C_TAIL, ls=":", lw=0.7)
        leak_mu0_f, leak_mu0_s = leak_estimate(x, c, 0.0)
        # leak at tau = mu_hat: threshold sits AT the measured core
        leak_hat_f, leak_hat_s = leak_estimate(x, c, c)
        ax.text(0.03, 0.76,
                f"sub-$\\tau$ signal {100*leak_mu0_f:.1f}{PCT} of frames "
                f"({100*leak_mu0_s:.0f}{PCT} of tail)",
                transform=ax.transAxes, fontsize=5.8)
    else:
        leak_mu0_f = leak_mu0_s = leak_hat_f = leak_hat_s = float("nan")
        ax.text(0.03, 0.76, "core untrusted", transform=ax.transAxes,
                fontsize=6, color="0.4")
    ax.text(0.03, 0.92,
            f"$\\Delta$={c:+.1f}  kept {100*kept_frac:.1f}{PCT}",
            transform=ax.transAxes, fontsize=6.2,
            color=C_SUP if ch in SUPPRESSED else "black")
    ax.text(0.03, 0.84, f"low {100*low:.1f}{PCT} / high {100*high:.1f}{PCT}",
            transform=ax.transAxes, fontsize=6)
    tcol = C_SUP if ch in SUPPRESSED else "black"
    ax.set_title(f"ch{ch} (fid {s['freq_id']})", fontsize=7.5, color=tcol,
                 pad=2)
    ax.tick_params(labelsize=5.5)
    ax.set_ylim(bottom=bottom)         # half a single-frame density
    ax.grid(color="0.94", lw=0.35)
    ax.set_axisbelow(True)
    rows_out.append({
        "atsc_channel": ch, "freq_id": s["freq_id"],
        "kept_frac_at_tau_mu0": f"{kept_frac:.4f}",
        "core_excess_1e3": f"{c:+.2f}",
        "subthreshold_signal_frac_of_valid_tau_mu0":
            f"{leak_mu0_f:.4f}" if trusted else "",
        "subthreshold_signal_frac_of_tail_tau_mu0":
            f"{leak_mu0_s:.4f}" if trusted else "",
        "subthreshold_signal_frac_of_valid_tau_muhat":
            f"{leak_hat_f:.4f}" if trusted else "",
        "zero_point_trusted": int(trusted),
    })
for j in range(len(chans), 24):
    axes.flat[j].axis("off")
for ax in axes[-1, :]:
    ax.set_xlabel(r"pilot excess $F/\mu_0-1\ [10^{-3}]$", fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("density", fontsize=7)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[
    Line2D([], [], color="0.15", lw=1.2, label=r"threshold $\tau=\mu_0$ (excess 0)"),
    Line2D([], [], color=HAT_C, ls="--", label=r"measured core $\hat{\mu}_0$"),
    Line2D([], [], color=C_TAIL, ls=":",
           label=r"$\hat{\mu}_0\pm12\times10^{-3}$ tail bounds"),
    Patch(facecolor=KEPT_C, alpha=0.18, label="kept (science data)"),
    Patch(facecolor=MASK_C, alpha=0.14, label="masked"),
], loc="lower center", ncol=5, fontsize=8, frameon=False,
    bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Valid-frame F distributions about the operating threshold, "
             "all 23 channels (kept vs masked; sub-threshold signal leak "
             "from mirrored-core estimate)", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.99))
fig.savefig(OUT / "fig_excess_threshold_all23.png", dpi=230,
            bbox_inches="tight")
fig.savefig(OUT / "fig_excess_threshold_all23.pdf", bbox_inches="tight")

with open(OUT / "subthreshold_leakage.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print("wrote fig_excess_threshold_all23 + subthreshold_leakage.csv")
for r in rows_out:
    if r["subthreshold_signal_frac_of_valid_tau_mu0"]:
        v = float(r["subthreshold_signal_frac_of_valid_tau_mu0"])
        t = float(r["subthreshold_signal_frac_of_tail_tau_mu0"])
        if v > 0.005:
            print(f"  ch{r['atsc_channel']}: sub-threshold signal "
                  f"{100*v:.1f}% of frames ({100*t:.0f}% of its tail)")
