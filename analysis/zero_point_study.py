#!/usr/bin/env python3
"""Empirical per-channel zero points from full-depth per-frame statistics.

Estimator: asymmetric sigma-clip. Signal only ever RAISES F, so iterate:
robust center (median) + lower-side scale (MAD of the sub-median half),
clip F > m + 4*s_lower, repeat to convergence. The survivors estimate the
H0 core; channels where clipping removes most frames (signal-dominated)
are flagged rather than trusted.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
DRAO_LON_H = -119.6175 / 15.0          # solar-time offset from UTC (hours)
COARSE_MHZ = 400.0 / 1024.0
C_NOM, C_SUP, C_ELE = "#0072B2", "#D55E00", "#009E73"

z = np.load(_paths.PERFRAME)
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})


def core_zero(f, mu0):
    """Mode-anchored H0 core: locate the distribution mode, then iterate the
    median inside a +/-6e-3*mu0 window (~2.5 sigma_H0). Robust to BOTH the
    signal high tail and the reference-contamination low tail."""
    lo, hi = np.percentile(f, [0.5, 99.5])
    hist, edges = np.histogram(f, bins=512, range=(lo, hi))
    m = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    win = 6e-3 * mu0
    for _ in range(20):
        w = f[np.abs(f - m) <= win]
        if w.size < 100:
            break
        m_new = float(np.median(w))
        if abs(m_new - m) < 1e-7:
            m = m_new
            break
        m = m_new
    w = f[np.abs(f - m) <= win]
    mu_hat = float(w.mean())
    err = float(w.std(ddof=1) / np.sqrt(w.size)) if w.size > 1 else float("nan")
    window_frac = w.size / f.size
    tail = 12e-3 * mu0            # ~5 sigma_H0 from the core
    low_frac = float((f < mu_hat - tail).mean())
    high_frac = float((f > mu_hat + tail).mean())
    return mu_hat, err, window_frac, low_frac, high_frac


def block_err(fv, fui_v, mu_hat, mu0, n_boot=400, seed=11):
    """Per-event (unit) block-bootstrap error on the core-window mean.

    Frames within a capture unit share conditions, so the naive SEM
    understates the zero-point uncertainty; resampling whole units is the
    honest interval."""
    win = 6e-3 * mu0
    sel = np.abs(fv - mu_hat) <= win
    fw, uw = fv[sel], fui_v[sel]
    if fw.size < 2:
        return float("nan")
    order = np.argsort(uw, kind="stable")
    fw_s, uw_s = fw[order], uw[order]
    units, starts = np.unique(uw_s, return_index=True)
    if units.size < 2:
        return float("nan")
    sums = np.add.reduceat(fw_s, starts)
    cnts = np.diff(np.append(starts, fw_s.size))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, units.size, units.size)
        means[b] = sums[pick].sum() / max(cnts[pick].sum(), 1)
    return float(np.std(means, ddof=1))


rows = []
frames_by_class_hour = {}
for ch in chans:
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    rej = z[f"ch{ch}_reject_mask"].astype(bool)
    fui = z[f"ch{ch}_frame_unit_index"].astype(int)
    t0 = z[f"ch{ch}_unit_time0_ctime"]
    mu0, tns, rnss, fid, pilot, center = z[f"ch{ch}_scalars"]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    okf = valid & np.isfinite(f)
    fv = f[okf]
    fui_v = fui[okf]
    mu_hat, mu_err, window_frac, low_frac, high_frac = core_zero(fv, mu0)
    mu_err_blk = block_err(fv, fui_v, mu_hat, mu0)
    mf_now = float(rej[valid].mean())
    # distrust the empirical zero point when the H0 core is not identifiable
    signal_dom = (mf_now > 0.9 or window_frac < 0.15
                  or abs(mu_hat - mu0) / mu0 > 20e-3)
    mf_hat = float((fv > mu_hat).mean())
    veto = 12e-3 * mu0
    kept_2sided = float(((fv <= mu_hat) & (fv >= mu_hat - veto)).mean())
    # solar-hour mask fractions (diurnal diagnostic)
    tf = t0[fui] + 0.0
    ok = valid & np.isfinite(tf)
    hours = ((tf[ok] / 3600.0 + DRAO_LON_H) % 24.0).astype(int)
    mask_ok = rej[ok]
    prof = np.full(24, np.nan)
    for h in range(24):
        selh = hours == h
        if selh.sum() > 50:
            prof[h] = mask_ok[selh].mean()
    rows.append(dict(
        ch=ch, fid=int(fid), n_valid=int(fv.size),
        mu0=float(mu0), mu_hat=mu_hat, mu_err=mu_err,
        mu_err_blk=mu_err_blk,
        gap_1e3=1e3 * (mu_hat - mu0) / mu0,
        window_frac=window_frac, low_frac=low_frac, high_frac=high_frac,
        signal_dominated=signal_dom,
        mask_frac_manifest=mf_now, mask_frac_empirical=mf_hat,
        kept_manifest=1 - mf_now, kept_empirical=1 - mf_hat,
        kept_2sided=kept_2sided,
        diurnal=prof,
    ))

# recovered bandwidth: manifest vs empirical vs empirical+low-veto
def _pick(r, key):
    return r["kept_manifest"] if r["signal_dominated"] else r[key]

rec_man = sum(r["kept_manifest"] * COARSE_MHZ for r in rows)
rec_emp = sum(_pick(r, "kept_empirical") * COARSE_MHZ for r in rows)
rec_2s = sum(_pick(r, "kept_2sided") * COARSE_MHZ for r in rows)
total = len(rows) * COARSE_MHZ

with open(OUT / "empirical_zero_points.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "n_valid", "mu0_analytic",
                "mu0_empirical", "mu0_empirical_err", "gap_1e3",
                "core_window_frac", "low_tail_frac", "high_tail_frac",
                "zero_point_trusted",
                "mask_frac_analytic_tau", "mask_frac_empirical_tau",
                "kept_frac_2sided_veto", "mu0_empirical_err_block"])
    for r in rows:
        w.writerow([r["ch"], r["fid"], r["n_valid"], f"{r['mu0']:.6f}",
                    f"{r['mu_hat']:.6f}", f"{r['mu_err']:.6f}",
                    f"{r['gap_1e3']:+.3f}", f"{r['window_frac']:.3f}",
                    f"{r['low_frac']:.4f}", f"{r['high_frac']:.4f}",
                    int(not r["signal_dominated"]),
                    f"{r['mask_frac_manifest']:.4f}",
                    f"{r['mask_frac_empirical']:.4f}",
                    f"{r['kept_2sided']:.4f}",
                    f"{r['mu_err_blk']:.6f}"])

print(f"{'ch':>3} {'n':>6} {'mu0':>8} {'mu_hat':>8} {'gap(1e-3)':>9} "
      f"{'mf@mu0':>7} {'mf@hat':>7} {'low%':>6} {'high%':>6} {'trust':>5}")
for r in rows:
    print(f"{r['ch']:>3} {r['n_valid']:>6} {r['mu0']:>8.5f} "
          f"{r['mu_hat']:>8.5f} {r['gap_1e3']:>+9.2f} "
          f"{r['mask_frac_manifest']:>7.3f} {r['mask_frac_empirical']:>7.3f} "
          f"{100*r['low_frac']:>5.1f}% {100*r['high_frac']:>5.1f}% "
          f"{'' if r['signal_dominated'] else 'Y':>5}")
print(f"\nrecovered MHz, tau = manifest mu0        : {rec_man:.3f} / {total:.3f}")
print(f"recovered MHz, tau = empirical mu0_hat   : {rec_emp:.3f} / {total:.3f}")
print(f"recovered MHz, empirical + low-side veto : {rec_2s:.3f} / {total:.3f}")

# ---- figure: gaps + mask-fraction correction ---------------------------------
SUPPRESSED = {14, 21, 25, 28, 36}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True,
                               gridspec_kw={"hspace": 0.09})
YLIM = 16.0
for r in rows:
    c = (C_SUP if r["ch"] in SUPPRESSED else
         C_ELE if r["signal_dominated"] or r["gap_1e3"] > 3 else C_NOM)
    g = r["gap_1e3"]
    if r["signal_dominated"]:
        # no identifiable H0 core: the empirical gap is not a calibration
        ax1.annotate(f"ch{r['ch']}: no null core",
                     xy=(r["ch"], YLIM * 0.95), xytext=(r["ch"], YLIM * 0.62),
                     ha="center", fontsize=7, color=c,
                     arrowprops=dict(arrowstyle="->", color=c, lw=1))
        continue
    if abs(g) > YLIM:
        ax1.annotate(f"ch{r['ch']}: ${g:+.0f}$",
                     xy=(r["ch"], YLIM * 0.95), xytext=(r["ch"], YLIM * 0.62),
                     ha="center", fontsize=7, color=c,
                     arrowprops=dict(arrowstyle="->", color=c, lw=1))
        continue
    yerr = r["mu_err_blk"] if np.isfinite(r["mu_err_blk"]) else r["mu_err"]
    ax1.errorbar(r["ch"], g, yerr=1e3 * yerr / r["mu0"],
                 fmt="o", ms=4.5, color=c, capsize=2, lw=1)
ax1.axhline(0, color="0.4", lw=0.8)
ax1.set_ylim(-YLIM, YLIM)
ax1.set_ylabel(r"$(\hat{\mu}_0-\mu_0)/\mu_0\ [10^{-3}]$")
ax1.set_title("Measured vs analytic zero point (mode-anchored H0 core, "
              "full depth; per-event block-bootstrap errors)", fontsize=10)
ax1.grid(axis="y", color="0.92", lw=0.6)
ax1.set_axisbelow(True)
w = 0.36
for r in rows:
    c = (C_SUP if r["ch"] in SUPPRESSED else
         C_ELE if r["signal_dominated"] or r["gap_1e3"] > 3 else C_NOM)
    ax2.bar(r["ch"] - w / 2, r["mask_frac_manifest"], width=w, color=c,
            alpha=0.45)
    ax2.bar(r["ch"] + w / 2, r["mask_frac_empirical"], width=w, color=c)
ax2.axhline(0.5, color="0.25", ls="--", lw=0.9)
ax2.set_ylabel(r"mask fraction (light: $\tau=\mu_0$; dark: $\tau=\hat{\mu}_0$)")
ax2.set_xlabel("ATSC physical channel")
ax2.set_xticks([r["ch"] for r in rows])
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="y", color="0.92", lw=0.6)
ax2.set_axisbelow(True)
from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_SUP, label="suppressed family"),
    Line2D([], [], marker="o", ls="", color=C_ELE, label="signal-elevated"),
    Line2D([], [], marker="o", ls="", mfc="none", color=C_ELE,
           label="no null-core calibration (arrows)"),
    Line2D([], [], marker="o", ls="", color=C_NOM, label="nominal")],
    fontsize=7.5, loc="upper left")
fig.savefig(OUT / "fig_empirical_zero_points.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_empirical_zero_points.pdf", bbox_inches="tight")
plt.close(fig)

# ---- diurnal diagnostic -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
for r in rows:
    if r["ch"] in (30, 24, 17):
        ax.plot(range(24), r["diurnal"], "-o", ms=3, lw=1.2,
                color=C_ELE, alpha=0.9)
        ax.annotate(f"ch{r['ch']}", xy=(23.2, r["diurnal"][~np.isnan(r['diurnal'])][-1]
                    if np.any(~np.isnan(r["diurnal"])) else 0.5),
                    fontsize=7, color=C_ELE)
    elif r["ch"] in SUPPRESSED:
        ax.plot(range(24), r["diurnal"], "-", lw=1.0, color=C_SUP, alpha=0.8)
    else:
        ax.plot(range(24), r["diurnal"], "-", lw=0.7, color=C_NOM, alpha=0.35)
ax.axhline(0.5, color="0.3", ls="--", lw=0.8)
ax.set_xlabel("local solar hour at DRAO")
ax.set_ylabel(r"mask fraction at $\tau=\mu_0$")
ax.set_title("Diurnal mask-fraction profiles (signal channels swing; "
             "H0-like channels stay flat)", fontsize=10)
ax.set_xticks(range(0, 24, 3))
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_diurnal_mask_fraction.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("\nwrote empirical_zero_points.csv + figures")
