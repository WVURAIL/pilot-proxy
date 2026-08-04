#!/usr/bin/env python3
"""Publication Fig. 3 + item-2 acceptance from a merged sweep summary CSV.

Usage: python3 fig3_publication.py <dtv_snr_summary.csv> [label]

Plots the threshold rule (filled markers, solid) and, when the columns are
present, the positive-excess rule (open markers, dashed). Wilson intervals
are taken from *_wilson95_lo/hi columns when available, else computed from
the rate and the trials column.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
BIN_HZ = 390625.0 / 128.0
COLORS = {0.0: "#0072B2", -1000.0: "#D55E00", 1000.0: "#009E73",
          -1500.0: "#A85C85", 1500.0: "#E69F00"}
MARKS = {0.0: "o", -1000.0: "v", 1000.0: "^",
         -1500.0: "<", 1500.0: ">"}

csv_path = sys.argv[1] if len(sys.argv) > 1 else str(_paths.SWEEP_CSV)
label = sys.argv[2] if len(sys.argv) > 2 else "1000 trials/point"
rows = list(csv.DictReader(open(csv_path)))
if not rows:
    raise SystemExit("empty CSV")
cols = rows[0].keys()


def trials_of(r):
    for k in ("noise_trials", "n_trials", "trials", "num_trials"):
        if k in r and r[k]:
            return int(float(r[k]))
    return None


def wilson(p, n, z=1.959963984540054):
    if n is None or n <= 0:
        return p, p
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - hw), min(1.0, ctr + hw)


def series(rule):
    """rule -> {offset: sorted [(snr, pd, lo, hi)]} or None if absent."""
    rate_k = f"{rule}_detection_rate"
    if rate_k not in cols:
        return None
    lo_k, hi_k = f"{rate_k}_wilson95_lo", f"{rate_k}_wilson95_hi"
    have_ci = lo_k in cols and hi_k in cols
    data = {}
    for r in rows:
        if not r.get(rate_k):
            continue
        off = float(r["frequency_offset_hz"])
        pd = float(r[rate_k])
        if have_ci and r.get(lo_k):
            lo, hi = float(r[lo_k]), float(r[hi_k])
        else:
            lo, hi = wilson(pd, trials_of(r))
        data.setdefault(off, []).append(
            (float(r["requested_snr_shelf_db"]), pd, lo, hi))
    for off in data:
        data[off].sort()
    return data


def crossing(snr, pd, level):
    for i in range(len(pd) - 1):
        if pd[i] < level <= pd[i + 1]:
            f = (level - pd[i]) / (pd[i + 1] - pd[i])
            return snr[i] + f * (snr[i + 1] - snr[i])
    return float("nan")


def acceptance(name, data):
    print(f"=== {name} rule ===")
    verdicts = {}
    for off, ro in sorted(data.items()):
        snr = np.array([x[0] for x in ro])
        pd = np.array([x[1] for x in ro])
        lo = np.array([x[2] for x in ro])
        hi = np.array([x[3] for x in ro])
        viol = [(snr[i], snr[i + 1]) for i in range(len(pd) - 1)
                if pd[i + 1] < pd[i] and hi[i + 1] < lo[i]]
        trans = (pd > 0.2) & (pd < 0.8)
        halfw = 100 * (hi - lo) / 2
        p50, p90 = crossing(snr, pd, 0.5), crossing(snr, pd, 0.9)
        verdicts[off] = dict(p50=p50, p90=p90)
        hw = (f"mean {halfw[trans].mean():.1f}% max {halfw[trans].max():.1f}%"
              if trans.any() else "n/a")
        print(f"  offset {off:+6.0f} Hz: monotone="
              f"{'PASS' if not viol else viol}  "
              f"P50 {p50:+.2f} dB  P90 {p90:+.2f} dB  "
              f"Wilson half-width in transition: {hw}")
    if 0.0 in verdicts and -1000.0 in verdicts and 1000.0 in verdicts:
        pred = -10 * np.log10(np.sinc(1000.0 / BIN_HZ) ** 2)
        s50 = [verdicts[o]["p50"] - verdicts[0.0]["p50"]
               for o in (-1000.0, 1000.0)]
        print(f"  capture loss: sinc^2 predicts +{pred:.2f} dB at +/-1 kHz; "
              f"measured P50 shifts {s50[0]:+.2f}, {s50[1]:+.2f} dB")
    return verdicts


thr = series("threshold")
pex = series("positive_excess")
if thr is None:
    raise SystemExit("no threshold_detection_rate column found")
v_thr = acceptance("threshold", thr)
v_pex = acceptance("positive-excess", pex) if pex else None
if pex is None:
    print("(no positive_excess columns in this CSV)")


def means():
    """{offset: sorted [(snr_db, mean_excess)]} from fstat_raw_mean."""
    if "fstat_raw_mean" not in cols:
        return None
    MU0_CH14 = 1.0020553
    data = {}
    for r in rows:
        if not r.get("fstat_raw_mean"):
            continue
        off = float(r["frequency_offset_hz"])
        exc = float(r["fstat_raw_mean"]) / MU0_CH14 - 1.0
        data.setdefault(off, []).append(
            (float(r["requested_snr_shelf_db"]), exc))
    for off in data:
        data[off].sort()
    return data


# Deterministic benchmark: F ~ mu0 * ncF(2R, 4R, lambda), lambda = 2 R C s_eff
R_TRIAL, C_ALLOC = 512, 145.7475
def bench_mean_excess(s_lin, off_hz):
    s_eff = s_lin * np.sinc(off_hz / BIN_HZ) ** 2
    d1, d2 = 2 * R_TRIAL, 4 * R_TRIAL
    lam = d1 * C_ALLOC * s_eff
    return (d2 * (d1 + lam)) / (d1 * (d2 - 2)) - 1.0


def bench_pd(s_lin, off_hz):
    from scipy.stats import ncx2, f as fdist
    s_eff = s_lin * np.sinc(off_hz / BIN_HZ) ** 2
    d1, d2 = 2 * R_TRIAL, 4 * R_TRIAL
    lam = d1 * C_ALLOC * s_eff
    # P(F_raw > mu0) with F_raw = mu0 * ncF: P(ncF(d1,d2,lam) > 1)
    from scipy.stats import ncf
    return float(ncf.sf(1.0, d1, d2, lam))

# ---- figure -------------------------------------------------------------------
fig, (axm, ax) = plt.subplots(2, 1, figsize=(7.6, 8.2), sharex=True,
                              gridspec_kw={"height_ratios": [1.0, 1.25],
                                           "hspace": 0.14})
# ---- panel (a): mean pilot excess vs shelf SNR (the deterministic meter) ----
mns = means()
sg = np.logspace(-6.0, -2.0, 300)
if mns:
    for off, ro in sorted(mns.items()):
        snr = np.array([x[0] for x in ro])
        exc = np.array([x[1] for x in ro])
        good = exc > 0
        axm.semilogy(snr[good], exc[good], MARKS.get(off, "s"), ms=4.2,
                     color=COLORS.get(off, "0.4"), mfc="none",
                     label=f"measured mean, {'0 Hz' if off==0 else f'{off/1000:+.1f} kHz'}")
    for off in sorted(mns):
        axm.semilogy(10*np.log10(sg), bench_mean_excess(sg, off), "-",
                     lw=0.9, color=COLORS.get(off, "0.4"), alpha=0.65)
axm.set_ylabel(r"mean pilot excess $\langle F\rangle/\mu_0 - 1$")
axm.set_title("(a) the deterministic meter: measured mean excess vs the "
              "ncF benchmark (lines)", fontsize=9.5)
axm.grid(color="0.92", lw=0.5)
axm.set_axisbelow(True)
axm.legend(fontsize=7, ncol=2, loc="upper left")
# reduced-variable top axis on panel (a): k = C s / sigma (512-row trials)
SIG_TRIAL = float(np.sqrt(1/R_TRIAL + 1/(2*R_TRIAL)))
def _snr_to_k(x):
    return C_ALLOC * 10**(np.asarray(x)/10.0) / SIG_TRIAL
def _k_to_snr(k):
    return 10*np.log10(np.maximum(np.asarray(k), 1e-12) * SIG_TRIAL / C_ALLOC)
secax = axm.secondary_xaxis("top", functions=(_snr_to_k, _k_to_snr))
secax.set_xscale("log")
secax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10])
secax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10"], fontsize=7)
secax.set_xlabel("mean shift [null widths]  (512-row trials)", fontsize=8)
for a_ in (axm, ax):
    a_.axvline(_k_to_snr(1.0), color="0.7", ls="--", lw=0.8)
ax.text(_k_to_snr(1.0) - 0.35, 0.06,
        "width crossing ($k{=}1$, $-34.3$ dB)",
        rotation=90, fontsize=7, color="0.4", va="bottom", ha="right")

# Primary: the deployed positive-excess rule (operational cleaning).
if pex:
    for off, ro in sorted(pex.items()):
        snr = np.array([x[0] for x in ro])
        pd = np.array([x[1] for x in ro])
        lo = np.array([x[2] for x in ro])
        hi = np.array([x[3] for x in ro])
        lbl = "0 Hz" if off == 0 else f"{off/1000:+.0f} kHz"
        ax.errorbar(snr, pd, yerr=[pd - lo, hi - pd], fmt=MARKS[off] + "-",
                    ms=4.5, lw=1.4, capsize=2, color=COLORS[off],
                    label=f"pos-excess (deployed), {lbl}")
for off in (0.0,):
    ax.plot(10*np.log10(sg), [bench_pd(x, off) for x in sg], "-",
            lw=0.9, color="0.55", alpha=0.9,
            label="ncF benchmark (0 Hz)")
# Fixed-threshold backup curves intentionally NOT drawn: the paper
# deploys the positive-excess rule only (backup mode documented in
# text; its sweep columns remain in the archived summary CSV).
ax.set_xlim(-60, -20)
ax.text(-59, 0.55, "2048-input deployment curves land here\n"
        "(GPU regeneration in progress)", fontsize=7.5,
        color="0.45", va="center")
for lev in (0.5, 0.9):
    ax.axhline(lev, color="0.8", ls=":", lw=0.8)
vp = v_pex.get(0.0) if v_pex else None
if vp and np.isfinite(vp["p90"]):
    ax.annotate(
        f"deployed rule: $P_d$=0.9 at $\\approx${vp['p90']:.1f} dB",
        xy=(vp["p90"], 0.9), xytext=(-38.3, 0.70), fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
ax.set_xlabel("requested shelf SNR [dB]")
ax.set_ylabel(r"detection probability $P_d$")
ax.set_title("(b) the derived consequence: positive-excess detection rate "
             "vs the same benchmark", fontsize=9.5)
fig.suptitle(f"Model validation of the deterministic chain, exact-integer "
             f"CPU reference ({label}; Wilson 95 per cent)", fontsize=10.5,
             y=0.995)
ax.set_ylim(-0.02, 1.04)
ax.legend(fontsize=7.5, loc="upper left", ncol=2 if pex else 1)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
ax.text(0.99, 0.015,
        "provisional: 4-stream (512-row) trials --- deployment-scale pending",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="0.45", style="italic")
fig.tight_layout()
fig.savefig(OUT / "fig3_detection_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_detection_curves.pdf", bbox_inches="tight")
print("\nwrote fig3_detection_curves (publication)")
