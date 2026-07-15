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
COLORS = {0.0: "#0072B2", -1000.0: "#D55E00", 1000.0: "#009E73"}
MARKS = {0.0: "o", -1000.0: "v", 1000.0: "^"}

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

# ---- figure -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for off, ro in sorted(thr.items()):
    snr = np.array([x[0] for x in ro])
    pd = np.array([x[1] for x in ro])
    lo = np.array([x[2] for x in ro])
    hi = np.array([x[3] for x in ro])
    lbl = "0 Hz" if off == 0 else f"{off/1000:+.0f} kHz"
    ax.errorbar(snr, pd, yerr=[pd - lo, hi - pd], fmt=MARKS[off] + "-",
                ms=4.5, lw=1.3, capsize=2, color=COLORS[off],
                label=f"threshold, {lbl}")
if pex:
    for off, ro in sorted(pex.items()):
        snr = np.array([x[0] for x in ro])
        pd = np.array([x[1] for x in ro])
        lo = np.array([x[2] for x in ro])
        hi = np.array([x[3] for x in ro])
        lbl = "0 Hz" if off == 0 else f"{off/1000:+.0f} kHz"
        ax.errorbar(snr, pd, yerr=[pd - lo, hi - pd], fmt=MARKS[off] + "--",
                    ms=4.5, lw=1.0, capsize=2, color=COLORS[off],
                    markerfacecolor="none", alpha=0.85,
                    label=f"pos-excess, {lbl}")
ax.axvline(-32.0, color="0.35", ls="--", lw=0.9)
ax.text(-32.05, 0.03, "science threshold $-32$ dB", rotation=90, fontsize=8,
        va="bottom", ha="right", color="0.35")
for lev in (0.5, 0.9):
    ax.axhline(lev, color="0.8", ls=":", lw=0.8)
v = v_thr.get(0.0)
if v and np.isfinite(v["p90"]):
    ax.annotate(
        f"$P_d$=0.5 at {v['p50']:.2f} dB\n$P_d$=0.9 at {v['p90']:.2f} dB",
        xy=(v["p90"], 0.9), xytext=(-38, 0.72), fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
ax.set_xlabel("requested shelf SNR [dB]")
ax.set_ylabel(r"detection probability $P_d$")
ax.set_title(f"Synthetic detection curves, exact-integer CPU reference "
             f"({label}; Wilson 95%)", fontsize=10.5)
ax.set_ylim(-0.02, 1.04)
ax.legend(fontsize=7.5, loc="upper left", ncol=2 if pex else 1)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig3_detection_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_detection_curves.pdf", bbox_inches="tight")
print("\nwrote fig3_detection_curves (publication)")
