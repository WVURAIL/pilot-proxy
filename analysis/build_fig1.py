#!/usr/bin/env python3
"""Fig. 1 draft: transmitter-census context for the ATSC 14-36 survey band."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
C_FULL, C_RELAY, C_LPTV = "#0072B2", "#009E73", "#D55E00"

rows = list(csv.DictReader(open(
    str(_paths.REPO / "data/census/census.csv"))))


def group(cls):
    if cls == "Full-power":
        return "full"
    if cls in ("Relay", "Class A"):
        return "relay"
    return "lptv"


counts = defaultdict(lambda: defaultdict(int))
det = defaultdict(list)
n_nodet = 0
for r in rows:
    ch = int(r["rf_channel"])
    g = group(r["service_class"])
    counts[ch][g] += 1
    if r["detectability_db"].strip():
        det[ch].append((float(r["detectability_db"]), g,
                        float(r["distance_km"] or "nan")))
    else:
        n_nodet += 1
print(f"{n_nodet} stations without detectability_db (counted, not plotted)")

chans = list(range(14, 37))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True,
                               gridspec_kw={"hspace": 0.24,
                                            "height_ratios": [1, 1.15]})
full = np.array([counts[c]["full"] for c in chans])
relay = np.array([counts[c]["relay"] for c in chans])
lptv = np.array([counts[c]["lptv"] for c in chans])
ax1.bar(chans, full, color=C_FULL,
        label="full-power ($\\pm$1 kHz tolerance)", width=0.72)
ax1.bar(chans, relay, bottom=full, color=C_RELAY,
        label="relay / Class A ($\\pm$1 kHz)", width=0.72)
ax1.bar(chans, lptv, bottom=full + relay, color=C_LPTV,
        label="translator / LPTV (no tolerance recorded)", width=0.72)
ax1.set_ylabel("transmitters within 500 mi")
ax1.set_ylim(0, 33)
ax1.legend(fontsize=7.2, loc="upper center", ncol=3,
           frameon=False)
ax1.grid(axis="y", color="0.92", lw=0.6)
ax1.set_axisbelow(True)
ax1.set_title("UHF DTV transmitter census within 500 mi of DRAO "
              f"({len(rows)} stations, merged emitters)", fontsize=10.5)

# ---- panel (b): measured carrier offsets vs the detector cells ----------
# The case-study question: are the off-nominal carriers consistent with
# loosely-toleranced transmitter classes, and on which channels?
FB = 3051.7578125
TGT_FILL, GUARD_FILL, REF_FILL = "#D55E00", "0.55", "#7B4FA6"
lines = defaultdict(list)
for r in csv.DictReader(open(_paths.RESULTS /
                             "transmitter_census/extracted_lines.csv")):
    lines[int(r["rf_channel"])].append((float(r["offset_hz"]) / FB,
                                        float(r["snr_db"])))
for lo, hi, col, al in ((-0.5, 0.5, TGT_FILL, 0.16),
                        (-1.5, -0.5, GUARD_FILL, 0.16),
                        (0.5, 1.5, GUARD_FILL, 0.16),
                        (-2.5, -1.5, REF_FILL, 0.20),
                        (1.5, 2.5, REF_FILL, 0.20)):
    ax2.axhspan(lo, hi, color=col, alpha=al, lw=0, zorder=0)
for c in chans:
    for off_bins, snr in lines.get(c, []):
        ax2.plot(c, off_bins, marker="o",
                 ms=max(2.2, 0.16 * snr + 1.2), color="0.15", mew=0,
                 alpha=0.85)
ax2.annotate("ch33: only carriers in the skipped guard", xy=(33, -1.18),
             xytext=(19.0, -2.15), fontsize=7,
             arrowprops=dict(arrowstyle="-", lw=0.7))
ax2.annotate("ch30 drift picket", xy=(30, -1.25), xytext=(33.6, -2.15),
             fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.7))
ax2.set_ylim(-2.6, 2.6)
ax2.set_xlabel("ATSC physical channel")
ax2.set_ylabel("extracted-line offset [fine bins]")
ax2.set_xticks(chans)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="x", color="0.94", lw=0.5)
ax2.set_axisbelow(True)
ax2.set_title("extracted carriers vs detector cells (marker size $\\propto$ "
              "integrated SNR;\ntarget / guard / reference bands shaded)",
              fontsize=9, pad=4)

fig.tight_layout()
fig.savefig(OUT / "fig1_census_context.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_census_context.pdf", bbox_inches="tight")
n_lptv33 = counts[33]["lptv"] + counts[33]["relay"]
print(f"wrote fig1_census_context (ch33 loose-tolerance candidates within "
      f"500 mi: {n_lptv33})")
