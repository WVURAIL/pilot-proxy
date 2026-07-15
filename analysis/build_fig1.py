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
                               gridspec_kw={"hspace": 0.08,
                                            "height_ratios": [1, 1.15]})
full = np.array([counts[c]["full"] for c in chans])
relay = np.array([counts[c]["relay"] for c in chans])
lptv = np.array([counts[c]["lptv"] for c in chans])
ax1.bar(chans, full, color=C_FULL, label="full-power", width=0.72)
ax1.bar(chans, relay, bottom=full, color=C_RELAY, label="relay / Class A",
        width=0.72)
ax1.bar(chans, lptv, bottom=full + relay, color=C_LPTV,
        label="translator / LPTV", width=0.72)
ax1.set_ylabel("transmitters within 500 mi")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="y", color="0.92", lw=0.6)
ax1.set_axisbelow(True)
ax1.set_title("UHF DTV transmitter census within 500 mi of DRAO "
              f"({len(rows)} stations, merged emitters)", fontsize=10.5)

rng = np.random.default_rng(20260715)
for c in chans:
    for d, g, dist in det[c]:
        col = {"full": C_FULL, "relay": C_RELAY, "lptv": C_LPTV}[g]
        ax2.plot(c + rng.uniform(-0.22, 0.22), d, marker="o", ms=2.6,
                 color=col, alpha=0.55, mew=0)
    if det[c]:
        dmax, gmax, distmax = max(det[c])
        ax2.plot(c, dmax, marker="_", ms=13, color="0.15", mew=1.6)
for c, lbl in ((30, "CHKL-1\n17.7 km"),):
    dmax = max(det[c])[0]
    ax2.annotate(lbl, xy=(c, dmax), xytext=(c + 0.7, dmax - 6),
                 fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.7))
ax2.set_xlabel("ATSC physical channel")
ax2.set_ylabel("detectability [dB]")
ax2.set_xticks(chans)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(axis="y", color="0.92", lw=0.6)
ax2.set_axisbelow(True)
fig.savefig(OUT / "fig1_census_context.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_census_context.pdf", bbox_inches="tight")
print("wrote fig1_census_context")
