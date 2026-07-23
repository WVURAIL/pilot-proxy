#!/usr/bin/env python3
"""figS_ch26_interior: quarterly hi/low-tail rates for ch26 from the
survey-extension product (fid 660), against the SNAPSHOT calibration.
Mirrors figS_ch27_interior.

Input:  $PP_OUT/ch26_extension_quarterly.csv (default ~/paper/out)
Output: figS_ch26_interior.pdf beside this script."""
import csv, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from pilot_proxy.plot_style import setup_matplotlib
plt = setup_matplotlib()

OUT = Path(os.environ.get("PP_OUT", "~/paper/out")).expanduser()
rows = list(csv.DictReader(open(OUT / "ch26_extension_quarterly.csv")))
q  = [float(r["quarter"]) for r in rows]
hi = [100*float(r["hi_rate"]) for r in rows]
lo = [100*float(r["lo_rate"]) for r in rows]

fig, ax = plt.subplots(figsize=(7.4, 3.1))
ax.axvspan(2021.0, 2022.72, color="0.92", zorder=0)
ax.text(2021.86, 6.4, "archive\ncoverage hole", ha="center", va="center",
        fontsize=7.5, color="0.45")
ax.plot(q, hi, marker="o", ms=4, lw=1.2, color="#D55E00",
        label="hi-rate (target-side)")
ax.plot(q, lo, marker="s", ms=4, lw=1.2, color="#0072B2",
        label="low-tail (reference-side)")
ax.axvline(2023.25, color="0.6", lw=0.8, ls=":")
ax.text(2023.30, 10.0, "2023 Q2:\nboth tails collapse\n(ch32 falls same window)",
        fontsize=7, color="0.35", va="top")
ax.set_xlabel("year (quarters with $\\geq 40$ valid frames)")
ax.set_ylabel("rate [per cent]")
ax.set_ylim(-0.4, 11.5)
ax.set_title("ch26 at full depth: weak two-sided flicker 2018--2023 Q1, "
             "extinguished 2023 Q2, class-sensitive uptick 2026 Q2",
             fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(HERE / "figS_ch26_interior.pdf", bbox_inches="tight")
print("wrote figS_ch26_interior")
