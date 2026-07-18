#!/usr/bin/env python3
"""ch30 two-component (mixture-route) test: dissect the bimodal F distribution.

The refused channel ch30 shows two cleanly separated F populations. This
script measures the minority (lower) population and answers whether the
mixture-model reclamation route of Sec. 5.2 applies:

  - split at F = 2 mu0 (the gap 1.5--10 x mu0 is essentially empty);
  - classify every capture unit as all-low / all-high / mixed (mixed = the
    fade hypothesis; all-low = the transmitter-silence hypothesis);
  - locate the minority against the ANALYTIC zero point and measure its
    within-unit width against the trusted-channel null width (2.39e-3).

Outputs (PP_OUT):
  ch30_offair_minority.csv   per off-air unit: date, n_frames, mean, std
  stdout                     the summary numbers quoted in Sec. 5.2
"""
import csv
import datetime

import numpy as np

import _paths  # noqa: F401

z = np.load(_paths.PERFRAME)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(_paths.OUT / "empirical_zero_points.csv"))}
mu0 = float(study[30]["mu0_analytic"])
IDEAL_WIDTH = 2.39e-3          # sqrt(1/R + 1/2R), R = 262144 (Sec. 5.2)
SPLIT = 2.0                    # in units of mu0; the 1.5--10 gap is empty

pt = z["ch30_p_target_u64"].astype(np.float64)
pr = z["ch30_p_ref_sum_u64"].astype(np.float64)
ok = z["ch30_valid"].astype(bool)
with np.errstate(divide="ignore", invalid="ignore"):
    f = 2.0 * pt / pr
ok &= np.isfinite(f)
F = f[ok] / mu0
fui = z["ch30_frame_unit_index"][ok]
t0 = z["ch30_unit_time0_ctime"]
n = F.size

hi_clump = ((F >= 10) & (F < 14)).sum()
gap = ((F >= 1.5) & (F < 10)).sum()
low = F < SPLIT
print(f"ch30: n_valid={n}  mu0_analytic={mu0:.6f}")
print(f"  high clump (10--14 x mu0): {hi_clump} ({100*hi_clump/n:.2f}%)")
print(f"  gap (1.5--10 x mu0):       {gap} ({100*gap/n:.2f}%)")
print(f"  minority (F < {SPLIT:.0f} mu0):     {int(low.sum())} "
      f"({100*low.sum()/n:.2f}%)")

kinds = {"all_low": [], "all_high": 0, "mixed": 0}
for u in np.unique(fui):
    m = fui == u
    fr = low[m].mean()
    if fr >= 0.99:
        kinds["all_low"].append(u)
    elif fr <= 0.01:
        kinds["all_high"] += 1
    else:
        kinds["mixed"] += 1
low_units = kinds["all_low"]
print(f"  unit composition: all_low={len(low_units)} "
      f"all_high={kinds['all_high']} mixed={kinds['mixed']} "
      f"(mixed=0 -> transmitter silences, not fades)")

rows, means, widths = [], [], []
for u in low_units:
    x = F[fui == u]
    d = datetime.datetime.utcfromtimestamp(t0[u]).strftime("%Y-%m-%d")
    rows.append({"unit": int(u), "date": d, "n_frames": int(x.size),
                 "mean_over_mu0": f"{x.mean():.5f}",
                 "std_over_mu0": (f"{x.std(ddof=1):.5f}" if x.size >= 2
                                  else "")})
    means.append(x.mean())
    if x.size >= 3:
        widths.append(x.std(ddof=1))
means = np.array(means)
med_frame = np.median(F[low])
print(f"  minority centre: frame median {(med_frame-1)*1e3:+.2f} e-3, "
      f"unit-mean median {(np.median(means)-1)*1e3:+.2f} e-3 "
      f"(vs analytic zero point)")
print(f"  unit-mean scatter: {means.std(ddof=1)*1e3:.1f} e-3 rms, range "
      f"[{(means.min()-1)*1e3:+.1f}, {(means.max()-1)*1e3:+.1f}] e-3")
w = np.median(widths)
print(f"  within-unit width: {w*1e3:.1f} e-3 = {w/IDEAL_WIDTH:.0f}x the "
      f"trusted-channel null width")
core = np.abs(F - med_frame) <= 6e-3
print(f"  occupancy of +/-6e-3 window at the minority centre: "
      f"{100*core.sum()/n:.2f}% of all frames (floor: 15%)")
dates = sorted({r["date"] for r in rows})
print(f"  off-air days ({len(dates)}): {dates[0]} .. {dates[-1]}")

with open(_paths.OUT / "ch30_offair_minority.csv", "w", newline="") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wcsv.writeheader()
    wcsv.writerows(sorted(rows, key=lambda r: r["date"]))
print("wrote ch30_offair_minority.csv")
