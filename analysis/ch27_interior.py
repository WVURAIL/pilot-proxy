#!/usr/bin/env python3
"""ch27 interior trajectory from the survey-extension product (fid 644).

The frozen-snapshot analysis could only say: 38 per cent detections in
2020 Q3, quiet 2025-26 endpoints, interior unsampled (depth cap). The
full-depth extension product fills the interior. This script computes
the quarterly hi/low-tail rates over the complete history against the
SNAPSHOT calibration (empirical_zero_points.csv -- deliberately, so the
statement is an extension of the frozen analysis, not a recalibration).

Input:  PP_CH27_PRODUCT (default ~/pilot_proxy_runs/chime-pilots/_per_pilot/644.npz)
Output: PP_OUT/ch27_extension_quarterly.csv + stdout summary
"""
import csv
import datetime
import os
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

PROD = Path(os.environ.get(
    "PP_CH27_PRODUCT",
    str(Path.home() / "pilot_proxy_runs/chime-pilots/_per_pilot/644.npz")))
z = np.load(PROD)
assert int(z["physical_channel"][0]) == 27 and int(z["freq_id"][0]) == 644
s = {int(r["atsc_channel"]): r for r in
     csv.DictReader(open(_paths.OUT / "empirical_zero_points.csv"))}[27]
mu0 = float(s["mu0_analytic"])
mh = float(s["mu0_empirical"])

pt = z["p_target_u64"].reshape(-1).astype(np.float64)
pr = z["p_ref_sum_u64"].reshape(-1).astype(np.float64)
ok = z["valid"].reshape(-1).astype(bool)
with np.errstate(divide="ignore", invalid="ignore"):
    f = 2.0 * pt / pr
ok &= np.isfinite(f)
F = f / mu0
fui = z["frame_unit_index"].reshape(-1)
t0 = z["unit_time0_ctime"]


def quarter(ts):
    d = datetime.datetime.utcfromtimestamp(ts)
    return d.year + (d.month - 1) // 3 / 4


fq = np.array([quarter(t) for t in t0])[fui]
hi = ok & (F > (mh / mu0 + 12e-3))
lo = ok & (F < (mh / mu0 - 12e-3))
rows = []
for q in sorted(set(fq)):
    m = ok & (fq == q)
    n = int(m.sum())
    if n < 40:
        continue
    rows.append({"quarter": f"{q:.2f}", "n_valid_frames": n,
                 "hi_rate": f"{hi[m].sum()/n:.4f}",
                 "lo_rate": f"{lo[m].sum()/n:.4f}"})
with open(_paths.OUT / "ch27_extension_quarterly.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
loud = [r for r in rows if float(r["hi_rate"]) > 0.2]
quiet = [r for r in rows if float(r["quarter"]) >= 2022.75]
print(f"ch27 extension: {len(t0)} units, {int(ok.sum())} valid frames")
print(f"  loud quarters (hi>20%): {len(loud)}  "
      f"[{loud[0]['quarter']} .. {loud[-1]['quarter']}]" if loud else "")
print(f"  2022Q4-on quarters: {len(quiet)}, max hi "
      f"{max(float(r['hi_rate']) for r in quiet)*100:.1f}%")
print("wrote ch27_extension_quarterly.csv")
