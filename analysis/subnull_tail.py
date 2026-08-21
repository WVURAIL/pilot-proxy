#!/usr/bin/env python3
"""Diagnose the population of frames sitting below the calibrated null.

F = 2 P_target / (P_ref_lo + P_ref_hi) can only fall well below its null if
the reference bins carry more power than the target bin.  Two very different
causes produce that, and they are separable with what the products retain:

* **broadband** -- the frame is loud everywhere (impulsive RFI, a saturating
  event).  Then baseband_power_linear is elevated and the target bin is not
  especially quiet.
* **reference-bin contamination** -- a carrier sits in one of the two guard
  references.  Then total power is ordinary and the deficit is confined to
  the ratio.

Reports, per channel, the size of the sub-null population and the ratio of
its median total power to the null bulk's.  On this archive the answer is
broadband: 0.24% of frames, at a median 3.35x the null bulk's total power,
spread across many acquisitions.  That population belongs to the incumbent
power/kurtosis flagger, not to this narrowband detector.
"""
from __future__ import annotations

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402

from ppcal import eras as E  # noqa: E402
from ppcal.calib import calibrate  # noqa: E402
from ppcal.products import load_all  # noqa: E402

D = str(P.PER_PILOT)
K = 5.0        # how many null scales below mu counts as "sub-null"


def main():
    print("%3s %9s %9s %10s %10s %9s %8s"
          % ("ch", "n_sub", "frac", "medF/mu", "power_ratio", "units_hit",
             "top_unit"))
    tot_sub = tot_n = 0
    rows = []
    for c in sorted(load_all(D), key=lambda c: c.ch):
        segs = E.segment(c)
        fmask = E.final_era_frame_mask(c, segs)
        cal = calibrate(c, fmask, segs[-1].label, 0)
        if cal.mu_shift_db > 3.0 or not np.isfinite(cal.sigma):
            continue                  # carrier-dominated: no null to sit under
        f = c.fstat[fmask]
        pwr = c._z["baseband_power_linear"][c.health_include, 0][fmask]
        unit = c.frame_unit[fmask]

        cut = cal.mu - K * cal.sigma
        sub = f < cut
        bulk = (f >= cal.mu - cal.sigma) & (f <= cal.mu + cal.sigma)
        n_sub = int(sub.sum())
        tot_sub += n_sub
        tot_n += int(f.size)
        if n_sub == 0:
            continue
        pr = (float(np.median(pwr[sub]) / np.median(pwr[bulk]))
              if bulk.any() else float("nan"))
        u, counts = np.unique(unit[sub], return_counts=True)
        top = float(counts.max() / n_sub)
        print("%3d %9d %9.5f %10.4f %10.3f %9d %8.2f"
              % (c.ch, n_sub, n_sub / f.size,
                 float(np.median(f[sub])) / cal.mu, pr, u.size, top))
        rows.append((c.ch, n_sub / f.size, pr, u.size, top))

    print("\ntotal sub-null frames: %d of %d (%.4f%%) over %d channels with a "
          "null" % (tot_sub, tot_n, 100 * tot_sub / max(tot_n, 1), len(rows)))
    pr = np.array([r[2] for r in rows])
    print("power ratio (sub-null median / null-bulk median): "
          "min %.2f  median %.2f  max %.2f" % (pr.min(), np.median(pr), pr.max()))
    print("channels where sub-null frames are NOT power-elevated (ratio < 1.5): "
          "%d of %d" % (int((pr < 1.5).sum()), len(rows)))
    conc = np.array([r[4] for r in rows])
    print("largest share from a single acquisition unit: median %.2f, max %.2f"
          % (np.median(conc), conc.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
