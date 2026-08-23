#!/usr/bin/env python3
"""Segment every channel into activity eras and check against the locked table.

``analysis/make_policy_data.py`` carries a hand-curated EPOCH table of
transmitter transitions.  This script recovers transitions from the data
alone and reports where the two agree.
"""
from __future__ import annotations

import _calibration_paths as P  # noqa: F401

from ppcal import eras as E  # noqa: E402
from ppcal.products import load_all  # noqa: E402

# from analysis/make_policy_data.py, EPOCH: ch -> transition text
LOCKED = {
    35: "sign-on Nov 2021",
    19: "sign-off Dec 2024",
    26: "sign-off Apr 2023",
    20: "step down Sep 2022",
    27: "sign-off in 2021-22 archive gap",
    32: "sign-off in 2021-22 archive gap",
    17: "level step Oct 2022, on-to-on",
}

D = str(P.PER_PILOT)


def main():
    chans = sorted(load_all(D), key=lambda c: -c.ch)
    print("%3s %6s  %-9s %s" % ("ch", "eras", "verdict", "segmentation"))
    found = {}
    for c in chans:
        segs = E.segment(c)
        found[c.ch] = segs
        desc = "  ->  ".join(
            "%s [%d u, med %+.2f dB]" % (s.label, s.n_units, s.level_median_db)
            for s in segs)
        locked = LOCKED.get(c.ch)
        if len(segs) > 1 and locked:
            verdict = "agrees"
        elif len(segs) > 1 and not locked:
            verdict = "NEW"
        elif len(segs) == 1 and locked:
            verdict = "MISSED"
        else:
            verdict = "single"
        print("%3d %6d  %-9s %s" % (c.ch, len(segs), verdict, desc))

    print()
    print("locked transitions:",
          ", ".join("ch%d (%s)" % (k, v) for k, v in sorted(LOCKED.items())))
    multi = [k for k, v in found.items() if len(v) > 1]
    print("data-driven multi-era channels:", sorted(multi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
