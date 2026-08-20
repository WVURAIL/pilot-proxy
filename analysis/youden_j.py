#!/usr/bin/env python3
"""Coarse-versus-fine ROC and Youden J, recomputed from the survey products.

The candidate-selection record (docs/DESIGN_DECISIONS.md, 2026-07) chose the
fine designated-set CFAR as the runtime decision from measured ROCs: every
calibrated coarse operating point either spends about half of verified-quiet
time (the positive-excess point holds P_d ~ 1 but rejects ~48.5% of the
null by construction) or collapses on weak channels. This script recomputes
that comparison on the released per-pilot products with the populations
stated, so the dissertation can cite a committed analysis rather than a
remembered one.

Populations:
  null   : channel 35 (freq_id 521) frames in its verified transmitter-off
           era (through 2021-10) --- the off state is established by the
           era, not by non-detection.
  signal : the 2025+ on-epoch frames of channels 36, 35, and 34.

Per-frame statistics:
  coarse : F / mu0 (the stored exact statistic over the exact null mean).
  fine   : designated-window maximum of the stored 256-bin fine statistic
           over anchor +/- 2 bins, normalized by the frame's bulk median;
           each channel's anchor is the argmax of its mean signal-epoch
           fine spectrum, and the null frames are scored in the same
           window (null bins are exchangeable, Section 5.5 of the
           dissertation).

Output: the ROC table at the record's null-quantile P_fa points, each
statistic's Youden J (max over threshold of P_d - P_fa), and the coarse
positive-excess point's measured null exceedance.

    python3 analysis/youden_j.py [--products DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  -- puts <repo>/src on sys.path
from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO, ARCHIVED_FINE_POWER_RATIO)

import _products as P

NULL_FID = 521                    # channel 35
NULL_OFF_THROUGH = "2021-10"      # verified transmitter-off era
SIGNAL = {36: 506, 35: 521, 34: 537}
SIGNAL_FROM = "2025-01"
PFA_POINTS = (0.10, 0.05, 0.015)
HALF = 2                          # designated window: anchor +/- 2 fine bins


def _ts(ym: str, end: bool = False) -> float:
    y, m = int(ym[:4]), int(ym[5:7])
    if end:                       # first instant after the month
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp()


def _frames(d):
    """(frame times, valid mask, coarse F/mu0, fine [nframes, 256])."""
    t = d["unit_time0_ctime"][d["frame_unit_index"].ravel()]
    valid = d["valid"].ravel().astype(bool)
    mu0 = float(np.ravel(d["mu0"])[0])
    coarse = d[ARCHIVED_COARSE_POWER_RATIO].ravel() / mu0
    fine = d[ARCHIVED_FINE_POWER_RATIO]
    return t, valid, coarse, fine


def _fine_stat(fine, window):
    """Designated-window max over the frame's bulk median."""
    med = np.nanmedian(fine, axis=1)
    win = np.nanmax(fine[:, window], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = win / med
    return s


def _roc(null_s, sig_s):
    """P_d at the P_fa grid plus Youden J over the swept threshold."""
    null_s = null_s[np.isfinite(null_s)]
    sig_s = sig_s[np.isfinite(sig_s)]
    thresholds = np.unique(null_s)
    pfa = np.array([(null_s > th).mean() for th in thresholds])
    pd_ = np.array([(sig_s > th).mean() for th in thresholds])
    j = pd_ - pfa
    best = int(np.argmax(j))
    at = {}
    for target in PFA_POINTS:
        k = int(np.argmin(np.abs(pfa - target)))
        at[target] = (float(pfa[k]), float(pd_[k]), float(thresholds[k]))
    return at, (float(j[best]), float(thresholds[best]),
                float(pfa[best]), float(pd_[best]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path, default=None)
    args = ap.parse_args()
    paths = P.paths(per_pilot=args.products or P.PER_PILOT)

    d_null = P.load_npz(paths[35])
    t, valid, coarse, fine = _frames(d_null)
    off = valid & (t < _ts(NULL_OFF_THROUGH, end=True))
    null_coarse = coarse[off]
    print(f"null: ch35 off-era frames n={int(off.sum())} "
          f"(through {NULL_OFF_THROUGH})")
    pe = float((null_coarse > 1.0).mean())
    print(f"coarse positive-excess point on the null: "
          f"P_fa = {pe:.3f} (the deployed rule's verified-quiet spend)")

    for ch, fid in SIGNAL.items():
        d = P.load_npz(paths[ch])
        ts, va, co, fi = _frames(d)
        on = va & (ts >= _ts(SIGNAL_FROM))
        # anchor: argmax of the mean signal-epoch fine spectrum
        anchor = int(np.nanargmax(np.nanmean(fi[on], axis=0)))
        window = (np.arange(anchor - HALF, anchor + HALF + 1)) % fi.shape[1]
        sig_fine = _fine_stat(fi[on], window)
        null_fine = _fine_stat(fine[off], window)
        at_c, j_c = _roc(null_coarse, co[on])
        at_f, j_f = _roc(null_fine, sig_fine)
        print(f"\nch{ch}: signal n={int(on.sum())} (from {SIGNAL_FROM}), "
              f"anchor bin {anchor}")
        print("  Pfa(null q)   coarse Pd   fine Pd")
        for target in PFA_POINTS:
            print(f"  {target:11.3f}   {at_c[target][1]:9.3f}   "
                  f"{at_f[target][1]:7.3f}")
        print(f"  Youden J: coarse {j_c[0]:.3f} "
              f"(Pfa {j_c[2]:.3f}, Pd {j_c[3]:.3f}) | "
              f"fine {j_f[0]:.3f} (Pfa {j_f[2]:.3f}, Pd {j_f[3]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
