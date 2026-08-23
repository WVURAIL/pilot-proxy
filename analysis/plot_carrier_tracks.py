#!/usr/bin/env python3
"""Track each channel's carrier frequency month by month.

The fine statistic is stored per frame, so the carrier's position in the
11.92 Hz envelope grid can be followed through the survey.  A month yields a
measurement when the monthly median fine spectrum has a peak standing
``MIN_DB`` clear of its own null bulk; the peak is refined to sub-bin
precision by a three-point parabolic fit.

Writes ``tables/carrier_tracks.csv``, ``tables/carrier_drift.csv`` and the
track figure.
"""
from __future__ import annotations

import argparse
import csv
import os

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ppcal import eras as E, spectra as S  # noqa: E402
from ppcal.plotting import (EXCISE_COLOR, INK, INK2, SERIES, save_fig,
                            setup_style)  # noqa: E402
from ppcal.products import M0, NMONTHS, load_all, month_label  # noqa: E402

MIN_DB = 3.0        # peak height above the month's own fine null bulk
MIN_MONTHS = 8      # months of track needed before a drift is fitted


def track(c):
    """(month, offset_hz, height_db) for every month with a detected line."""
    months, rf, img = S.fine_spectrogram(c)
    out = []
    for j, m in enumerate(months):
        col = img[:, j]
        if not np.isfinite(col).all():
            continue
        base = float(np.median(col))
        i = int(np.argmax(col))
        h = col[i] - base
        if h < MIN_DB or i == 0 or i == col.size - 1:
            continue
        y0, y1, y2 = col[i - 1], col[i], col[i + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        step = float(rf[1] - rf[0])
        out.append((int(m), float(rf[i] + delta * step), float(h)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.products = args.products or str(P.PER_PILOT)
    args.out = args.out or str(P.OUT)
    setup_style()
    chans = sorted(load_all(args.products), key=lambda c: c.ch)

    rows, tracks, drifts = [], {}, []
    for c in chans:
        t = track(c)
        tracks[c.ch] = t
        for m, hz, h in t:
            rows.append(dict(ch=c.ch, month=month_label(m),
                             offset_hz=round(hz, 2), height_db=round(h, 2)))
        if len(t) >= MIN_MONTHS:
            segs = E.segment(c)
            last = segs[-1]
            tl = [(m, hz) for m, hz, _ in t if m >= last.month_start]
            if len(tl) >= MIN_MONTHS:
                x = np.array([m for m, _ in tl], float)
                y = np.array([hz for _, hz in tl])
                slope, icpt = np.polyfit(x, y, 1)
                resid = y - (slope * x + icpt)
                scatter = float(resid.std())
                swing = abs(slope) * (x.max() - x.min())
                total = float(np.std(y))
                # a drift is claimed only when the straight line explains the
                # track: the fitted swing must dominate the scatter about it,
                # and the fit must remove most of the raw variance.
                r2 = 1.0 - (resid.var() / y.var()) if y.var() > 0 else 0.0
                drifts.append(dict(
                    ch=c.ch, era=last.label, n_months=len(tl),
                    slope_hz_per_year=slope * 12.0,
                    fitted_swing_hz=swing, span_hz=float(y.max() - y.min()),
                    scatter_hz=scatter, r2=r2, track_sd_hz=total,
                    coherent_drift=bool(swing > 3.0 * scatter and r2 > 0.7)))

    tdir = os.path.join(args.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "carrier_tracks.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ch", "month", "offset_hz",
                                           "height_db"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(tdir, "carrier_drift.csv"), "w", newline="",
              encoding="utf-8") as fh:
        cols = ["ch", "era", "n_months", "slope_hz_per_year",
                "fitted_swing_hz", "span_hz", "scatter_hz", "r2",
                "track_sd_hz", "coherent_drift"]
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for d in sorted(drifts, key=lambda d: -abs(d["slope_hz_per_year"])):
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in d.items()})

    live = [c for c in chans if len(tracks[c.ch]) >= MIN_MONTHS]
    dmap = {d["ch"]: d for d in drifts}
    ncol = 4
    nrow = (len(live) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.4, 2.3 * nrow),
                             sharex=True)
    for ax in axes.ravel()[len(live):]:
        ax.set_visible(False)
    for ax, c in zip(axes.ravel(), live):
        t = tracks[c.ch]
        x = np.array([m for m, _, _ in t], float)
        y = np.array([hz for _, hz, _ in t])
        d = dmap.get(c.ch)
        coherent = bool(d and d["coherent_drift"])
        col = SERIES[1] if coherent else INK
        ax.plot(x, y, marker="o", ms=2.4, lw=1.0, color=col, alpha=0.9)
        if d:
            xs = np.array([x.min(), x.max()])
            sl = d["slope_hz_per_year"] / 12.0
            icpt = np.mean(y) - sl * np.mean(x)
            ax.plot(xs, sl * xs + icpt, color=EXCISE_COLOR, lw=1.2,
                    ls=(0, (4, 3)))
            ax.annotate("ch %d\n%+.0f Hz/yr%s"
                        % (c.ch, d["slope_hz_per_year"],
                           "" if coherent else "  (scatter-dominated)"),
                        xy=(0.03, 0.94), xycoords="axes fraction", va="top",
                        fontsize=8.4, color=INK2)
        else:
            ax.annotate("ch %d" % c.ch, xy=(0.03, 0.94),
                        xycoords="axes fraction", va="top", fontsize=8.4,
                        color=INK2)
        pos = [m for m in range(NMONTHS) if (M0 + m) % 24 == 0]
        ax.set_xticks(pos)
        ax.set_xticklabels([str((M0 + m) // 12) for m in pos], fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("offset  [Hz]", fontsize=9)
    fig.suptitle("Carrier frequency tracked month by month\n"
                 "monthly median fine spectrum's peak, parabolically refined "
                 "inside the 11.92 Hz grid; orange = drift the straight line "
                 "explains, black = scatter-dominated", fontsize=12.5,
                 x=0.5, y=1.004)
    fig.tight_layout()
    p = save_fig(fig, os.path.join(args.out, "figures",
                                   "fig10_carrier_tracks.png"))

    print("%d channels carry a trackable carrier" % len(live))
    print("%3s %-18s %7s %12s %9s %9s %7s %9s"
          % ("ch", "era", "months", "Hz/yr", "span_Hz", "scatter", "r2",
             "coherent"))
    for d in sorted(drifts, key=lambda d: -abs(d["slope_hz_per_year"])):
        print("%3d %-18s %7d %12.1f %9.1f %9.1f %7.3f %9s"
              % (d["ch"], d["era"], d["n_months"], d["slope_hz_per_year"],
                 d["span_hz"], d["scatter_hz"], d["r2"],
                 "yes" if d["coherent_drift"] else "no"))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
