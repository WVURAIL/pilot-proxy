#!/usr/bin/env python3
"""One calibration card per channel: eras, spectrograms, spectra, statistics.

    python3 analysis/plot_calibration_channels.py [--only 17,30]
"""
from __future__ import annotations

import argparse
import os

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ppcal import spectra as S, state as ST  # noqa: E402
from ppcal.plotting import (CRITICAL, EXCISE_COLOR, INK, INK2, KEEP_COLOR,
                            MUTED, OCC_CMAP, SERIES, save_fig,
                            setup_style)  # noqa: E402
from ppcal.products import COARSE_HZ, FINE_HZ, M0, NMONTHS  # noqa: E402

ZOOM_HZ = 420.0          # half-width of the zoomed spectrogram


def _carrier_hz(c, fmask):
    """RF offset of the strongest fine line in the latest era, or the
    geometric prediction when no line stands above the null bulk."""
    rf, med, _ = S.era_fine_spectrum(c, fmask)
    predicted = float(c.sense * c.centered(c.predicted_fine_bin) * FINE_HZ)
    if not np.isfinite(med).any():
        return predicted, False
    base = float(np.median(med))
    j = int(np.argmax(med))
    if med[j] - base >= 1.0:
        return float(rf[j]), True
    return predicted, False


def _spectrogram(ax, months, rf, img, ylim=None, cbar_ax=None, segs=None):
    if not np.isfinite(img).any():
        return None
    hi = max(float(np.nanpercentile(img, 99.7)), 1.0)
    cmap = OCC_CMAP.copy()
    cmap.set_bad("#f2f1ec")
    im = ax.imshow(np.ma.masked_invalid(img), aspect="auto", origin="lower",
                   cmap=cmap, vmin=0, vmax=hi, interpolation="nearest",
                   extent=[months[0] - 0.5, months[-1] + 0.5,
                           rf[0] - FINE_HZ / 2, rf[-1] + FINE_HZ / 2])
    if ylim:
        ax.set_ylim(*ylim)
    if segs:
        lo, hi_y = ax.get_ylim()
        for seg in segs[1:]:
            ax.axvline(seg.month_start - 0.5, color=INK, lw=2.0)
        last = segs[-1]
        ax.add_patch(Rectangle(
            (last.month_start - 0.5, lo),
            last.month_end - last.month_start + 1, hi_y - lo, fill=False,
            edgecolor=INK, lw=1.0, ls=(0, (3, 2)), zorder=5))
    pos = [m for m in range(int(months[0]), int(months[-1]) + 1)
           if (M0 + m) % 12 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([str((M0 + m) // 12) for m in pos], fontsize=8.5)
    ax.grid(False)
    if cbar_ax is not None:
        cb = ax.figure.colorbar(im, cax=cbar_ax)
        cb.set_label("dB rel. $\\mu_0$", fontsize=8.5, color=INK2)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)
    return im


def _spec(ax, c, span_khz, center="channel", peaks=False):
    c0 = 0.0 if center == "channel" else c.pilot_offset_hz
    rf, dbv = S.spectrum_db(c)
    x = (rf - c0) / 1e3
    sel = np.abs(x) <= span_khz
    ax.plot(x[sel], dbv[sel], color=INK, lw=0.7)
    t = (S.target_coarse_offset_hz(c) - c0) / 1e3
    ax.axvspan(t - COARSE_HZ / 2e3, t + COARSE_HZ / 2e3, color=SERIES[0],
               alpha=0.16, lw=0)
    for g in S.guard_reference_offsets_hz(c):
        g = (g - c0) / 1e3
        ax.axvspan(g - COARSE_HZ / 2e3, g + COARSE_HZ / 2e3, color=SERIES[1],
                   alpha=0.16, lw=0)
    ax.axvline((c.pilot_offset_hz - c0) / 1e3, color=SERIES[2], lw=1.1,
               ls=(0, (4, 3)))
    ax.axvline((0.0 - c0) / 1e3, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax.set_xlim(-span_khz, span_khz)
    if peaks:
        rows = []
        for p in S.peak_census(c):
            d = (p.rf_offset_hz - c0) / 1e3
            if abs(d) > span_khz:
                continue
            lab = {"pilot": "pilot", "secondary": "carrier",
                   "channel-centre artefact": "instrumental DC"}[p.kind]
            col = {"pilot": SERIES[2], "secondary": SERIES[1],
                   "channel-centre artefact": MUTED}[p.kind]
            ax.plot([d], [p.db_rel_median], marker="v", ms=6, color=col,
                    zorder=5)
            rows.append(("%s  %+.2f kHz  %.1f dB"
                         % (lab, d, p.db_rel_median), col))
        for k, (txt, col) in enumerate(rows):
            ax.annotate(txt, xy=(0.02, 0.97 - 0.075 * k),
                        xycoords="axes fraction", va="top", fontsize=8.2,
                        color=col)


def card(s, out):
    c, segs, fmask, cal = s.c, s.segs, s.fmask, s.cal
    carrier_hz, carrier_found = _carrier_hz(c, fmask)
    verdict_col = EXCISE_COLOR if s.verdict == "excise" else KEEP_COLOR
    eta = s.eta_channel

    fig = plt.figure(figsize=(12.6, 15.2))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.028],
                          height_ratios=[0.78, 1.05, 1.0, 1.0],
                          hspace=0.36, wspace=0.24,
                          top=0.925, bottom=0.035, left=0.075, right=0.955)

    # ---- (a) occupancy history -------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    um, lvl = c.unit_month, c.unit_level_db
    ax.scatter(um + np.random.default_rng(c.ch).uniform(-0.35, 0.35, um.size),
               lvl, s=3, color=MUTED, alpha=0.30, lw=0, zorder=1)
    months, med, _ = c.monthly_level_db()
    ax.plot(months, med, color=INK, lw=1.6, zorder=3)
    bits = []
    for i, seg in enumerate(segs):
        col = SERIES[i % len(SERIES)]
        ax.axvspan(seg.month_start - 0.5, seg.month_end + 0.5, color=col,
                   alpha=0.10, lw=0, zorder=0)
        ax.plot([seg.month_start - 0.5, seg.month_end + 0.5],
                [seg.level_median_db] * 2, color=col, lw=2.4, zorder=4)
        bits.append("era %d %s (%+.2f dB)"
                    % (i + 1, seg.label, seg.level_median_db))
    ax.axhline(0.0, color=CRITICAL, lw=1.0, ls=(0, (1, 2)))
    ax.axhline(10 * np.log10(eta * cal.mu / c.mu0), color=KEEP_COLOR, lw=1.0,
               ls=(0, (4, 3)))
    pos = [m for m in range(NMONTHS) if (M0 + m) % 12 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([str((M0 + m) // 12) for m in pos])
    ax.set_xlim(-0.5, NMONTHS - 0.5)
    ax.set_ylabel("unit level  [dB rel. $\\mu_0$]")
    ax.set_title("(a) occupancy history:  " + "   |   ".join(bits),
                 fontsize=10.5, loc="left")

    # ---- (b, c) spectrograms ---------------------------------------------
    grid, rf, img = S.fine_spectrogram(c)
    axb = fig.add_subplot(gs[1, 0])
    cbb = fig.add_subplot(gs[1, 2])
    _spectrogram(axb, grid, rf, img, cbar_ax=cbb, segs=segs)
    axb.set_ylabel("RF offset from target bin  [Hz]")
    axb.set_title("(b) fine spectrogram, full $\\pm 1.53$ kHz band",
                  fontsize=11, loc="left")

    axc = fig.add_subplot(gs[1, 1])
    _spectrogram(axc, grid, rf, img,
                 ylim=(carrier_hz - ZOOM_HZ, carrier_hz + ZOOM_HZ), segs=segs)
    axc.axhline(carrier_hz, color="#ffffff", lw=0.8, ls=(0, (4, 4)))
    axc.set_title("(c) zoom: $\\pm %d$ Hz about the %s line (%.1f Hz bins)"
                  % (ZOOM_HZ, "measured" if carrier_found else "predicted",
                     FINE_HZ), fontsize=11, loc="left")

    # ---- (d, e) integrated spectra ---------------------------------------
    axd = fig.add_subplot(gs[2, 0])
    _spec(axd, c, 195.3, center="channel")
    axd.set_xlabel("RF offset from channel centre  [kHz]")
    axd.set_ylabel("dB rel. median")
    axd.set_title("(d) archive-integrated channel spectrum", fontsize=11,
                  loc="left")

    axe = fig.add_subplot(gs[2, 1])
    _spec(axe, c, 15.0, center="pilot", peaks=True)
    axe.set_xlabel("RF offset from synthesized pilot  [kHz]")
    axe.set_title("(e) zoom on the pilot, $\\pm 15$ kHz", fontsize=11,
                  loc="left")

    # ---- (f) latest-era fine spectrum, before and after masking ----------
    axf = fig.add_subplot(gs[3, 0])
    rff, beforef, afterf, mstats = S.era_fine_spectrum_masked(
        c, fmask, eta * cal.mu)
    axf.plot(rff, beforef, color=MUTED, lw=1.2, label="before masking")
    axf.plot(rff, afterf, color=INK, lw=1.4, label="after masking")
    axf.fill_between(rff, afterf, beforef, color=KEEP_COLOR, alpha=0.16, lw=0)
    lines = S.fine_line_census(c, fmask)
    _, medf, _ = S.era_fine_spectrum(c, fmask)
    base = float(np.median(medf)) if np.isfinite(medf).any() else 0.0
    for hz, dbm, _ in lines:
        axf.plot([hz], [base + dbm], marker="v", ms=6, color=SERIES[2],
                 zorder=5)
    if lines:
        axf.annotate("\n".join("%+.0f Hz   %.1f dB" % (hz, dbm)
                               for hz, dbm, _ in lines),
                     xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                     fontsize=8.4, color=SERIES[2])
    axf.set_xlim(-1600, 1600)
    axf.set_xlabel("RF offset from target coarse bin  [Hz]")
    axf.set_ylabel("dB rel. $\\mu_0$")
    axf.legend(loc="upper right", fontsize=8.5)
    axf.set_title("(f) latest-era mean fine spectrum, before and after "
                  "masking: $-$%.1f dB over the band, %d resolved line%s"
                  % (mstats["band_suppression_db"], len(lines),
                     "" if len(lines) == 1 else "s"),
                  fontsize=11, loc="left")

    # ---- (g) statistic and the ladder ------------------------------------
    axg = fig.add_subplot(gs[3, 1])
    f = c.fstat[fmask] / c.mu0
    edges = np.logspace(np.log10(0.85), np.log10(max(400, f.max() * 1.1)), 170)
    axg.hist(np.clip(f, edges[0], edges[-1]), bins=edges, color=INK,
             alpha=0.82, lw=0)
    axg.set_xscale("log")
    axg.set_yscale("log")
    axg.axvline(1.0, color=CRITICAL, lw=1.4,
                label="$F > 1$ (provisional, as collected)")
    axg.axvline(cal.mu / c.mu0, color=SERIES[0], lw=1.4, ls=(0, (1, 2)),
                label="$\\mu = %.4g$ (calibrated)" % cal.mu)
    eta_basis = ("supplied science-priced" if s.eta_is_per_channel
                 else "historical report fallback")
    axg.axvline(eta * cal.mu / c.mu0, color=KEEP_COLOR, lw=1.4, ls=(0, (4, 3)),
                label="$F > %.3f\\,\\mu$ (%s $\\eta$)" % (eta, eta_basis))
    axg.set_xlabel("$F/\\mu_0$   (the collection-time scale)")
    axg.set_ylabel("frames")
    axg.legend(loc="best", fontsize=8.5)
    axg.set_title("(g) latest-era statistic and the threshold ladder",
                  fontsize=11, loc="left")

    # ---- header ----------------------------------------------------------
    d_pilot, db_pilot = S.measured_pilot_offset_hz(c)
    nsec = sum(1 for p in S.peak_census(c) if p.kind == "secondary")
    head = ("ch %d      freq id %d      pilot %.5f MHz      %+.1f kHz from "
            "channel centre" % (c.ch, c.fid, c.pilot_hz / 1e6,
                                c.pilot_offset_hz / 1e3))
    science = ""
    if s.thresholds and s.thresholds.get("r_tol_dilation"):
        rt = s.thresholds["r_tol_dilation"]
        cap = s.thresholds.get("r_cost_cap")
        th = s.thresholds.get("r_cost_thermal")
        science = ("     $z$ %.2f-%.2f, $r/r_{\\rm tol}$ = %s (cap) .. %s "
               "(thermal)"
               % (s.thresholds.get("z_low", float("nan")),
                  s.thresholds.get("z_high", float("nan")),
                  "%.3g" % (cap / rt) if cap else "n/a",
                  "%.3g" % (th / rt) if th else "n/a"))
    sub = ("$\\mu_0$ provisional = %.6f     $\\mu$ calibrated = %.5g  "
           "(%+.3f dB)     $\\hat{\\sigma}/\\mu$ = %.2e\n"
           "masked: %.1f%% at $F>1$, %.1f%% at $F>%.3f\\mu$     "
           "%d era%s, latest %s (%s frames)     carrier %+.2f kHz / %.1f dB"
           "     %d secondary carrier%s%s\n%s"
           % (c.mu0, cal.mu, cal.mu_shift_db, cal.sigma_over_mu,
              100 * cal.occupancy_provisional, 100 * s.occ_working, eta,
              len(segs), "" if len(segs) == 1 else "s", segs[-1].label,
              "{:,}".format(int(fmask.sum())), d_pilot / 1e3, db_pilot,
              nsec, "" if nsec == 1 else "s", science, s.reason))
    fig.text(0.008, 0.995, head, fontsize=15, color=INK, ha="left", va="top")
    fig.text(0.008, 0.978, sub, fontsize=9.4, color=INK2, ha="left",
             va="top", linespacing=1.6)
    fig.text(0.992, 0.995, s.disposition.upper(), fontsize=13,
             color=verdict_col, ha="right", va="top", fontweight="semibold")
    return save_fig(fig, os.path.join(out, "ch%02d_card.png" % c.ch))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated physical channels")
    args = ap.parse_args(argv)
    args.products = args.products or str(P.PER_PILOT)
    args.out = args.out or str(P.OUT / "figures" / "channels")
    args.thresholds = args.thresholds or str(P.THRESHOLD_TABLE)
    setup_style()
    only = ({int(x) for x in args.only.split(",")} if args.only else None)
    for s in ST.build(args.products, threshold_csv=args.thresholds):
        if only and s.ch not in only:
            continue
        print("wrote", card(s, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
