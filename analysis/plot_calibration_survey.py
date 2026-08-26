#!/usr/bin/env python3
"""Survey-wide calibration plates, one figure per question.

    python3 analysis/plot_calibration_survey.py [--products DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import os

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from ppcal import spectra as S, state as ST  # noqa: E402
from ppcal.plotting import (CRITICAL, EXCISE_COLOR, INK, INK2, KEEP_COLOR,
                            MASK_COLOR, MUTED, OCC_CMAP, SERIES, SURFACE,
                            annotate_channel, save_fig,
                            setup_style)  # noqa: E402
from ppcal.products import COARSE_HZ, M0, NMONTHS  # noqa: E402

ETA = ST.FALLBACK_ETA


def col_of(s):
    if s.verdict == "excise":
        return EXCISE_COLOR
    return MASK_COLOR if s.occ_working > 0.10 else KEEP_COLOR


# ---------------------------------------------------------------------------

def fig_timeline(st, out):
    n = len(st)
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    grid = np.full((n, NMONTHS), np.nan)
    for r, s in enumerate(st):
        months, med, _ = s.c.monthly_level_db()
        grid[r, months] = med
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=OCC_CMAP,
                   vmin=0, vmax=22, interpolation="nearest")
    for r, s in enumerate(st):
        for seg in s.segs[1:]:
            ax.plot([seg.month_start - 0.5] * 2, [r - 0.5, r + 0.5],
                    color=INK, lw=2.4, solid_capstyle="butt", zorder=4)
        last = s.segs[-1]
        c = CRITICAL if s.verdict == "excise" else KEEP_COLOR
        ax.add_patch(Rectangle((last.month_start - 0.5, r - 0.5),
                               last.month_end - last.month_start + 1, 1.0,
                               fill=False, edgecolor=c, lw=1.8, zorder=5))
    ax.set_yticks(range(n))
    ax.set_yticklabels(["ch %d" % s.ch for s in st], fontsize=9)
    pos = [m for m in range(NMONTHS) if (M0 + m) % 12 == 0]
    ax.set_xticks(pos)
    ax.set_xticklabels([str((M0 + m) // 12) for m in pos])
    ax.set_xlim(-0.5, NMONTHS - 0.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.026)
    cb.set_label("monthly median pilot level  [dB rel. the provisional "
                 "constant]", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Activity eras across the completed survey\n"
                 "blank = no baseband holding that month; black rule = "
                 "recovered era boundary; outlined box = latest era "
                 "(green keep, red excise)", fontsize=12.5, loc="left")
    return save_fig(fig, os.path.join(out, "fig01_occupancy_timeline.png"))


def fig_mu(st, out):
    """The calibrated mu against the provisional constant."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                 gridspec_kw=dict(width_ratios=[1.3, 1]))
    chs = [s.ch for s in st]
    shift = [max(s.cal.mu_shift_db, 3e-3) for s in st]
    cols = [col_of(s) for s in st]
    a0.bar(chs, shift, color=cols, width=0.72)
    a0.set_yscale("log")
    a0.axhline(ST.CARRIER_DOMINATED_LEVEL_DB, color=INK2, lw=1.2,
               ls=(0, (4, 3)))
    a0.annotate("carrier-dominated above %.0f dB"
                % ST.CARRIER_DOMINATED_LEVEL_DB,
                xy=(13.6, 3.6), color=INK2, fontsize=9.5)
    a0.axhspan(3e-3, 1.0, color=SERIES[0], alpha=0.12, lw=0)
    a0.annotate("null recovered: every kept channel lies below 1 dB",
                xy=(13.6, 0.0042), color=SERIES[0], fontsize=9.5)
    a0.set_xticks(chs)
    a0.set_xticklabels([str(c) for c in chs], fontsize=8)
    a0.set_xlabel("DTV physical channel")
    a0.set_ylabel("$10\\log_{10}(\\mu / \\mu_{0})$  [dB]")
    a0.set_title("How far the calibrated $\\mu$ sits above the provisional "
                 "constant", fontsize=12, loc="left")

    keep = [s for s in st if s.verdict == "keep"]
    a1.scatter([s.cal.mu_shift_db for s in keep],
               [s.cal.sigma_over_mu for s in keep], s=48, color=KEEP_COLOR,
               zorder=3, edgecolor=SURFACE, linewidth=0.6)
    for s in keep:
        a1.annotate(str(s.ch), (s.cal.mu_shift_db, s.cal.sigma_over_mu),
                    textcoords="offset points", xytext=(5, 3), fontsize=8,
                    color=INK2)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("null offset  $10\\log_{10}(\\mu/\\mu_0)$  [dB]")
    a1.set_ylabel("null width  $\\hat{\\sigma}/\\mu$")
    a1.set_title("Null width of the 17 kept channels", fontsize=12, loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig02_mu_calibration.png"))


def fig_ladder_summary(st, out):
    """The three rungs of the threshold ladder, per channel."""
    keep = sorted([s for s in st if s.verdict == "keep"],
                  key=lambda s: s.occ_working)
    exc = sorted([s for s in st if s.verdict == "excise"],
                 key=lambda s: s.occ_working)
    order = exc + keep
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11.4, 8.4))
    h = 0.26
    p1 = [s.cal.occupancy_provisional for s in order]
    pm = [s.cal.occupancy["1"] for s in order]
    pw = [s.occ_working for s in order]
    ax.barh(y + h, p1, height=h, color=CRITICAL)
    ax.barh(y, pm, height=h, color=SERIES[0])
    ax.barh(y - h, pw, height=h, color=KEEP_COLOR)
    for k, s in enumerate(order):
        if s.verdict == "excise":
            ax.barh(k - h, pw[k], height=h, color="none", edgecolor=INK,
                    hatch="////", lw=0.0)
        ax.annotate("$\\eta$=%.2f" % s.eta_channel, (pw[k], k - h),
                    textcoords="offset points", xytext=(5, -3), fontsize=7.6,
                    color=INK2)
    ax.axhline(len(exc) - 0.5, color=INK2, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(["ch %d" % s.ch for s in order], fontsize=9.5)
    for lab, s in zip(ax.get_yticklabels(), order):
        lab.set_color(EXCISE_COLOR if s.verdict == "excise" else INK2)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("fraction of latest-era frames masked")
    kept_med = np.median([s.occ_working for s in keep])
    kept_prov = np.median([s.cal.occupancy_provisional for s in keep])
    ax.annotate("excised: $\\mu$ is the carrier itself,\nso the report-rule "
                "bar (hatched) is not\na report point",
                xy=(0.30, len(exc) - 3.4), fontsize=9.5, color=EXCISE_COLOR)
    handles = [Patch(color=CRITICAL, label="$F > 1$  (provisional, as "
                                           "collected)"),
               Patch(color=SERIES[0], label="$F > \\mu$  (calibrated on the "
                                            "collected data)"),
               Patch(color=KEEP_COLOR, label="$F > \\eta\\,\\mu$  (report "
                                             "threshold for each channel)")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.025))
    etas = [s.eta_channel for s in st]
    ax.set_title("The threshold ladder, per channel\n"
                 "across the 17 kept channels the median masked fraction "
                 "falls from %.1f%% under the provisional rule to %.1f%% "
                 "at $\\eta$ = %.2f-%.2f"
                 % (100 * kept_prov, 100 * kept_med, min(etas), max(etas)),
                 fontsize=12.5, loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig11_ladder_summary.png"))


def fig_histograms(st, out):
    n = len(st)
    ncol, nrow = 4, (n + 3) // 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.4, 2.35 * nrow),
                             sharex=True)
    edges = np.logspace(np.log10(0.85), np.log10(400), 150)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, s in zip(axes.ravel(), st):
        c, cal = s.c, s.cal
        f = c.fstat[s.fmask] / c.mu0
        ax.hist(np.clip(f, edges[0], edges[-1]), bins=edges, color=INK,
                alpha=0.80, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color=CRITICAL, lw=1.4)
        ax.axvline(cal.mu / c.mu0, color=SERIES[0], lw=1.4, ls=(0, (1, 2)))
        ax.axvline(s.eta_channel * cal.mu / c.mu0, color=KEEP_COLOR, lw=1.4,
                   ls=(0, (4, 3)))
        annotate_channel(ax, "ch %d" % c.ch, loc=(0.015, 0.885))
        ax.annotate("$\\eta\\,%.2f$   masks $%.1f\\%%$"
                    % (s.eta_channel, 100 * s.occ_working),
                    xy=(0.985, 0.895), xycoords="axes fraction", ha="right",
                    fontsize=8.0, color=INK2)
        ax.tick_params(labelsize=8.5)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("$F / \\mu_0$   (the collection-time scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frames")
    handles = [plt.Line2D([], [], color=CRITICAL, lw=1.6,
                          label="$F > 1$ (provisional, as collected)"),
               plt.Line2D([], [], color=SERIES[0], lw=1.6, ls=(0, (1, 2)),
                          label="$\\mu$ calibrated on this era"),
               plt.Line2D([], [], color=KEEP_COLOR, lw=1.6, ls=(0, (4, 3)),
                          label="$F > \\eta\\,\\mu$, report threshold")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Latest-era statistic per channel: a narrow null core, a "
                 "heavy transient tail, and a carrier lobe where one exists",
                 fontsize=13, x=0.5, y=1.002)
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig03_histograms.png"))


def _spectrum_panel(ax, c, span_khz, center="channel"):
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


def fig_spectra(st, out, span_khz=195.3, name="fig04_wide_spectra.png",
                title=None, center="channel", xlabel=None):
    n = len(st)
    ncol, nrow = 4, (n + 3) // 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.4, 2.3 * nrow),
                             sharex=True)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, s in zip(axes.ravel(), st):
        _spectrum_panel(ax, s.c, span_khz, center=center)
        annotate_channel(ax, "ch %d" % s.ch)
        ax.tick_params(labelsize=8.5)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(xlabel or
                          "RF offset from CHIME channel centre  [kHz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("dB rel. median")
    handles = [Patch(color=SERIES[0], alpha=0.35, label="target coarse bin"),
               Patch(color=SERIES[1], alpha=0.35,
                     label="$\\pm 2$-bin guard references"),
               plt.Line2D([], [], color=SERIES[2], lw=1.4, ls=(0, (4, 3)),
                          label="synthesized pilot position"),
               plt.Line2D([], [], color=MUTED, lw=1.4, ls=(0, (1, 2)),
                          label="channel centre (instrumental)")]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(title or ("Archive-integrated channel spectra: every carrier "
                           "in the 390.6 kHz CHIME channel"),
                 fontsize=13, x=0.5, y=1.002)
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, name))


def fig_ladder(st, out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    etas = np.geomspace(0.9, 60, 120)
    for s in st:
        f = s.c.fstat[s.fmask] / s.cal.mu
        occ = np.array([(f > e).mean() for e in etas])
        ax = a1 if s.verdict == "excise" else a0
        ax.plot(etas, np.maximum(occ, 1e-4), color=col_of(s), lw=1.5,
                alpha=0.85)
        ax.scatter([s.eta_channel], [max(s.occ_working, 1e-4)], s=30,
                   color=col_of(s), zorder=4, edgecolor=SURFACE, linewidth=0.6)
        j = int(np.argmin(np.abs(etas - 2.4)))
        ax.annotate(str(s.ch), (etas[j], max(occ[j], 1e-4)),
                    textcoords="offset points", xytext=(3, 2), fontsize=8,
                    color=INK2)
    for ax, ttl in ((a0, "17 kept channels"), (a1, "6 excised channels")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(1.0, color=SERIES[0], lw=1.3, ls=(0, (1, 2)))
        ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (1, 2)))
        ax.set_xlabel("threshold multiplier $\\eta$   ($F > \\eta\\,\\mu$)")
        ax.set_title(ttl, fontsize=12, loc="left")
    a0.set_ylabel("fraction of latest-era frames masked")
    a0.annotate("$\\eta=1$", (1.01, 1.3e-4), color=SERIES[0], fontsize=9)
    fig.suptitle("Threshold ladder on the calibrated null: what each channel "
                 "costs at each $\\eta$\n"
                 "markers are report points, not an operational export",
                 fontsize=13, x=0.5, y=1.02)
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig06_threshold_ladder.png"))


def fig_era_effect(st, out):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for s in st:
        # same threshold rule on both populations: each calibrates its own mu
        f_all = s.c.fstat
        x = float(np.mean(f_all > s.eta_channel * s.blind.mu))
        y = s.occ_working
        col = col_of(s)
        multi = len(s.segs) > 1
        ax.plot([x, x], [x, y], color=col, lw=1.1, alpha=0.5, zorder=1)
        ax.scatter([x], [y], s=72 if multi else 34, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=0.7,
                   marker="o" if multi else ".")
        if multi or abs(x - y) > 0.05:
            ax.annotate("ch %d" % s.ch, (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=9, color=INK2)
    lim = [1e-3, 1.15]
    ax.plot(lim, lim, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("masked fraction, era-blind (whole archive)")
    ax.set_ylabel("masked fraction, latest era only")
    ax.annotate("below the diagonal:\nera resolution frees the channel",
                xy=(0.24, 0.0055), fontsize=10, color=INK2)
    ax.annotate("above:\nthe latest era\nis the dirty one", xy=(0.0055, 0.42),
                fontsize=10, color=INK2)
    ax.set_title("Why the latest era is the right population\n"
                 "large markers are the seven channels with a recovered "
                 "transition; $F > \\eta\\,\\mu$ with $\\mu$ calibrated "
                 "separately on each population", fontsize=12.5, loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig07_era_effect.png"))


def fig_dispositions(st, out):
    order = sorted(st, key=lambda s: (s.verdict == "excise", s.occ_working))
    n = len(order)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.8, 7.4), sharey=True)
    y = np.arange(n)
    cols = [col_of(s) for s in order]
    a0.barh(y, [s.occ_working for s in order], color=cols, height=0.66)
    a0.axvline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)))
    a0.annotate("excision line (50%)", xy=(0.51, 0.6), color=INK2,
                fontsize=9.5)
    a0.set_yticks(y)
    a0.set_yticklabels(["ch %d" % s.ch for s in order], fontsize=9.5)
    a0.set_xlabel("latest-era frames masked at $F > \\eta\\,\\mu$")
    a0.set_xlim(0, 1.04)
    a0.set_title("Masking load", fontsize=12, loc="left")

    a1.barh(y, [max(s.cal.mu_shift_db, 3e-3) for s in order], color=cols,
            height=0.66)
    a1.set_xscale("log")
    a1.axvline(ST.CARRIER_DOMINATED_LEVEL_DB, color=INK2, lw=1.2,
               ls=(0, (4, 3)))
    a1.annotate("carrier-dominated", xy=(3.4, 0.6), color=INK2, fontsize=9.5)
    a1.set_xlabel("$10\\log_{10}(\\mu/\\mu_0)$  [dB]")
    a1.set_title("Evidence: does a null exist in this era at all?",
                 fontsize=12, loc="left")
    handles = [Patch(color=KEEP_COLOR, label="keep, light masking"),
               Patch(color=MASK_COLOR, label="keep, heavy masking"),
               Patch(color=EXCISE_COLOR, label="excise")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Historical report labels from the latest era alone -- all "
                 "23 reproduce the published policy", fontsize=13,
                 x=0.5, y=1.0)
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig08_dispositions.png"))


def fig_census(st, out):
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    for s in st:
        c = s.c
        for p in S.peak_census(c):
            d = p.offset_from_pilot_hz / 1e3
            if p.kind == "pilot":
                m, col, sz = "o", SERIES[0], 90
            elif p.kind == "channel-centre artefact":
                m, col, sz = "x", MUTED, 52
            else:
                m, col, sz = "^", SERIES[1], 74
            ax.scatter([d], [c.ch], marker=m, s=sz, color=col, zorder=3,
                       edgecolor=SURFACE if m != "x" else None, linewidth=0.7)
            if p.kind == "secondary":
                ax.annotate("%.0f dB" % p.db_rel_median, (d, c.ch),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=INK2)
        t = S.target_coarse_offset_hz(c) - c.pilot_offset_hz
        ax.plot([(t - COARSE_HZ / 2) / 1e3, (t + COARSE_HZ / 2) / 1e3],
                [c.ch, c.ch], color=SERIES[0], lw=5, alpha=0.20,
                solid_capstyle="butt", zorder=1)
        for g in S.guard_reference_offsets_hz(c):
            gg = g - c.pilot_offset_hz
            ax.plot([(gg - COARSE_HZ / 2) / 1e3, (gg + COARSE_HZ / 2) / 1e3],
                    [c.ch, c.ch], color=SERIES[1], lw=5, alpha=0.20,
                    solid_capstyle="butt", zorder=1)
    ax.set_yticks([s.ch for s in st])
    ax.set_xlabel("RF offset from the synthesized pilot position  [kHz]")
    ax.set_ylabel("DTV physical channel")
    ax.set_xlim(-30, 30)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0],
                          label="pilot carrier"),
               plt.Line2D([], [], marker="^", ls="", color=SERIES[1],
                          label="secondary carrier"),
               plt.Line2D([], [], marker="x", ls="", color=MUTED,
                          label="channel-centre instrumental line"),
               Patch(color=SERIES[0], alpha=0.35, label="target coarse bin"),
               Patch(color=SERIES[1], alpha=0.35, label="guard references")]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title("Carrier census within $\\pm 30$ kHz of each pilot\n"
                 "the line at every channel's centre is instrumental, not a "
                 "transmitter", fontsize=12.5, loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig09_carrier_census.png"))


def fig_residual_tolerance(st, out):
    """Residual against tolerance at both ends of the coherence bracket."""
    have = [s for s in st
            if s.thresholds and s.thresholds.get("r_tol_dilation")]
    if not have:
        return None
    order = sorted(
        have, key=lambda s: -(s.thresholds.get("r_cost_cap") or 0))
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11.0, 7.6))
    for k, s in enumerate(order):
        rt = s.thresholds["r_tol_dilation"]
        cap = (s.thresholds.get("r_cost_cap") or np.nan) / rt
        th = (s.thresholds.get("r_cost_thermal") or np.nan) / rt
        ax.plot([th, cap], [k, k], color=MUTED, lw=1.6, zorder=1,
                solid_capstyle="round")
        ax.scatter([cap], [k], s=52, color=EXCISE_COLOR, zorder=3,
                   edgecolor=SURFACE, linewidth=0.6)
        ax.scatter([th], [k], s=52, color=SERIES[0], zorder=3,
                   edgecolor=SURFACE, linewidth=0.6,
                   marker="o" if th <= 1 else "X")
    ax.axvline(1.0, color=INK, lw=1.6)
    ax.annotate("tolerance", xy=(1.12, len(order) - 1.4), color=INK,
                fontsize=10)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(["ch %d%s" % (s.ch, "  $\\tau$ measured"
                                     if s.thresholds.get("tau_measured") else "")
                        for s in order], fontsize=9.5)
    ax.set_xlabel("masked residual $r$ / channel bias tolerance $r_{\\rm tol}$")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[0],
                          label="thermal end ($n_{\\rm coh}=1$)"),
               plt.Line2D([], [], marker="X", ls="", color=SERIES[0],
                          label="thermal end, still outside tolerance"),
               plt.Line2D([], [], marker="o", ls="", color=EXCISE_COLOR,
                          label="measured-or-sidereal-cap end (adopted)")]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("The coherence bracket, and the end this analysis stands "
                 "on\nevery residual at the red end is an upper bound, so a "
                 "channel outside tolerance there is uncertified rather than "
                 "disqualified", fontsize=12.5, loc="left")
    fig.tight_layout()
    return save_fig(
        fig, os.path.join(out, "fig12_residual_tolerance_bracket.png"))


def fig_eta(st, out):
    """The per-channel threshold, and what actually sets it."""
    have = [s for s in st if s.eta_is_per_channel]
    if not have:
        return None
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12.6, 8.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.5, 1]))
    chs = [s.ch for s in have]
    etas = [s.eta_channel for s in have]
    a0.bar(chs, etas, color=[col_of(s) for s in have], width=0.72)
    for s in have:
        a0.annotate("%.2f" % s.eta_channel, (s.ch, s.eta_channel),
                    ha="center", va="bottom", fontsize=8, color=INK2)
    a0.axhline(ETA, color=INK2, lw=1.2, ls=(0, (4, 3)))
    a0.annotate("the single global rule, $\\eta = %.1f$" % ETA,
                xy=(14.0, ETA + 0.03), color=INK2, fontsize=9.5)
    a0.set_ylim(0.95, max(etas) * 1.12)
    a0.set_ylabel("$\\eta$   ($F > \\eta\\,\\mu$)")
    a0.set_title("Each channel gets its own threshold\n"
                 "cost-optimal $\\eta$ from the survey-time trade "
                 "$(1+r)/(1-f)$ at the conservative coherence bound, with the "
                 "10 dB fine-stage credit applied", fontsize=12.5, loc="left")

    dil = [s.thresholds.get("r_tol_dilation") for s in have]
    gro = [s.thresholds.get("r_tol_growth") for s in have]
    a1.plot(chs, dil, marker="o", ms=4, color=SERIES[1], lw=1.4,
            label="acoustic-dilation tolerance")
    a1.plot(chs, gro, marker="s", ms=4, color=SERIES[0], lw=1.4,
            label="growth-rate tolerance")
    a1.set_yscale("log")
    a1.set_xticks(chs)
    a1.set_xticklabels([str(c) for c in chs], fontsize=8.5)
    a1.set_xlabel("DTV physical channel   (rising channel number = rising "
                  "frequency = falling redshift)")
    a1.set_ylabel("$r_{\\rm tol}$")
    a1.legend(loc="upper right", ncol=2)
    a1.set_title("The redshift-bin tolerances behind it: the growth-rate tier "
                 "varies by only 1.3x across the whole band", fontsize=11,
                 loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig13_eta_per_channel.png"))


def fig_mask_effect(st, out):
    """Latest-era fine PSD before and after masking, one panel per channel."""
    n = len(st)
    ncol, nrow = 4, (n + 3) // 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.4, 2.35 * nrow),
                             sharex=True)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, s in zip(axes.ravel(), st):
        thr = s.eta_channel * s.cal.mu
        rf, before, after, stats = S.era_fine_spectrum_masked(
            s.c, s.fmask, thr)
        ax.plot(rf, before, color=MUTED, lw=1.2)
        ax.plot(rf, after, color=INK, lw=1.2)
        ax.fill_between(rf, after, before, color=col_of(s), alpha=0.18, lw=0)
        annotate_channel(ax, "ch %d" % s.ch, loc=(0.015, 0.885))
        ax.annotate("$-$%.1f dB  %.0f%% kept"
                    % (stats["band_suppression_db"],
                       100 * stats["kept_fraction"]),
                    xy=(0.985, 0.90), xycoords="axes fraction", ha="right",
                    fontsize=7.6, color=INK2)
        ax.set_xlim(-1600, 1600)
        ax.tick_params(labelsize=8.5)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("RF offset from target coarse bin  [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("dB rel. $\\mu_0$")
    handles = [plt.Line2D([], [], color=MUTED, lw=1.8,
                          label="before masking (all latest-era frames)"),
               plt.Line2D([], [], color=INK, lw=1.8,
                          label="after masking ($F \\leq \\eta\\,\\mu$)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("What the threshold removes: latest-era time-averaged fine "
                 "spectrum, before and after masking\n"
                 "means over frames, because mean power is what integrates "
                 "into a map", fontsize=13, x=0.5, y=1.002)
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig14_mask_effect.png"))


def fig_bracket_stability(st, out):
    """How far eta moves between the two ends of the coherence bracket."""
    have = [x for x in st if x.eta_bracket_ratio == x.eta_bracket_ratio]
    if not have:
        return None
    order = sorted(have, key=lambda x: -x.eta_bracket_ratio)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11.2, 7.8))
    for k, x in enumerate(order):
        col = EXCISE_COLOR if x.verdict == "excise" else KEEP_COLOR
        ax.plot([x.eta_channel, x.eta_thermal], [k, k], color=MUTED, lw=1.6,
                zorder=1, solid_capstyle="round")
        ax.scatter([x.eta_channel], [k], s=46, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=0.6)
        ax.scatter([x.eta_thermal], [k], s=46, color=col, zorder=3,
                   marker="D", edgecolor=SURFACE, linewidth=0.6)
        if x.eta_is_identified:
            ax.annotate("identified", (max(x.eta_thermal, x.eta_channel), k),
                        textcoords="offset points", xytext=(9, -3),
                        fontsize=8.2, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["ch %d%s" % (x.ch, "  $\\tau$ meas."
                      if x.thresholds.get("tau_measured")
                      else "") for x in order], fontsize=9.5)
    ax.set_xlabel("$\\eta$   (circle: adopted cap end;  diamond: thermal end)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=KEEP_COLOR,
                          label="kept channel"),
               plt.Line2D([], [], marker="o", ls="", color=EXCISE_COLOR,
                          label="excised channel")]
    ax.legend(handles=handles, loc="upper right")
    n_id = sum(1 for x in have if x.eta_is_identified)
    ax.set_title("Is the per-channel threshold identified?\n"
                 "the bracket collapses on %d of %d channels -- and every one "
                 "of them is excised, so $\\eta$ is pinned exactly where it "
                 "no longer matters" % (n_id, len(have)),
                 fontsize=12.5, loc="left")
    fig.tight_layout()
    return save_fig(fig, os.path.join(out, "fig15_bracket_stability.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresholds", default=None)
    args = ap.parse_args(argv)
    args.products = args.products or str(P.PER_PILOT)
    args.out = args.out or str(P.OUT / "figures")
    args.thresholds = args.thresholds or str(P.THRESHOLD_TABLE)
    setup_style()
    st = ST.build(args.products, threshold_csv=args.thresholds)
    made = [fig_timeline(st, args.out),
            fig_mu(st, args.out),
            fig_histograms(st, args.out),
            fig_spectra(st, args.out),
            fig_spectra(st, args.out, span_khz=15.0, center="pilot",
                        name="fig05_zoom_spectra.png",
                        xlabel="RF offset from the synthesized pilot "
                               "position  [kHz]",
                        title="Zoom on each pilot: the main carrier, its "
                              "neighbours, and the guard references"),
            fig_ladder(st, args.out),
            fig_era_effect(st, args.out),
            fig_dispositions(st, args.out),
            fig_census(st, args.out),
            fig_ladder_summary(st, args.out),
            fig_residual_tolerance(st, args.out),
            fig_eta(st, args.out),
            fig_mask_effect(st, args.out),
            fig_bracket_stability(st, args.out)]
    for p in made:
        if p:
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
