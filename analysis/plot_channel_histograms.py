#!/usr/bin/env python3
"""Per-channel distributions of the decision statistic and the reported level.

One row per channel. The left panel is the coarse statistic F near its null;
the right panel is the shelf level the product reports for each frame. Both
are log-count, because the interesting parts are the tails.

Two vertical rules matter on the left panel, and the whole plate exists to
show where they sit relative to one another:

    F = mu0    the decision line. mu0 = 2||w0||^2 / (||w1||^2 + ||w2||^2) is an
               exact rational constant fixed by the quantised weight bank, and
               the deployed rule is reject <=> F > mu0.

    F = 1      where the *level* estimator becomes defined. The product sets
               pnr_bin_db = 10 log10(F - 1) exactly, so it references the null
               to unity rather than to mu0.

Those are not the same number, and on some channels they are in the opposite
order. Everything the right panel can and cannot say follows from that,
including missing floors on channels where mu0 < 1, which are an arithmetic
consequence rather than a statement about the transmitter.

The floor-provenance checks come from the released ``baonoise`` package
(bao-noise-tolerance): floor_provenance re-derives the shelf offset from the
product and verifies both the deployed rule and the level formula before
answering, so the numbers annotated on the plate are checked rather than
assumed. Install baonoise into the analysis environment to run this script.

    python3 analysis/plot_channel_histograms.py --out ~/paper/out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _style import (
    BASELINE, CRITICAL, INK, INK2, MUTED, SERIES, SURFACE,
    save_fig, setup_style)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from baonoise import residual as R

import _products as P

DEFAULT_CHANNELS = (32, 33, 34, 35, 36)


def allocation_mhz(ch: int) -> str:
    """The 6 MHz ATSC allocation for a physical channel (14-36)."""
    lo = 470 + 6 * (ch - 14)
    return f"{lo}-{lo + 6}"


# ------------------------------------------------- figure code
def _hist_stat(ax, row, xlim):
    """Decision statistic near its null, scaled by the null's own scatter."""
    x, edges = row["x"], np.linspace(*xlim, 131)
    inside = (x >= xlim[0]) & (x <= xlim[1])
    top = 1
    for m, c in ((row["kept"], SERIES[0]), (row["hit"], SERIES[1])):
        h, _, _ = ax.hist(x[m & inside], bins=edges, color=c, alpha=0.8,
                          lw=0, zorder=3)
        top = max(top, h.max())

    d = row["mu0_sigma"]
    ax.axvspan(min(0.0, d), max(0.0, d), lw=0, zorder=2, alpha=0.22,
               color=BASELINE if d > 0 else CRITICAL)
    ax.axvline(d, color=INK, lw=1.3, zorder=6)
    ax.axvline(0.0, color=MUTED, lw=1.1, ls=(0, (3, 2)), zorder=6)

    note = (f"{row['n_band']:,} kept,\nwith a level" if d > 0
            else f"{row['n_band']:,} masked,\nnone with a level")
    ax.annotate(note, xy=(0.5 * d, top * 1.25), xytext=(2.7, top * 2.0),
                ha="left", va="center", fontsize=8.5,
                color=INK2 if d > 0 else CRITICAL,
                arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=1, shrinkB=1,
                                color=MUTED if d > 0 else CRITICAL))

    for m, out, ha, xf, c in ((row["hit"], x > xlim[1], "right", 0.99, SERIES[1]),
                              (row["kept"], x < xlim[0], "left", 0.01, SERIES[0])):
        n = int((m & out).sum())
        if n:
            arrow = r"$\rightarrow$" if ha == "right" else r"$\leftarrow$"
            txt = f"{n:,} off-scale {arrow}" if ha == "right" else f"{arrow} {n:,} off-scale"
            ax.annotate(txt, xy=(xf, 0.02), xycoords="axes fraction", ha=ha,
                        va="bottom", fontsize=8.5, color=c, zorder=7,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.55, top * 5.0)
    ax.set_axisbelow(True)


def _hist_level(ax, row, xlim):
    """Reported shelf level, where the product defines one."""
    edges = np.linspace(*xlim, 121)
    for m, c in ((row["sliver"], SERIES[0]), (row["hit_lev"], SERIES[1])):
        if m.sum():
            ax.hist(np.clip(row["shelf"][m], *xlim), bins=edges, color=c,
                    alpha=0.8, lw=0, zorder=3)
    if np.isfinite(row["floor_db"]):
        ax.axvline(row["floor_db"], color=INK, lw=1.3, zorder=6)
    ax.axvline(row["floor_sigma_db"], color=INK, lw=1.1, ls=(0, (1, 1.6)),
               zorder=6)
    if row["n_undef"]:
        ax.annotate(rf"$\leftarrow$ {row['n_undef']:,} with no level "
                    rf"($F \leq 1$)",
                    xy=(0.015, 0.03), xycoords="axes fraction", ha="left",
                    va="bottom", fontsize=8.5, color=MUTED, zorder=7,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_axisbelow(True)


def _hist_titles(axL, axR, row):
    d = row["mu0_sigma"]
    where = (rf"decision line $+{d:.2f}\,\sigma$ from $F=1$" if d > 0 else
             rf"decision line ${d:.2f}\,\sigma$, below $F=1$")
    axL.set_title(rf"$\mu_0 = {row['mu0']:.6f}$,  {where}", fontsize=9.5,
                  color=INK if d > 0 else CRITICAL, pad=6)
    if np.isfinite(row["floor_db"]):
        head = (rf"{row['f_masked']*100:.1f}\% masked  ---  {row['n_sliver']:,} "
                rf"frames set the floor at ${row['floor_db']:.1f}$ dB")
    else:
        head = (rf"{row['f_masked']*100:.1f}\% masked  ---  "
                rf"no frame can set a floor")
    axR.set_title(head.replace(r"\%", "%"), fontsize=9.5, color=INK2, pad=6)


def _hist_legend(fig, y):
    fig.legend(handles=[
        Patch(facecolor=SERIES[0], alpha=0.8, label="kept by the mask"),
        Patch(facecolor=SERIES[1], alpha=0.8, label="masked"),
        Line2D([], [], color=INK, lw=1.3, label=r"$F = \mu_0$, the decision line"),
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (3, 2)),
               label=r"$F = 1$, above which a level exists"),
        Line2D([], [], color=INK, lw=1.1, ls=(0, (1, 1.6)),
               label=r"null scatter $\sigma_{\rm null}$"),
    ], loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False,
        fontsize=9.5, handlelength=2.2, columnspacing=2.6)


def fig_channel_histograms(rows: list[dict], outfile: Path,
                           xlim_stat=(-5.0, 8.0), xlim_level=(-92.0, 6.0)):
    """One row per channel: the decision statistic, and the level reported.

    The floor is read from frames lying between $F=1$ and $F=\\mu_0$, so it
    exists only where mu0 > 1, and where it exists it is under half a sigma
    of the null wide, which is why it tracks the weight bank rather than the
    sky. Channels with mu0 < 1 are titled in the critical color: for those the
    interval is empty for any dataset.
    """
    setup_style()
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 2.05 * n + 1.9),
                             gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    for i, (row, (axL, axR)) in enumerate(zip(rows, axes)):
        _hist_stat(axL, row, xlim_stat)
        _hist_level(axR, row, xlim_level)
        _hist_titles(axL, axR, row)
        axL.set_ylabel(f"ch{row['ch']}\n{row['band']} MHz", color=INK,
                       fontsize=10.5, labelpad=10)
        if i == n - 1:
            axL.set_xlabel(r"$(F-1) \, / \, \sigma_{\rm null}$")
            axR.set_xlabel("Reported shelf level "
                           r"$10\log_{10}(F-1) - 21.64$ [dB rel. system noise]")
        axR.tick_params(axis="x", labelbottom=(i == n - 1))
    fig.align_ylabels(axes[:, 0])
    fig.suptitle("Where each channel's frames fall, either side of the "
                 "decision line\n"
                 r"(the floor can only be read between $F=1$ and $F=\mu_0$; "
                 "on channels 35 and 36 those are reversed)",
                 fontsize=12, color=INK, y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.088,
                        hspace=0.55, wspace=0.17)
    _hist_legend(fig, 0.012)
    return save_fig(fig, outfile)


def fig_channel_histogram(row: dict, outfile: Path,
                          xlim_stat=(-5.0, 8.0), xlim_level=(-92.0, 6.0)):
    """A single channel's pair of panels, at the house landscape size."""
    setup_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    _hist_stat(axL, row, xlim_stat)
    _hist_level(axR, row, xlim_level)
    axL.set_xlabel(r"$(F-1) \, / \, \sigma_{\rm null}$")
    axR.set_xlabel("Reported shelf level [dB rel. system noise]")
    axL.set_ylabel("Frames per bin")
    floor = (rf"floor ${row['floor_db']:.1f}$ dB"
             if np.isfinite(row["floor_db"]) else "no floor available")
    fig.suptitle(f"Channel {row['ch']} ({row['band']} MHz, freq id "
                 f"{row['freq_id']})\n"
                 rf"$\mu_0 = {row['mu0']:.6f}$, "
                 rf"{row['f_masked']*100:.1f}% masked, {floor}",
                 fontsize=12, color=INK, y=0.99)
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.2)
    _hist_legend(fig, -0.06)
    return save_fig(fig, outfile)


# ------------------------------------------------- per-channel rows
def channel_row(path) -> dict:
    """Everything the figure needs, with the provenance checks run first.

    floor_provenance re-derives the shelf offset from the product and verifies
    both the deployed rule and the level formula before answering, so the
    numbers annotated on the plate are checked rather than assumed.
    """
    prov = R.floor_provenance(path)
    d = P.load_npz(path)
    v = d["valid"][:, 0].astype(bool)
    rej = d["reject_mask"][:, 0].astype(bool)
    F = d["fstat_raw"][:, 0]
    shelf = d["snr_shelf_db"][:, 0]
    mu0 = prov.mu0

    kept, hit = v & ~rej, v & rej
    lev = v & np.isfinite(shelf)
    band = v & (F > min(1.0, mu0)) & (F <= max(1.0, mu0))

    return dict(
        ch=prov.channel, freq_id=prov.freq_id,
        band=allocation_mhz(prov.channel),
        mu0=mu0, mu0_sigma=(mu0 - 1.0) / prov.sigma_null,
        x=(F - 1.0) / prov.sigma_null, shelf=shelf,
        kept=kept, hit=hit, sliver=kept & lev, hit_lev=hit & lev,
        n_valid=int(v.sum()), n_kept=prov.n_kept, n_sliver=prov.n_sliver,
        n_band=int(band.sum()), n_undef=int((v & ~lev).sum()),
        f_masked=float(rej[v].mean()),
        floor_db=prov.reported_db, floor_mu0_db=prov.mu0_implied_db,
        sigma=prov.sigma_null, sigma_spread=prov.sigma_spread,
        floor_sigma_db=prov.sigma_implied_db, verdict=prov.verdict,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=None,
                    help="per-pilot survey products (default: channels "
                         f"{DEFAULT_CHANNELS} under $PP_PER_PILOT)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--singles", action="store_true",
                    help="also write one figure per channel")
    args = ap.parse_args(argv)
    products = (args.products if args.products is not None
                else list(P.paths(channels=DEFAULT_CHANNELS).values()))

    rows = [channel_row(p) for p in products]
    args.out.mkdir(parents=True, exist_ok=True)

    out = fig_channel_histograms(rows, args.out / "fig5_channel_histograms.png")
    print(f"wrote {out} (+ .pdf)")
    if args.singles:
        for r in rows:
            fig_channel_histogram(
                r, args.out / f"fig5{chr(ord('a') + r['ch'] - 32)}_hist_ch{r['ch']}.png")
        print(f"wrote {args.out}/fig5*_hist_ch*.png (+ .pdf)")

    print(f"\n{'ch':>3} {'mu0':>12} {'masked':>7} {'kept':>7} {'sliver':>7} "
          f"{'no level':>9} {'floor':>8} {'10lg(mu0-1)':>12} "
          f"{'sigma_null':>11} {'x-spread':>8} {'floor(sig)':>10}")
    for r in rows:
        print(f"{r['ch']:3d} {r['mu0']:12.9f} {r['f_masked']*100:6.1f}% "
              f"{r['n_kept']:7,d} {r['n_sliver']:7,d} {r['n_undef']:9,d} "
              f"{r['floor_db']:8.2f} {r['floor_mu0_db']:12.2f} "
              f"{r['sigma']:11.3e} {r['sigma_spread']:8.1f} "
              f"{r['floor_sigma_db']:10.2f}")
    print()
    for r in rows:
        print(f"  ch{r['ch']}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
