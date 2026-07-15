#!/usr/bin/env python3
"""Full 23-channel spectra grids from all_spectra.npz (full-depth per-pilot
accumulations), with detector reference positions marked on the zoom."""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
B = _paths.RESULTS
INK, AFTER, PILOT_C, LINE_C, REF_C = "0.35", "#0072B2", "#D55E00", "#009E73", "#7B4FA6"
SUPPRESSED = {14, 21, 25, 28, 36}
NFFT, SR, SENSE = 16384, 390625.0, -1.0
REF_KHZ = 2 * 3051.7578125 / 1e3          # references at +/-2 fine bins

z = np.load(_paths.SPECTRA)
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})
lines = {}
for r in csv.DictReader(open(B / "transmitter_census/extracted_lines.csv")):
    lines.setdefault(int(r["rf_channel"]), []).append(float(r["offset_hz"]))

k = np.arange(NFFT)
f_bb = np.where(k < NFFT // 2, k, k - NFFT).astype(np.float64) * (SR / NFFT)


def panel(ax, ch, span_khz, zoom):
    pilot, center, fid, n_valid, n_masked = z[f"ch{ch}_meta"]
    mf = n_masked / n_valid if n_valid else float("nan")
    before = z[f"ch{ch}_before"] / max(n_valid, 1.0)
    after = z[f"ch{ch}_after"] / max(n_valid - n_masked, 1.0)
    off = (center + SENSE * f_bb - pilot) / 1e3
    s = np.argsort(off)
    o, b, a = off[s], before[s], after[s]
    ref = np.median(b[b > 0])
    sel = np.abs(o) <= span_khz
    ax.plot(o[sel], 10 * np.log10(np.maximum(b[sel], ref * 1e-4) / ref),
            color=INK, lw=0.55, label="before mask")
    ax.plot(o[sel], 10 * np.log10(np.maximum(a[sel], ref * 1e-4) / ref),
            color=AFTER, lw=0.55, alpha=0.85, label="after mask")
    ax.axvline(0.0, color=PILOT_C, ls="--", lw=0.7)
    if zoom:
        for rk in (-REF_KHZ, REF_KHZ):
            ax.axvline(rk, color=REF_C, ls="-.", lw=0.6, alpha=0.8)
    dc = (center - pilot) / 1e3
    if abs(dc) <= span_khz:
        ax.axvline(dc, color="0.6", ls=":", lw=0.7)
    if zoom:
        for lo in lines.get(ch, []):
            if abs(lo / 1e3) <= span_khz:
                ax.plot([lo / 1e3], [ax.get_ylim()[1]], marker="v", ms=2.6,
                        color=LINE_C, clip_on=False)
    tcol = PILOT_C if ch in SUPPRESSED else "black"
    ax.set_title(f"ch{ch} (fid {int(fid)})  $f_{{\\rm mask}}$={mf:.2f}",
                 fontsize=7, color=tcol, pad=2)
    ax.tick_params(labelsize=5.5)
    ax.grid(color="0.93", lw=0.35)
    ax.set_axisbelow(True)


for span, zoom, fname, title in (
        (200.0, False, "fig_spectra_all23_fullspan",
         "Integrated spectra, full coarse channel, all 23 channels "
         "(mean per valid frame, dB rel. channel median)"),
        (20.0, True, "fig_spectra_all23_pilot_zoom",
         "Integrated spectra, $\\pm$20 kHz about the nominal pilot "
         "(references at $\\pm$6.1 kHz marked; census lines as ticks)")):
    fig, axes = plt.subplots(6, 4, figsize=(11.5, 12.6), sharex=True)
    for j, ch in enumerate(chans):
        panel(axes.flat[j], ch, span, zoom)
    for j in range(len(chans), 24):
        axes.flat[j].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("offset from pilot [kHz]", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("dB", fontsize=6.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, label="before mask"),
               Line2D([], [], color=AFTER, label="after mask"),
               Line2D([], [], color=PILOT_C, ls="--", label="nominal pilot"),
               Line2D([], [], color="0.6", ls=":", label="coarse-channel DC")]
    if zoom:
        handles += [Line2D([], [], color=REF_C, ls="-.",
                           label="reference bins ($\\pm$2)"),
                    Line2D([], [], color=LINE_C, marker="v", ls="",
                           label="extracted line")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(OUT / f"{fname}.png", dpi=230, bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    print("wrote", fname)
