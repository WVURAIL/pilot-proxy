#!/usr/bin/env python3
"""Full 23-channel spectra grids from all_spectra.npz (full-depth per-pilot
accumulations), with the detector cells (target, skipped guards, references)
shaded from the shipped weight-bank manifest."""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib
from pilot_proxy.detector_weights import DetectorWeightBank

plt = setup_matplotlib()
OUT = _paths.OUT
B = _paths.RESULTS
INK, AFTER, PILOT_C, LINE_C, REF_C = "0.35", "#0072B2", "#D55E00", "#009E73", "#7B4FA6"
TGT_FILL, GUARD_FILL, REF_FILL = "#D55E00", "0.55", "#7B4FA6"
SUPPRESSED = {14, 21, 25, 28, 36}
NFFT, SR, SENSE = 16384, 390625.0, -1.0
FB_KHZ = 3051.7578125 / 1e3               # fine-bin (cell) width

z = np.load(_paths.SPECTRA)
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})
lines = {}
for r in csv.DictReader(open(B / "transmitter_census/extracted_lines.csv")):
    lines.setdefault(int(r["rf_channel"]), []).append(float(r["offset_hz"]))

# Detector cell placement from the shipped bank manifest. The bank's
# normalized-frequency convention puts the coarse-channel DC at 0.5, so a
# term at normalized nu sits at sky frequency center + (nu - 0.5) * SR
# (verified: target lands on the nominal pilot to <1 Hz on all 23 channels).
bank = DetectorWeightBank(
    explicit_path=str(_paths.REPO / "weights/chime_dtv_weights_k128.bin"))
cells = {}
for ch in chans:
    pilot, center, *_ = z[f"ch{ch}_meta"]
    lay = bank.layout_for_physical_channel(ch)

    def _off_khz(nu):
        return (center + (nu - 0.5) * SR - pilot) / 1e3

    nu_t = float(lay["target_normalized_frequency"])
    t = _off_khz(nu_t)
    refs = []
    guards = []
    for side, nkey, bkey in (
            ("lower", "lower_reference_normalized_frequency",
             "lower_reference_offset_bins"),
            ("upper", "upper_reference_normalized_frequency",
             "upper_reference_offset_bins")):
        r_off = _off_khz(lay[nkey])
        nb = int(lay[bkey])
        wrapped = abs(r_off - (t + nb * FB_KHZ)) > 0.5 * FB_KHZ
        refs.append((r_off, wrapped, side))
        # skipped guard bins: same modular walk as the placement rule
        sgn = 1 if nb > 0 else -1
        guards.extend(_off_khz((nu_t + sgn * g / 128.0) % 1.0)
                      for g in range(1, abs(nb)))
    cells[ch] = {"target": t, "refs": refs, "guards": guards}

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
    # detector cells (one fine bin wide, centred on each term / skipped bin)
    cc = cells[ch]
    h = 0.5 * FB_KHZ
    ax.axvspan(cc["target"] - h, cc["target"] + h, color=TGT_FILL,
               alpha=0.16, lw=0, zorder=0)
    for g in cc["guards"]:
        if abs(g) <= span_khz:
            ax.axvspan(g - h, g + h, color=GUARD_FILL, alpha=0.16, lw=0,
                       zorder=0)
    for r_off, wrapped, side in cc["refs"]:
        if abs(r_off) <= span_khz:
            ax.axvspan(r_off - h, r_off + h, color=REF_FILL, alpha=0.20,
                       lw=0, zorder=0)
        else:
            ax.text(0.985, 0.06, f"{side} ref "
                    + ("wrapped to " if wrapped else "at ")
                    + f"{r_off:+.0f} kHz $\\rightarrow$",
                    transform=ax.transAxes, fontsize=5.0, color=REF_C,
                    ha="right")
    ax.axvline(0.0, color=PILOT_C, ls="--", lw=0.7)
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
         "(mean per valid frame, dB rel. channel median; detector cells "
         "shaded)"),
        (20.0, True, "fig_spectra_all23_pilot_zoom",
         "Integrated spectra, $\\pm$20 kHz about the nominal pilot "
         "(target / skipped-guard / reference cells shaded, one fine bin "
         "wide; census lines as ticks)")):
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
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK, label="before mask"),
               Line2D([], [], color=AFTER, label="after mask"),
               Line2D([], [], color=PILOT_C, ls="--", label="nominal pilot"),
               Line2D([], [], color="0.6", ls=":", label="coarse-channel DC"),
               Patch(facecolor=TGT_FILL, alpha=0.16, label="target cell"),
               Patch(facecolor=GUARD_FILL, alpha=0.16,
                     label="skipped guard"),
               Patch(facecolor=REF_FILL, alpha=0.20,
                     label="reference cells ($\\pm$2 bins)")]
    if zoom:
        handles += [Line2D([], [], color=LINE_C, marker="v", ls="",
                           label="extracted line")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(OUT / f"{fname}.png", dpi=230, bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    print("wrote", fname)
