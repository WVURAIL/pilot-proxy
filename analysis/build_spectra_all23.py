#!/usr/bin/env python3
"""Full 23-channel spectra grids from all_spectra.npz (full-depth per-pilot
accumulations), with the detector cells (target, skipped guards, references)
shaded from the shipped weight-bank manifest.

Three variants are written per view (suffix _before / _instrument_removed /
_after), all with shared per-channel y-limits so they can be flipped
between at the same scale:

  before             raw before-mask mean spectrum
  instrument_removed before-mask with the identified instrumental tones and
                     the coarse-channel DC spur notched to the local
                     running-median background
  after              mean spectrum after the ORIGINAL ANALYTIC
                     POSITIVE-EXCESS production mask (the masking recorded
                     in the archived products; NOT the candidate three-arm
                     rule), same notching applied

Notching is COSMETIC: it is applied only to these supplementary plotting
copies. Every detector and mask result in the paper uses unnotched data.

Tone search domain: outside +/-10 kHz of the nominal pilot (which covers
the +/-7.63 kHz detector-cell support plus a 7.63--10 kHz buffer that is
also excluded) and outside +/-100 Hz of the coarse-channel DC. Five of
the six lines found sit within 9.5 Hz (< half a 23.84 Hz spectral bin)
of rational fractions of the coarse sample rate (+/- SR/5, SR/3, 2SR/5)
in baseband and appear in one channel each -- consistent with
instrumental / digital-processing spurs, pending operations
confirmation. The nearest line is 73.6 kHz from detector-cell support.
The unidentified ch17 line (+157.29 kHz baseband) is retained. The
prominence columns in instrument_tones.csv are each measured against
that spectrum's own local running-median background, so the
before/after difference is a prominence change under the analytic mask,
not an absolute-amplitude change.
"""
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
INSTR_C = "#CC79A7"
SUPPRESSED = {14, 21, 25, 28, 36}
NFFT, SR, SENSE = 16384, 390625.0, -1.0
FB_KHZ = 3051.7578125 / 1e3               # fine-bin (cell) width

VARIANTS = ("before", "instrument_removed", "after")
VCOLOR = {"before": INK, "instrument_removed": INSTR_C, "after": AFTER}
VLABEL = {"before": "before mask",
          "instrument_removed": "before mask, instr.\\ tones notched",
          "after": "after analytic pos.-excess mask, instr.\\ tones notched"}
VTITLE = {"before": "before mask (raw)",
          "instrument_removed": "before mask, instrumental tones $+$ DC "
                                "notched to local background (plotting "
                                "copies only)",
          "after": "after the original analytic positive-excess mask "
                   "(not the candidate three-arm rule), instrumental "
                   "tones $+$ DC notched"}

# instrumental-tone detection / identification / notching parameters
INSTR_FRACS = {"SR/5": SR / 5, "SR/3": SR / 3, "2SR/5": 2 * SR / 5}
DET_DB = 4.0            # detection threshold above running-median background
IDENT_TOL_HZ = 30.0     # ~ half a 23.84 Hz bin, with margin
NOTCH_PAD = 3           # extra bins notched either side of a detected group
MED_WIN = 401           # running-median window (~9.6 kHz)

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


def _running_median(p, w=MED_WIN):
    """Circular running median (the spectrum is periodic in baseband)."""
    from numpy.lib.stride_tricks import sliding_window_view
    half = w // 2
    pad = np.concatenate([p[-half:], p, p[:half]])
    return np.median(sliding_window_view(pad, w), axis=1)


def _notch(power, bg, groups):
    """Replace the bins of each group (+/- NOTCH_PAD) with the background."""
    out = power.copy()
    for grp in groups:
        i0 = max(int(grp[0]) - NOTCH_PAD, 0)
        i1 = min(int(grp[-1]) + NOTCH_PAD, power.size - 1)
        out[i0:i1 + 1] = bg[i0:i1 + 1]
    return out


# ---- per-channel streams: detect + document + notch instrumental tones ----
data = {}
tone_rows = []
for ch in chans:
    pilot, center, fid, n_valid, n_masked = z[f"ch{ch}_meta"]
    mf = n_masked / n_valid if n_valid else float("nan")
    before = z[f"ch{ch}_before"] / max(n_valid, 1.0)
    after = z[f"ch{ch}_after"] / max(n_valid - n_masked, 1.0)
    off = (center + SENSE * f_bb - pilot) / 1e3
    s = np.argsort(off)
    o, b, a, fb = off[s], before[s], after[s], f_bb[s]
    bg_b, bg_a = _running_median(b), _running_median(a)
    exc_b = 10 * np.log10(np.maximum(b, 1e-30) / np.maximum(bg_b, 1e-30))
    exc_a = 10 * np.log10(np.maximum(a, 1e-30) / np.maximum(bg_a, 1e-30))
    # coarse-channel DC spur: grow from the DC bin while above 3 dB
    k0 = int(np.argmin(np.abs(fb)))
    i0 = i1 = k0
    while i0 > max(k0 - 10, 1) and exc_b[i0 - 1] > 3.0:
        i0 -= 1
    while i1 < min(k0 + 10, b.size - 2) and exc_b[i1 + 1] > 3.0:
        i1 += 1
    kill = [np.arange(i0, i1 + 1)]
    # narrow-tone candidates outside the detector cells and away from DC
    cand = np.flatnonzero((exc_b > DET_DB) & (np.abs(o) > 10.0)
                          & (np.abs(fb) > 100.0))
    if cand.size:
        for grp in np.split(cand, np.flatnonzero(np.diff(cand) > 3) + 1):
            p = grp[int(np.argmax(exc_b[grp]))]
            ident = next((("$-$" if fb[p] < 0 else "$+$") + name
                          for name, fv in INSTR_FRACS.items()
                          if abs(abs(fb[p]) - fv) <= IDENT_TOL_HZ), "")
            removed = bool(ident)
            if removed:
                kill.append(grp)
            tone_rows.append({
                "atsc_channel": ch, "freq_id": int(fid),
                "f_bb_hz": f"{fb[p]:.1f}",
                "offset_from_pilot_khz": f"{o[p]:.2f}",
                "prominence_db_before": f"{exc_b[p]:.1f}",
                "prominence_db_after_analytic_mask": f"{exc_a[p]:.1f}",
                "width_bins": int(len(grp)),
                "identification": (ident.replace("$", "")
                                   if ident else "unidentified"),
                "action": ("notched_in_supplement_only" if removed
                           else "retained")})
    data[ch] = {
        "meta": (pilot, center, fid, mf),
        "o": o,
        "before": b,
        "instrument_removed": _notch(b, bg_b, kill),
        "after": _notch(a, bg_a, kill),
        "ref": np.median(b[b > 0]),     # shared dB zero for all variants
    }

with open(OUT / "instrument_tones.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tone_rows[0].keys()))
    w.writeheader()
    w.writerows(tone_rows)
n_rm = sum(r["action"] == "notched_in_supplement_only" for r in tone_rows)
print(f"wrote instrument_tones.csv: {len(tone_rows)} tones "
      f"({n_rm} notched in supplement only, {len(tone_rows) - n_rm} "
      f"retained) + DC notch on all channels (plotting copies only)")


def panel(ax, ch, span_khz, zoom, which):
    d = data[ch]
    pilot, center, fid, mf = d["meta"]
    o, ref = d["o"], d["ref"]
    dc = (center - pilot) / 1e3
    if span_khz is None:            # true full coarse channel (dc-centred)
        xlo, xhi = dc - SR / 2e3, dc + SR / 2e3
    else:
        xlo, xhi = -span_khz, span_khz
    sel = (o >= xlo) & (o <= xhi)

    def dbrel(p):
        return 10 * np.log10(np.maximum(p[sel], ref * 1e-4) / ref)

    ys = {v: dbrel(d[v]) for v in VARIANTS}
    ax.plot(o[sel], ys[which], color=VCOLOR[which], lw=0.55,
            label=VLABEL[which])
    ax.set_xlim(xlo, xhi)
    # shared y-limits across all three variants so the set can be flipped
    # between at the same scale
    lo_y = min(float(y.min()) for y in ys.values())
    hi_y = max(float(y.max()) for y in ys.values())
    pad = 0.05 * max(hi_y - lo_y, 1.0)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    # detector cells (one fine bin wide, centred on each term / skipped bin)
    cc = cells[ch]
    h = 0.5 * FB_KHZ
    ax.axvspan(cc["target"] - h, cc["target"] + h, color=TGT_FILL,
               alpha=0.16, lw=0, zorder=0)
    for g in cc["guards"]:
        if xlo <= g <= xhi:
            ax.axvspan(g - h, g + h, color=GUARD_FILL, alpha=0.16, lw=0,
                       zorder=0)
    for r_off, wrapped, side in cc["refs"]:
        if xlo <= r_off <= xhi:
            ax.axvspan(r_off - h, r_off + h, color=REF_FILL, alpha=0.20,
                       lw=0, zorder=0)
        else:
            ax.text(0.985, 0.06, f"{side} ref "
                    + ("wrapped to " if wrapped else "at ")
                    + f"{r_off:+.0f} kHz $\\rightarrow$",
                    transform=ax.transAxes, fontsize=5.0, color=REF_C,
                    ha="right")
    ax.axvline(0.0, color=PILOT_C, ls="--", lw=0.7)
    if xlo <= dc <= xhi:
        ax.axvline(dc, color="0.6", ls=":", lw=0.7)
    if zoom:
        for lo in lines.get(ch, []):
            if xlo <= lo / 1e3 <= xhi:
                ax.plot([lo / 1e3], [ax.get_ylim()[1]], marker="v", ms=2.6,
                        color=LINE_C, clip_on=False)
    tcol = PILOT_C if ch in SUPPRESSED else "black"
    ax.set_title(f"ch{ch} (fid {int(fid)})  $f_{{\\rm mask}}$={mf:.2f}",
                 fontsize=7, color=tcol, pad=2)
    ax.tick_params(labelsize=5.5)
    ax.grid(color="0.93", lw=0.35)
    ax.set_axisbelow(True)


for span, zoom, fbase, tbase in (
        (None, False, "fig_spectra_all23_fullspan",
         "Integrated spectra, full coarse channel (dc-centred), all 23 "
         "channels (mean per valid frame, dB rel. channel median; detector "
         "cells shaded)"),
        (20.0, True, "fig_spectra_all23_pilot_zoom",
         "Integrated spectra, $\\pm$20 kHz about the nominal pilot "
         "(target / skipped-guard / reference cells shaded, one fine bin "
         "wide; extracted spectral lines as ticks)")):
    for which in VARIANTS:
        fig, axes = plt.subplots(6, 4, figsize=(11.5, 12.6), sharex=zoom)
        for j, ch in enumerate(chans):
            panel(axes.flat[j], ch, span, zoom, which)
        for j in range(len(chans), 24):
            axes.flat[j].axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("offset from pilot [kHz]", fontsize=6.5)
        for ax in axes[:, 0]:
            ax.set_ylabel("dB", fontsize=6.5)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([], [], color=VCOLOR[which], label=VLABEL[which]),
                   Line2D([], [], color=PILOT_C, ls="--",
                          label="nominal pilot"),
                   Line2D([], [], color="0.6", ls=":",
                          label="coarse-channel DC"),
                   Patch(facecolor=TGT_FILL, alpha=0.16,
                         label="target cell"),
                   Patch(facecolor=GUARD_FILL, alpha=0.16,
                         label="skipped guard"),
                   Patch(facecolor=REF_FILL, alpha=0.20,
                         label="reference cells ($\\pm$2 bins)")]
        if zoom:
            handles += [Line2D([], [], color=LINE_C, marker="v", ls="",
                               label="extracted line")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(f"{tbase} --- {VTITLE[which]}", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0.02, 1, 0.99))
        fname = f"{fbase}_{which}"
        fig.savefig(OUT / f"{fname}.png", dpi=230, bbox_inches="tight")
        fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", fname)
