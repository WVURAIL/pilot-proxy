#!/usr/bin/env python3
"""Fig. 2 draft: detector weight response |W(f)|^2 from the shipped K=128
int4 bank, plus exact manifest reference-placement geometry."""
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.detector_reference import unpack_packed_complex

plt = setup_matplotlib()
OUT = _paths.OUT
BANK = str(_paths.REPO / "weights/chime_dtv_weights_k128.bin")
K = 128
C_T, C_L, C_U = "#0072B2", "#D55E00", "#009E73"

bank = DetectorWeightBank(explicit_path=BANK)

# ---------- panel (a): |W(f)|^2 for a nominal channel (ch18) ----------------
w_packed, _ = bank.get_weights_for_physical_channel(18)
lay18 = bank.layout_for_physical_channel(18)
w = unpack_packed_complex(w_packed, 4)          # (3, K) complex, int4 components
OS = 64                                          # DTFT oversampling
f = np.linspace(0, 1, K * OS, endpoint=False)    # normalized fine-band freq
E = np.exp(-2j * np.pi * np.outer(f, np.arange(K)))
W = (E @ w.T.conj()).T                           # (3, K*OS) -- conj-matched
P = np.abs(W) ** 2
P /= P.max()

nf_t = lay18["target_normalized_frequency"]
nf_l = lay18["lower_reference_normalized_frequency"]
nf_u = lay18["upper_reference_normalized_frequency"]
# verify term order against the manifest before trusting labels
peaks = [f[np.argmax(P[i])] for i in range(3)]
order_ok = (abs(peaks[0] - nf_t) < 1e-2 and abs(peaks[1] - nf_l) < 1e-2
            and abs(peaks[2] - nf_u) < 1e-2)
print("term-order check (target, lower, upper):",
      [f"{p:.5f}" for p in peaks], "manifest:",
      [f"{v:.5f}" for v in (nf_t, nf_l, nf_u)], "OK" if order_ok else "MISMATCH")
assert order_ok

x = (f - nf_t) * K                               # fine bins from the target
fig = plt.figure(figsize=(8.6, 7.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1, 1], hspace=0.42,
                      wspace=0.18)
ax = fig.add_subplot(gs[0, :])
for i, (lbl, c) in enumerate((("target", C_T), ("lower reference", C_L),
                              ("upper reference", C_U))):
    sel = np.abs(x) <= 8
    ax.plot(x[sel], 10 * np.log10(np.maximum(P[i][sel], 1e-8)),
            color=c, lw=1.3, label=lbl)
for g in (-1, 1):
    ax.axvspan(g - 0.5, g + 0.5, color="0.9", zorder=0)
ax.axvline(0, color="0.75", lw=0.6)
ax.set_xlim(-8, 8)
ax.set_ylim(-58, 2)
ax.set_xlabel("fine bins from the target (bin width 3051.76 Hz)")
ax.set_ylabel(r"$|W(f)|^2$ [dB rel. target peak]")
ax.set_title(f"(a) int4 weight-term responses, K=128 "
             f"(ch18; guards shaded, refs at $\\pm2$ bins; "
             f"$\\mu_0$={lay18['mu0']:.4f})", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)

# ---------- panels (b): manifest geometry strips ------------------------------
CASES = ((18, "ch18 -- nominal"),
         (14, "ch14 -- DC in skipped guard"),
         (28, "ch28 -- reference 2.1 bins from DC"),
         (21, "ch21 -- lower reference wrapped across edge"))


def strip(ax, ch, title, lo, hi, show_labels):
    lay = bank.layout_for_physical_channel(ch)
    t = (lay["target_normalized_frequency"] - 0.5) * K
    lr = (lay["lower_reference_normalized_frequency"] - 0.5) * K
    ur = (lay["upper_reference_normalized_frequency"] - 0.5) * K
    ax.axvline(0, color="0.55", ls=":", lw=1.0)
    for e in (-K / 2, K / 2):
        if lo <= e <= hi:
            ax.axvline(e, color="0.2", lw=1.2)
    guards = []
    for s in (-1, 1):
        g = t + s * (1 + 0)          # skipped guard bins at target +/- 1
        guards.append(g)
        if lo <= g <= hi:
            ax.plot([g], [0], marker="s", ms=5, mfc="none", mec="0.45")
    for v, c, m in ((t, C_T, "o"), (lr, C_L, "v"), (ur, C_U, "^")):
        if lo <= v <= hi:
            ax.plot([v], [0], marker=m, ms=7, color=c)
    if ch == 21 and hi >= K / 2 - 2:  # wrap arrow: below-edge request -> far side
        ax.annotate("", xy=(lr - 2.5, 0.32), xytext=(t - 2.0, 0.32),
                    arrowprops=dict(arrowstyle="->", color=C_L, lw=1.0,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text((t + lr) / 2, 0.62, "wrap", fontsize=7, color=C_L,
                ha="center")
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.tick_params(labelsize=7)
    if show_labels:
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=8)
    ax.grid(color="0.95", lw=0.4)
    ax.set_axisbelow(True)


for row, (ch, title) in enumerate(CASES[:3], start=1):
    pass
axes_full = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
axes_zoom = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])]
# top strip row: ch18 + ch14; bottom: ch28 + ch21 (two per axis via offsets)
# simpler: 4 thin strips stacked inside two axes-> instead build 4 subaxes
for a in axes_full + axes_zoom:
    a.remove()
sub = fig.add_gridspec(4, 2, top=0.52, bottom=0.10, hspace=0.75, wspace=0.14)
for i, (ch, title) in enumerate(CASES):
    axl = fig.add_subplot(sub[i, 0])
    axr = fig.add_subplot(sub[i, 1])
    strip(axl, ch, title, -K / 2 - 4, K / 2 + 4, True)
    strip(axr, ch, title, -9, 9, False)
    if i == 0:
        axl.set_title("(b) manifest placement, full coarse channel "
                      "[fine bins from DC]", fontsize=9)
        axr.set_title("DC zoom ($\\pm9$ bins)", fontsize=9)
    if i == len(CASES) - 1:
        axl.set_xlabel("fine bins from coarse-channel DC", fontsize=8)
        axr.set_xlabel("fine bins from coarse-channel DC", fontsize=8)

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_T, label="target"),
    Line2D([], [], marker="v", ls="", color=C_L, label="lower ref"),
    Line2D([], [], marker="^", ls="", color=C_U, label="upper ref"),
    Line2D([], [], marker="s", ls="", mfc="none", mec="0.45",
           label="skipped guard"),
    Line2D([], [], color="0.55", ls=":", label="DC"),
    Line2D([], [], color="0.2", label="channel edge")],
    loc="lower center", ncol=6, fontsize=7.5, frameon=False,
    bbox_to_anchor=(0.5, 0.012))
fig.suptitle("Detector geometry: quantized weight responses and shipped "
             "reference placement", fontsize=11, y=0.985)
fig.savefig(OUT / "fig2_detector_geometry.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2_detector_geometry.pdf", bbox_inches="tight")
print("wrote fig2_detector_geometry")
