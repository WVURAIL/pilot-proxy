#!/usr/bin/env python3
"""Deterministic zero-point correction from the PFB noise shape.

White noise -> 4-tap/2048 sinc-Hamming PFB (the repo's reference prototype)
-> critically sampled coarse channel with aliased PSD
S(nu) = sum_k |H(nu+k)|^2 -> detector fine bins (K=128 rectangular window)
integrate S through the Dirichlet kernel -> per-fine-bin noise gain g(d).
Predicted zero-point ratio per channel from the manifest geometry:
E[F]/mu0 = g(d_t)*(l_ns+u_ns) / (l_ns*g(d_l) + u_ns*g(d_u)).
Compare against the measured mu_hat/mu0 - 1.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib
from pilot_proxy.reference_channelizer import sinc_hamming_pfb_response
from pilot_proxy.detector_weights import DetectorWeightBank

plt = setup_matplotlib()
OUT = _paths.OUT
K = 128
BANK = str(_paths.REPO / "weights/chime_dtv_weights_k128.bin")

# --- prototype transfer function on a dense grid (channel-width units) -------
h = np.asarray(sinc_hamming_pfb_response(4, 2048, dtype=np.float64),
               dtype=np.float64).reshape(-1)
P = 1 << 22
H = np.fft.rfft(h, P)
freq_ch = np.arange(H.size) * (2048.0 / P)          # in channel widths
H2 = np.abs(H) ** 2
H2 /= H2[0]

def H2_at(f):
    """|H|^2 at frequency f (channel widths), even in f."""
    return np.interp(np.abs(f), freq_ch, H2, right=0.0)

# --- aliased in-channel PSD S(nu), nu in [-0.5, 0.5) --------------------------
M = 1 << 17
nu = (np.arange(M) / M) - 0.5
S = np.zeros(M)
for k in range(-3, 4):
    S += H2_at(nu + k)
S /= S[M // 2]                                       # normalize at channel DC

# --- fine-bin gains: circular convolution with the K-point Dirichlet kernel ---
eps = 1e-30
D2 = (np.sin(np.pi * K * nu) / (K * np.sin(np.pi * nu) + eps)) ** 2
D2[np.isclose(nu % (1.0 / 1.0), 0.0)] = 1.0          # nu=0 limit
D2 /= D2.sum()
g_dense = np.real(np.fft.ifft(np.fft.fft(np.fft.ifftshift(S)) *
                              np.fft.fft(np.fft.ifftshift(D2))))
g_dense = np.fft.fftshift(g_dense)                   # aligned with nu grid
g_mid = g_dense[M // 2]
g_dense /= g_mid

def g_at(d_bins):
    """Noise gain at d fine bins from coarse-channel DC (periodic)."""
    nu_d = ((d_bins / K) + 0.5) % 1.0 - 0.5
    return float(np.interp(nu_d, nu, g_dense))

# --- per-channel prediction from the manifest ---------------------------------
bank = DetectorWeightBank(explicit_path=BANK)
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}
rows = []
for ch in sorted(study):
    lay = bank.layout_for_physical_channel(ch)
    d_t = (lay["target_normalized_frequency"] - 0.5) * K
    d_l = (lay["lower_reference_normalized_frequency"] - 0.5) * K
    d_u = (lay["upper_reference_normalized_frequency"] - 0.5) * K
    lns = float(lay["lower_reference_norm_sq"])
    uns = float(lay["upper_reference_norm_sq"])
    ratio = (g_at(d_t) * (lns + uns)) / (lns * g_at(d_l) + uns * g_at(d_u))
    pred = 1e3 * (ratio - 1.0)
    s = study[ch]
    meas = float(s["gap_1e3"]) if s["zero_point_trusted"] == "1" else None
    rows.append((ch, d_t, d_l, d_u, pred, meas,
                 lay["reference_placement_status"]))

print("edge/center gain snapshot: "
      f"g(0)=1.000  g(32)={g_at(32):.4f}  g(60)={g_at(60):.4f}  "
      f"g(62.4)={g_at(62.44):.4f}  g(63.56)={g_at(63.56):.4f}  "
      f"g(64)={g_at(64.0):.4f}")
print(f"\n{'ch':>3} {'d_t':>7} {'d_l':>7} {'d_u':>7} "
      f"{'pred(1e-3)':>10} {'meas(1e-3)':>10}  placement")
for ch, dt_, dl_, du_, pred, meas, st in rows:
    m = f"{meas:+10.2f}" if meas is not None else "  untrusted"
    print(f"{ch:>3} {dt_:>7.2f} {dl_:>7.2f} {du_:>7.2f} {pred:>+10.3f} {m}  {st}")

# --- figure -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
d_grid = np.linspace(-64, 64, 2001)
ax1.plot(d_grid, [g_at(d) for d in d_grid], color="#0072B2", lw=1.4)
for ch, dt_, dl_, du_, pred, meas, st in rows:
    if ch == 21:
        ax1.plot([dt_], [g_at(dt_)], "o", color="#D55E00", ms=6)
        ax1.plot([dl_], [g_at(dl_)], "v", color="#D55E00", ms=6)
        ax1.plot([du_], [g_at(du_)], "^", color="#D55E00", ms=6)
ax1.annotate("ch21 target", xy=(-62.4, g_at(-62.44)), xytext=(-45, 0.9),
             fontsize=8, color="#D55E00",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#D55E00"))
ax1.annotate("ch21 wrapped lower ref", xy=(63.56, g_at(63.56)),
             xytext=(18, 0.78), fontsize=8, color="#D55E00",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#D55E00"))
ax1.set_xlabel("fine bins from coarse-channel DC")
ax1.set_ylabel("relative noise gain $g(d)$")
ax1.set_title("(a) PFB aliased-noise shape through the K=128 window",
              fontsize=10)
ax1.grid(color="0.92", lw=0.6)
ax1.set_axisbelow(True)

SPUR = {14, 25, 28}
for ch, dt_, dl_, du_, pred, meas, st in rows:
    if meas is None:
        continue
    c = ("#D55E00" if ch == 21 else "#7B4FA6" if ch in SPUR else "#0072B2")
    ax2.plot(pred, meas, "o", ms=5, color=c)
    if ch == 21 or ch in SPUR or abs(meas) > 3:
        ax2.annotate(f"ch{ch}", xy=(pred, meas), xytext=(pred + 0.3, meas),
                     fontsize=7.5, color=c)
lim = 14
ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="prediction = measurement")
ax2.axhline(0, color="0.85", lw=0.6)
ax2.axvline(0, color="0.85", lw=0.6)
ax2.set_xlabel(r"predicted $\Delta$ from PFB noise shape $[10^{-3}]$")
ax2.set_ylabel(r"measured $(\hat{\mu}_0-\mu_0)/\mu_0\ [10^{-3}]$")
ax2.set_title("(b) deterministic prediction vs measured zero-point shift",
              fontsize=10)
ax2.set_xlim(-lim, 3)
ax2.set_ylim(-lim, 3)
ax2.legend(fontsize=8)
ax2.grid(color="0.92", lw=0.6)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_pfb_zero_point_prediction.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_pfb_zero_point_prediction.pdf", bbox_inches="tight")
print("\nwrote fig_pfb_zero_point_prediction")
