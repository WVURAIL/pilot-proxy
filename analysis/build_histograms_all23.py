
#!/usr/bin/env python3
"""MVP per-channel normalized-F histograms, all 23 channels.

The plotted statistic is

    F_norm = p_target * ref_norm_sum_sq /
             (p_ref_sum * target_norm_sq),

so the fixed analytic decision is F_norm > 1. Crossing counts are evaluated
with exact integer products; floating point is used only for plotting. No
archive-fitted centre, tail model, or empirical threshold enters this figure.
"""
import csv

import numpy as np

import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
OUT = _paths.OUT
INK, FLAG_C, KEEP_C = "0.25", "#D55E00", "#0072B2"
SPAN = 60.0  # +/- 1e-3 units for the shared panels

z = np.load(_paths.PERFRAME)
chans = sorted({int(k[2:].split("_")[0]) for k in z.files})
summary = []

fig, axes = plt.subplots(6, 4, figsize=(11.5, 12.6))
for j, ch in enumerate(chans):
    ax = axes.flat[j]
    pt_u = z[f"ch{ch}_p_target_u64"].astype(np.uint64)
    pr_u = z[f"ch{ch}_p_ref_sum_u64"].astype(np.uint64)
    valid = z[f"ch{ch}_valid"].astype(bool) & (pr_u > 0)
    _, tns_f, rnss_f, fid_f, _, _ = z[f"ch{ch}_scalars"]
    tns, rnss, fid = int(tns_f), int(rnss_f), int(fid_f)
    if tns <= 0 or rnss <= 0:
        raise ValueError(f"ch{ch}: non-positive weight-energy constant")

    # Python integers preserve the exact decision if future products approach
    # the uint64 multiplication limit.
    crossing = np.fromiter(
        (int(pt) * rnss > int(pr) * tns for pt, pr in zip(pt_u, pr_u)),
        dtype=bool,
        count=pt_u.size,
    ) & valid
    equality = np.fromiter(
        (int(pt) * rnss == int(pr) * tns for pt, pr in zip(pt_u, pr_u)),
        dtype=bool,
        count=pt_u.size,
    ) & valid

    pt = pt_u.astype(np.float64)
    pr = pr_u.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_norm = pt * rnss / (pr * tns)
    x = 1e3 * (f_norm[valid] - 1.0)

    if ch == 30:
        lo, hi = np.percentile(x, [0.2, 99.8])
        pad = 0.06 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 320)
    else:
        bins = np.linspace(-SPAN, SPAN, 600)
    cnt, edges = np.histogram(x, bins=bins, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ax.semilogy(mids, np.where(cnt > 0, cnt, np.nan), color=INK, lw=0.7,
                drawstyle="steps-mid")
    ax.axvspan(edges[0], 0.0, color=KEEP_C, alpha=0.08)
    ax.axvspan(0.0, edges[-1], color=FLAG_C, alpha=0.08)
    ax.axvline(0.0, color="0.15", lw=0.9)

    n_valid = int(valid.sum())
    n_cross = int(crossing.sum())
    frac = n_cross / n_valid if n_valid else float("nan")
    ax.text(0.03, 0.92, f"$N={n_valid:,}$", transform=ax.transAxes,
            fontsize=6.3)
    ax.text(0.03, 0.81, f"$F>1$: {100*frac:.1f}\\%",
            transform=ax.transAxes, fontsize=6.3)
    ax.set_title(f"ch{ch} (fid {fid})", fontsize=7.5, pad=2)
    ax.tick_params(labelsize=5.5)
    ax.set_ylim(bottom=1e-5)
    ax.grid(color="0.94", lw=0.35)
    ax.set_axisbelow(True)
    summary.append({
        "atsc_channel": ch,
        "freq_id": fid,
        "n_valid": n_valid,
        "n_crossing_strict": n_cross,
        "crossing_fraction_strict": f"{frac:.8f}",
        "n_exact_equality": int(equality.sum()),
        "target_norm_sq": tns,
        "ref_norm_sum_sq": rnss,
    })

for j in range(len(chans), 24):
    axes.flat[j].axis("off")
for ax in axes[-1, :]:
    ax.set_xlabel(r"$(F-1)\ [10^{-3}]$", fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("density", fontsize=7)

from matplotlib.lines import Line2D

fig.legend(handles=[
    Line2D([], [], color=KEEP_C, lw=5, alpha=0.35,
           label=r"unflagged: $F\leq1$"),
    Line2D([], [], color=FLAG_C, lw=5, alpha=0.35,
           label=r"flagged: $F>1$"),
], loc="lower center", ncol=2, fontsize=8, frameon=False,
    bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Normalized $F$ distributions for valid archive blocks",
             fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.99))
fig.savefig(OUT / "fig_excess_threshold_all23.png", dpi=230,
            bbox_inches="tight")
fig.savefig(OUT / "fig_excess_threshold_all23.pdf", bbox_inches="tight")

with open(OUT / "archive_unity_crossings.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
print("wrote fig_excess_threshold_all23 and archive_unity_crossings.csv")
