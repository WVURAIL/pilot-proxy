#!/usr/bin/env python3
"""Build Table 3, appendix tables, diagnostic figures, and memo inputs from
the uploaded results bundle."""
import csv
import json
from pathlib import Path

import numpy as np

import sys
import _paths  # noqa: F401  (repo src on sys.path + shared locations)
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()

B = _paths.RESULTS
OUT = _paths.OUT
OUT.mkdir(exist_ok=True)

FRAME_S = 16384 / 390625.0
COARSE_KHZ = 400e3 / 1024.0
REF_OFF = 2          # reference_offset_bins
HALF = 64.0          # K/2 fine bins to the coarse-channel edge

# ---- load -----------------------------------------------------------------
h0 = list(csv.DictReader(open(B / "h0_quicklook/h0_fulldepth.csv")))
fd_rows = list(csv.DictReader(open(B / "fulldepth/fulldepth_cleaning_tradeoff.csv")))
fd_sum = json.load(open(B / "fulldepth/fulldepth_summary.json"))
dec = json.load(open(B / "combine_subset_decision.json"))
census_sum = json.load(open(B / "transmitter_census/summary.json"))
lines = list(csv.DictReader(open(B / "transmitter_census/extracted_lines.csv")))
hist = np.load(B / "h0_quicklook/h0_fulldepth_fstat_histograms.npz")

kept_stack = set(dec["kept_channels"])
x0 = {int(r["freq_id"]): r for r in fd_rows if float(r["excess_db"]) == 0.0}
lines_per_ch = {}
for r in lines:
    ch = int(r["rf_channel"])
    lines_per_ch[ch] = lines_per_ch.get(ch, 0) + 1

rows = []
for r in h0:
    ch = int(r["physical_channel"])
    fid = int(r["freq_id"])
    mu0 = float(r["mu0"])
    mean = float(r["mean_fstat"])
    sem = float(r["sem_fstat"])
    n = int(r["n_valid"])
    mf = float(r["mask_fraction_valid"])
    tdc = float(r["pilot_offset_from_dc_finebins"])
    dev = mean / mu0 - 1.0
    dev_sem = sem / mu0
    refs = [tdc - REF_OFF, tdc + REF_OFF]
    ref_dc = min(abs(x) for x in refs)
    ref_edge = min(HALF - abs(x) for x in refs)
    wrapped = any(abs(x) > HALF for x in refs)
    f = x0[fid]
    kept_frac = float(f["kept_fraction"])
    cls = ("suppressed" if dev < -2e-3 else
           "elevated" if dev > 3e-3 else "nominal")
    rows.append(dict(
        ch=ch, fid=fid, n_valid=n, hours=n * FRAME_S / 3600.0,
        mu0=mu0, mean=mean, sem=sem, dev=dev, dev_sem=dev_sem,
        mask_frac=mf, kept_frac=kept_frac,
        recovered_khz=kept_frac * COARSE_KHZ,
        in_stack=fid in kept_stack,
        n_lines=lines_per_ch.get(ch, 0),
        t_dc=tdc, ref_dc=ref_dc, ref_edge=ref_edge, wrapped=wrapped,
        cls=cls,
    ))
rows.sort(key=lambda r: r["ch"])

# consistency: full-depth recovered on the 16 stacked channels vs stack tradeoff
rec_stack_fd = sum(r["recovered_khz"] for r in rows if r["in_stack"]) / 1e3
print(f"full-depth recovered over the 16 stacked channels: {rec_stack_fd:.3f} MHz "
      f"(stack tradeoff on common frames gave 3.736 MHz)")
print(f"total processed exposure: {sum(r['hours'] for r in rows):.2f} h")

# ---- Table 3 CSV ------------------------------------------------------------
with open(OUT / "table3_fulldepth.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "n_valid_frames", "exposure_hours",
                "mu0", "mean_F", "sem_F", "dev_1e3", "dev_sem_1e3",
                "mask_fraction", "kept_fraction", "recovered_khz_at_mu0",
                "in_stack", "census_lines",
                "pilot_offset_dc_bins", "min_ref_dist_dc_bins",
                "min_ref_dist_edge_bins", "ref_wrapped", "class"])
    for r in rows:
        w.writerow([r["ch"], r["fid"], r["n_valid"], f"{r['hours']:.4f}",
                    f"{r['mu0']:.6f}", f"{r['mean']:.6f}", f"{r['sem']:.6f}",
                    f"{1e3*r['dev']:.3f}", f"{1e3*r['dev_sem']:.3f}",
                    f"{r['mask_frac']:.4f}", f"{r['kept_frac']:.4f}",
                    f"{r['recovered_khz']:.1f}",
                    int(r["in_stack"]), r["n_lines"],
                    f"{r['t_dc']:.2f}", f"{r['ref_dc']:.2f}",
                    f"{r['ref_edge']:.2f}", int(r["wrapped"]), r["cls"]])

# ---- Table 3 LaTeX ----------------------------------------------------------
def tex_num(x, fmt):
    return f"${x:{fmt}}$"

tex = [r"""% Table 3: per-channel survey summary (full depth, tau = mu0).
% Requires \usepackage{booktabs}. Deviation is (mean F - mu0)/mu0 in units of 1e-3.
\begin{table*}
  \centering
  \caption{Per-channel survey summary at full depth. $N_{\rm valid}$ is the
  number of valid 41.9\,ms detector frames; $\langle F\rangle$ is the mean
  F-statistic over valid frames; $\Delta$ is the fractional zero-point
  deviation $(\langle F\rangle-\mu_0)/\mu_0$ in units of $10^{-3}$ (statistical
  uncertainty in parentheses); $f_{\rm mask}$ is the masked fraction of valid
  frames at the shipped operating point $\tau=\mu_0$; recovered bandwidth is
  $(1-f_{\rm mask})\times390.6$\,kHz. ``Stack'' marks membership in the
  16-channel event-matched stack (amended selection; Appendix~X). Lines is the
  number of carrier lines extracted for the transmitter-census case study.}
  \label{tab:survey_summary}
  \begin{tabular}{rrrrlllrrcr}
    \toprule
    Ch & freq\_id & $N_{\rm valid}$ & Hours & $\mu_0$ & $\langle F\rangle$ &
    $\Delta\,[10^{-3}]$ & $f_{\rm mask}$ & Rec.\,[kHz] & Stack & Lines \\
    \midrule
"""]
for r in rows:
    stack_mark = r"\checkmark" if r["in_stack"] else "--"
    dev_cell = f"${1e3*r['dev']:+.2f}" + r"\," + f"({1e3*r['dev_sem']:.2f})$"
    tex.append(
        f"    {r['ch']} & {r['fid']} & {r['n_valid']} & {r['hours']:.3f} & "
        f"{r['mu0']:.4f} & {r['mean']:.4f} & "
        f"{dev_cell} & "
        f"{r['mask_frac']:.3f} & {r['recovered_khz']:.0f} & "
        f"{stack_mark} & {r['n_lines']} \\\\\n")
tex.append(r"""    \midrule
    \multicolumn{8}{l}{Total recovered at $\tau=\mu_0$ (23 channels)} &
    """ + f"{sum(r['recovered_khz'] for r in rows):.0f}" + r""" & & \\
    \bottomrule
  \end{tabular}
\end{table*}
""")
(OUT / "table3_fulldepth.tex").write_text("".join(tex))

# ---- appendix: subset selection ----------------------------------------------
reg = dec["registered_rule"]
with open(OUT / "appendix_dropcurve.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step", "dropped_freq_id", "channels_remaining",
                "common_events_after"])
    for i, s in enumerate(reg["full_drop_curve"], 1):
        w.writerow([i, s["drop_fid"], s["n_channels_after"],
                    s["intersection_events_after"]])
with open(OUT / "appendix_exact_by_k.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["k", "common_events", "excluded_freq_ids"])
    for r in dec["exact_best_by_k"]:
        w.writerow([r["k"], r["common_events"],
                    " ".join(str(f) for f in r["excluded"])])

atex = [r"""% Appendix: stacked-subset selection.
\begin{table}
  \centering
  \caption{Registered greedy drop-curve (decision 1, registered 2026-07-08):
  each step removes the most event-constraining channel. Starting from all 23
  completed channels the common-event count is 0; the rule's $\geq$50\%%-growth
  walk stops after dropping freq\_id 767 at 12 common events. The greedy path
  is myopic on this archive: the per-channel node-uptime epochs mean three
  blockers must be shed simultaneously before the intersection grows.}
  \label{tab:dropcurve}
  \begin{tabular}{rrrr}
    \toprule
    Step & Dropped freq\_id & Channels left & Common events \\
    \midrule
"""]
for i, s in enumerate(reg["full_drop_curve"], 1):
    atex.append(f"    {i} & {s['drop_fid']} & {s['n_channels_after']} & "
                f"{s['intersection_events_after']} \\\\\n")
atex.append(r"""    \bottomrule
  \end{tabular}
\end{table}

\begin{table}
  \centering
  \caption{Exact maximum common-event count as a function of subset size $k$
  (search over observed event-presence signatures, closed over covered
  events). The amended selection (maximize common events subject to the
  registered $\geq$16-channel floor) is the $k=16$ row: 1{,}548 shared events,
  a factor $\sim$130 over the greedy stopping point.}
  \label{tab:exactsubsets}
  \begin{tabular}{rrp{4.2cm}}
    \toprule
    $k$ & Common events & Excluded freq\_ids \\
    \midrule
""")
for r in dec["exact_best_by_k"]:
    if r["k"] >= 10:
        ex = ", ".join(str(f) for f in r["excluded"]) or "--"
        sel = r" $\leftarrow$ adopted" if r["k"] == 16 else ""
        atex.append(f"    {r['k']} & {r['common_events']} & {ex}{sel} \\\\\n")
atex.append(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
(OUT / "appendix_subset_selection.tex").write_text("".join(atex))

# ---- Figure A: zero-point deviation + mask fraction ---------------------------
C_NOM, C_SUP, C_ELE = "#0072B2", "#D55E00", "#009E73"
col = {"nominal": C_NOM, "suppressed": C_SUP, "elevated": C_ELE}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                               gridspec_kw={"hspace": 0.08})
chs = [r["ch"] for r in rows]
YLIM = 14.0
for r in rows:
    d = 1e3 * r["dev"]
    c = col[r["cls"]]
    if d > YLIM:  # off-scale (ch24 +36, ch30 +10430)
        lbl = (f"ch{r['ch']}: ${d:+.0f}$" if d < 100 else
               f"ch{r['ch']}: $" + f"{d/1e3:+.1f}" + r"\times10^{3}$")
        ax1.annotate(lbl,
                     xy=(r["ch"], YLIM * 0.96), xytext=(r["ch"], YLIM * 0.68),
                     ha="center", fontsize=7, color=c,
                     arrowprops=dict(arrowstyle="->", color=c, lw=1))
    else:
        ax1.errorbar(r["ch"], d, yerr=1e3 * r["dev_sem"], fmt="o", ms=4.5,
                     color=c, capsize=2, lw=1)
ax1.axhline(0, color="0.4", lw=0.8)
ax1.set_ylim(-YLIM, YLIM)
ax1.set_ylabel(r"$(\langle F\rangle-\mu_0)/\mu_0\ \ [10^{-3}]$")
ax1.grid(axis="y", color="0.92", lw=0.6)
ax1.set_axisbelow(True)
for ch, lbl, dy in ((21, "wrapped ref", +1.0), (14, "DC in guard", -1.8),
                    (28, "ref near DC", -1.8), (25, "near DC", -1.8)):
    r = next(x for x in rows if x["ch"] == ch)
    ax1.annotate(lbl, xy=(ch, 1e3 * r["dev"]),
                 xytext=(ch + 0.3, 1e3 * r["dev"] + dy),
                 fontsize=7, color=C_SUP)
ax2.bar(chs, [r["mask_frac"] for r in rows],
        color=[col[r["cls"]] for r in rows], width=0.72)
ax2.axhline(0.5, color="0.25", ls="--", lw=0.9)
ax2.set_ylabel(r"mask fraction at $\tau=\mu_0$")
ax2.set_xlabel("ATSC physical channel")
ax2.set_xticks(chs)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_ylim(0, 1.02)
ax2.grid(axis="y", color="0.92", lw=0.6)
ax2.set_axisbelow(True)
from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_NOM, label="nominal"),
    Line2D([], [], marker="o", ls="", color=C_SUP,
           label=r"suppressed ($\Delta<-2\times10^{-3}$)"),
    Line2D([], [], marker="o", ls="", color=C_ELE,
           label=r"signal-elevated ($\Delta>+3\times10^{-3}$)")],
    fontsize=7.5, loc="lower right", framealpha=0.9)
fig.suptitle("Full-depth H0 zero-point by channel (all valid frames)",
             y=0.94, fontsize=11)
fig.savefig(OUT / "fig_zero_point_fulldepth.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig_zero_point_fulldepth.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Figure B: F distributions around mu0 -------------------------------------
mu0_by_ch = {r["ch"]: r["mu0"] for r in rows}
fig, ax = plt.subplots(figsize=(7.0, 4.4))
PAL4 = {"21": "#D55E00", "14": "#7B4FA6", "28": "#00795A", "25": "#0072B2"}
for ch in (35,):
    cnt, edg = hist[f"ch{ch}_counts"], hist[f"ch{ch}_edges"]
    mu = mu0_by_ch[ch]
    x = 1e3 * ((0.5 * (edg[:-1] + edg[1:])) / mu - 1.0)
    dens = cnt / cnt.sum() / np.diff(1e3 * (edg / mu - 1.0))
    ax.plot(x, dens, color="0.25", ls="--", lw=1.4,
            label=f"ch{ch} (healthy reference)")
for ch_s, c in PAL4.items():
    ch = int(ch_s)
    cnt, edg = hist[f"ch{ch}_counts"], hist[f"ch{ch}_edges"]
    mu = mu0_by_ch[ch]
    x = 1e3 * ((0.5 * (edg[:-1] + edg[1:])) / mu - 1.0)
    dens = cnt / cnt.sum() / np.diff(1e3 * (edg / mu - 1.0))
    ax.plot(x, dens, color=c, lw=1.6, label=f"ch{ch}")
ax.axvline(0, color="0.4", lw=0.8)
ax.set_xlim(-45, 45)
ax.set_xlabel(r"$(F-\mu_0)/\mu_0\ \ [10^{-3}]$")
ax.set_ylabel("probability density")
ax.set_title("Valid-frame F distributions, suppressed family vs healthy channel")
ax.legend(fontsize=8)
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_f_distributions_suppressed.png", dpi=300,
            bbox_inches="tight")
fig.savefig(OUT / "fig_f_distributions_suppressed.pdf", bbox_inches="tight")
plt.close(fig)

# distribution width stats for the memo
print("\np5-p95 widths in (F/mu0-1) x 1e3:")
for ch in (35, 33, 21, 14, 25, 28):
    cnt, edg = hist[f"ch{ch}_counts"], hist[f"ch{ch}_edges"]
    mu = mu0_by_ch[ch]
    mids = 0.5 * (edg[:-1] + edg[1:])
    cum = np.cumsum(cnt) / cnt.sum()
    q = lambda p: 1e3 * (mids[np.searchsorted(cum, p)] / mu - 1)
    print(f"  ch{ch}: median {q(.5):+7.2f}  width {q(.95)-q(.05):6.2f}")

print("\nwrote:", sorted(p.name for p in OUT.iterdir()))
