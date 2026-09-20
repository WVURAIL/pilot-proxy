#!/usr/bin/env python3
"""Per-channel spectrograms of the DTV pilot region, 2018 to 2026, with the latest era marked.

For each of the 24 archive per-pilot products this reads the per-frame PSD (16,384 fine bins across
the coarse channel, coded in units of 0.01 dB relative to each frame's own reference), windows it
around the channel's nominal pilot frequency, takes the median in monthly time bins, and draws the
result as a spectrogram. A carrier shows as a horizontal line at zero; a transmitter switching off
shows as that line stopping.

Every panel is drawn on the same time axis, and any month with no frames is GREY, so that a gap in
coverage cannot be mistaken for a quiet transmitter. That distinction matters: several channels
stop being recorded years before the archive ends.

The dashed line marks the start of the current era as the archive's own dating run recorded it
(sections/era/current_first_month in the per-channel ledger). That is the boundary the ruling relies
on when it calls a transmitter switched off, so it is drawn here to be checked by eye.

The monthly medians are cached in spectrogram_cache.npz; delete it to re-reduce from the products.

usage: build_spectrograms.py [out dir]
"""
import datetime as dt
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

SRC = "/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot"
LEDGER = "/home/djg/rail/results/archive_v5_2026-09-08_corrected/ledger/channels"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "spectrogram_cache.npz")
BW = 400e6 / 1024 / 16384          # 23.84 Hz per fine bin
HALF = 250                          # +-250 bins = +-6.0 kHz, wide enough for an off-nominal carrier
BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NODATA = "#b9b8b3"                  # neutral grey, off the blue ramp, so it cannot read as a level
CMAP = LinearSegmentedColormap.from_list("seqblue", BLUE)
CMAP.set_bad(NODATA)

def era_info(ch):
    """The current-era boundary AND what cut it. The dating run cuts an era on three different
    grounds and they do not mean the same thing: a spectral-state transition is a transmitter
    changing state, a station change is only the carrier's peak drifting in frequency, and an
    archive start is no change at all. Drawing them identically invites the reader to treat a
    frequency drift as a switch-off."""
    f = glob.glob(os.path.join(LEDGER, f"ch{ch}_fid*.json"))
    if not f: return None, None, 0.0
    e = json.load(open(f[0])).get("sections", {}).get("era", {})
    return e.get("current_first_month"), e.get("current_evidence"), float(e.get("current_peak_drift_bins_per_month") or 0.0)

STYLE = {"spectral-state transition": dict(color="#0d366b", ls="-",  lw=2.0, tag="state"),
         "station change":            dict(color="#b06000", ls=":",  lw=2.0, tag="drift"),
         "archive start":             dict(color="#52514e", ls="--", lw=1.0, tag="start")}

def month_index(ts):
    d = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    return d.year * 12 + (d.month - 1)

if os.path.exists(CACHE):
    c = np.load(CACHE, allow_pickle=True)
    chans, imgs, m0, m1 = list(c["chans"]), list(c["imgs"]), int(c["m0"]), int(c["m1"])
    print("loaded cache", CACHE)
else:
    raw = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        ch = int(z["physical_channel"][0])
        c0 = int(round(-(float(z["pilot_frequency_hz"]) - float(z["chime_frequency_hz"])) / BW)) % 16384
        idx = np.arange(c0 - HALF, c0 + HALF + 1) % 16384
        psd = z["psd_frame_db_i16"][:, idx].astype(np.float32) * 0.01
        t = z["unit_time0_ctime"][z["frame_unit_index"]]
        del z
        mi = np.array([month_index(x) for x in t])
        raw.append((ch, mi, psd))
        print(f"ch{ch:2d}: {psd.shape[0]} frames, {np.unique(mi).size} months with data", flush=True)
    m0 = min(int(mi.min()) for _, mi, _ in raw)
    m1 = max(int(mi.max()) for _, mi, _ in raw)
    months = np.arange(m0, m1 + 1)
    chans, imgs = [], []
    for ch, mi, psd in raw:
        img = np.full((months.size, psd.shape[1]), np.nan, np.float32)
        for k, m in enumerate(months):
            sel = mi == m
            if sel.any(): img[k] = np.median(psd[sel], axis=0)
        chans.append(ch); imgs.append(img)
    del raw
    np.savez_compressed(CACHE, chans=np.array(chans), imgs=np.array(imgs), m0=m0, m1=m1)
    print("wrote cache", CACHE)

order = np.argsort(chans)
chans = [chans[i] for i in order]; imgs = [imgs[i] for i in order]
n = len(chans); ncol = 4; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.5 * nrow), facecolor="#fcfcfb")
ext = [m0 / 12.0, (m1 + 1) / 12.0, -HALF * BW / 1e3, HALF * BW / 1e3]
for ax, ch, img in zip(axes.ravel(), chans, imgs):
    ax.imshow(np.ma.masked_invalid(img.T), aspect="auto", origin="lower", extent=ext,
              cmap=CMAP, vmin=0.0, vmax=12.0, interpolation="nearest")
    st, evid, drift = era_info(ch)
    if st:
        y, mo = st.split("-")
        x = int(y) + (int(mo) - 1) / 12.0
        sty = STYLE.get(str(evid), STYLE["archive start"])
        ax.axvline(x, color=sty["color"], ls=sty["ls"], lw=sty["lw"])
        lab = sty["tag"] if sty["tag"] != "drift" else f"drift {drift:.2f} b/mo"
        ax.text(x, HALF * BW / 1e3 * 0.72, " " + lab, fontsize=7, color=sty["color"], va="top")
    ax.set_title(f"channel {ch}", fontsize=10, color="#0b0b0b", pad=3)
    ax.tick_params(labelsize=7, colors="#52514e")
    for s in ax.spines.values(): s.set_color("#c9c8c3")
for ax in axes.ravel()[n:]: ax.axis("off")
for i, ax in enumerate(axes.ravel()[:n]):
    if i // ncol == nrow - 1: ax.set_xlabel("year", fontsize=8, color="#52514e")
    if i % ncol == 0: ax.set_ylabel("kHz from nominal pilot", fontsize=8, color="#52514e")
fig.suptitle("DTV pilot region per channel, monthly median PSD relative to each frame's reference\n"
             "grey: no frames archived that month.  the era line is coloured by what actually cut it",
             fontsize=12, color="#0b0b0b", y=0.995)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 12.0), cmap=CMAP),
                  ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
cb.set_label("dB over frame reference", fontsize=8, color="#52514e")
cb.ax.tick_params(labelsize=7, colors="#52514e")
from matplotlib.lines import Line2D
handles = [Patch(facecolor=NODATA, edgecolor="#c9c8c3", label="no data archived"),
           Line2D([0], [0], color="#0d366b", ls="-", lw=2.0, label="era cut by a transmitter state change"),
           Line2D([0], [0], color="#b06000", ls=":", lw=2.0, label="era cut only by carrier frequency drift"),
           Line2D([0], [0], color="#52514e", ls="--", lw=1.0, label="no era change: archive start")]
axes.ravel()[n - 1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.08, 1.0),
                           fontsize=9, frameon=False)
p = os.path.join(OUT, "channel_spectrograms.png")
fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print("wrote", p)
