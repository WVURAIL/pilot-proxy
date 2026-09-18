#!/usr/bin/env python3
"""Per-epoch, per-class, per-polarisation in-band excess for every channel, and the channel 37 control per class
(amendment 8). Classes: the ten of baseline_floor.py. For each of the fourteen epochs and each coarse bin the
noise-bias-free coherent amplitude over all frames, the 10th-percentile sky reference within 40 MHz (per class and
polarisation), the excess; per channel the in-band median over its bins; for channel 37 also the per-bin values (the
floor sample). Output: class_excess_epochs.csv (channel, epoch, ew, ns, pol, excess) and class_floor_bins.csv
(epoch, ew, ns, pol, freq_id, excess for the channel 37 bins)."""
import csv, glob, json, os
import numpy as np
NFFT = 16384
CLASSES = [(0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0)]
EPOCHS = [l.split("=")[1].split("_")[-1] for l in open("epochs_14.txt").read().split()]
rows, floor = [], []
for ev in EPOCHS:
    d = f"/home/djg/rail/datasets/pilot_reduce_{ev}"; out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        a = {}
        for c in CLASSES:
            for p in (0, 1):
                w = np.where((k[:, 0] == c[0]) & (k[:, 1] == c[1]) & (k[:, 2] == p) & (k[:, 3] == p))[0]
                if w.size == 0: a[(c, p)] = np.nan; continue
                v = z["stacks"][:, w[0]] / (NFFT * P); n = v.size; s = v.sum()
                C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1)) if n > 1 else np.nan
                a[(c, p)] = float(np.sqrt(max(C, 0.0))) if np.isfinite(C) else np.nan
        out[int(m["freq_id"])] = dict(ch=int(m["channel"]), f=float(m["freq_mhz"]), a=a)
    fids = sorted(out); fr = np.array([out[f]["f"] for f in fids])
    for c in CLASSES:
        for p in (0, 1):
            aa = np.array([out[f]["a"][(c, p)] for f in fids])
            x = {}
            for i, f in enumerate(fids):
                near = np.abs(fr - fr[i]) <= 40.0
                x[f] = out[f]["a"][(c, p)] - float(np.nanpercentile(aa[near], 10))
            for ch in sorted({out[f]["ch"] for f in fids}):
                v = [x[f] for f in fids if out[f]["ch"] == ch and np.isfinite(x[f])]
                rows.append(dict(channel=ch, epoch=ev, ew=c[0], ns=c[1], pol=p, excess=f"{np.median(v):.6e}" if v else "", n_bins=len(v)))
            for f in fids:
                if out[f]["ch"] == 37 and np.isfinite(x[f]): floor.append(dict(epoch=ev, ew=c[0], ns=c[1], pol=p, freq_id=f, excess=f"{x[f]:.6e}"))
    print(ev, "done", flush=True)
with open("class_excess_epochs.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open("class_floor_bins.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(floor[0])); w.writeheader(); w.writerows(floor)
print("wrote class_excess_epochs.csv", len(rows), "rows; class_floor_bins.csv", len(floor), "rows")
