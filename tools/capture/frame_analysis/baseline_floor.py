#!/usr/bin/env python3
"""Baseline dependence of the residual and the control-channel floor (2026-09-18, after the ruling audit, finding S1).

For each of the four science dumps, each coarse bin and each baseline class in CLASSES (both polarisations, read
separately): the noise-bias-free coherent amplitude ratio over all frames (item 2 of the predeclaration), the sky
reference (10th percentile over the bins within 40 MHz, per class and polarisation) and the excess. Per channel the
in-band median excess on the larger polarisation. The control is channel 37 (608 to 614 MHz, no DTV allocation): its
per-bin excess gives the floor of the estimator per class (mean and scatter over its bins and the four dumps).
Output: baseline_floor.csv (per channel and class: A, floor mean, floor scatter, A over floor) and a printed table."""
import csv, glob, json, os, sys
import numpy as np
NFFT = 16384
CLASSES = [(0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0)]
DUMPS = ["20260916162300", "20260917040230", "20260917090230", "20260917140230"]
def load(ev):
    out = {}
    d = f"/home/djg/rail/datasets/pilot_reduce_{ev}"
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        a = {}
        for c in CLASSES:
            for p in (0, 1):
                w = np.where((k[:, 0] == c[0]) & (k[:, 1] == c[1]) & (k[:, 2] == p) & (k[:, 3] == p))[0]
                if w.size == 0: a[(c, p)] = np.nan; continue
                v = z["stacks"][:, w[0]] / (NFFT * P); n = v.size; s = v.sum()
                C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1))
                a[(c, p)] = float(np.sqrt(max(C, 0.0)))
        out[int(m["freq_id"])] = dict(ch=int(m["channel"]), f=float(m["freq_mhz"]), a=a)
    fids = sorted(out); fr = np.array([out[f]["f"] for f in fids])
    for key in [(c, p) for c in CLASSES for p in (0, 1)]:
        aa = np.array([out[f]["a"][key] for f in fids])
        for i, f in enumerate(fids):
            near = np.abs(fr - fr[i]) <= 40.0
            out[f].setdefault("x", {})[key] = out[f]["a"][key] - float(np.nanpercentile(aa[near], 10))
    return out
eps = {ev: load(ev) for ev in DUMPS}
chans = sorted({v["ch"] for e in eps.values() for v in e.values()})
rows = []
for c in CLASSES:
    for p in (0, 1):
        key = (c, p)
        # control floor: channel 37 bins, all dumps
        ctrl = [e[f]["x"][key] for e in eps.values() for f in e if e[f]["ch"] == 37]
        fl_mean = float(np.nanmean(ctrl)); fl_sd = float(np.nanstd(ctrl))
        for ch in chans:
            per_dump = []
            for e in eps.values():
                v = [e[f]["x"][key] for f in e if e[f]["ch"] == ch]
                if v: per_dump.append(float(np.nanmedian(v)))
            A = float(np.nanmedian(per_dump)) if per_dump else np.nan
            rows.append(dict(channel=ch, ew=c[0], ns=c[1], baseline_m=round(c[0] * 22.0 + c[1] * 0.3048, 2) if c[0] == 0 or c[1] == 0 else round(np.hypot(c[0] * 22.0, c[1] * 0.3048), 2), pol=p, A=A, floor_mean=fl_mean, floor_sd=fl_sd, A_over_floor=(A / fl_mean if fl_mean > 0 else np.nan), A_minus_floor_over_sd=((A - fl_mean) / fl_sd if fl_sd > 0 else np.nan)))
with open("baseline_floor.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("A over the channel-37 floor (larger polarisation), per class; '*' = more than 3 floor scatters above the floor")
hdr = "ch  " + "  ".join(f"{('ns%d' % c[1] if c[0] == 0 else 'ew%d' % c[0] + ('ns%d' % c[1] if c[1] else '')):>10s}" for c in CLASSES)
print(hdr)
for ch in chans:
    cells = []
    for c in CLASSES:
        best = None
        for p in (0, 1):
            r = next(x for x in rows if x["channel"] == ch and x["ew"] == c[0] and x["ns"] == c[1] and x["pol"] == p)
            if best is None or (r["A"] == r["A"] and r["A"] > best["A"]): best = r
        cells.append(f"{best['A_over_floor']:8.1f}{'*' if best['A_minus_floor_over_sd'] > 3 else ' '} ")
    print(f"{ch:2d}  " + "  ".join(cells))
print("floor (ch37) per class, larger pol: " + ", ".join(f"{c}: {max(next(x for x in rows if x['channel']==37 and x['ew']==c[0] and x['ns']==c[1] and x['pol']==p)['floor_mean'] for p in (0,1)):.1e}" for c in CLASSES))
