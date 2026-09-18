#!/usr/bin/env python3
"""Coherence of the DTV residual between dumps, as a function of the lag between them.

usage: cadence_lags.py <out csv> <label>=<products dir> [...]

For each channel, class (EW 0, NS 1, same polarisation, both pols averaged) and bin set (pilot bin; in-band median),
the dump's mean phasor V_d = mean over its frames of the stack mean. For every pair of dumps (a, b) at lag dt the
normalised coherence is rho_ab = Re(V_a conj V_b) / sqrt(|V_a|^2 |V_b|^2), noise-bias free in the cross term because
the two dumps share no samples; the per-dump noise floor is estimated from the frame-to-frame scatter and reported.
rho near 1 at a lag means the residual's coherent part has not decorrelated over that lag (it counts toward G);
rho near 0 means it has. Lags are grouped to the nearest predeclared class (15, 30, 60, 120, 240, 480, 720, 960,
1200 s, then hours) and averaged over pairs."""
import glob, json, os, sys
import numpy as np
NFFT = 16384
PILOT = {14:844,15:829,16:813,17:798,18:783,19:767,20:752,21:736,22:721,23:706,24:690,25:675,26:660,27:644,28:629,29:614,30:598,31:583,32:568,33:552,34:537,35:521,36:506,37:491}
LAG_CLASSES = [15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600, 720, 960, 1200, 3600, 18000, 36000, 86400]
CLASSES = {"ns1": (0, 1), "ns8": (0, 8), "ew1": (1, 0)}
def load_dump(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        V = {}
        for name, (ew, ns) in CLASSES.items():
            idx = [int(np.where((k[:, 0] == ew) & (k[:, 1] == ns) & (k[:, 2] == p) & (k[:, 3] == p))[0][0]) for p in (0, 1)]
            V[name] = z["stacks"][:, idx].mean(axis=1) / (NFFT * P)      # [frames], both pols averaged, in A units
        out[m["freq_id"]] = dict(ch=m["channel"], V=V, f=m["freq_mhz"], t0=m["time0_ctime"] + m["j0"] * 2.56e-6, n=m["n_frames"])
    # reference phasor per class: mean over the ch37 and ch34 bins (no transmitter), subtracted from every bin
    for name in CLASSES:
        ref = [v["V"][name].mean() for v in out.values() if v["ch"] in (34, 37)]
        r = np.mean(ref) if ref else 0.0
        for v in out.values(): v["V"][name + "_x"] = v["V"][name] - r
    return out
dumps = {}
for arg in sys.argv[2:]:
    lbl, d = arg.split("=", 1); dumps[lbl] = load_dump(d)
labels = list(dumps); t = {l: np.median([v["t0"] for v in dumps[l].values()]) for l in labels}
chans = sorted(set(v["ch"] for d in dumps.values() for v in d.values()))
rows = []
for ch in chans:
  for cls in ("ns1", "ns1_x", "ns8_x", "ew1_x"):
    for binset in ("pilot", "inband"):
        for a_i in range(len(labels)):
            for b_i in range(a_i + 1, len(labels)):
                a, b = labels[a_i], labels[b_i]; dt = abs(t[b] - t[a])
                fids = [f for f in dumps[a] if dumps[a][f]["ch"] == ch and f in dumps[b]]
                if binset == "pilot": fids = [f for f in fids if f == PILOT.get(ch)]
                if not fids: continue
                rhos = []; floors = []
                for f in fids:
                    Va, Vb = dumps[a][f]["V"][cls], dumps[b][f]["V"][cls]
                    ma, mb = Va.mean(), Vb.mean()
                    if abs(ma) == 0 or abs(mb) == 0: continue
                    rho = (ma * np.conj(mb)).real / (abs(ma) * abs(mb))
                    # noise floor: scatter of the per-frame phasors around the mean, relative to the mean
                    fl = np.sqrt((abs(Va - ma) ** 2).mean() / Va.size + (abs(Vb - mb) ** 2).mean() / Vb.size) / np.sqrt(abs(ma) * abs(mb))
                    rhos.append(rho); floors.append(fl)
                if not rhos: continue
                lag_class = min(LAG_CLASSES, key=lambda L: abs(np.log(max(dt, 1) / L)))
                rows.append(dict(channel=ch, cls=cls, bins=binset, a=a, b=b, lag_s=round(dt, 1), lag_class=lag_class, rho=float(np.median(rhos)), rho_noise=float(np.median(floors)), n_bins=len(rhos)))
with open(sys.argv[1], "w") as fh:
    cols = list(rows[0].keys()) if rows else ["channel"]; fh.write(",".join(cols) + "\n")
    for r in rows: fh.write(",".join(str(r[c]) for c in cols) + "\n")
print(f"{len(rows)} pairs -> {sys.argv[1]}")
if rows:
    for cls in ("ns1", "ns1_x", "ns8_x", "ew1_x"):
        print(f"\nin-band coherence between dumps, class {cls} (x = reference-subtracted excess): ch | lag class s: rho (noise)")
        for ch in chans:
            rr = [r for r in rows if r["channel"] == ch and r["bins"] == "inband" and r["cls"] == cls]
            by = {}
            for r in rr: by.setdefault(r["lag_class"], []).append(r)
            print(f"  {ch:2d} | " + "  ".join(f"{L}s: {np.mean([x['rho'] for x in v]):+.2f} ({np.mean([x['rho_noise'] for x in v]):.2f})" for L, v in sorted(by.items())))
