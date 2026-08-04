#!/usr/bin/env python3
"""Full-depth per-channel H0 table + exact subset search over event presence.

Run on the CANFAR notebook session (venv active):

    python fulldepth_and_subsets.py [WORK_DIR]

WORK_DIR defaults to ~/pilot_proxy_runs/chime-pilots/_per_pilot.

Outputs (written to the current directory):
  h0_fulldepth.csv                per-channel mean F vs mu0 over ALL valid
                                  frames (Fig. 4 / Table 3 inputs at depth)
  event_presence_signatures.npz   compact presence structure (freq_id list +
                                  per-signature bitmask counts) for offline
                                  analysis

Printed: the full-depth H0 table, and for each subset size k the exact
maximum common-event count over observed presence signatures (the greedy
drop-curve is myopic; this finds the block greedy cannot reach).
"""
import sys
import collections
from pathlib import Path

import numpy as np

work = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path.home() / "pilot_proxy_runs" / "chime-pilots" / "_per_pilot")
paths = sorted(p for p in work.glob("*.npz") if p.stem.isdigit())
if not paths:
    raise SystemExit(f"no per-pilot products under {work}")

fids, chans, events = [], [], []
rows = []
for p in paths:
    with np.load(str(p)) as z:
        fid = int(np.asarray(z["freq_id"]).reshape(-1)[0])
        ch = int(np.asarray(z["physical_channel"]).reshape(-1)[0])
        valid = np.asarray(z["valid"]).reshape(-1).astype(bool)
        rej = np.asarray(z["reject_mask"]).reshape(-1).astype(bool)
        f = np.asarray(z["fstat_raw"]).reshape(-1)[valid]
        f = f[np.isfinite(f)]
        mu0 = float(np.asarray(z["mu0"]).reshape(-1)[0])
        ev = set(np.asarray(z["source_event_keys"]).reshape(-1)
                 .astype(str).tolist())
    n = f.size
    mean = float(f.mean()) if n else float("nan")
    sem = float(f.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    gap, bound = abs(mean - mu0), abs(mu0 - 1.0) / 3.0
    rows.append((ch, fid, n, mean, sem, mu0, gap, bound,
                 bool(n and gap < bound),
                 float(rej.sum() / valid.sum()) if valid.any() else float("nan")))
    fids.append(fid)
    chans.append(ch)
    events.append(ev)

rows.sort()
hdr = ("ch,freq_id,n_valid,mean_fstat,sem_fstat,mu0,abs_mean_minus_mu0,"
       "bound_abs_mu0_minus_1_over_3,mean_tracks_mu0,mask_fraction_valid")
lines = [hdr] + [",".join(str(x) for x in r) for r in rows]
Path("h0_fulldepth.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Full-depth H0 zero-point table (all valid frames per channel):")
print(f"{'ch':>3} {'fid':>4} {'n_valid':>8} {'meanF':>9} {'sem':>8} "
      f"{'mu0':>8} {'|m-mu0|':>9} {'bound':>8} {'ok':>3} {'maskfrac':>8}")
for ch, fid, n, mean, sem, mu0, gap, bound, ok, mf in rows:
    print(f"{ch:>3} {fid:>4} {n:>8} {mean:>9.5f} {sem:>8.5f} {mu0:>8.5f} "
          f"{gap:>9.5f} {bound:>8.5f} {str(ok):>3} {mf:>8.4f}")

# ---- exact subset search over observed presence signatures ---------------
order = np.argsort(fids)
fids = [fids[i] for i in order]
chans = [chans[i] for i in order]
events = [events[i] for i in order]
presence: dict[str, int] = {}
for i, s in enumerate(events):
    for e in s:
        presence[e] = presence.get(e, 0) | (1 << i)
sig_counts = collections.Counter(presence.values())
sigs = np.asarray(list(sig_counts.keys()), dtype=np.int64)
cnts = np.asarray([sig_counts[int(s)] for s in sigs], dtype=np.int64)
np.savez_compressed("event_presence_signatures.npz",
                    freq_ids=np.asarray(fids, dtype=np.int64),
                    physical_channels=np.asarray(chans, dtype=np.int64),
                    signature=sigs, count=cnts)

best: dict[int, tuple[int, int]] = {}
for S in sigs.tolist():
    total = int(cnts[(sigs & S) == S].sum())
    k = int(bin(S).count("1"))
    if k not in best or total > best[k][0]:
        best[k] = (total, S)

print(f"\n{len(sigs)} distinct presence signatures over "
      f"{int(cnts.sum())} events, {len(fids)} channels")
print("exact best common-event count per subset size "
      "(candidates = observed signatures):")
print(f"{'k':>3} {'common':>7}  excluded channels")
for k in sorted(best, reverse=True):
    total, S = best[k]
    excl = sorted(fids[i] for i in range(len(fids)) if not (S >> i) & 1)
    print(f"{k:>3} {total:>7}  {excl}")
print("\nwrote h0_fulldepth.csv and event_presence_signatures.npz "
      "-- upload both with the results bundle.")
