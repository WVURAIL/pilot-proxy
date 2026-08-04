#!/usr/bin/env python3
"""Dump per-frame baseband power from the per-pilot stack products.

Run on CANFAR (venv active):  python dump_power.py
Defaults: reads ~/pilot_proxy_runs/chime-pilots/_per_pilot/*.npz,
writes ~/paper/dumps/power.npz. Override:  python dump_power.py <in_dir> <out.npz>
"""
import glob
import sys
from pathlib import Path

import numpy as np

in_dir = (sys.argv[1] if len(sys.argv) > 1 else
          str(Path.home() / "pilot_proxy_runs/chime-pilots/_per_pilot"))
out_dir = Path.home() / "paper/dumps"
out_path = (sys.argv[2] if len(sys.argv) > 2 else str(out_dir / "power.npz"))
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

out = {}
paths = sorted(glob.glob(f"{in_dir}/*.npz"))
if not paths:
    raise SystemExit(f"no per-pilot npz files found under {in_dir}")
n_ch = 0
for p in paths:
    with np.load(p) as z:
        ch = int(z["physical_channel"][0])
        if "baseband_power_linear" not in z.files:
            print(f"WARNING: ch{ch} ({p}) has no baseband_power_linear; skipped")
            continue
        out[f"ch{ch}_baseband_power_linear"] = np.asarray(
            z["baseband_power_linear"], dtype=np.float64).reshape(-1)
        out[f"ch{ch}_p_target_u64"] = np.asarray(
            z["p_target_u64"], dtype=np.uint64).reshape(-1)
        out[f"ch{ch}_scalars"] = np.asarray([
            float(np.asarray(z["freq_id"]).reshape(-1)[0]),
            float(np.asarray(z["nfft"]).reshape(-1)[0]),
            float(np.asarray(z["num_input_streams"]).reshape(-1)[0]),
            float(np.asarray(z["sense"]).reshape(-1)[0]),
        ])
        n = out[f"ch{ch}_baseband_power_linear"].size
        if n != out[f"ch{ch}_p_target_u64"].size:
            raise SystemExit(f"ch{ch}: power/p_target length mismatch")
        print(f"ch{ch}: {n} frames")
        n_ch += 1
if not n_ch:
    raise SystemExit("no channels dumped")
np.savez_compressed(out_path, **out)
print(f"wrote {out_path} ({n_ch} channels)")
