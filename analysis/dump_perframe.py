#!/usr/bin/env python3
"""Dump per-frame detector statistics from the per-pilot stack products.

Run on CANFAR (venv active):  python dump_perframe.py
Defaults: reads ~/pilot_proxy_runs/chime-pilots/_per_pilot/*.npz,
writes ~/paper/dumps/perframe.npz. Override:  python dump_perframe.py <in_dir> <out.npz>
"""
import glob
import sys
from pathlib import Path

import numpy as np

in_dir = (sys.argv[1] if len(sys.argv) > 1 else
          str(Path.home() / "pilot_proxy_runs/chime-pilots/_per_pilot"))
out_dir = Path.home() / "paper/dumps"
out_path = (sys.argv[2] if len(sys.argv) > 2 else str(out_dir / "perframe.npz"))
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

out = {}
paths = sorted(glob.glob(f"{in_dir}/*.npz"))
if not paths:
    raise SystemExit(f"no per-pilot npz files found under {in_dir}")
for p in paths:
    with np.load(p) as z:
        ch = int(z["physical_channel"][0])
        for k in ("p_target_u64", "p_ref_sum_u64", "valid", "reject_mask",
                  "frame_unit_index"):
            out[f"ch{ch}_{k}"] = np.asarray(z[k]).reshape(-1)
        out[f"ch{ch}_unit_time0_ctime"] = np.asarray(z["unit_time0_ctime"])
        out[f"ch{ch}_scalars"] = np.asarray([
            float(z["mu0"][0]),
            float(z["target_norm_sq"][0]),
            float(z["ref_norm_sum_sq"][0]),
            float(np.asarray(z["freq_id"]).reshape(-1)[0]),
            float(z["pilot_frequency_hz"][0]),
            float(z["chime_frequency_hz"][0]),
        ])
np.savez_compressed(out_path, **out)
print(f"wrote {out_path} ({len(paths)} channels)")
