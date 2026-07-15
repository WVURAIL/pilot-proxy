#!/usr/bin/env python3
"""Dump full-depth integrated spectra from the per-pilot stack products.

Run on CANFAR (venv active):  python dump_spectra.py
Defaults: reads ~/pilot_proxy_runs/chime-pilots/_per_pilot/*.npz,
writes ~/paper/dumps/all_spectra.npz. Override:  python dump_spectra.py <in_dir> <out.npz>
"""
import glob
import sys
from pathlib import Path

import numpy as np

in_dir = (sys.argv[1] if len(sys.argv) > 1 else
          str(Path.home() / "pilot_proxy_runs/chime-pilots/_per_pilot"))
out_dir = Path.home() / "paper/dumps"
out_path = (sys.argv[2] if len(sys.argv) > 2 else str(out_dir / "all_spectra.npz"))
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

out = {}
paths = sorted(glob.glob(f"{in_dir}/*.npz"))
if not paths:
    raise SystemExit(f"no per-pilot npz files found under {in_dir}")
for p in paths:
    with np.load(p) as z:
        ch = int(z["physical_channel"][0])
        out[f"ch{ch}_before"] = np.asarray(z["integrated_spectrum_before_mask"])
        out[f"ch{ch}_after"] = np.asarray(z["integrated_spectrum_after_mask"])
        out[f"ch{ch}_meta"] = np.asarray([
            float(z["pilot_frequency_hz"][0]),
            float(z["chime_frequency_hz"][0]),
            float(np.asarray(z["freq_id"]).reshape(-1)[0]),
            float(np.asarray(z["valid"]).reshape(-1).astype(bool).sum()),
            float(np.asarray(z["reject_mask"]).reshape(-1).astype(bool).sum()),
        ])
np.savez_compressed(out_path, **out)
print(f"wrote {out_path} ({len(paths)} channels)")
