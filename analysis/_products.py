"""Per-pilot survey products for the survey-plate scripts.

One more location, following the ``_paths.py`` convention:

  PP_PER_PILOT  directory of per-pilot survey products (*.npz)
                (default: ~/pilot_proxy_runs/chime-pilots/_per_pilot)

Products are external inputs, so loading never enables pickle.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

PER_PILOT = Path(os.environ.get(
    "PP_PER_PILOT", "~/pilot_proxy_runs/chime-pilots/_per_pilot")).expanduser()


def load_npz(path) -> dict[str, np.ndarray]:
    """Load a product archive without pickle, all arrays materialised."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True)
                for name in archive.files}


def paths(channels=None, per_pilot: Path = PER_PILOT) -> dict[int, str]:
    """physical channel -> product path, optionally filtered to `channels`."""
    found: dict[int, str] = {}
    for p in sorted(glob.glob(str(per_pilot / "*.npz"))):
        with np.load(p, allow_pickle=False) as archive:
            ch = int(np.ravel(archive["physical_channel"])[0])
        found[ch] = p
    if not found:
        raise SystemExit(f"no per-pilot products (*.npz) under {per_pilot}; "
                         "set PP_PER_PILOT or pass --products")
    if channels is not None:
        missing = [c for c in channels if c not in found]
        if missing:
            raise SystemExit(f"no product for channel(s) {missing} "
                             f"under {per_pilot}")
        found = {c: found[c] for c in channels}
    return found
