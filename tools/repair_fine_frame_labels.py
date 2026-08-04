#!/usr/bin/env python3
"""Repair fine_detected_frame labels in existing per-pilot products.

Products written before the frame-labeling fix stamp every detection in a
unit with the unit's FIRST global frame index. The authoritative partition
is fine_detected_count (per frame, in frame order), so the correct column
is exactly np.repeat(arange(n_frames), counts). This script rewrites only
that one array, atomically (tmp + rename, same np.savez_compressed flavor
the analyzer uses), and leaves every other array byte-identical.

Safe to run on already-correct or mixed-label products: it is idempotent,
verifies row-count consistency before touching a file, and skips files
whose column already matches.

Usage:
    python repair_fine_frame_labels.py ~/pilot_proxy_runs/chime-pilots-v2/_per_pilot
    python repair_fine_frame_labels.py <dir> --dry-run

Run it only on quiescent products (scan finished or stopped) -- not while
the analyzer might checkpoint the same file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def repair_file(path: Path, dry_run: bool) -> str:
    with np.load(str(path), allow_pickle=False) as z:
        if "fine_detected_frame" not in z.files or "fine_detected_count" not in z.files:
            return "skip (no fine detection arrays)"
        counts = np.asarray(z["fine_detected_count"]).reshape(-1)
        stored = np.asarray(z["fine_detected_frame"]).reshape(-1)
        expected = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        if stored.size != expected.size:
            return (f"REFUSE: rows ({stored.size}) != sum of counts "
                    f"({expected.size}) -- inspect by hand")
        if np.array_equal(stored, expected):
            return "ok (already correct)"
        if dry_run:
            return (f"would repair: {np.count_nonzero(stored != expected)} of "
                    f"{stored.size} labels differ")
        data = {name: z[name] for name in z.files}
    data["fine_detected_frame"] = expected
    tmp = str(path) + ".repair.tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, str(path))
    return f"repaired ({np.count_nonzero(stored != expected)} labels rewritten)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("per_pilot_dir", type=Path,
                    help="_per_pilot directory containing <freq_id>.npz products")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()
    files = sorted(args.per_pilot_dir.glob("*.npz"))
    if not files:
        print(f"no .npz products under {args.per_pilot_dir}", file=sys.stderr)
        return 1
    worst = 0
    for p in files:
        verdict = repair_file(p, args.dry_run)
        print(f"{p.name:12s} {verdict}")
        if verdict.startswith("REFUSE"):
            worst = 2
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
