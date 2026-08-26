#!/usr/bin/env python3
"""Generate the per-channel residual-chain table (dissertation Table 9.6 and
its lower-band extension) from per-pilot survey products.

The chain follows the dissertation's stated conventions, verified by
reproducing the published first-measured block from raw products:

* kept-frame floor = the ``--floor-percentile`` (default 90th) percentile of
  the null population (coarse-kept frames whose fine-stage shelf estimate is
  finite), or of the transmitter-off era when ``--off-from``/``--off-through``
  applies;
* nested variance split into DC / inter-day / intra-day / fast shares;
* component budget r = 10^(floor/10) * (rho_intra * n_coh(tau) + rho_fast),
  with n_coh = min(tau_c, sidereal day) / 41.94 ms and tau_c measured where
  the structure-function estimator's gates pass, else the sidereal-day cap;
* r_keep uses the same gain on the on-air shelf level.

Running with ``--self-test`` first reproduces the published first-block
constants and aborts if any of them moves: the table's provenance is this
reproduction, not a remembered analysis.

Requires the released ``rfisher`` package (RFIsher).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

from pilot_proxy.archive_health import (
    FRAME_HEALTH_GATE_SCHEMA_VERSION,
    temporary_residual_health_views,
)

# Transmitter-era boundaries from the monthly occupancy analysis.
OFF_THROUGH = {35: "2021-10"}
OFF_FROM = {19: "2024-12", 20: "2022-09", 26: "2023-04", 27: "2022-10",
            32: "2022-10"}
# Channels whose science-relevant chain row is the transmitter-off era.
OFF_EPOCH_CANONICAL = {19, 20, 26, 27}
THIN_FLOOR_FRAMES = 30      # parenthesize floors from fewer null frames
BINDING_RTOL = 1.5e-3       # binding growth-rate tolerance (Table 9.4)

SELF_TEST = {  # channel: (floor_db, n_null, x_over_binding) published values
    33: (-44.95, 1388, 24.0),
    29: (-48.2, 8119, 6620.0),
    32: (-48.5, 2818, 44.0),
    34: (-45.6, 168, 1603.0),
}


def chain_row(res, path, off_through=None, off_from=None):
    _, st, ct = res.budget_from_products(path, off_through=off_through,
                                         off_from=off_from)
    with np.load(path, allow_pickle=False) as view:
        health_included = int(view["health_included_frame_count"])
        health_excluded = int(view["health_excluded_frame_count"])
    n_intra = res.n_coh_from_correlation_time(ct.tau_for_budget)
    gain = st.intraday_fraction * n_intra + st.fast_fraction
    r_keep = 10.0 ** (st.on_shelf_db / 10.0) * gain
    r_proxy = (10.0 ** (st.floor_db / 10.0) * gain
               if np.isfinite(st.floor_db) else None)
    return dict(
        channel=st.channel,
        epoch=("off-epoch" if off_from else
               "on-epoch" if off_through else "full"),
        masked_pct=round(100.0 * st.masked_fraction, 2),
        on_shelf_db=round(st.on_shelf_db, 2),
        n_null=st.n_off_frames,
        floor_db=(round(st.floor_db, 2) if np.isfinite(st.floor_db) else None),
        floor_thin=bool(0 < st.n_off_frames < THIN_FLOOR_FRAMES),
        intra_share_pct=round(100.0 * st.intraday_fraction, 2),
        ground_filter_db=round(st.ground_filter_db, 1),
        tau_minutes=round(ct.tau_for_budget / 60.0, 1),
        tau_quality=str(getattr(ct, "quality", "")),
        tau_capped=not ct.is_measured,
        r_keep=float(f"{r_keep:.4g}"),
        r_proxy=(None if r_proxy is None else float(f"{r_proxy:.4g}")),
        x_over_binding=(None if r_proxy is None
                        else float(f"{r_proxy / BINDING_RTOL:.4g}")),
        health_gate_schema_version=FRAME_HEALTH_GATE_SCHEMA_VERSION,
        health_included_frames=health_included,
        health_excluded_frames=health_excluded,
        health_floor_status=(
            "available"
            if st.n_off_frames > 0
            else "unavailable_no_health_included_stored_mask_kept_frames"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("exports/dissertation/inputs/"
                                     "bao_channel_chain.csv"))
    parser.add_argument("--floor-percentile", type=float, default=90.0)
    parser.add_argument("--skip-self-test", action="store_true")
    args = parser.parse_args()

    from rfisher import residual as res

    paths = [
        Path(path)
        for path in sorted(glob.glob(os.path.join(str(args.products), "*.npz")))
    ]
    if not paths:
        raise SystemExit(f"no per-pilot products under {args.products}")
    rows = []
    # RFIsher owns the residual calculation but accepts only paths. Derived
    # minimal views apply the archive gate before any floor, variance, or
    # correlation-time statistic reaches that package.
    with temporary_residual_health_views(paths) as health_views:
        for path in health_views:
            with np.load(path, allow_pickle=False) as z:
                ch = int(z["physical_channel"][0])
            epochs = []
            if ch in OFF_EPOCH_CANONICAL:
                epochs.append(dict(off_from=OFF_FROM[ch]))
                epochs.append(dict())       # era-mixture view, flagged by user
            elif ch in OFF_THROUGH:
                # published first-block row is the full-archive view; the
                # transmitter-off era rides along as the supplementary epoch row
                epochs.append(dict())
                epochs.append(dict(off_through=OFF_THROUGH[ch]))
            else:
                epochs.append(dict())
            for kw in epochs:
                rows.append(chain_row(res, path, **kw))

    if not args.skip_self_test:
        got = {r["channel"]: r for r in rows if r["epoch"] == "full"}
        for ch, (floor, n_null, mult) in SELF_TEST.items():
            r = got[ch]
            assert abs(r["floor_db"] - floor) < 0.06, (ch, r["floor_db"])
            assert r["n_null"] == n_null, (ch, r["n_null"])
            assert abs(r["x_over_binding"] / mult - 1.0) < 0.02, \
                (ch, r["x_over_binding"])
        print(f"self-test: first-block constants reproduced "
              f"({len(SELF_TEST)} channels)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} chain rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
