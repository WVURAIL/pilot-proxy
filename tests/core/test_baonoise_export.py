# coding=utf-8
"""Gate for the baonoise calibration exporter.

Synthetic product with a known geometry: a persistent line at bin 50
during an on-quarter, a transmitter-off quarter at the null bulk, and
exact frame counts. Asserts the spec's ingest-side invariants (bin sums
equal meta totals, contiguous edges ending at inf), the same-statistic
rule (the histogrammed value is the window max), anchor selection, and
the off-epoch null classification.
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "export_baonoise_calibration.py"

N_FRAMES = 240
LINE_BIN = 50
LINE_VALUE = 21.7
T_OFF = 1595000000.0  # 2020Q3
T_ON = 1740000000.0   # 2025Q1


def _write_product(path: pathlib.Path, rng: np.random.Generator) -> None:
    fine = rng.normal(1.0, 0.03, size=(N_FRAMES, 256)).clip(0.5, 1.4)
    on = np.arange(N_FRAMES) >= N_FRAMES // 2
    fine[on, LINE_BIN] = LINE_VALUE
    unit_index = np.arange(N_FRAMES) // 4
    n_units = N_FRAMES // 4
    unit_t0 = np.where(np.arange(n_units) >= n_units // 2, T_ON, T_OFF)
    det_frames = np.flatnonzero(on)
    np.savez(
        path,
        freq_id=np.array([521]),
        physical_channel=np.array([35], dtype=np.int32),
        valid=np.ones((N_FRAMES, 1), dtype=np.uint8),
        fstat_fine=fine.astype(np.float32),
        frame_unit_index=unit_index.astype(np.int32),
        unit_time0_ctime=unit_t0.astype(np.float64),
        baseband_power_linear=np.full((N_FRAMES, 1), 2.5),
        detector_version=np.asarray("pilot-proxy/test kernel=test K=128"),
        fine_detected_frame=det_frames.astype(np.int64),
        fine_detected_bin=np.full(det_frames.size, LINE_BIN, dtype=np.int64),
        unit_order=np.asarray([f"u{i}" for i in range(n_units)]),
    )


def _write_census(path: pathlib.Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rf_channel", "callsign", "service_class",
                    "detectability_db", "distance_km", "bearing_deg",
                    "frequency_tolerance", "chime_ch_index",
                    "nominal_pilot_mhz", "city", "state_prov"])
        w.writerow([35, "TESTCALL", "Full-power", 90.0, 10.0, 0.0,
                    "±1 kHz", 521, 599.309, "Testville", "WA"])
        w.writerow([36, "OTHERCALL", "Full-power", 80.0, 10.0, 0.0,
                    "±1 kHz", 506, 605.309, "Testville", "WA"])


def test_exporter_end_to_end(tmp_path):
    rng = np.random.default_rng(12345)
    prod_dir = tmp_path / "per_pilot"
    prod_dir.mkdir()
    _write_product(prod_dir / "521.npz", rng)
    census = tmp_path / "census.csv"
    _write_census(census)
    out = tmp_path / "out"

    res = subprocess.run(
        [sys.executable, str(TOOL),
         "--per-pilot-dir", str(prod_dir),
         "--out", str(out),
         "--census", str(census)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ingest-side validations: all passed" in res.stdout

    # provenance: measured anchor at the line bin
    prov = json.loads((out / "provenance.json").read_text())
    w = prov["window_definitions"]["35"]
    assert w["anchored"] is True and w["anchor_bin"] == LINE_BIN

    # File 1 invariants: contiguity, inf tail, exact sums vs meta
    rows = list(csv.DictReader(open(out / "onsky_frame_hist.csv")))
    total = 0
    by_epoch: dict[str, list] = {}
    for r in rows:
        by_epoch.setdefault(r["epoch"], []).append(r)
        total += int(r["n_frames"])
    assert total == N_FRAMES
    for epoch, er in by_epoch.items():
        assert er[-1]["bin_hi"] == "inf"
        for a, b in zip(er[:-1], er[1:]):
            assert a["bin_hi"] == b["bin_lo"]

    meta = list(csv.DictReader(open(out / "channel_meta.csv")))
    m35 = next(m for m in meta if m["atsc_channel"] == "35")
    assert int(m35["n_valid_frames"]) == N_FRAMES
    assert m35["census_candidate"] == "TESTCALL"
    # absent channel listed with reason
    m36 = next(m for m in meta if m["atsc_channel"] == "36")
    assert int(m36["n_valid_frames"]) == 0 and m36["notes"]

    # same-statistic rule: every on-frame's window max is the line value,
    # so the on-quarter histogram puts all its mass in the bin holding it
    on_rows = [r for r in rows if r["epoch"] == "2025Q1"]
    hot = [r for r in on_rows if int(r["n_frames"])]
    assert len(hot) == 1
    lo, hi = float(hot[0]["bin_lo"]), float(hot[0]["bin_hi"])
    assert lo <= LINE_VALUE < hi
    assert int(hot[0]["n_frames"]) == N_FRAMES // 2

    # off-epoch null: 2020Q3 classified transmitter-off, anchor-window rows
    nulls = list(csv.DictReader(open(out / "null_frame_hist.csv")))
    off_rows = [r for r in nulls
                if r["source"] == "off_epoch_anchor_window"]
    assert off_rows and all(r["epoch"] == "2020Q3" for r in off_rows)
    assert sum(int(r["n_frames"]) for r in off_rows) == N_FRAMES // 2

    # event-window products: per-event max sums to the number of events
    # with valid frames; the integrated product puts every on-event at the
    # line value (all frames in an on-event carry the line)
    ev = list(csv.DictReader(open(out / "window_event_hist.csv")))
    n_events = N_FRAMES // 4
    assert sum(int(r["n_windows"]) for r in ev) == n_events
    integ = list(csv.DictReader(open(out / "window_event_integrated_hist.csv")))
    on_int = [r for r in integ if r["epoch"] == "2025Q1" and int(r["n_windows"])]
    assert len(on_int) == 1
    lo, hi = float(on_int[0]["bin_lo"]), float(on_int[0]["bin_hi"])
    assert lo <= LINE_VALUE < hi
    assert int(on_int[0]["n_windows"]) == n_events // 2
    int_null = list(csv.DictReader(open(out / "null_event_integrated_hist.csv")))
    off_int = [r for r in int_null
               if r["source"] == "off_epoch_anchor_window"]
    assert sum(int(r["n_windows"]) for r in off_int) == n_events // 2
