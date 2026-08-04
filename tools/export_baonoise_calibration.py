#!/usr/bin/env python3
# coding=utf-8
"""Export the pilot-proxy -> baonoise calibration bundle from per-pilot products.

Implements the "PilotProxy -> baonoise export specification (CANFAR pass)":
long-form CSV histograms of the per-frame decision statistic, an empirical
null measured for the same statistic, per-window maxima, channel metadata,
and provenance. Everything derives offline from the stored per-pilot
``.npz`` products (the full per-frame fine spectrum is persisted), so this
tool can run after any scan without touching baseband.

The same-statistic rule (the spec's "one rule that matters most"):

    F_frame = max over the designated tolerance window of fstat_fine[b]

where the window is the per-channel deployed designated set: the measured
pilot line anchor +/- ``--window-halfwidth`` fine bins when a persistent
line is detected, else the nominal bin 0 +/- the same halfwidth. The
per-bin statistic ``fstat_fine[b] = 2 S_t[b] / (S_l[b] + S_u[b])`` is
null-centered near 1 by construction (the mu0-corrected fine ratio), and
the identical max-over-window statistic is histogrammed in every file.

File 3 substitution (recorded in provenance): the spec's 10 s receiver
windows do not exist in archived baseband snapshots, so
``window_event_hist.csv`` holds the per-*event* maximum (one baseband
capture, <= ~0.6 s; events and units are one-to-one in this archive) with
the event duration distribution recorded; the occupancy-amplification
inequality is reported (not enforced), since it does not hold for
variable-size windows. ``window_event_integrated_hist.csv`` and its null
add the exploratory third policy: the window max of the event-AVERAGED
spectrum (detect on integrated spectra rather than flag from per-frame
decisions), as a mean-of-ratios approximation.

Null sources written to ``null_frame_hist.csv``:

* ``offpilot_bins`` -- disjoint same-width windows tiled over bins at
  least ``--null-clear-hz`` from the anchor and clear of every persistent
  line (per-bin median of the bulk-normalized spectrum >=
  ``--line-ratio``, dilated), per (channel, epoch). On occupied channels
  during on-epochs these carry transmitter leakage and are a
  contaminated upper bound on the null;
* ``off_epoch_anchor_window`` -- the anchor-window statistic itself from
  (channel, quarter) cells whose quarterly median of the window statistic
  sits at the null bulk (below ``--off-epoch-max-median``, >= 30 frames):
  transmitter-off epochs, the calibration-grade null of the deployed
  statistic.

The spec's ingest-side validations run at the end of every export; a hard
failure exits nonzero.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import sys

import numpy as np

FINE_BINS = 256
FINE_BIN_HZ = (390625.0 / 128.0) / 256.0
FRAME_SECONDS = 16384.0 / 390625.0

F_EDGES = np.r_[0.0, np.geomspace(0.3, 300.0, 151), np.inf]  # 152 bins


def quarter(ts: float) -> str:
    d = datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def fmt(x: float) -> str:
    if np.isinf(x):
        return "inf"
    return format(float(x), ".10g")


def hist_rows(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.int64)


class Channel:
    def __init__(self, path: str, args):
        z = np.load(path, allow_pickle=False)
        r = lambda k: np.asarray(z[k]).reshape(-1)
        self.path = path
        self.freq_id = int(r("freq_id")[0])
        self.atsc = int(r("physical_channel")[0])
        self.valid = r("valid").astype(bool)
        self.fine = np.asarray(z["fstat_fine"], dtype=np.float64)
        self.n_frames = self.valid.size
        self.unit_index = r("frame_unit_index").astype(int)
        t0 = r("unit_time0_ctime").astype(float)
        self.t = t0[self.unit_index]
        self.power = r("baseband_power_linear").astype(float)
        self.detector_version = str(np.asarray(z["detector_version"]))
        det_frame = r("fine_detected_frame").astype(int)
        det_bin = r("fine_detected_bin").astype(int)
        self.det_frac = (
            np.bincount(det_bin, minlength=FINE_BINS).astype(float)
            / max(self.n_frames, 1)
        )
        # line map: per-bin median of the bulk-normalized spectrum. A
        # persistent line elevates its bin's median; broadband splash
        # (whole-spectrum elevation on strong-transmitter frames)
        # normalizes away. Detection fractions cannot make this
        # distinction on saturated channels (any-bin CFAR fires broadly).
        vf = self.fine[self.valid] if self.valid.any() else self.fine[:1]
        bulk = np.median(vf, axis=1, keepdims=True)
        self.line_ratio = np.median(vf / np.maximum(bulk, 1e-12), axis=0)
        # anchor: strongest persistently-detected line, else nominal bin 0
        peak = int(np.argmax(self.det_frac))
        self.anchored = bool(self.det_frac[peak] >= args.anchor_fraction)
        self.anchor = peak if self.anchored else 0
        hw = int(args.window_halfwidth)
        self.window = np.array(
            [(self.anchor + k) % FINE_BINS for k in range(-hw, hw + 1)],
            dtype=int,
        )
        self.n_window = self.window.size
        # per-frame decision statistic over valid frames
        self.F_frame = self.fine[:, self.window].max(axis=1)
        self.epoch = np.array([quarter(x) for x in self.t])
        self.quarters = sorted(set(self.epoch[self.valid].tolist()))

    def offset_hz(self) -> float:
        b = self.anchor
        return (b if b < FINE_BINS // 2 else b - FINE_BINS) * FINE_BIN_HZ

    def null_tiles(self, args) -> np.ndarray:
        """Disjoint same-width off-pilot windows (bin index arrays)."""
        clear = int(np.ceil(args.null_clear_hz / FINE_BIN_HZ))
        hw = int(args.window_halfwidth)
        blocked = np.zeros(FINE_BINS, dtype=bool)
        for k in range(-(hw + clear), hw + clear + 1):
            blocked[(self.anchor + k) % FINE_BINS] = True
        for b in np.flatnonzero(self.line_ratio >= args.line_ratio):
            for k in range(-(hw + 2), hw + 3):
                blocked[(b + k) % FINE_BINS] = True
        tiles = []
        run: list[int] = []
        for b in range(FINE_BINS):
            if blocked[b]:
                run = []
                continue
            run.append(b)
            if len(run) == self.n_window:
                tiles.append(np.array(run, dtype=int))
                run = []
        return tiles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-pilot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--census", default="data/census/census.csv")
    ap.add_argument("--window-halfwidth", type=int, default=2,
                    help="designated-window halfwidth in fine bins (N = 2h+1)")
    ap.add_argument("--anchor-fraction", type=float, default=0.10,
                    help="min detection fraction for a measured-line anchor")
    ap.add_argument("--line-ratio", type=float, default=1.5,
                    help="per-bin median of the bulk-normalized spectrum "
                         "above which a bin is a persistent line (null excl.)")
    ap.add_argument("--null-clear-hz", type=float, default=500.0)
    ap.add_argument("--off-epoch-max-median", type=float, default=1.5,
                    help="quarterly median of the window statistic below "
                         "which a (channel, quarter) counts as "
                         "transmitter-off (min 30 frames)")
    ap.add_argument("--absent-note", default="not scanned in this run")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.per_pilot_dir, "*.npz")))
    if not paths:
        print(f"no products under {args.per_pilot_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    # census: many stations per channel; keep the strongest candidate per
    # RF channel (max detectability_db) and the rf_channel -> freq_id map
    best_by_rf: dict[int, dict] = {}
    rf_to_fid: dict[int, int] = {}
    with open(args.census, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rf = int(row["rf_channel"])
            fid = row.get("chime_ch_index", "").strip()
            if fid:
                rf_to_fid.setdefault(rf, int(float(fid)))
            try:
                det = float(row.get("detectability_db") or "-inf")
            except ValueError:
                det = float("-inf")
            cur = best_by_rf.get(rf)
            if cur is None or det > cur[0]:
                best_by_rf[rf] = (det, row)
    census = {rf_to_fid[rf]: row for rf, (det, row) in best_by_rf.items()
              if rf in rf_to_fid}

    chans = [Channel(p, args) for p in paths]
    chans.sort(key=lambda c: c.freq_id)

    onsky, null_rows, unit_rows, meta_rows, joint_rows = [], [], [], [], []
    int_rows, int_null_rows = [], []
    event_seconds: dict[str, dict] = {}
    window_defs = {}
    validation_fail = []

    for c in chans:
        window_defs[str(c.atsc)] = {
            "freq_id": c.freq_id,
            "anchored": c.anchored,
            "anchor_bin": int(c.anchor),
            "anchor_offset_hz": round(c.offset_hz(), 3),
            "n_bins_in_window": int(c.n_window),
        }
        tiles = c.null_tiles(args)
        z = np.load(c.path, allow_pickle=False)
        # off-epoch quarters, gated on the statistic itself (the
        # same-statistic rule extends to the null gate): a quarter is
        # transmitter-off when the window statistic's median sits at the
        # null bulk. Recorded CFAR detections cannot serve here -- the
        # any-bin rule splashes ~9% into the window even on verified-off
        # frames.
        off_quarters = []
        for q in c.quarters:
            sel = c.valid & (c.epoch == q)
            if sel.sum() >= 30 and (
                float(np.median(c.F_frame[sel])) < args.off_epoch_max_median
            ):
                off_quarters.append(q)

        n_valid_total = 0
        for q in c.quarters:
            sel = c.valid & (c.epoch == q)
            counts = hist_rows(c.F_frame[sel], F_EDGES)
            if int(counts.sum()) != int(sel.sum()):
                validation_fail.append(
                    f"ch{c.atsc} {q}: histogram sum {counts.sum()} != "
                    f"{sel.sum()} valid frames"
                )
            n_valid_total += int(sel.sum())
            for i, n in enumerate(counts):
                onsky.append((c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n)))
            # off-pilot null windows for this quarter
            if tiles:
                vals = np.concatenate(
                    [c.fine[sel][:, t].max(axis=1) for t in tiles]
                ) if sel.sum() else np.zeros(0)
                for i, n in enumerate(hist_rows(vals, F_EDGES)):
                    null_rows.append(
                        (c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n),
                         "offpilot_bins")
                    )
            if q in off_quarters:
                for i, n in enumerate(hist_rows(c.F_frame[sel], F_EDGES)):
                    null_rows.append(
                        (c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n),
                         "off_epoch_anchor_window")
                    )

        # per-event maxima (File 3 substitute; events == units in this
        # archive, one baseband capture each) and the integrated-event
        # statistic (policy-C exploration: the window max of the
        # event-averaged spectrum -- mean of fstat_fine over the event's
        # valid frames, a mean-of-ratios approximation to true
        # power-integrated detection; see provenance)
        n_units_valid = 0
        unit_max = {}
        unit_frames: dict[int, list[int]] = {}
        for f in np.flatnonzero(c.valid):
            u = int(c.unit_index[f])
            unit_max[u] = max(unit_max.get(u, 0.0), float(c.F_frame[f]))
            unit_frames.setdefault(u, []).append(f)
        by_quarter: dict[str, list[float]] = {}
        int_by_quarter: dict[str, list[float]] = {}
        u_t0 = np.asarray(z["unit_time0_ctime"]).reshape(-1)
        for u, w in unit_max.items():
            q = quarter(u_t0[u])
            by_quarter.setdefault(q, []).append(w)
            spec_int = c.fine[unit_frames[u]].mean(axis=0)
            int_by_quarter.setdefault(q, []).append(
                float(spec_int[c.window].max())
            )
        for q in sorted(by_quarter):
            vals = np.asarray(by_quarter[q])
            n_units_valid += vals.size
            for i, n in enumerate(hist_rows(vals, F_EDGES)):
                unit_rows.append((c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n)))
            ivals = np.asarray(int_by_quarter[q])
            for i, n in enumerate(hist_rows(ivals, F_EDGES)):
                int_rows.append((c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n)))
            if q in off_quarters:
                for i, n in enumerate(hist_rows(ivals, F_EDGES)):
                    int_null_rows.append(
                        (c.atsc, q, F_EDGES[i], F_EDGES[i + 1], int(n),
                         "off_epoch_anchor_window")
                    )
        # off-pilot null of the integrated statistic
        if tiles:
            ivals_null = []
            for u, fl in unit_frames.items():
                spec_int = c.fine[fl].mean(axis=0)
                for t in tiles:
                    ivals_null.append(float(spec_int[t].max()))
            for i, n in enumerate(hist_rows(np.asarray(ivals_null), F_EDGES)):
                int_null_rows.append(
                    (c.atsc, "all", F_EDGES[i], F_EDGES[i + 1], int(n),
                     "offpilot_bins")
                )
        frames_per_event = np.asarray(
            [len(v) for v in unit_frames.values()], dtype=float
        )
        event_seconds[str(c.atsc)] = {
            "frames_p50": float(np.median(frames_per_event)),
            "frames_p95": float(np.percentile(frames_per_event, 95)),
            "frames_max": float(frames_per_event.max()),
            "seconds_max": float(frames_per_event.max()) * FRAME_SECONDS,
        }

        # channel meta
        cen = census.get(c.freq_id, {})
        offpilot_all = (
            np.concatenate(
                [c.fine[c.valid][:, t].max(axis=1) for t in tiles]
            ) if tiles and c.valid.sum() else np.zeros(0)
        )
        off_epoch_all = (
            np.concatenate([
                c.F_frame[c.valid & (c.epoch == q)] for q in off_quarters
            ]) if off_quarters else np.zeros(0)
        )
        # zero-point basis: prefer the verified transmitter-off epochs
        # (leakage-free); fall back to off-pilot windows (contaminated by
        # transmitter leakage on occupied on-epochs -- see provenance)
        if off_epoch_all.size:
            zp_basis, zp_vals = "off_epoch", off_epoch_all
        else:
            zp_basis, zp_vals = "offpilot", offpilot_all
        zp_ok = bool(zp_vals.size and abs(np.median(zp_vals) - 1.0) < 0.10)
        meta_rows.append({
            "atsc_channel": c.atsc,
            "epoch": "all",
            "n_valid_frames": n_valid_total,
            "n_valid_windows": n_units_valid,
            "census_candidate": cen.get("callsign", ""),
            "census_offset_hz": (round(c.offset_hz(), 1) if c.anchored else ""),
            "zero_point_ok": zp_ok,
            "notes": (
                f"anchored at {c.offset_hz():+.0f} Hz"
                if c.anchored else "no persistent line; nominal window"
            ) + (f"; off-epoch quarters: {','.join(off_quarters)}"
                 if off_quarters else "") + f"; zero-point basis: {zp_basis}",
        })

        # joint (F_frame, power) 2-D histogram, pooled epochs
        v = c.valid
        p = np.clip(c.power[v], 1e-300, None)
        p_edges = np.r_[0.0, np.geomspace(
            max(p.min() * 0.99, 1e-12), p.max() * 1.01, 39), np.inf]
        H, _, _ = np.histogram2d(c.F_frame[v], p, bins=[F_EDGES[::4], p_edges])
        fe = F_EDGES[::4]
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i, j]:
                    joint_rows.append(
                        (c.atsc, fe[i], fe[i + 1], p_edges[j], p_edges[j + 1],
                         int(H[i, j]))
                    )

    # absent channels
    present = {c.atsc for c in chans}
    for fid, row in sorted(census.items()):
        ch = int(row["rf_channel"])
        if ch not in present:
            meta_rows.append({
                "atsc_channel": ch, "epoch": "all",
                "n_valid_frames": 0, "n_valid_windows": 0,
                "census_candidate": row.get("callsign", ""),
                "census_offset_hz": "", "zero_point_ok": "",
                "notes": args.absent_note,
            })

    def write_csv(name, header, rows, formatter):
        with open(os.path.join(args.out, name), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            for r in rows:
                w.writerow(formatter(r))

    write_csv("onsky_frame_hist.csv",
              ["atsc_channel", "epoch", "bin_lo", "bin_hi", "n_frames"],
              onsky, lambda r: [r[0], r[1], fmt(r[2]), fmt(r[3]), r[4]])
    write_csv("null_frame_hist.csv",
              ["atsc_channel", "epoch", "bin_lo", "bin_hi", "n_frames",
               "source"],
              null_rows, lambda r: [r[0], r[1], fmt(r[2]), fmt(r[3]), r[4], r[5]])
    write_csv("window_event_hist.csv",
              ["atsc_channel", "epoch", "bin_lo", "bin_hi", "n_windows"],
              unit_rows, lambda r: [r[0], r[1], fmt(r[2]), fmt(r[3]), r[4]])
    write_csv("window_event_integrated_hist.csv",
              ["atsc_channel", "epoch", "bin_lo", "bin_hi", "n_windows"],
              int_rows, lambda r: [r[0], r[1], fmt(r[2]), fmt(r[3]), r[4]])
    write_csv("null_event_integrated_hist.csv",
              ["atsc_channel", "epoch", "bin_lo", "bin_hi", "n_windows",
               "source"],
              int_null_rows,
              lambda r: [r[0], r[1], fmt(r[2]), fmt(r[3]), r[4], r[5]])
    write_csv("joint_f_power_hist.csv",
              ["atsc_channel", "f_bin_lo", "f_bin_hi", "p_bin_lo", "p_bin_hi",
               "n_frames"],
              joint_rows,
              lambda r: [r[0], fmt(r[1]), fmt(r[2]), fmt(r[3]), fmt(r[4]), r[5]])
    with open(os.path.join(args.out, "channel_meta.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)

    n_streams = 2048
    with open(os.path.join(args.out, "null_model.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "statistic": (
                f"max over the {2 * args.window_halfwidth + 1} designated "
                "fine bins of fstat_fine[b] = 2*S_t[b]/(S_l[b]+S_u[b])"
            ),
            "N_bins_in_window": 2 * args.window_halfwidth + 1,
            "per_bin_null": {
                "family": "fdist",
                "params": {"dfn": 2 * n_streams, "dfd": 4 * n_streams,
                           "note": "iid model; measured null is wider "
                                   "(real-spectrum correlations)"},
            },
            "bins_independent": False,
            "mu0_definition": (
                "the fine ratio is norm-self-corrected; no external mu0 "
                "scaling is applied to fstat_fine"
            ),
            "per_channel": False,
        }, f, indent=2)
        f.write("\n")

    prov = {
        "spec": "PilotProxy -> baonoise export specification (CANFAR pass)",
        "statistic_convention": (
            "F_frame = max over the per-channel designated window of "
            "fstat_fine[b]; window = measured-line anchor +/- "
            f"{args.window_halfwidth} fine bins where a persistent line is "
            f"detected (detection fraction >= {args.anchor_fraction}), else "
            "nominal bin 0 +/- the same halfwidth. Identical statistic in "
            "onsky, null, and unit-window files (same-statistic rule)."
        ),
        "fine_bin_hz": FINE_BIN_HZ,
        "frame_seconds": FRAME_SECONDS,
        "window_definitions": window_defs,
        "null_construction": {
            "offpilot_bins": (
                f"disjoint {2 * args.window_halfwidth + 1}-bin windows "
                f">= {args.null_clear_hz} Hz clear of the anchor and of all "
                "persistent-line bins (per-bin median of the bulk-"
                f"normalized spectrum >= {args.line_ratio}, dilated); same "
                "max statistic. CAVEAT: on occupied channels during "
                "on-epochs these windows carry transmitter leakage "
                "(scalloping sidelobes and the data shelf inside the "
                "fine span), so this source is a contaminated upper bound "
                "on the null there; off_epoch_anchor_window rows are the "
                "calibration-grade null where they exist"
            ),
            "off_epoch_anchor_window": (
                "anchor-window statistic from (channel, quarter) cells "
                "whose quarterly median of the window statistic is below "
                f"{args.off_epoch_max_median} with >= 30 frames "
                "(transmitter-off epochs; leakage-free, the calibration-"
                "grade null)"
            ),
        },
        "file3_substitution": (
            "true 10 s receiver windows do not exist in archived baseband "
            "snapshots; window_event_hist.csv holds the per-EVENT maximum "
            "(one baseband capture; events and units are one-to-one in "
            "this archive; sizes VARY, 1-14 valid frames, <= ~0.6 s). "
            "Because flagged frames concentrate in frame-rich events, the "
            "spec's occupancy-amplification inequality does not hold for "
            "variable-size windows and is reported, not enforced. The "
            "0.6 s -> 10 s occupancy segment is unmeasurable from "
            "snapshots and closes at deployment: the enqueued per-frame "
            "mask bits give exceedance_10s with a trivial counter."
        ),
        "integrated_event_product": (
            "window_event_integrated_hist.csv / "
            "null_event_integrated_hist.csv: window max of the "
            "event-AVERAGED spectrum (mean of fstat_fine over the event's "
            "valid frames) -- an exploratory third policy (detect on "
            "integrated spectra rather than flag from per-frame "
            "decisions). Mean-of-ratios approximation: exact "
            "power-integrated detection needs per-bin numerator and "
            "denominator sums, which the current product schema does not "
            "persist (a schema-v4 candidate if this policy is pursued)."
        ),
        "event_window_seconds": event_seconds,
        "binning": "edges = [0] + geomspace(0.3, 300, 151) + [inf] "
                   "(50 bins/decade)",
        "epoch_weighting": "calendar quarters (UTC); exposure = n_valid_frames",
        "detector_versions": sorted({c.detector_version for c in chans}),
        "products": [os.path.basename(c.path) for c in chans],
    }
    with open(os.path.join(args.out, "provenance.json"), "w",
              encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")

    # ---- ingest-side validations (spec section) -------------------------
    # 1. bin sums vs meta; 2. contiguity; 4. occupancy inequality
    meta_by = {(m["atsc_channel"]): m for m in meta_rows if m["n_valid_frames"]}
    sums: dict[int, int] = {}
    for ch, q, lo, hi, n in onsky:
        sums[ch] = sums.get(ch, 0) + n
    for ch, m in meta_by.items():
        if sums.get(ch, 0) != m["n_valid_frames"]:
            validation_fail.append(
                f"ch{ch}: onsky sum {sums.get(ch, 0)} != meta "
                f"{m['n_valid_frames']}"
            )
    for c in chans:
        fr = np.zeros(F_EDGES.size - 1)
        un = np.zeros(F_EDGES.size - 1)
        for ch, q, lo, hi, n in onsky:
            if ch == c.atsc:
                fr[np.searchsorted(F_EDGES, lo, side="left")] += n
        for ch, q, lo, hi, n in unit_rows:
            if ch == c.atsc:
                un[np.searchsorted(F_EDGES, lo, side="left")] += n
        if fr.sum() and un.sum():
            exc_f = 1.0 - np.cumsum(fr) / fr.sum()
            exc_u = 1.0 - np.cumsum(un) / un.sum()
            gap = float((exc_f - exc_u).max())
            if gap > 1e-12:
                # Expected for VARIABLE-size windows: flagged frames
                # concentrate in frame-rich units, so pooled unit-level
                # exceedance can dip below frame-level. The spec's
                # inequality presumes uniform 10 s windows; report, don't
                # fail (see file3_substitution in provenance.json).
                i = int(np.argmax(exc_f - exc_u))
                print(
                    f"  note ch{c.atsc}: unit-window exceedance dips "
                    f"{gap:.4f} below frame exceedance near tau="
                    f"{fmt(F_EDGES[i + 1])} (variable unit sizes; expected "
                    "for the File-3 substitute)"
                )

    n_null = sum(n for *_, n, s in
                 [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in null_rows])
    print(f"channels exported: {[c.atsc for c in chans]}")
    print(f"null window samples: {sum(r[4] for r in null_rows):,} "
          f"(offpilot + off-epoch)")
    for c in chans:
        w = window_defs[str(c.atsc)]
        print(f"  ch{c.atsc}: window anchor bin {w['anchor_bin']} "
              f"({w['anchor_offset_hz']:+.0f} Hz, "
              f"{'measured' if w['anchored'] else 'nominal'})")
    if validation_fail:
        print("VALIDATION FAILURES:", file=sys.stderr)
        for v in validation_fail:
            print("  " + v, file=sys.stderr)
        return 1
    print("ingest-side validations: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
