#!/usr/bin/env python3
"""Run the detector calibration over the completed per-pilot survey.

Writes ``calibration.json`` and the CSV tables the figures and report are
built from.  Run bao-noise-tolerance's ``eta_bao.py`` first if the per-channel
thresholds should be merged in; without it the ladder and dispositions are
still complete, on a single global eta.

    python3 analysis/make_calibration_data.py [--products DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402

from ppcal import spectra as S, state as ST  # noqa: E402
from ppcal.calib import ETA_LADDER  # noqa: E402
from ppcal.products import FRAME_SECONDS, month_label  # noqa: E402


def write_csv(path, rows, columns=None):
    if not rows:
        return 0
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bao", default=None)
    args = ap.parse_args(argv)
    args.products = args.products or str(P.PER_PILOT)
    args.out = args.out or str(P.OUT)
    args.bao = args.bao or str(P.ETA_BAO)
    tables = os.path.join(args.out, "tables")
    os.makedirs(tables, exist_ok=True)

    st = ST.build(args.products, bao_csv=args.bao)
    inv, era_rows, cal_rows, tx_rows, line_rows = [], [], [], [], []
    doc = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(),
           "products": args.products, "eta_ladder": list(ETA_LADDER),
           "eta_working_fallback": ST.ETA_WORKING,
           "carrier_dominated_db": ST.CARRIER_DOMINATED_DB,
           "channels": {}}

    tot_frames = tot_units = 0
    for s in st:
        c, cal, segs = s.c, s.cal, s.segs
        tot_frames += c.n_frames_raw
        tot_units += c.n_units_raw
        peaks = S.peak_census(c)
        d_pilot, db_pilot = S.measured_pilot_offset_hz(c)
        lines = S.fine_line_census(c, s.fmask)
        nsec = sum(1 for p in peaks if p.kind == "secondary")
        _, _, _, mstats = S.era_fine_spectrum_masked(
            c, s.fmask, s.eta_channel * cal.mu)
        wide_ok = S.wide_pair_is_era_resolved(segs)

        inv.append(dict(
            ch=c.ch, freq_id=c.fid, pilot_mhz=round(c.pilot_hz / 1e6, 6),
            chime_center_mhz=round(c.center_hz / 1e6, 6),
            pilot_offset_khz=round(c.pilot_offset_hz / 1e3, 3),
            frames=c.n_frames_raw, frames_healthy=int(c.health_include.sum()),
            frames_excluded=int((~c.health_include).sum()),
            units=c.n_units_raw,
            span="%s..%s" % (month_label(c.frame_month.min()),
                             month_label(c.frame_month.max())),
            integration_hours=round(c.n_frames_raw * FRAME_SECONDS / 3600, 3),
            n_eras=len(segs), final_era=segs[-1].label,
            final_era_frames=int(s.fmask.sum()),
            final_era_units=int(s.umask.sum()),
            secondary_carriers=nsec, fine_lines=len(lines),
            wide_pair_is_latest_era=wide_ok,
            measured_carrier_offset_khz=round(d_pilot / 1e3, 4),
            measured_carrier_db=round(db_pilot, 2)))

        for seg in segs:
            era_rows.append(dict(
                ch=c.ch, era=seg.index + 1, n_eras=len(segs), span=seg.label,
                month_start=month_label(seg.month_start),
                month_end=month_label(seg.month_end), units=seg.n_units,
                level_median_db=round(seg.level_median_db, 3),
                level_p90_db=round(seg.level_p90_db, 3),
                is_final=(seg.index == len(segs) - 1),
                locked_transition=ST.LOCKED_EPOCH.get(c.ch, "")))

        row = cal.row()
        row.update(
            mask_peak_suppression_db=round(mstats["peak_suppression_db"], 3),
            mask_band_suppression_db=round(mstats["band_suppression_db"], 3),
            wide_pair_is_latest_era=wide_ok,
            eta_channel=round(s.eta_channel, 4),
            eta_thermal=round(s.eta_thermal, 4),
            eta_bracket_ratio=round(s.eta_bracket_ratio, 4),
            eta_is_identified=s.eta_is_identified,
            eta_is_per_channel=s.eta_is_per_channel,
            occ_at_eta_channel=round(s.occ_working, 6),
            occ_at_eta_global=round(s.occ_global, 6),
            verdict=s.verdict, disposition=s.disposition, reason=s.reason,
            carrier_dominated=s.carrier_dominated,
            published_verdict=("excise" if c.ch in ST.PUBLISHED_EXCISED
                               else "keep"),
            agrees_with_published=s.agrees_with_published,
            inclusive_keep_override=(c.ch in ST.PUBLISHED_INCLUSIVE_KEEP),
            collection_ceased=ST.COLLECTION_CEASED.get(c.ch, ""),
            # same eta on both populations, each with its own calibrated mu
            era_blind_occ_working=round(float(np.mean(
                c.fstat > s.eta_channel * s.blind.mu)), 6),
            era_blind_mu_shift_db=round(s.blind.mu_shift_db, 4),
            residual_basis=s.bao.get("residual_basis", ""),
            r_tol_dilation=s.bao.get("r_tol_dilation", ""),
            tau_seconds=s.bao.get("tau_seconds", ""),
            tau_measured=s.bao.get("tau_measured", ""),
            r_cost_cap=s.bao.get("r_cost_cap", ""))
        cal_rows.append({k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in row.items()})

        for p in peaks:
            tx_rows.append(dict(
                ch=c.ch,
                rf_offset_from_centre_khz=round(p.rf_offset_hz / 1e3, 4),
                offset_from_pilot_khz=round(p.offset_from_pilot_hz / 1e3, 4),
                db_rel_median=round(p.db_rel_median, 2), kind=p.kind))
        for hz, dbm, dbp in lines:
            line_rows.append(dict(ch=c.ch, era=segs[-1].label,
                                  fine_offset_hz=round(hz, 2),
                                  median_db=round(dbm, 2), p90_db=round(dbp, 2)))

        g_lo, g_hi = S.guard_reference_offsets_hz(c)
        doc["channels"][str(c.ch)] = dict(
            freq_id=c.fid, mu0_provisional=c.mu0, sense=c.sense,
            pilot_offset_hz=c.pilot_offset_hz,
            target_coarse_offset_hz=S.target_coarse_offset_hz(c),
            guard_refs_hz=[g_lo, g_hi],
            measured_pilot_offset_hz=d_pilot, measured_pilot_db=db_pilot,
            predicted_fine_bin=c.predicted_fine_bin,
            eras=[dict(span=e.label, units=e.n_units,
                       level_median_db=e.level_median_db,
                       level_p90_db=e.level_p90_db) for e in segs],
            calibration=cal.row(), era_blind=s.blind.row(), bao=s.bao,
            mask_effect=mstats, wide_pair_is_latest_era=wide_ok,
            verdict=s.verdict, disposition=s.disposition, reason=s.reason,
            agrees_with_published=s.agrees_with_published,
            n_secondary=nsec, n_fine_lines=len(lines))

        print("ch%-3d eras=%d  mu=%9.4f (%+7.3f dB)  eta=%5.3f  occ>1=%.3f  "
              "occ>eta.mu=%.3f  %-20s pub=%-7s %s"
              % (c.ch, len(segs), cal.mu, cal.mu_shift_db, s.eta_channel,
                 cal.occupancy_provisional, s.occ_working, s.disposition,
                 "excise" if c.ch in ST.PUBLISHED_EXCISED else "keep",
                 "ok" if s.agrees_with_published else "DIFFERS"))

    keep = [s for s in st if s.verdict == "keep"]
    doc["totals"] = dict(
        channels=len(st), frames=tot_frames, units=tot_units,
        integration_hours=round(tot_frames * FRAME_SECONDS / 3600, 2),
        kept=len(keep), excised=len(st) - len(keep),
        agree_with_published=sum(1 for s in st if s.agrees_with_published),
        median_occ_provisional=float(np.median(
            [s.cal.occupancy_provisional for s in st])),
        median_occ_working=float(np.median([s.occ_working for s in st])),
        kept_median_occ_working=float(np.median(
            [s.occ_working for s in keep])),
        kept_median_occ_provisional=float(np.median(
            [s.cal.occupancy_provisional for s in keep])),
        eta_min=float(min(s.eta_channel for s in st)),
        eta_max=float(max(s.eta_channel for s in st)),
        eta_median=float(np.median([s.eta_channel for s in st])),
        eta_per_channel=sum(1 for s in st if s.eta_is_per_channel))
    with open(os.path.join(args.out, "calibration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, default=float)
        fh.write("\n")

    tau_rows = []
    for s_ in sorted(st, key=lambda z: -(z.eta_bracket_ratio
                                         if z.eta_bracket_ratio == z.eta_bracket_ratio
                                         else 0)):
        tau_rows.append(dict(
            ch=s_.ch,
            tau_measured=s_.bao.get("tau_measured", ""),
            tau_seconds=s_.bao.get("tau_seconds", ""),
            eta_cap=round(s_.eta_channel, 4),
            eta_thermal=round(s_.eta_thermal, 4),
            bracket_ratio=round(s_.eta_bracket_ratio, 4),
            identified=s_.eta_is_identified,
            verdict=s_.verdict,
            masked_at_eta_cap=round(s_.occ_working, 4)))
    write_csv(os.path.join(tables, "tau_priority.csv"), tau_rows)

    write_csv(os.path.join(tables, "channel_inventory.csv"), inv)
    write_csv(os.path.join(tables, "eras.csv"), era_rows)
    write_csv(os.path.join(tables, "calibration.csv"), cal_rows)
    write_csv(os.path.join(tables, "transmitter_census.csv"), tx_rows)
    write_csv(os.path.join(tables, "fine_lines.csv"), line_rows)
    t = doc["totals"]
    print("\n%d channels, %s frames (%.2f h), %s units"
          % (t["channels"], "{:,}".format(t["frames"]), t["integration_hours"],
             "{:,}".format(t["units"])))
    print("kept %d / excised %d; %d of %d agree with the published policy"
          % (t["kept"], t["excised"], t["agree_with_published"], t["channels"]))
    print("per-channel eta: %.3f .. %.3f (median %.3f), %d of %d priced "
          "individually"
          % (t["eta_min"], t["eta_max"], t["eta_median"],
             t["eta_per_channel"], t["channels"]))
    ident = [x for x in st if x.eta_is_identified]
    ratios = sorted(x.eta_bracket_ratio for x in st
                    if x.eta_bracket_ratio == x.eta_bracket_ratio)
    print("eta identified (bracket ratio < 1.1) on %d of %d channels; "
          "ratio median %.2f, max %.2f"
          % (len(ident), len(st), ratios[len(ratios) // 2], ratios[-1]))
    print("kept channels, median masked fraction: %.1f%% under F > 1, "
          "%.1f%% under F > eta*mu"
          % (100 * t["kept_median_occ_provisional"],
             100 * t["kept_median_occ_working"]))
    print("tables ->", tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
