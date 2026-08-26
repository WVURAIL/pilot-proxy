#!/usr/bin/env python3
"""Build policy_data.json for the historical DTV masking comparison.

Three stages over the per-pilot survey products (*.npz under $PP_PER_PILOT
or --products), all through the released RFIsher package --- the same dependency
``tools/make_dissertation_tables.py`` and ``plot_channel_histograms.py``
already carry:

1. per-channel threshold sweeps, coherence times, Fisher-forecast pricing,
   and the historical report labels;
2. the fixed report rule applied post-hoc to every archived frame
   (monthly masked fractions under the recorded and historical rules,
   kept-frame leakage);
3. the threshold-ladder fields (the literal F > 1 window fraction and each
   channel's mu0).

Locked report methodology --- do not change without an artifact revision:

* Window: since="2025-01" --- every month after the last detected
  transmitter transition (ch19 sign-off, Dec 2024).
* Coherence bracket: each sweep runs at the measured-or-day-cap tau AND at
  the thermal floor (tau = one frame, n_coh = 1); historical labels that
  agree at both ends are reported, and the rest are bracketed.
* INCLUSIVE_KEEP / COLLECTION_CEASED overrides and the ETAS grid below.

This artifact is report-only. It does not select or export an operational
threshold.

Render the page afterwards with ``render_artifacts.py``.

    python3 analysis/make_policy_data.py --products DIR --out DIR
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401  -- puts <repo>/src on sys.path
from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO)
from pilot_proxy.archive_health import (
    FRAME_HEALTH_GATE_SCHEMA_VERSION,
    evaluate_frame_health,
    temporary_residual_health_views,
)
from rfisher import api, channels as chn, residual as res, scenarios

import _products as P

SINCE = "2025-01"
# Historical labels kept-and-masked under the inclusive-for-now rule (see the
# override below). 14 and 15 joined 36 at the complete-23 snapshot:
# both satisfy the same condition (excise picked only at the coherence
# cap, bracket disagreement immaterial, the report rule frees ~75%
# of current-epoch frames -- an episodic faint carrier, not a pinned
# occupancy), and excising a ~75%-recoverable allocation would break
# the excision-only-where-recoverable-rounds-to-zero posture.
INCLUSIVE_KEEP = {14, 15, 36}
# channels whose current-epoch absence is an operations exclusion, not backlog
COLLECTION_CEASED = {30: "September 2023"}
ETAS = [1.0, 1.2, 1.4, 2.0, 5.0]
FRAME_S = 16384 * 2.56e-6          # 41.94 ms -> thermal end of the bracket
DAY_CAP = 86164.0
HISTORICAL_REPORT_ETA = 1.4
# survey span 2018-12 .. 2026-07 (extend M1 when a newer snapshot adds months)
M0 = 2018 * 12 + 11
M1 = 2026 * 12 + 6
NMONTHS = M1 - M0 + 1

# Transmitter epochs from the monthly occupancy analysis.
#
# The two epoch slots encode "the transmitter was off before this month"
# (off_through, a sign-on channel) and "off from this month" (off_from, a
# sign-off or step-down channel). Between them they cover every transition
# where one side of the boundary is a transmitter-off state -- which is what
# the residual machinery needs, because the off epoch is where a sensitivity
# floor can be measured.
#
# Channel 17's transition is of a third kind: both sides are on, at +12.6 dB
# through 2022-09 and +16.4 dB from 2022-10, which reads as a change of
# transmitter configuration rather than a sign-on or a sign-off. Neither slot
# can express it without asserting an off epoch that does not exist, so both
# are left empty and only the text is carried. The consequence is worth
# stating: ch17 has no epoch mask here, so any era-blind quantity computed
# for it -- its measured carrier offset in particular -- averages two
# configurations. Its disposition does not depend on this; it is excised on
# occupancy either way.
EPOCH = {  # ch -> (off_through, off_from, transition text)
    35: ("2021-08", None, "sign-on Nov 2021"),
    19: (None, "2024-12", "sign-off Dec 2024"),
    26: (None, "2023-04", "sign-off Apr 2023"),
    20: (None, "2022-09", "step down Sep 2022"),
    27: (None, "2022-10", "sign-off in 2021-22 archive gap"),
    32: (None, "2022-10", "sign-off in 2021-22 archive gap"),
    17: (None, None, "level step Oct 2022, on-to-on"),
}
CLASS = {17: "persistent", 22: "persistent", 24: "persistent", 30: "persistent",
         31: "persistent", 35: "persistent",
         19: "signed off", 20: "signed off", 26: "signed off",
         27: "signed off", 32: "signed off",
         34: "burst", 36: "burst",
         28: "weak", 33: "weak",
         16: "trace", 18: "trace", 21: "trace", 23: "trace", 25: "trace",
         29: "trace"}


def historical_report(paths, by_ch):
    """Stage 1: sweeps, forecast pricing, and historical report labels."""
    tables = {}
    for eta in ETAS:
        tables[eta] = chn.mask_table_from_products(paths, since=SINCE, eta=eta)
    n_frames_window = tables[1.0].n_frames

    fc = api.load()

    def req_hours(sc, zbin=None):
        bins = None if zbin is None else [int(zbin)]
        return fc.required_hours_metric(
            lambda t: fc.significance(sc, t, bins=bins), 5.0)

    clean_s = req_hours(scenarios.clean())
    clean_w = req_hours(scenarios.clean(), 6)

    def price(chn_id, f, r):
        sc = scenarios.at_threshold({chn_id: (f, r)},
                                    residual_excise_threshold=np.inf)
        return req_hours(sc) / clean_s

    def price_excise(chn_id):
        return req_hours(scenarios.single_channel(chn_id, 1.0, keep=False)) / clean_s

    out = {
        "since": SINCE,
        "etas": ETAS,
        "clean_hours": round(clean_s, 1),
        "decision_scope": "historical_report_only",
        "historical_report_eta": HISTORICAL_REPORT_ETA,
        "channels": {},
        "policy": {},
    }

    for ch in sorted(by_ch):
        info = by_ch[ch]
        ot, of, transition = EPOCH.get(ch, (None, None, None))
        z_lo, z_hi = chn.channel_z_range(ch)
        rec = dict(fid=info["fid"], pilot_mhz=round(info["pilot_mhz"], 4),
                   z=[round(z_lo, 3), round(z_hi, 3)],
                   cls=CLASS.get(ch, "trace"), transition=transition,
                   window_frac={str(e): (round(tables[e].fractions[ch], 4)
                                         if ch in tables[e].n_frames else None)
                                for e in ETAS},
                   window_n=int(n_frames_window.get(ch, 0)))

        # sweeps at both bracket ends
        path = info["path"]
        with np.load(path, allow_pickle=False) as health_view:
            health_valid = health_view["valid"][:, 0].astype(bool)
            health_rejected = health_view["reject_mask"][:, 0].astype(bool)
            health_kept = int(np.count_nonzero(health_valid & ~health_rejected))
            rec["health_included_frames"] = int(health_valid.sum())
            rec["health_included_stored_keep_frames"] = health_kept
        kw = dict(off_through=ot, off_from=of)
        sweeps = {}
        for tag, tau in (("cap", None), ("thermal", FRAME_S)):
            if health_kept == 0:
                sweeps[tag] = []
                continue
            try:
                s = res.threshold_sweep(path, tau_intraday=tau, **kw)
                if not s:
                    fp = res.floor_provenance(path)
                    s = res.threshold_sweep(path, tau_intraday=tau,
                                            floor_db=float(fp.sigma_implied_db),
                                            **kw)
                    rec["floor_substituted"] = True
                sweeps[tag] = s
            except Exception as e:
                sweeps[tag] = []
                rec.setdefault("sweep_errors", []).append(f"{tag}: {e}")
        ct = res.correlation_time(path, off_through=ot, off_from=of)
        rec["tau"] = dict(seconds=round(float(ct.tau_for_budget)),
                          measured=bool(ct.tau_for_budget < DAY_CAP * 0.99),
                          quality=str(getattr(ct, "quality", "")),
                          reason=str(getattr(ct, "reason", "")))

        def row_at(s, eta):
            if not s:
                return None
            r0 = min(s, key=lambda row: abs(row["eta"] - eta))
            return dict(eta=round(r0["eta"], 2), f=round(r0["f"], 4),
                        r=float(f"{r0['r_masked']:.4g}"), net=round(r0["net"], 3))

        for tag in ("cap", "thermal"):
            s = sweeps[tag]
            rec[f"sweep_{tag}"] = dict(
                r_unmasked=float(f"{s[0]['r_unmasked']:.4g}") if s else None,
                best=(lambda b: dict(eta=round(b["eta"], 2), f=round(b["f"], 4),
                                     net=round(b["net"], 3)) if b else None)(
                    res.best_operating_point(s)),
                at14=row_at(s, 1.4),
            )

        # Historical report label. The output key is frozen for compatibility.
        f1 = rec["window_frac"]["1.0"]
        f14 = rec["window_frac"]["1.4"]
        prices = {}
        if not any(sweeps.values()):
            if health_kept != 0:
                raise RuntimeError(
                    f"ch{ch}: both health-filtered residual sweeps are empty "
                    f"despite {health_kept} stored-mask kept frames; refusing "
                    "to manufacture a historical report label"
                )
            action = "excise"
            why = (
                "conservative health-gate decision: every frame kept by the "
                "stored coarse rule was a v1-excluded full-scale artifact, so "
                "no health-included kept/null frame remains to calibrate a "
                "residual floor or threshold sweep; excise pending new clean "
                "or transmitter-off calibration data"
            )
            prices["excise"] = round(price_excise(ch), 4)
            chosen = ("excised", 1.0, 0.0)
            agrees = None
            rec["health_floor_status"] = "unavailable_no_health_included_kept_frames"
        elif not all(sweeps.values()):
            raise RuntimeError(
                f"ch{ch}: only one coherence-bracket sweep succeeded; refusing "
                "to manufacture a historical report label"
            )
        elif f14 is None:
            if ch in COLLECTION_CEASED:
                action = "excise"
                why = ("no current-epoch frames because CHIME ceased baseband "
                       f"collection on this channel in {COLLECTION_CEASED[ch]} due "
                       "to the contamination itself; the archive record supports "
                       "the historical label — "
                       "shelf at system noise, the strongest carrier measured, "
                       "transmitter in Penticton — and the operational exclusion "
                       "corroborates excision; revisit only via periodic "
                       "spot-collection")
            else:
                action = "excise (pending data)"
                why = ("no processed frames in the current window; persistent "
                       "refused channel — price as excised, revisit")
            prices["excise"] = round(price_excise(ch), 4)
            chosen = ("excised", 1.0, 0.0)
            agrees = True
        elif f14 < 0.15:
            action = "clean"
            why = (f"under the historical eta=1.4 report rule, only "
                   f"{100*f14:.1f}% of current-epoch frames mask "
                   f"(vs {100*f1:.1f}% at the deployed rule); no transmitter to "
                   f"fight — the channel is effectively clean")
            r_fwd = 0.0
            prices["mask_at_1.4"] = round(price(ch, f14, r_fwd), 4)
            chosen = ("kept", f14, r_fwd)
            agrees = True
        else:
            opts = {}
            for tag in ("cap", "thermal"):
                s = rec[f"sweep_{tag}"]
                cand = {"excise": price_excise(ch)}
                if s["r_unmasked"] is not None:
                    cand["keep dirty"] = price(ch, 0.0, s["r_unmasked"])
                if s["at14"]:
                    cand["mask at best"] = price(
                        ch, f14, (s["best"] and next(
                            (row["r"] for row in [row_at(sweeps[tag], s["best"]["eta"])] if row),
                            0.0)) or 0.0)
                opts[tag] = cand
            pick = {tag: min(v, key=v.get) for tag, v in opts.items()}
            agrees = pick["cap"] == pick["thermal"]
            action = pick["cap"]
            spread = abs(opts["cap"][pick["cap"]] - opts["thermal"][pick["thermal"]])
            why = (f"cheapest option at the coherence cap: {pick['cap']} "
                   f"(x{opts['cap'][pick['cap']]:.3f}); thermal end picks "
                   f"{pick['thermal']} (x{opts['thermal'][pick['thermal']]:.3f}); "
                   + ("bracket agrees" if agrees else
                      f"bracket disagrees but spread is {100*spread:.1f}% of "
                      f"survey time — immaterial"))
            prices = {k: round(v, 4) for k, v in opts["cap"].items()}
            if action == "excise":
                chosen = ("excised", 1.0, 0.0)
            elif action == "keep dirty":
                chosen = ("kept", 0.0, rec["sweep_cap"]["r_unmasked"])
            else:
                b = rec["sweep_cap"]["best"]
                r_b = row_at(sweeps["cap"], b["eta"])["r"] if b else 0.0
                chosen = ("kept", f14, r_b)
            # Historical override where the coherence cap, not a live carrier,
            # drives the label and the report rule frees most frames.
            if (ch in INCLUSIVE_KEEP and action == "excise"
                    and f14 is not None and f14 < 0.5):
                r14 = (row_at(sweeps["cap"], 1.4) or {}).get("r", 0.0)
                action = "keep and mask (coherence unverified)"
                chosen = ("kept", f14, r14)
                why = ("historical inclusive-for-now label: the excision case "
                       "is the unverified coherence cap, not a live carrier; "
                       "kept and masked under the eta=1.4 report rule, priced at "
                       "the cap, pending the burst-resolved tau_c measurement")
        rec["recommendation"] = dict(action=action, why=why, agrees=agrees,
                                     prices=prices,
                                     decision_scope="historical_report_only")
        rec["chosen"] = chosen
        out["channels"][str(ch)] = rec
        print(f"ch{ch:>3} {rec['cls']:<10} f1={f1} f14={f14} -> {action}")

    # Whole-survey historical pricing at both bracket ends where applicable.
    per_channel = {}
    for ch, rec in ((int(c), r) for c, r in out["channels"].items()):
        kind, f, r = rec["chosen"]
        per_channel[ch] = (1.0, 0.0) if kind == "excised" else (f, r)
    sc_pol = scenarios.at_threshold(per_channel)
    out["policy"] = dict(
        decision_scope="historical_report_only",
        survey_x=round(req_hours(sc_pol) / clean_s, 4),
        worst_bin_x=round(req_hours(sc_pol, 6) / clean_w, 4),
        n_excised=sum(1 for r in out["channels"].values()
                      if r["chosen"][0] == "excised"),
    )
    # Recorded-rule comparison: windowed eta=1 fractions, masking only.
    dep = {ch: (tables[1.0].fractions[ch], 0.0)
           for ch in per_channel if ch in tables[1.0].n_frames}
    sc_dep = scenarios.at_threshold(dep)
    w = req_hours(sc_dep, 6)
    out["policy"]["deployed_survey_x"] = round(req_hours(sc_dep) / clean_s, 4)
    out["policy"]["deployed_worst_bin_x"] = (round(w / clean_w, 4)
                                             if np.isfinite(w) else None)
    return out


def apply_to_archive(pd, paths):
    """Stage 2: apply the historical report rule to every archived frame.

    The archived products store the raw coarse power ratio per frame, so the
    report masks are exact post-hoc recomputations, not approximations. Every
    denominator and distribution is restricted by the v1 frame-health gate.
    The frozen output field names retain the earlier artifact schema.
    """
    pd["month_labels"] = [f"{(M0+i)//12}-{(M0+i)%12+1:02d}"
                          for i in range(NMONTHS)]

    tot_dep_kept = tot_report_kept = tot_valid = 0
    for p in sorted(paths):
        with np.load(p, allow_pickle=False) as z:
            ch = str(int(z["physical_channel"][0]))
            rec = pd["channels"][ch]
            action = rec["recommendation"]["action"]
            F = z[ARCHIVED_COARSE_POWER_RATIO][:, 0]
            mu0 = float(np.asarray(z["mu0"]).ravel()[0])
            health = evaluate_frame_health(z)
            valid = health.include
            t0 = z["unit_time0_ctime"]
            months = np.array([
                dt.datetime.fromtimestamp(float(t), dt.timezone.utc).year * 12
                + dt.datetime.fromtimestamp(float(t), dt.timezone.utc).month - 1
                for t in t0])[z["frame_unit_index"]] - M0

        dep = valid & (F > mu0)
        report_rule_applies = (
            action.startswith("clean")
            or action.startswith("retune")
            or action.startswith("keep and mask")
        )
        report_mask = (
            valid & (F > HISTORICAL_REPORT_ETA * mu0)
            if report_rule_applies else None
        )

        def monthly(mask_arr):
            out = []
            for i in range(NMONTHS):
                sel = valid & (months == i)
                n = int(sel.sum())
                out.append(round(float(mask_arr[sel].mean()), 4) if n >= 5 else None)
            return out

        rec["applied"] = dict(
            monthly_deployed=monthly(dep),
            monthly_policy=(monthly(report_mask)
                            if report_rule_applies else None),
            excised=not report_rule_applies,
            health_included_frames=int(valid.sum()),
            health_excluded_frames=int(health.excluded.sum()),
        )
        # Kept-frame excess under the historical report rule.
        if report_rule_applies:
            win = valid & (months >= (2025 * 12 + 0 - M0))
            kept = win & ~report_mask
            if kept.sum():
                exc = 10 * np.log10(np.maximum(F[kept], 1e-9) / mu0)
                rec["applied"]["kept_excess_db"] = dict(
                    med=round(float(np.median(exc)), 3),
                    p99=round(float(np.percentile(exc, 99)), 3),
                    max=round(float(exc.max()), 3),
                    n=int(kept.sum()))
            tot_dep_kept += int((win & ~dep).sum())
            tot_report_kept += int(kept.sum())
            tot_valid += int(win.sum())

    pd["applied_summary"] = dict(
        window=SINCE,
        retuned_valid_frames=tot_valid,
        kept_deployed=tot_dep_kept,
        kept_policy=tot_report_kept,
        kept_deployed_frac=round(tot_dep_kept / tot_valid, 4),
        kept_policy_frac=round(tot_report_kept / tot_valid, 4),
        decision_scope="historical_report_only",
    )
    s = pd["applied_summary"]
    print(f"historical report channels, window {SINCE}->: "
          f"{s['retuned_valid_frames']} valid frames")
    print(f"  deployed rule keeps {100*s['kept_deployed_frac']:.1f}%")
    print(f"  historical report rule keeps {100*s['kept_policy_frac']:.1f}%")


def add_ladder(pd, by_ch):
    """Stage 3: step-1 (F > 1) window fractions and mu0, then the ladder table.

    The step-1 fraction takes F > 1 literally, i.e. eta = 1/mu0 per channel,
    with mu0 at report precision (5 decimals) so the two pages quote the
    same numbers.
    """
    for ch_s, rec in pd["channels"].items():
        ch = int(ch_s)
        mu0 = by_ch[ch]["mu0"]
        try:
            t = chn.mask_table_from_products([by_ch[ch]["path"]],
                                             since=pd["since"], eta=1.0 / mu0)
            frac = t.fractions.get(ch)
        except ValueError:
            frac = None
        rec["window_frac_f1"] = None if frac is None else round(frac, 4)
        rec["mu0"] = mu0

    print(f"\n{'ch':>4} {'mu0':>8} {'F>1 frac':>9} {'F>mu0':>7} {'eta':>5} "
          f"{'rho_thr':>8} {'kept rho-hat med/p99 (dB)':>26}")
    for ch_s in sorted(pd["channels"], key=int):
        rec = pd["channels"][ch_s]
        wf = rec["window_frac"]
        f1raw = rec["window_frac_f1"]
        ap = rec.get("applied") or {}
        exc = ap.get("excised", True)
        ke = ap.get("kept_excess_db")
        eta = "---" if exc else "1.4"
        rho = "---" if exc else "0.4 (+1.46 dB)"
        kept = f"{ke['med']:+.2f} / {ke['p99']:+.2f}" if ke else "---"
        f1c = "---" if f1raw is None else f"{100*f1raw:.1f}%"
        fmu = "---" if wf["1.0"] is None else f"{100*wf['1.0']:.1f}%"
        print(f"ch{ch_s:>2} {rec['mu0']:>8.5f} {f1c:>9} {fmu:>7} {eta:>5} "
              f"{rho:>8} {kept:>26}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", type=Path, default=None,
                    help="directory of per-pilot survey products "
                         "(default: $PP_PER_PILOT)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    per_pilot = P.PER_PILOT if args.products is None else args.products
    paths = sorted(glob.glob(str(per_pilot / "*.npz")))
    if not paths:
        raise SystemExit(f"no per-pilot products (*.npz) under {per_pilot}; "
                         "set PP_PER_PILOT or pass --products")

    source_by_ch = {}
    total_frames = total_health_included = 0
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            ch = int(z["physical_channel"][0])
            health = evaluate_frame_health(z)
            total_frames += int(health.include.size)
            total_health_included += int(health.include.sum())
            source_by_ch[ch] = dict(
                source_path=p,
                fid=int(z["freq_id"][0]),
                pilot_mhz=float(z["pilot_frequency_hz"][0]) / 1e6,
                mu0=round(float(z["mu0"][0]), 5),
            )

    # RFIsher owns the residual and forecast machinery, but its public APIs
    # accept paths and load frame arrays internally. Transient minimal views
    # make the v1 inclusion explicit at that boundary.
    with temporary_residual_health_views([Path(path) for path in paths]) as views:
        view_by_name = {view.name: str(view) for view in views}
        by_ch = {
            ch: {
                **metadata,
                "path": view_by_name[Path(metadata["source_path"]).name],
            }
            for ch, metadata in source_by_ch.items()
        }
        pd = historical_report([str(view) for view in views], by_ch)
        print("\nhistorical report:", pd["policy"])
        apply_to_archive(pd, paths)
        add_ladder(pd, by_ch)

    pd["health_gate"] = dict(
        schema_version=FRAME_HEALTH_GATE_SCHEMA_VERSION,
        policy="fail_closed",
        applied_to_all_residual_inputs=True,
        applied_to_all_baonoise_inputs=True,
        stored_frames=total_frames,
        included_frames=total_health_included,
        excluded_frames=total_frames - total_health_included,
    )

    out_path = args.out / "policy_data.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pd, fh)
    print("\nwrote", out_path)
    return out_path


if __name__ == "__main__":
    main()
