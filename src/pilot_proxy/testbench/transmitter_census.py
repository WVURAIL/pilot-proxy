# coding=utf-8
"""Transmitter-census case study: carrier-offset dispersion by service class.

Consumes two declared inputs and produces the paper's two case-study figures
plus their supporting tables. Nothing here touches the archive; both inputs
already exist (pilotcal line lists; the FCC/ISED 500-mile census).

Input schemas (CSV with a header row; extra columns are ignored)
-----------------------------------------------------------------
census:  rf_channel:int, callsign:str, service_class:str, and either
         detectability_db:float (a precomputed received-strength score, e.g.
         a propagation-model field strength; blanks rank last, tie-broken by
         distance_km when present) or erp_kw:float + distance_km:float
         (score = ERP / distance^2) [, bearing_deg, lat, lon, ...]
         service_class is normalized case-insensitively (punctuation and
         parentheticals collapse to underscores):
           full_service / full-power / primary        -> primary
           translator (lptv) / repeater / relay /
           lptv / low-power (lptv) / class a          -> non_primary
           anything else                              -> other
lines:   rf_channel:int, offset_hz:float (about the channel's nominal
         pilot), snr_db:float [, epoch, ...]

Association rule (pluggable, recorded per line in association.csv)
------------------------------------------------------------------
Measured lines cannot be matched to transmitters by frequency alone --
co-channel transmitters share the nominal pilot; their *offsets* are the
unknown being studied. Per RF channel, lines are ranked by SNR and census
entries by a detectability score (ERP / distance^2); ranks are paired
positionally. Confidence tiers:
  dominant     strongest line paired with the top-scoring entry when that
               entry is primary (the usual case: one full-power station
               dominates the channel)
  ranked       remaining positional pairs
  unassociated line count exceeds census count (a finding, reported)
The alternative "dominant/secondary" strategy labels only the strongest
line per channel (primary where the census has one) and pools the rest as
non_primary without per-transmitter claims.

Outputs (into --output-dir)
---------------------------
association.csv                per-line class label, tier, paired callsign
summary.json                   per-class dispersion, per-channel stats,
                               Spearman rho + bootstrap CI, run parameters
fig_offset_by_class.png/.pdf   class-split offset distribution (Figure A)
fig_spread_vs_composition.png/.pdf
                               per-channel spread vs non-primary fraction
                               among associated lines (Figure B)
threshold_stability.csv        per-channel MAD vs SNR threshold sweep
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIMARY_ALIASES = {"primary", "full_service", "full-service", "full_power",
                   "full-power", "fs", "dt", "dtv"}
NON_PRIMARY_ALIASES = {"non_primary", "translator", "translator_lptv",
                       "tx_translator", "repeater", "rebroadcaster", "relay",
                       "lptv", "low_power", "low_power_lptv", "ld", "dc",
                       "class_a"}


def normalize_class(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if key in PRIMARY_ALIASES:
        return "primary"
    if key in NON_PRIMARY_ALIASES:
        return "non_primary"
    return "other"


@dataclass
class Transmitter:
    rf_channel: int
    callsign: str
    service_class: str          # normalized
    raw_class: str
    detectability: float        # higher ranks first; -inf when unscored
    distance_km: float          # tiebreak among unscored; inf when absent


@dataclass
class Line:
    rf_channel: int
    offset_hz: float
    snr_db: float


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_census(path: Path) -> list[Transmitter]:
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f"census: {path} is empty")
    scored_mode = "detectability_db" in rows[0]
    out = []
    for r in rows:
        dist = float(r["distance_km"]) if str(r.get("distance_km", "")).strip() else float("inf")
        if dist <= 0:
            raise SystemExit(f"census: non-positive distance for {r.get('callsign')}")
        if scored_mode:
            raw = str(r.get("detectability_db", "")).strip()
            score = float(raw) if raw else float("-inf")
        else:
            score = float(r["erp_kw"]) / (dist * dist)
        out.append(Transmitter(
            rf_channel=int(r["rf_channel"]),
            callsign=str(r.get("callsign", "")).strip(),
            service_class=normalize_class(r["service_class"]),
            raw_class=str(r["service_class"]).strip(),
            detectability=score,
            distance_km=dist,
        ))
    return out


def load_lines(path: Path) -> list[Line]:
    return [Line(rf_channel=int(r["rf_channel"]),
                 offset_hz=float(r["offset_hz"]),
                 snr_db=float(r["snr_db"]))
            for r in _read_rows(path)]


# ------------------------------------------------------------- association

def associate(lines: list[Line], census: list[Transmitter], *,
              strategy: str = "ranked") -> list[dict]:
    """Per-line association records. See module docstring for the rule."""
    if strategy not in ("ranked", "dominant_secondary"):
        raise SystemExit(f"unknown association strategy: {strategy!r}")
    records: list[dict] = []
    channels = sorted({l.rf_channel for l in lines})
    for ch in channels:
        ch_lines = sorted((l for l in lines if l.rf_channel == ch),
                          key=lambda l: -l.snr_db)
        ch_census = sorted((t for t in census if t.rf_channel == ch),
                           key=lambda t: (-t.detectability, t.distance_km))
        for rank, line in enumerate(ch_lines):
            rec = {"rf_channel": ch, "offset_hz": line.offset_hz,
                   "snr_db": line.snr_db, "line_rank": rank,
                   "callsign": "", "tier": "unassociated",
                   "service_class": "unassociated"}
            if strategy == "ranked" and rank < len(ch_census):
                t = ch_census[rank]
                rec.update(callsign=t.callsign, service_class=t.service_class,
                           tier=("dominant" if rank == 0 and
                                 t.service_class == "primary" else "ranked"))
            elif strategy == "dominant_secondary" and ch_census:
                if rank == 0:
                    t = ch_census[0]
                    rec.update(callsign=t.callsign,
                               service_class=t.service_class, tier="dominant")
                else:
                    rec.update(service_class="non_primary", tier="pooled")
            records.append(rec)
    return records


# --------------------------------------------------------------- statistics

def mad_hz(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def rms_hz(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, average ranks for ties (no scipy)."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=np.float64)
        ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
        # average ties
        for val in np.unique(a):
            sel = a == val
            if sel.sum() > 1:
                ranks[sel] = ranks[sel].mean()
        return ranks
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def bootstrap_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = spearman(x[idx], y[idx])
        if np.isfinite(r):
            stats.append(r)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def per_channel_stats(records: list[dict]) -> list[dict]:
    out = []
    for ch in sorted({r["rf_channel"] for r in records}):
        rs = [r for r in records if r["rf_channel"] == ch]
        offsets = np.asarray([r["offset_hz"] for r in rs], np.float64)
        assoc = [r for r in rs if r["service_class"] != "unassociated"]
        n_np = sum(1 for r in assoc if r["service_class"] == "non_primary")
        out.append({
            "rf_channel": ch,
            "n_lines": len(rs),
            "n_unassociated": sum(1 for r in rs
                                  if r["service_class"] == "unassociated"),
            "mad_hz": mad_hz(offsets),
            "rms_hz": rms_hz(offsets),
            "span_hz": float(offsets.max() - offsets.min()) if offsets.size else float("nan"),
            "non_primary_fraction": (n_np / len(assoc)) if assoc else float("nan"),
        })
    return out


def threshold_sweep(records: list[dict], thresholds: np.ndarray) -> list[dict]:
    rows = []
    for t in thresholds:
        for ch in sorted({r["rf_channel"] for r in records}):
            offs = np.asarray([r["offset_hz"] for r in records
                               if r["rf_channel"] == ch and r["snr_db"] >= t],
                              np.float64)
            rows.append({"snr_threshold_db": float(t), "rf_channel": ch,
                         "n_lines": int(offs.size), "mad_hz": mad_hz(offs)})
    return rows


# ------------------------------------------------------------------ figures

def _figures(records, chan_stats, rho, ci, out_dir: Path) -> None:
    from pilot_proxy.plot_style import setup_matplotlib
    setup_matplotlib()
    import matplotlib.pyplot as plt

    # Figure A: class-split offset distribution (per detected line)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    groups = [("primary", "C0"), ("non_primary", "C1"), ("unassociated", "C7")]
    rng = np.random.default_rng(1)
    for i, (cls, color) in enumerate(groups):
        offs = np.asarray([r["offset_hz"] for r in records
                           if r["service_class"] == cls], np.float64)
        if offs.size == 0:
            continue
        x = i + rng.uniform(-0.14, 0.14, size=offs.size)
        ax.plot(x, offs, ".", ms=4, alpha=0.6, color=color)
        q1, q2, q3 = np.percentile(offs, [25, 50, 75])
        ax.hlines([q1, q2, q3], i - 0.22, i + 0.22, color=color,
                  lw=[1.0, 1.8, 1.0])
        ax.text(i, ax.get_ylim()[0], "", ha="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0].replace("_", " ") for g in groups])
    ax.set_ylabel("pilot offset from nominal (Hz)")
    ax.set_title("Carrier offset by service class (per detected line)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_offset_by_class.{ext}", dpi=200)
    plt.close(fig)

    # Figure B: per-channel spread vs composition
    xs = np.asarray([c["non_primary_fraction"] for c in chan_stats], np.float64)
    ys = np.asarray([c["mad_hz"] for c in chan_stats], np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(xs[ok], ys[ok], "o", ms=5)
    for c in chan_stats:
        if np.isfinite(c["non_primary_fraction"]) and np.isfinite(c["mad_hz"]):
            ax.annotate(str(c["rf_channel"]),
                        (c["non_primary_fraction"], c["mad_hz"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    label = (f"Spearman $\\rho$ = {rho:.2f} "
             f"[{ci[0]:.2f}, {ci[1]:.2f}], N = {int(ok.sum())}")
    ax.set_xlabel("non-primary fraction among associated lines")
    ax.set_ylabel("per-channel offset MAD (Hz)")
    ax.set_title(label)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_spread_vs_composition.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze-transmitter-census",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--census", type=Path, required=True,
                   help="Census CSV (schema in the module docstring).")
    p.add_argument("--lines", type=Path, required=True,
                   help="Detected-line CSV from the pilotcal high-res spectra.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--association", choices=["ranked", "dominant_secondary"],
                   default="ranked")
    p.add_argument("--snr-threshold-db", type=float, default=None,
                   help="Drop lines below this SNR before analysis "
                        "(default: keep all; sweep is reported regardless).")
    p.add_argument("--sweep-start-db", type=float, default=0.0)
    p.add_argument("--sweep-stop-db", type=float, default=20.0)
    p.add_argument("--sweep-step-db", type=float, default=2.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    census = load_census(args.census)
    lines = load_lines(args.lines)
    if args.snr_threshold_db is not None:
        lines = [l for l in lines if l.snr_db >= args.snr_threshold_db]
    if not lines:
        raise SystemExit("no lines above threshold; nothing to analyze")

    records = associate(lines, census, strategy=args.association)
    chan_stats = per_channel_stats(records)

    xs = np.asarray([c["non_primary_fraction"] for c in chan_stats], np.float64)
    ys = np.asarray([c["mad_hz"] for c in chan_stats], np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    rho = spearman(xs[ok], ys[ok])
    ci = bootstrap_ci(xs[ok], ys[ok], n_boot=args.bootstrap, seed=args.seed)

    by_class = {}
    for cls in ("primary", "non_primary", "unassociated"):
        offs = np.asarray([r["offset_hz"] for r in records
                           if r["service_class"] == cls], np.float64)
        by_class[cls] = {"n": int(offs.size), "mad_hz": mad_hz(offs),
                         "rms_hz": rms_hz(offs),
                         "span_hz": (float(offs.max() - offs.min())
                                     if offs.size else float("nan"))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "association.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    sweep = threshold_sweep(records, np.arange(args.sweep_start_db,
                                               args.sweep_stop_db + 1e-9,
                                               args.sweep_step_db))
    with open(args.output_dir / "threshold_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)
    summary = {"schema_version": "transmitter_census_v1",
               "association_strategy": args.association,
               "snr_threshold_db": args.snr_threshold_db,
               "by_class": by_class, "per_channel": chan_stats,
               "spearman_rho": rho,
               "spearman_ci95": list(ci),
               "n_channels": int(ok.sum())}
    with open(args.output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    _figures(records, chan_stats, rho, ci, args.output_dir)

    print(f"lines: {len(records)}  channels: {len(chan_stats)}  "
          f"primary MAD {by_class['primary']['mad_hz']:.1f} Hz vs "
          f"non-primary MAD {by_class['non_primary']['mad_hz']:.1f} Hz  "
          f"| rho={rho:.2f} CI95=[{ci[0]:.2f},{ci[1]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
