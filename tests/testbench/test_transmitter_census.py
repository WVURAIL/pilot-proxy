# coding=utf-8
"""Transmitter-census case study: association, dispersion, correlation.

Synthetic fixture encodes the physical expectation the analysis is built to
test: primaries hold tight offsets (disciplined exciters), non-primaries
scatter wide, and channels with more translators show larger spread.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.testbench.transmitter_census import (
    associate,
    bootstrap_ci,
    load_census,
    load_lines,
    mad_hz,
    main,
    normalize_class,
    per_channel_stats,
    spearman,
)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _fixture(tmp_path, rng=None):
    """14 channels; channel c has c-13 translators; primary tight, tx wide."""
    rng = rng or np.random.default_rng(7)
    census, lines = [], []
    for ch in range(14, 28):
        census.append({"rf_channel": ch, "callsign": f"K{ch}AA",
                       "service_class": "full_service", "erp_kw": 1000.0,
                       "distance_km": 60.0})
        lines.append({"rf_channel": ch,
                      "offset_hz": float(rng.normal(0, 8)),
                      "snr_db": 30.0})
        for k in range(ch - 14):  # ch14 has none: anchors the low end
            census.append({"rf_channel": ch, "callsign": f"K{ch}T{k}",
                           "service_class": "translator", "erp_kw": 1.0,
                           "distance_km": 120.0 + 10 * k})
            lines.append({"rf_channel": ch,
                          "offset_hz": float(rng.normal(0, 150.0 * (1 + k))),
                          "snr_db": 12.0 - k})
    return (_write_csv(tmp_path / "census.csv", census),
            _write_csv(tmp_path / "lines.csv", lines))


def test_class_normalization_aliases():
    assert normalize_class("Full Service") == "primary"
    assert normalize_class("TRANSLATOR") == "non_primary"
    assert normalize_class("LPTV") == "non_primary"
    assert normalize_class("experimental") == "other"
    # real-world census strings (FCC/ISED exports)
    assert normalize_class("Translator (LPTV)") == "non_primary"
    assert normalize_class("Low-power (LPTV)") == "non_primary"
    assert normalize_class("Relay") == "non_primary"
    assert normalize_class("Class A") == "non_primary"
    assert normalize_class("Full-power") == "primary"


def test_scored_census_ranks_by_detectability_then_distance(tmp_path):
    path = _write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "A", "service_class": "Full-power",
         "detectability_db": "84.1", "distance_km": "60"},
        {"rf_channel": 20, "callsign": "B", "service_class": "Translator (LPTV)",
         "detectability_db": "", "distance_km": "90"},
        {"rf_channel": 20, "callsign": "C", "service_class": "Relay",
         "detectability_db": "", "distance_km": "40"},
    ])
    census = load_census(path)
    order = [t.callsign for t in
             sorted(census, key=lambda t: (-t.detectability, t.distance_km))]
    assert order == ["A", "C", "B"]  # scored first; unscored by distance


def test_association_tiers_and_unassociated(tmp_path):
    census = load_census(_write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "KPRI", "service_class": "primary",
         "erp_kw": 1000.0, "distance_km": 50.0},
        {"rf_channel": 20, "callsign": "KTRX", "service_class": "translator",
         "erp_kw": 0.5, "distance_km": 150.0},
    ]))
    lines = load_lines(_write_csv(tmp_path / "l.csv", [
        {"rf_channel": 20, "offset_hz": 3.0, "snr_db": 30.0},
        {"rf_channel": 20, "offset_hz": -450.0, "snr_db": 12.0},
        {"rf_channel": 20, "offset_hz": 900.0, "snr_db": 8.0},
    ]))
    recs = associate(lines, census, strategy="ranked")
    by_rank = {r["line_rank"]: r for r in recs}
    assert by_rank[0]["tier"] == "dominant" and by_rank[0]["callsign"] == "KPRI"
    assert by_rank[1]["service_class"] == "non_primary"
    assert by_rank[2]["tier"] == "unassociated"  # more lines than census: a finding


def test_dominant_secondary_pools_the_rest(tmp_path):
    census = load_census(_write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "KPRI", "service_class": "primary",
         "erp_kw": 1000.0, "distance_km": 50.0}]))
    lines = load_lines(_write_csv(tmp_path / "l.csv", [
        {"rf_channel": 20, "offset_hz": 1.0, "snr_db": 30.0},
        {"rf_channel": 20, "offset_hz": 600.0, "snr_db": 9.0}]))
    recs = associate(lines, census, strategy="dominant_secondary")
    assert recs[0]["tier"] == "dominant"
    assert recs[1]["tier"] == "pooled"
    assert recs[1]["service_class"] == "non_primary"


def test_spearman_and_bootstrap_recover_monotone_relation():
    x = np.arange(20, dtype=float)
    y = x ** 2 + 0.1
    rho = spearman(x, y)
    assert rho == pytest.approx(1.0)
    lo, hi = bootstrap_ci(x, y, n_boot=300, seed=1)
    assert lo > 0.8 and hi <= 1.0


def test_mad_is_robust_to_one_outlier():
    tight = np.asarray([0.0, 1.0, -1.0, 2.0, -2.0, 1000.0])
    assert mad_hz(tight) < 10.0


def test_end_to_end_recovers_class_and_correlation(tmp_path):
    census_path, lines_path = _fixture(tmp_path)
    out = tmp_path / "out"
    rc = main(["--census", str(census_path), "--lines", str(lines_path),
               "--output-dir", str(out), "--bootstrap", "300"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    by = summary["by_class"]
    # the constructed physics: primaries tight, translators wide
    assert by["primary"]["mad_hz"] < 40.0 < by["non_primary"]["mad_hz"]
    # more translators -> larger spread, by construction
    assert summary["spearman_rho"] > 0.6
    assert summary["spearman_ci95"][0] > 0.2
    for name in ("association.csv", "threshold_stability.csv",
                 "fig_offset_by_class.png", "fig_offset_by_class.pdf",
                 "fig_spread_vs_composition.png"):
        assert (out / name).exists(), name
    # per-channel table covers all channels with finite spread
    chans = per_channel_stats(
        associate(load_lines(lines_path), load_census(census_path)))
    assert len(chans) == 14
    assert all(np.isfinite(c["mad_hz"]) for c in chans if c["n_lines"] > 1)


def test_snr_threshold_filters_before_analysis(tmp_path):
    census_path, lines_path = _fixture(tmp_path)
    out = tmp_path / "out_thresh"
    main(["--census", str(census_path), "--lines", str(lines_path),
          "--output-dir", str(out), "--snr-threshold-db", "25",
          "--bootstrap", "50"])
    summary = json.loads((out / "summary.json").read_text())
    # only the 30 dB primaries survive a 25 dB cut
    assert summary["by_class"]["non_primary"]["n"] == 0
    assert summary["by_class"]["primary"]["n"] == 14
