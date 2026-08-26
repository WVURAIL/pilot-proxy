# coding=utf-8
"""Unit tests for the calibration suite in ``analysis/ppcal``.

These exercise the estimators and the segmentation against synthetic inputs
with known answers, so they run without any survey product. The tests that
need a product are skipped unless ``PP_PER_PILOT`` points at a directory
holding some.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
sys.path.insert(0, str(ANALYSIS))

from ppcal.calib import (  # noqa: E402
    NULL_SCALE_PROBES, half_sample_mode, null_scale_about)
from ppcal.eras import Era, _mann_whitney_z, segment  # noqa: E402
from ppcal.era_view import quiet_era_floor_db  # noqa: E402
from ppcal import state as calibration_state  # noqa: E402
from ppcal.products import (  # noqa: E402
    COARSE_HZ, FINE_HZ, month_index, month_label)
from ppcal.spectra import wide_pair_is_era_resolved  # noqa: E402


def _era(i, start, end):
    return Era(index=i, month_start=start, month_end=end, n_units=10,
               level_median_db=0.0, level_p90_db=1.0)


# --------------------------------------------------------------------------
# half-sample mode
# --------------------------------------------------------------------------

def test_half_sample_mode_recovers_a_narrow_core_under_heavy_contamination():
    """The mode must ignore a right tail that outweighs the core in mass."""
    rng = np.random.default_rng(11)
    core = rng.normal(1.0, 0.003, 4000)
    tail = rng.uniform(1.5, 400.0, 6000)          # 60% of the sample
    mu = half_sample_mode(np.concatenate([core, tail]))
    assert abs(mu - 1.0) < 0.002


def test_half_sample_mode_follows_the_dominant_population():
    """Where a carrier lobe is the densest population, the mode lands on it.

    This is the property the disposition logic relies on: a channel with no
    recoverable null reports a mu far from the analytic constant rather than
    a spuriously clean one.
    """
    rng = np.random.default_rng(12)
    null = rng.normal(1.0, 0.003, 500)
    lobe = rng.normal(40.0, 2.0, 9500)
    mu = half_sample_mode(np.concatenate([null, lobe]))
    assert 38.0 < mu < 42.0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_half_sample_mode_handles_tiny_samples(n):
    assert np.isfinite(half_sample_mode(np.arange(n, dtype=float)))


def test_half_sample_mode_is_empty_safe():
    assert np.isnan(half_sample_mode(np.array([])))


# --------------------------------------------------------------------------
# null scale
# --------------------------------------------------------------------------

def test_null_scale_recovers_a_known_gaussian_width():
    rng = np.random.default_rng(13)
    sigma = 0.004
    f = rng.normal(1.0, sigma, 200000)
    est, spread = null_scale_about(f, 1.0)
    assert est == pytest.approx(sigma, rel=0.05)
    assert spread < 1.15                       # probes agree on a clean null


def test_null_scale_is_blind_to_the_right_tail():
    """A carrier can only add power, so the left side must not move."""
    rng = np.random.default_rng(14)
    f = rng.normal(1.0, 0.004, 100000)
    contaminated = np.concatenate([f, rng.uniform(2.0, 500.0, 60000)])
    a, _ = null_scale_about(f, 1.0)
    b, _ = null_scale_about(contaminated, 1.0)
    assert b == pytest.approx(a, rel=0.02)


def test_null_scale_refuses_a_starved_sample():
    est, spread = null_scale_about(np.array([1.0, 0.99, 0.98]), 1.0)
    assert np.isnan(est) and np.isnan(spread)


def test_null_scale_probes_match_the_released_convention():
    """Guard against the probe table drifting from RFIsher's."""
    assert NULL_SCALE_PROBES == ((32.0, 1.0000), (5.0, 1.9600),
                                 (0.3, 2.9677))


# --------------------------------------------------------------------------
# segmentation statistic
# --------------------------------------------------------------------------

def test_mann_whitney_z_is_zero_for_identical_populations():
    a = np.arange(40, dtype=float)
    assert abs(_mann_whitney_z(a, a.copy())) < 1e-9


def test_mann_whitney_z_detects_a_clean_step():
    z = _mann_whitney_z(np.zeros(30), np.ones(30))
    assert abs(z) > 6.0


def test_mann_whitney_z_survives_ties():
    """All-tied input has zero variance and must not divide by zero."""
    a = np.ones(20)
    assert _mann_whitney_z(a, a.copy()) == 0.0


def test_mann_whitney_z_handles_an_empty_side():
    assert _mann_whitney_z(np.array([]), np.ones(5)) == 0.0


class _SegmentChannel:
    def __init__(self, levels, spacing_days=31.0, within_month_days=None):
        self._levels = np.asarray(levels, dtype=float)
        months = np.arange(self._levels.size)
        offsets = ((0.0,) if within_month_days is None
                   else tuple(within_month_days))
        self.unit_month = np.repeat(months, len(offsets))
        self.unit_level_db = np.repeat(self._levels, len(offsets))
        times = np.concatenate([
            (month * spacing_days + np.asarray(offsets)) * 86400.0
            for month in months
        ])
        self.units = (times, np.ones(times.size), np.ones(times.size),
                      np.ones(times.size, dtype=int))

    def monthly_level_db(self):
        count = np.ones(self._levels.size, dtype=int)
        return np.arange(self._levels.size), self._levels.copy(), count


def test_segment_accepts_a_supported_station_step():
    channel = _SegmentChannel([0.0] * 10 + [3.0] * 10)
    eras = segment(channel)
    assert [(era.month_start, era.month_end) for era in eras] == [(0, 9),
                                                                  (10, 19)]


def test_segment_uses_observed_time_span_not_month_approximation():
    channel = _SegmentChannel([0.0] * 10 + [3.0] * 10, spacing_days=20.0)
    assert len(segment(channel)) == 1


def test_segment_includes_first_and_last_acquisition_within_each_month():
    channel = _SegmentChannel(
        [0.0] * 10 + [3.0] * 10,
        spacing_days=29.0,
        within_month_days=(0.0, 10.0),
    )
    assert len(segment(channel)) == 2


def test_segment_accepts_the_exact_elapsed_span_boundary():
    channel = _SegmentChannel([0.0] * 10 + [3.0] * 10, spacing_days=30.0)
    assert len(segment(channel, min_days=270.0)) == 2
    assert len(segment(channel, min_days=270.01)) == 1


def test_segment_step_margin_is_explicit():
    channel = _SegmentChannel([0.0] * 10 + [1.9] * 10)
    assert len(segment(channel)) == 1
    assert len(segment(channel, min_step_db=1.5)) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_months", 0),
        ("min_days", -1.0),
        ("min_step_db", float("nan")),
        ("z_crit", -1.0),
        ("max_eras", 0),
    ],
)
def test_segment_validates_policy_parameters(field, value):
    channel = _SegmentChannel([0.0] * 10 + [3.0] * 10)
    with pytest.raises(ValueError, match=field):
        segment(channel, **{field: value})


def test_quiet_floor_validates_its_explicit_parameters():
    with pytest.raises(ValueError, match="minimum_frames"):
        quiet_era_floor_db(object(), [], minimum_frames=0)
    with pytest.raises(ValueError, match="percentile"):
        quiet_era_floor_db(object(), [], percentile=101.0)
    with pytest.raises(TypeError, match="integer"):
        quiet_era_floor_db(object(), [], minimum_frames=1.5)


def test_threshold_table_rejects_an_invalid_supplied_eta(tmp_path):
    path = tmp_path / "thresholds.csv"
    path.write_text("ch,eta_cost_cap\n14,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="channel 14.*eta_cost_cap"):
        calibration_state._read_thresholds(path)


def test_threshold_table_reads_positive_supplied_etas(tmp_path):
    path = tmp_path / "thresholds.csv"
    path.write_text(
        "ch,eta_cost_cap,eta_cost_thermal,tau_measured\n"
        "14,1.25,1.1,True\n",
        encoding="utf-8",
    )
    rows = calibration_state._read_thresholds(path)
    assert rows[14]["eta_cost_cap"] == pytest.approx(1.25)
    assert rows[14]["eta_cost_thermal"] == pytest.approx(1.1)
    assert rows[14]["tau_measured"] is True


def test_threshold_build_aliases_are_mutually_exclusive():
    with pytest.raises(ValueError, match="legacy bao_csv alias"):
        calibration_state.build(
            "unused", threshold_csv="thresholds.csv", bao_csv="legacy.csv")


# --------------------------------------------------------------------------
# geometry, the month grid, and the era container
# --------------------------------------------------------------------------

def test_detector_grid_constants():
    """The fine grid is the coarse bin split 256 ways; both are exact."""
    assert COARSE_HZ == pytest.approx(390625.0 / 128)
    assert FINE_HZ == pytest.approx(COARSE_HZ / 256)
    assert FINE_HZ == pytest.approx(11.9209, abs=1e-4)


def test_month_index_and_label_round_trip():
    for label in ("2018-12", "2021-01", "2024-06", "2026-07"):
        year, month = (int(x) for x in label.split("-"))
        ts = datetime.datetime(year, month, 15,
                               tzinfo=datetime.timezone.utc).timestamp()
        assert month_label(month_index(ts)[0]) == label


def test_month_grid_starts_at_the_survey_epoch():
    assert month_label(0) == "2018-12"


def test_era_label_spans_both_ends():
    assert _era(0, 0, 11).label == "2018-12..2019-11"


# --------------------------------------------------------------------------
# the wide-spectrum era caveat
# --------------------------------------------------------------------------

def test_wide_pair_is_era_resolved_only_for_a_single_era():
    """The archive-accumulated pair is a latest-era quantity only when the
    channel never transitioned."""
    assert wide_pair_is_era_resolved([_era(0, 0, 91)])
    assert not wide_pair_is_era_resolved([_era(0, 0, 40), _era(1, 41, 91)])


# --------------------------------------------------------------------------
# end-to-end, only when products are available
# --------------------------------------------------------------------------

def _product_dir():
    raw = os.environ.get("PP_PER_PILOT")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() and any(path.glob("*.npz")) else None


@pytest.mark.skipif(_product_dir() is None,
                    reason="set PP_PER_PILOT to a directory of products")
def test_a_real_channel_calibrates_and_segments():
    from ppcal import eras as E
    from ppcal.calib import calibrate
    from ppcal.products import Channel, product_paths

    path = product_paths(str(_product_dir()))[0]
    c = Channel(path)
    segs = E.segment(c)
    assert segs and segs[0].month_start <= segs[-1].month_end
    assert all(a.month_end < b.month_start
               for a, b in zip(segs, segs[1:]))      # eras are disjoint

    fmask = E.final_era_frame_mask(c, segs)
    assert fmask.any()
    cal = calibrate(c, fmask, segs[-1].label, int(fmask.sum()))
    assert cal.mu > 0
    assert 0.0 <= cal.occupancy_provisional <= 1.0
    # the ladder is monotone: a higher threshold can never mask more
    ladder = [cal.occupancy[k] for k in ("1", "1.2", "1.4", "2", "5")]
    assert all(a >= b for a, b in zip(ladder, ladder[1:]))


@pytest.mark.skipif(_product_dir() is None,
                    reason="set PP_PER_PILOT to a directory of products")
def test_masking_can_only_lower_the_mean_fine_spectrum():
    """Masking removes frames from a mean of non-negative powers, so the
    band-integrated suppression can never be negative."""
    from ppcal import eras as E
    from ppcal.calib import calibrate
    from ppcal.products import Channel, product_paths
    from ppcal.spectra import era_fine_spectrum_masked

    path = product_paths(str(_product_dir()))[0]
    c = Channel(path)
    segs = E.segment(c)
    fmask = E.final_era_frame_mask(c, segs)
    cal = calibrate(c, fmask, segs[-1].label, int(fmask.sum()))

    rf, before, after, stats = era_fine_spectrum_masked(c, fmask,
                                                        1.4 * cal.mu)
    assert rf.size == 256
    assert 0.0 <= stats["kept_fraction"] <= 1.0
    # the fine statistic is stored float32, so equality is only ever
    # exact to about 1e-4 dB after a 30k-frame accumulation
    assert stats["band_suppression_db"] >= -1e-4
    assert np.all(np.isfinite(before))
    # an unreachable threshold keeps every frame and removes nothing
    _, b2, a2, s2 = era_fine_spectrum_masked(c, fmask, np.inf)
    assert s2["kept_fraction"] == pytest.approx(1.0)
    assert s2["band_suppression_db"] == pytest.approx(0.0, abs=1e-4)
    # before and after reduce two different copies of the same float32
    # data (the block and its boolean-indexed copy), and numpy's pairwise
    # summation is layout sensitive, so they agree only to float32
    # precision -- measured at 3.3e-5 dB on this archive
    assert np.allclose(b2, a2, rtol=0, atol=1e-4)
