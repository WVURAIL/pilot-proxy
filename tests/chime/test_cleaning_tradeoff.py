# coding=utf-8
"""Tests for the post-hoc cleaning-tradeoff sweep."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime.cleaning_tradeoff import (
    CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ,
    OPERATING_CURVE_FIGURE,
    RECOVERED_BANDWIDTH_FIGURE,
    TRADEOFF_CSV_FILENAME,
    TRADEOFF_SUMMARY_FILENAME,
    control_floor_db,
    sweep_cleaning_tradeoff,
    write_outputs,
)
from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)
from pilot_proxy.detector_contract import norm_corrected_positive_excess

# Two channels with deliberately unequal norms (mu0 = 10/9 and 10/11), six
# frames each, p_ref fixed at 90 so the exact rule is p_target*nrs > nt*90.
TARGET_NORM_SQ = np.asarray([5, 5], dtype=np.int64)
REF_NORM_SUM_SQ = np.asarray([9, 11], dtype=np.int64)
P_REF = 90
# Channel 0 threshold: p_t > 50; channel 1: p_t > 450/11 ~ 40.9.
P_TARGET = np.asarray(
    [[45, 30], [51, 41], [60, 45], [80, 60], [120, 90], [49, 40]],
    dtype=np.uint64,
)
POWER = np.asarray(
    [[10.0, 10.0], [40.0, 40.0], [40.0, 40.0], [40.0, 40.0], [40.0, 40.0],
     [10.0, 10.0]]
)


def _write_run(run_dir, *, corrupt_stored_mask=False, omit_norms=False) -> None:
    run_dir.mkdir(parents=True)
    n_frames, n_pilots = P_TARGET.shape
    valid = np.ones((n_frames, n_pilots), dtype=np.uint8)
    p_ref = np.full((n_frames, n_pilots), P_REF, dtype=np.uint64)
    mask = np.zeros((n_frames, n_pilots), dtype=np.uint8)
    for f in range(n_frames):
        for c in range(n_pilots):
            mask[f, c] = norm_corrected_positive_excess(
                int(P_TARGET[f, c]), P_REF,
                target_norm_sq=int(TARGET_NORM_SQ[c]),
                ref_norm_sum_sq=int(REF_NORM_SUM_SQ[c]),
            )
    if corrupt_stored_mask:
        mask[0, 0] ^= 1
    detector = {
        "physical_channel": np.asarray([14, 15], dtype=np.int32),
        "p_target_u64": P_TARGET,
        "p_ref_sum_u64": p_ref,
        "valid": valid,
        "mask": mask,
        "target_norm_sq": TARGET_NORM_SQ,
        "ref_norm_sum_sq": REF_NORM_SUM_SQ,
        "mu0": 2.0 * TARGET_NORM_SQ / REF_NORM_SUM_SQ,
    }
    if omit_norms:
        for key in ("target_norm_sq", "ref_norm_sum_sq", "mu0"):
            detector.pop(key)
    np.savez_compressed(run_dir / CHIME_DETECTOR_OUTPUTS_FILENAME, **detector)
    np.savez_compressed(
        run_dir / CHIME_SPECTROGRAM_CACHE_FILENAME,
        baseband_power_linear=POWER, valid=valid, mask=mask,
    )


def _write_control(run_dir, mean_power: float) -> None:
    run_dir.mkdir(parents=True)
    valid = np.ones((4, 1), dtype=np.uint8)
    detector = {
        "physical_channel": np.asarray([99], dtype=np.int32),
        "p_target_u64": np.full((4, 1), 40, dtype=np.uint64),
        "p_ref_sum_u64": np.full((4, 1), 90, dtype=np.uint64),
        "valid": valid,
        "mask": np.zeros((4, 1), dtype=np.uint8),
        "target_norm_sq": np.asarray([5], dtype=np.int64),
        "ref_norm_sum_sq": np.asarray([9], dtype=np.int64),
        "mu0": np.asarray([10.0 / 9.0]),
    }
    np.savez_compressed(run_dir / CHIME_DETECTOR_OUTPUTS_FILENAME, **detector)
    np.savez_compressed(
        run_dir / CHIME_SPECTROGRAM_CACHE_FILENAME,
        baseband_power_linear=np.full((4, 1), mean_power),
        valid=valid, mask=np.zeros((4, 1), dtype=np.uint8),
    )


def test_sweep_anchors_on_stored_mask_and_counts_exactly(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run)

    report = sweep_cleaning_tradeoff(run, excess_db_grid=[0.0, 3.0, 6.0])

    by_key = {
        (row["physical_channel"], row["excess_db"]): row for row in report["rows"]
    }
    # Hand counts at x = 0: ch14 masks p_t in {51,60,80,120} -> 4/6;
    # ch15 masks {41,45,60,90} -> 4/6.
    assert by_key[(14, 0.0)]["masked_fraction"] == pytest.approx(4 / 6)
    assert by_key[(15, 0.0)]["masked_fraction"] == pytest.approx(4 / 6)
    # Kept frames all have power 10, masked all 40.
    assert by_key[(14, 0.0)]["cleaned_power_db"] == pytest.approx(10.0)
    assert by_key[(14, 0.0)]["input_power_db"] == pytest.approx(
        10.0 * np.log10(POWER[:, 0].mean())
    )
    # x = 3 dB: ch14 threshold p_t > 50*10^0.3 ~ 99.8 -> only 120 masked.
    assert by_key[(14, 3.0)]["masked_fraction"] == pytest.approx(1 / 6)
    # Kept fraction is nondecreasing with threshold, per channel.
    for channel in (14, 15):
        kept = [by_key[(channel, x)]["kept_fraction"] for x in (0.0, 3.0, 6.0)]
        assert kept == sorted(kept)
    # Recovered bandwidth at the operating point.
    expected = (2 / 6 + 2 / 6) * CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ
    assert report["operating_point"]["recovered_mhz"] == pytest.approx(expected)
    assert report["recovered_mhz_by_excess_db"][6.0] >= expected


def test_sweep_refuses_when_anchor_mismatches(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run, corrupt_stored_mask=True)
    with pytest.raises(SystemExit, match="x=0 recompute disagrees"):
        sweep_cleaning_tradeoff(run, excess_db_grid=[0.0])


def test_sweep_refuses_legacy_products_without_norms(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run, omit_norms=True)
    with pytest.raises(SystemExit, match="norm-corrected products"):
        sweep_cleaning_tradeoff(run, excess_db_grid=[0.0])


def test_residual_uses_control_floor_and_hours_scale(tmp_path) -> None:
    run = tmp_path / "run"
    control = tmp_path / "control"
    _write_run(run)
    _write_control(control, mean_power=10.0)  # floor = 10 dB

    assert control_floor_db(control) == pytest.approx(10.0)
    report = sweep_cleaning_tradeoff(
        run, excess_db_grid=[0.0], control_run_dir=control, survey_hours=100.0
    )
    row = next(r for r in report["rows"] if r["physical_channel"] == 14)
    assert row["residual_db"] == pytest.approx(row["cleaned_power_db"] - 10.0)
    operating = report["operating_point"]
    assert operating["recovered_mhz_hours"] == pytest.approx(
        operating["recovered_mhz"] * 100.0
    )


def test_write_outputs_produces_csv_json_and_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run)
    report = sweep_cleaning_tradeoff(run, excess_db_grid=[0.0, 6.0])

    written = write_outputs(report, tmp_path / "out")

    names = {path.name for path in written}
    assert names == {
        TRADEOFF_CSV_FILENAME, TRADEOFF_SUMMARY_FILENAME,
        OPERATING_CURVE_FIGURE, RECOVERED_BANDWIDTH_FIGURE,
    }
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    summary = json.loads((tmp_path / "out" / TRADEOFF_SUMMARY_FILENAME).read_text())
    assert summary["schema_version"] == "pilot_proxy_cleaning_tradeoff_v1"
