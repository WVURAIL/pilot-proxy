# coding=utf-8
from __future__ import annotations

import csv

import numpy as np
import pytest

from pilot_proxy.chime.choose_detector_k import (
    capture_loss_allowed_bin_fraction,
    choose_detector_k,
)


def test_capture_loss_allowed_bin_fraction_matches_one_db_rule() -> None:
    assert capture_loss_allowed_bin_fraction(1.0) == pytest.approx(0.2615, abs=5e-4)


def test_choose_detector_k_writes_candidate_summary(tmp_path) -> None:
    frequency_offset = tmp_path / "frequency_offset_outputs.npz"
    output = tmp_path / "tables" / "k_candidate_summary.csv"
    physical_channel = np.asarray([17, 30, 35], dtype=np.int32)
    # Channel 17 is reliable and passes K=256 but fails K=512 after recentering.
    # Channel 30 is reliable and tight after median recentering.
    # Channel 35 has large offsets but is below the prominence reliability cut.
    offsets = np.asarray(
        [
            [-250.0, 80.0, -3000.0],
            [0.0, 95.0, 2000.0],
            [0.0, 100.0, 500.0],
            [0.0, 105.0, -1000.0],
            [250.0, 120.0, 4000.0],
        ],
        dtype=np.float64,
    )
    prominence = np.asarray(
        [
            [30.0, 40.0, 8.0],
            [31.0, 41.0, 9.0],
            [32.0, 42.0, 10.0],
            [31.0, 41.0, 9.0],
            [30.0, 40.0, 8.0],
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        frequency_offset,
        physical_channel=physical_channel,
        frequency_offset_hz=offsets,
        peak_prominence_db=prominence,
        valid=np.ones_like(offsets, dtype=np.uint8),
        sample_rate_hz=np.asarray(390_625.0),
    )

    result = choose_detector_k(
        frequency_offset=frequency_offset,
        output=output,
        candidate_k=[64, 128, 256, 512],
        candidate_reference_spacing_policy=[
            "fixed_skipped_guard",
            "fixed_hz_reference_spacing",
        ],
        min_peak_prominence_db=25.0,
        max_capture_loss_db=1.0,
    )

    assert result == output
    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8

    k256_fixed_hz = [
        row
        for row in rows
        if row["K"] == "256"
        and row["reference_spacing_policy"] == "fixed_hz_reference_spacing"
    ][0]
    assert k256_fixed_hz["reference_offset_bins"] == "4"
    assert k256_fixed_hz["skipped_guard_bins"] == "3"
    assert k256_fixed_hz["reference_offset_bins_computed"] == "4"
    assert k256_fixed_hz["reference_offset_bins_effective"] == "4"
    assert k256_fixed_hz["reference_spacing_policy_note"] == "none"
    assert k256_fixed_hz["num_reliable_channels"] == "2"
    assert k256_fixed_hz["num_passing_channels"] == "2"
    assert k256_fixed_hz["passing_physical_channels"] == "17;30"
    assert k256_fixed_hz["failing_physical_channels"] == ""
    assert k256_fixed_hz["reason"] == "passes_reliable_channels_candidate_for_canfar"

    k512_fixed_hz = [
        row
        for row in rows
        if row["K"] == "512"
        and row["reference_spacing_policy"] == "fixed_hz_reference_spacing"
    ][0]
    assert k512_fixed_hz["reference_offset_bins"] == "8"
    assert k512_fixed_hz["num_passing_channels"] == "1"
    assert k512_fixed_hz["passing_physical_channels"] == "30"
    assert k512_fixed_hz["failing_physical_channels"] == "17"
    assert k512_fixed_hz["reason"] == "fails_reliable_channel_residual_criterion"

    baseline = [
        row
        for row in rows
        if row["K"] == "128"
        and row["reference_spacing_policy"] == "fixed_hz_reference_spacing"
    ][0]
    assert baseline["recommended"] == "True"
    assert baseline["reason"] == "validated_baseline"

    k64_fixed_hz = [
        row
        for row in rows
        if row["K"] == "64"
        and row["reference_spacing_policy"] == "fixed_hz_reference_spacing"
    ][0]
    assert k64_fixed_hz["reference_offset_bins"] == "2"
    assert k64_fixed_hz["reference_offset_bins_computed"] == "1"
    assert k64_fixed_hz["reference_offset_bins_effective"] == "2"
    assert (
        k64_fixed_hz["reference_spacing_policy_note"]
        == "clamped_to_min_reference_offset_bins"
    )


def test_choose_detector_k_reports_reference_placement_summary(tmp_path) -> None:
    frequency_offset = tmp_path / "frequency_offset_outputs.npz"
    output = tmp_path / "tables" / "k_candidate_summary.csv"
    physical_channel = np.asarray([14, 21, 30], dtype=np.int32)
    offsets = np.zeros((4, 3), dtype=np.float64)
    prominence = np.full((4, 3), 35.0, dtype=np.float64)
    np.savez_compressed(
        frequency_offset,
        physical_channel=physical_channel,
        frequency_offset_hz=offsets,
        peak_prominence_db=prominence,
        valid=np.ones_like(offsets, dtype=np.uint8),
        sample_rate_hz=np.asarray(390_625.0),
    )

    choose_detector_k(
        frequency_offset=frequency_offset,
        output=output,
        candidate_k=[128, 256],
        candidate_reference_spacing_policy=["fixed_skipped_guard"],
        min_peak_prominence_db=25.0,
        max_capture_loss_db=1.0,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    k128 = [row for row in rows if row["K"] == "128"][0]
    assert k128["reference_placement_status"] == "mixed:edge_wrapped;nominal"
    assert k128["num_channels_with_adaptive_reference"] == "1"
    assert k128["channels_with_adaptive_reference"] == "21"
    assert k128["num_edge_wrapped_references"] == "1"
    assert k128["channels_with_edge_wrapped_reference"] == "21"
    assert k128["num_forbidden_tone_in_skipped_guard"] == "1"
    assert k128["channels_with_forbidden_tone_in_skipped_guard"] == "14"
    assert "DTV 21: upper reference wrapped across coarse-channel edge" in k128[
        "placement_warnings"
    ]

    k256 = [row for row in rows if row["K"] == "256"][0]
    assert k256["reference_placement_status"] == "mixed:dc_shifted;nominal"
    assert k256["num_channels_with_adaptive_reference"] == "1"
    assert k256["channels_with_adaptive_reference"] == "14"
    assert k256["num_dc_shifted_references"] == "1"
    assert k256["channels_with_dc_shifted_reference"] == "14"
