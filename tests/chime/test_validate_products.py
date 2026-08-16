# coding=utf-8
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_TOKEN,
    CHIME_STATS_SCHEMA_TOKEN,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_chime_detector_contract,
    normalized_positive_excess_policy,
)
from pilot_proxy.chime.validate_products import validate_products


def _write_products(
    run_dir,
    *,
    corrupt_cache_mask: bool = False,
    corrupt_normalized_positive_excess_mask: bool = False,
    corrupt_detector_window_metadata: bool = False,
    omit_norms: bool = False,
) -> None:
    run_dir.mkdir()
    physical_channel = np.asarray([14, 15], dtype=np.int32)
    pilot_frequency_hz = np.asarray([470_309_441.0, 476_309_441.0])
    chime_frequency_hz = np.asarray([470_312_500.0, 476_171_875.0])
    frame_index = np.asarray([0, 1, 2], dtype=np.int64)
    valid = np.asarray([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)
    target_norm_sq = np.asarray([5, 5], dtype=np.int64)
    reference_norm_sum_sq = np.asarray([9, 11], dtype=np.int64)
    null_power_ratio = (
        2.0 * target_norm_sq.astype(np.float64)
        / reference_norm_sum_sq.astype(np.float64)
    )
    p_ref = valid.astype(np.uint64) * np.uint64(20)
    p_target = np.asarray([[11, 10], [40, 0], [9, 50]], dtype=np.uint64)
    coarse_power_ratio = np.asarray(
        [[1.1, 1.0], [4.0, np.nan], [0.9, 5.0]],
        dtype=np.float64,
    )
    mask = np.asarray([[0, 1], [1, 0], [0, 1]], dtype=np.uint8)
    if corrupt_normalized_positive_excess_mask:
        mask = np.array(mask, copy=True)
        mask[0, 0] = 1

    normalized_ratio = coarse_power_ratio / null_power_ratio[np.newaxis, :]
    normalized_pilot_excess = normalized_ratio - 1.0
    pilot_excess_db = np.full(normalized_pilot_excess.shape, np.nan)
    positive = normalized_pilot_excess > 0.0
    pilot_excess_db[positive] = 10.0 * np.log10(
        normalized_pilot_excess[positive]
    )

    detector_payload = {
        "physical_channel": physical_channel,
        "pilot_frequency_hz": pilot_frequency_hz,
        "chime_frequency_hz": chime_frequency_hz,
        "frame_index": frame_index,
        "p_target_u64": p_target,
        "p_ref_sum_u64": p_ref,
        "coarse_power_ratio": coarse_power_ratio,
        "normalized_coarse_power_ratio_db": 10.0 * np.log10(normalized_ratio),
        "pilot_excess_db": pilot_excess_db,
        "estimated_data_shelf_snr_db": np.zeros((3, 2)),
        "mask": mask,
        "valid": valid,
    }
    if not omit_norms:
        detector_payload["target_norm_sq"] = target_norm_sq
        detector_payload["reference_norm_sum_sq"] = reference_norm_sum_sq
        detector_payload["null_power_ratio"] = null_power_ratio
        detector_payload["normalized_pilot_excess"] = normalized_pilot_excess
    np.savez_compressed(run_dir / "chime_detector_outputs.npz", **detector_payload)
    cache_mask = np.array(mask, copy=True)
    if corrupt_cache_mask:
        cache_mask[0, 0] = 1
    baseband = np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    np.savez_compressed(
        run_dir / "chime_spectrogram_cache.npz",
        baseband_power_linear=baseband,
        baseband_power_db=10.0 * np.log10(baseband),
        mask=cache_mask,
        valid=valid,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        relative_time_s=np.asarray([0.0, 0.04194304, 0.08388608]),
    )
    valid_flags = valid != 0
    mask_flags = mask != 0
    valid_count = np.sum(valid_flags, axis=0).reshape(1, -1)
    invalid_count = (valid.shape[0] - valid_count).astype(np.int64)
    masked_count_valid = np.sum(mask_flags & valid_flags, axis=0).reshape(1, -1)
    unmasked_count_valid = np.sum((~mask_flags) & valid_flags, axis=0).reshape(1, -1)
    mask_fraction_valid = masked_count_valid.astype(float) / valid_count
    mask_fraction_total = np.sum(mask_flags, axis=0).reshape(1, -1).astype(float) / float(
        valid.shape[0]
    )
    np.savez_compressed(
        run_dir / "chime_reductions_10s.npz",
        chunk_index=np.asarray([0], dtype=np.int64),
        chunk_start_frame=np.asarray([0], dtype=np.int64),
        chunk_stop_frame=np.asarray([3], dtype=np.int64),
        input_power_mean=np.asarray([[20.0, 21.0]]),
        cleaned_power_mean=np.asarray([[20.0, 11.0]]),
        valid_count=valid_count.astype(np.int64),
        invalid_count=invalid_count.astype(np.int64),
        masked_count_valid=masked_count_valid.astype(np.int64),
        unmasked_count_valid=unmasked_count_valid.astype(np.int64),
        mask_fraction_valid=mask_fraction_valid,
        mask_fraction_total=mask_fraction_total,
    )
    mask_policy = dict(normalized_positive_excess_policy())
    detector_contract = build_chime_detector_contract(
        detector_window_samples=128,
        skipped_guard_bins=1,
        reference_offset_bins=2,
        num_weight_terms=3,
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        time_reverse_detector_windows_before_kernel=True,
    )
    run_config: dict[str, object] = {
        "schema_version": CHIME_RUN_CONFIG_SCHEMA_TOKEN,
        "detector_contract": detector_contract,
        "detector_window_samples": 256 if corrupt_detector_window_metadata else 128,
    }
    run_config["mask_policy"] = mask_policy
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (run_dir / "input_manifest.json").write_text(
        json.dumps({"schema_version": "pilotproxy_chime_input_manifest_v1"}),
        encoding="utf-8",
    )
    (run_dir / "stats.json").write_text(
        json.dumps(
            {
                "schema_version": CHIME_STATS_SCHEMA_TOKEN,
                "num_frames": 3,
                "num_pilots": 2,
                "detector_window_samples": 128,
                "rational_overflow_count_by_pilot": [0, 0],
                "detector_contract": detector_contract,
                "mask_policy": mask_policy,
            }
        ),
        encoding="utf-8",
    )


def test_validate_products_accepts_consistent_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    report_path = tmp_path / "product_validation.json"
    _write_products(run_dir)

    report = validate_products(run_dir=run_dir, output_json=report_path)

    assert report["valid"] is True
    assert report["num_errors"] == 0
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["valid"] is True


def test_validate_products_reports_cross_product_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_products(run_dir, corrupt_cache_mask=True)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "spectrogram.mask" in checks


def test_validate_products_checks_normalized_positive_excess_mask_rule(tmp_path) -> None:
    run_dir = tmp_path / "positive"
    _write_products(run_dir)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is True


def test_validate_products_reports_bad_normalized_positive_excess_mask(tmp_path) -> None:
    run_dir = tmp_path / "bad_positive"
    _write_products(
        run_dir,
        corrupt_normalized_positive_excess_mask=True,
    )

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "detector.mask.normalized_positive_excess_rule" in checks


def test_validate_products_reports_detector_window_contract_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "bad_window"
    _write_products(run_dir, corrupt_detector_window_metadata=True)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "run_config.detector_window_samples" in checks


def test_validate_products_accepts_normalized_rule(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_products(run_dir)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is True, report["errors"]


def test_validate_products_checks_normalized_mask_rule(tmp_path) -> None:
    # The corrupted cell [0, 0] (mask forced to 1) is legal under the legacy
    # F>1 rule (2*11 > 20) but illegal under the declared normalized rule
    # (11*9 <= 5*20), so this only fails if the validator dispatches to the
    # normalized rule.
    run_dir = tmp_path / "run"
    _write_products(run_dir, corrupt_normalized_positive_excess_mask=True)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "detector.mask.normalized_positive_excess_rule" in checks


def test_validate_products_normalized_rule_requires_norms(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_products(run_dir, omit_norms=True)

    report = validate_products(run_dir=run_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "detector.mask.norms_missing" in checks
