# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_proxy.provenance import file_sha256
from pilot_proxy.detector_contract import (
    INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED,
    INPUT_COORDINATE_RAW_INPUT,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
)
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CHIME_FINE_OFFSET_HZ = -3_059.0


def _layout_by_channel(manifest: dict, channel: int) -> dict:
    for row in manifest["target_reference_layout"]:
        if int(row["physical_channel"]) == int(channel):
            return row
    raise AssertionError(f"missing physical channel {channel}")


def test_export_runtime_weight_bundle_writes_compact_profiles(tmp_path) -> None:
    output_dir = tmp_path / "bundle"

    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )

    for path in outputs.values():
        assert path.exists()

    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))

    assert contract["schema_version"] == "pilotproxy_chime_detector_contract_v1"
    assert contract["per_frequency_threshold"] is False
    assert contract["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    assert (
        contract["input_coordinate_system"]
        == INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED
    )
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )
    assert profiles["schema_version"] == "pilotproxy_runtime_pilot_profiles_v1"
    assert manifest["schema_version"] == "pilotproxy_runtime_weights_manifest_v1"
    assert profiles["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    assert manifest["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    assert profiles["detector_contract_sha256"] == manifest["detector_contract_sha256"]
    assert profiles["weights_sha256"] == file_sha256(outputs["weights"])
    assert [row["physical_channel"] for row in profiles["profiles"]] == [14, 21]
    assert profiles["profiles"][0]["weight_bank_index"] == 0
    assert profiles["profiles"][0]["weight_bank_offset_bytes"] == 0
    assert profiles["profiles"][1]["weight_bank_index"] == 1
    assert profiles["profiles"][1]["weight_bank_offset_bytes"] == (
        manifest["weight_profile_nbytes"]
    )
    assert outputs["weights"].stat().st_size == (
        2 * manifest["weight_profile_nbytes"]
    )
    sha_text = outputs["sha256sums"].read_text("utf-8")
    assert "detector_contract.json" in sha_text
    assert "pilot_profiles.json" in sha_text
    assert "weights.bin" in sha_text
    assert "weights.manifest.json" in sha_text

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is True
    assert report["num_errors"] == 0


def test_chime_runtime_bundle_post_coordinate_uses_detector_coordinate(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    channel_14 = _layout_by_channel(manifest, 14)
    channel_21 = _layout_by_channel(manifest, 21)

    assert contract["input_coordinate_system"] == (
        INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED
    )
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is True
    )
    assert channel_14["target_offset_hz"] == pytest.approx(-CHIME_FINE_OFFSET_HZ)
    assert channel_14["lower_reference_edge_wrapped"] is True
    assert channel_14["upper_reference_edge_wrapped"] is False
    assert channel_21["lower_reference_edge_wrapped"] is False
    assert channel_21["upper_reference_edge_wrapped"] is False


def test_chime_runtime_bundle_raw_coordinate_uses_native_inverted_coordinate(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_RAW_INPUT,
        output_dir=output_dir,
    )
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    channel_14 = _layout_by_channel(manifest, 14)
    channel_21 = _layout_by_channel(manifest, 21)

    assert contract["input_coordinate_system"] == INPUT_COORDINATE_RAW_INPUT
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )
    assert channel_14["target_offset_hz"] == pytest.approx(CHIME_FINE_OFFSET_HZ)
    assert channel_14["lower_reference_edge_wrapped"] is False
    assert channel_14["upper_reference_edge_wrapped"] is True
    assert channel_21["lower_reference_edge_wrapped"] is False
    assert channel_21["upper_reference_edge_wrapped"] is False


def test_validate_runtime_weight_bundle_reports_bad_offset(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["profiles"][1]["weight_bank_offset_bytes"] = 1
    outputs["pilot_profiles"].write_text(json.dumps(profiles), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "sha256sums.pilot_profiles.json" in checks
    assert "pilot_profiles.detector_contract_sha256" not in checks
    assert "pilot_profiles.profile_offset_alignment" in checks


def test_validate_runtime_weight_bundle_reports_coordinate_mismatch(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )
    manifest = json.loads(outputs["weights_manifest"].read_text("utf-8"))
    manifest["weight_coordinate_system"] = "raw_input_frequency_coordinate"
    outputs["weights_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "weight_coordinate_system.consistency" in checks


def test_validate_runtime_weight_bundle_reports_input_coordinate_mismatch(
    tmp_path,
) -> None:
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    profiles["input_coordinate_system"] = "raw_input_frequency_coordinate"
    outputs["pilot_profiles"].write_text(json.dumps(profiles), encoding="utf-8")

    report = validate_runtime_weight_bundle(bundle_dir=output_dir)

    assert report["valid"] is False
    checks = {error["check"] for error in report["errors"]}
    assert "input_coordinate_system.consistency" in checks
    assert "pilot_profiles.input_coordinate_system" in checks


def _export_reference_bundle(tmp_path):
    output_dir = tmp_path / "bundle"
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=(
            CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
        ),
        detector_core_profile_path=(
            CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
        ),
        physical_channels=[14, 21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        output_dir=output_dir,
    )
    return output_dir, outputs


def _rewrite_profiles(outputs, profiles) -> None:
    outputs["pilot_profiles"].write_text(
        json.dumps(profiles), encoding="utf-8"
    )


def test_exported_bundle_carries_pending_fine_calibration(tmp_path) -> None:
    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    for row in profiles["profiles"]:
        block = row["fine_calibration"]
        assert block["status"] == "pending_campaign"
        assert block["decision_version"] == "fine_decision_v1"
        assert block["anchor_bin"] is None
        assert block["designated_half_width"] == 2
        assert block["cfar_rank"] is None
        assert block["cfar_multiplier_q16"] is None
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is True


def test_calibrated_fine_block_validates_when_consistent(tmp_path) -> None:
    from pilot_proxy.fine_decision import pack_bulk_mask
    from pilot_proxy.fine_reduction import independent_bin_mask

    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    bulk = independent_bin_mask(256, designated_bins=[62])
    words = pack_bulk_mask(bulk)
    profiles["profiles"][0]["fine_calibration"] = {
        "status": "calibrated",
        "decision_version": "fine_decision_v1",
        "anchor_bin": 62,
        "designated_half_width": 2,
        "bulk_mask_words_hex": [f"0x{w:016x}" for w in words],
        "cfar_rank": int(sum(bulk) // 2),
        "cfar_multiplier_q16": int(1.5 * 65536),
        "provenance": {
            "epochs": ["2026Q1"],
            "null_quantile": 0.999,
            "source_product_sha256": ["deadbeef"],
        },
    }
    _rewrite_profiles(outputs, profiles)
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    # only the sha256sums entry fails (profiles were rewritten in place);
    # the fine_calibration block itself must contribute no errors
    checks = {e["check"] for e in report["errors"]}
    assert not any("fine_calibration" in c for c in checks), checks


def test_calibrated_fine_block_rejects_inconsistencies(tmp_path) -> None:
    from pilot_proxy.fine_decision import pack_bulk_mask
    from pilot_proxy.fine_reduction import independent_bin_mask

    output_dir, outputs = _export_reference_bundle(tmp_path)
    profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    bulk = independent_bin_mask(256, designated_bins=[200])
    words = pack_bulk_mask(bulk)
    base = {
        "status": "calibrated",
        "decision_version": "fine_decision_v1",
        "anchor_bin": 62,  # anchor INSIDE a mask that never excluded it
        "designated_half_width": 2,
        "bulk_mask_words_hex": [f"0x{w:016x}" for w in words],
        "cfar_rank": 5000,  # deeper than the bulk population
        "cfar_multiplier_q16": 0,  # not positive
        "provenance": {},
    }
    profiles["profiles"][0]["fine_calibration"] = dict(base)
    profiles["profiles"][1]["fine_calibration"] = {
        "status": "unheard_of",
        "decision_version": "fine_decision_v2",
    }
    _rewrite_profiles(outputs, profiles)
    report = validate_runtime_weight_bundle(bundle_dir=output_dir)
    assert report["valid"] is False
    checks = {e["check"] for e in report["errors"]}
    assert any("cfar_rank" in c for c in checks)
    assert any("cfar_multiplier_q16" in c for c in checks)
    assert any("bulk_mask_words_hex" in c for c in checks)
    assert any("status" in c for c in checks)
    assert any("decision_version" in c for c in checks)
