# coding=utf-8
from __future__ import annotations

import json
import shutil

import pytest

from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.detector_core import DetectorCoreProfile
from pilot_proxy.integration.weight_generation import DetectorCoreLayout, target_layout
from pilot_proxy.paths import CONFIGS_DIR
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

FIRST_SHIPPED_PHYSICAL_CHANNEL = 14
LAST_SHIPPED_PHYSICAL_CHANNEL_EXCLUSIVE = 37
FIRST_SHIPPED_PILOT_MHZ = 470.309441
UNKNOWN_PILOT_MHZ_NEAR_CHANNEL_14 = 470.310000
PILOT_FREQUENCY_TOLERANCE_HZ = 10.0

EDGE_WRAP_PHYSICAL_CHANNEL = 21
EDGE_WRAP_PILOT_MHZ = 512.309441
EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS = -2
EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS = 2
REFERENCE_COARSE_CHANNEL_WIDTH_HZ = 390_625.0


def _core(*, k: int = 128, reference_offset_bins: int = 2) -> DetectorCoreLayout:
    return DetectorCoreLayout(
        detector_window_samples=int(k),
        skipped_guard_bins=int(reference_offset_bins) - 1,
        reference_offset_bins=int(reference_offset_bins),
    )


def _reference_profile():
    return load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
    )


def test_weight_bank_validates_known_physical_channel() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(
        FIRST_SHIPPED_PHYSICAL_CHANNEL
    )

    assert valid
    assert weights is not None
    assert bank.known_pilot_frequencies_mhz[0] == FIRST_SHIPPED_PILOT_MHZ


def test_weight_bank_rejects_unknown_pilot_frequency() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    with pytest.raises(ValueError, match="not in the weight manifest"):
        bank.get_weights_for_pilot_frequency(
            UNKNOWN_PILOT_MHZ_NEAR_CHANNEL_14,
            tolerance_hz=PILOT_FREQUENCY_TOLERANCE_HZ,
        )


def test_deprecated_detector_spacing_field_is_rejected() -> None:
    old_key = "_".join(("reference", "guard", "bins"))

    with pytest.raises(ValueError, match="Deprecated detector-spacing field"):
        DetectorCoreProfile.from_dict({old_key: 2})


def test_weight_bank_reports_channel_21_wrapped_boundary_layout() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    layout = bank.layout_for_physical_channel(EDGE_WRAP_PHYSICAL_CHANNEL)

    assert layout["target_frequency_mhz"] == EDGE_WRAP_PILOT_MHZ
    assert layout["adaptive_reference_placement"] is True
    assert layout["lower_reference_offset_bins"] == EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS
    assert layout["upper_reference_offset_bins"] == EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS
    assert layout["lower_reference_edge_wrapped"] is True
    assert layout["upper_reference_edge_wrapped"] is False
    assert layout["edge_reference_wrapped"] is True
    assert layout["reference_placement_status"] == "edge_wrapped"
    assert layout["strict_reference_offset_pass"] is True
    assert layout["detector_fine_bin_width_hz"] == pytest.approx(
        REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 128.0
    )
    assert layout["lower_reference_relative_to_target_hz"] == pytest.approx(
        EDGE_WRAP_LOWER_REFERENCE_OFFSET_BINS
        * REFERENCE_COARSE_CHANNEL_WIDTH_HZ
        / 128.0
    )
    assert layout["upper_reference_relative_to_target_hz"] == pytest.approx(
        EDGE_WRAP_UPPER_REFERENCE_OFFSET_BINS
        * REFERENCE_COARSE_CHANNEL_WIDTH_HZ
        / 128.0
    )
    assert "lower_reference_offset_hz" in layout
    assert "upper_reference_offset_hz" in layout


def test_k128_dtv14_dc_in_skipped_guard_not_reference() -> None:
    layout = target_layout(
        physical_channel=14,
        profile=_reference_profile(),
        core=_core(k=128, reference_offset_bins=2),
    )

    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_offset_bins"] == 2
    assert layout["dc_reference_collision"] is False
    assert layout["dc_reference_shifted"] is False
    assert layout["forbidden_tone_in_skipped_guard"] is True
    assert layout["reference_placement_status"] == "nominal"


def test_target_on_forbidden_tone_hard_fails(monkeypatch) -> None:
    profile = _reference_profile()
    center_hz = profile.coarse_channel_center_hz(843)
    monkeypatch.setattr(
        "pilot_proxy.integration.weight_generation.physical_channel_to_pilot_hz",
        lambda channel: center_hz,
    )

    with pytest.raises(ValueError, match="target pilot bin collides"):
        target_layout(
            physical_channel=14,
            profile=profile,
            core=_core(k=128, reference_offset_bins=2),
        )


def test_k256_offset2_dtv14_reference_shifts_away_from_dc() -> None:
    layout = target_layout(
        physical_channel=14,
        profile=_reference_profile(),
        core=_core(k=256, reference_offset_bins=2),
    )

    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_requested_offset_bins"] == 2
    # The configured candidate remains reference_offset_bins=2. The selected
    # upper reference moves to +3 only as the adaptive DC-avoidance correction.
    assert layout["upper_reference_offset_bins"] == 3
    assert layout["upper_reference_requested_dc_collision"] is True
    assert layout["upper_reference_dc_shifted"] is True
    assert layout["dc_reference_collision"] is True
    assert layout["dc_reference_shifted"] is True
    assert layout["reference_placement_status"] == "dc_shifted"
    assert layout["strict_reference_offset_pass"] is True
    assert layout["upper_reference_requested_relative_to_target_hz"] == pytest.approx(
        2 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 256.0
    )
    assert layout["upper_reference_relative_to_target_hz"] == pytest.approx(
        3 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / 256.0
    )


def test_dtv21_edge_reference_wraps_without_moving_closer() -> None:
    layout = target_layout(
        physical_channel=21,
        profile=_reference_profile(),
        core=_core(k=128, reference_offset_bins=2),
    )

    assert layout["lower_reference_requested_edge_wrapped"] is True
    assert layout["lower_reference_edge_wrapped"] is True
    assert layout["lower_reference_offset_bins"] == -2
    assert layout["upper_reference_offset_bins"] == 2
    assert (layout["lower_reference_offset_bins"], layout["upper_reference_offset_bins"]) != (
        -1,
        1,
    )
    assert layout["strict_reference_offset_pass"] is True


def test_weight_bank_lists_shipped_physical_channels() -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    assert bank.supported_physical_channels() == list(
        range(
            FIRST_SHIPPED_PHYSICAL_CHANNEL,
            LAST_SHIPPED_PHYSICAL_CHANNEL_EXCLUSIVE,
        )
    )


def test_physical_channel_lookup_requires_manifest(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    bank = DetectorWeightBank(explicit_path=copied)
    with pytest.raises(ValueError, match="require the adjacent weight manifest"):
        bank.get_weights_for_physical_channel(FIRST_SHIPPED_PHYSICAL_CHANNEL)


def test_weight_bank_rejects_manifest_binary_binding_mismatch(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    # The shipped manifest binds the binary via artifacts.weights_sha256 (v2);
    # corrupt that field so the loader's manifest/binary check must reject it.
    # (The legacy weights_git_blob_sha1 path is only reached when weights_sha256
    # is absent, so tampering it here would be silently skipped.)
    manifest["artifacts"]["weights_sha256"] = "0" * 64
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest/binary"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_v2_manifest_without_coordinate_system(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("weight_coordinate_system", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires weight_coordinate_system"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_v2_manifest_without_input_coordinate(tmp_path) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("input_coordinate_system", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires input_coordinate_system"):
        DetectorWeightBank(explicit_path=copied)


def test_weight_bank_rejects_v2_manifest_without_input_preprocessing(
    tmp_path,
) -> None:
    copied = tmp_path / DEFAULT_WEIGHTS_PATH.name
    shutil.copyfile(DEFAULT_WEIGHTS_PATH, copied)
    source_manifest = DEFAULT_WEIGHTS_PATH.with_suffix(
        DEFAULT_WEIGHTS_PATH.suffix + ".manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.pop("input_preprocessing", None)
    copied.with_suffix(copied.suffix + ".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires input_preprocessing"):
        DetectorWeightBank(explicit_path=copied)
