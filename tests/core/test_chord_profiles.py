# coding=utf-8
"""CHORD / CHORD-pathfinder receiver-profile and runtime-bundle checks.

These tests pin the CHORD integration parameters against independently
computed expectations:

* channelizer grid: first Nyquist zone of a 3.2 GS/s ADC with a 16384-point
  PFB, so 8192 coarse channels of exactly 195312.5 Hz with channel RF center
  ``coarse_channel_index * 195312.5 Hz`` (upright spectral sense, ascending
  order, kotekan ``freq_id == coarse_channel_index``);
* ATSC 14-36 pilot placement: 23 unique coarse channels 2408..3084 with
  exact half-Hz fine offsets; physical channel 14 is the single adaptive
  reference case (upper reference wraps through the forbidden DC bin);
* framing: 16384-sample detector blocks give windows_per_stream = 128, the
  frozen fine-reduction transform length, and detector row counts
  131072 (CHORD, 1024 streams) / 16384 (pathfinder, 128 streams);
* runtime bundles: the declared ``chord_freq_id`` channel-id map populates
  ``chord_channel_id`` / ``receiver_channel_id`` for kotekan first-frame
  profile selection, while CHIME bundles keep null channel ids.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_geometry import DetectorInputLayout
from pilot_proxy.integration import (
    load_detector_core_profile,
    load_receiver_profile,
)
from pilot_proxy.integration.receiver_profile import (
    FREQUENCY_ORDER_ASCENDING_RF,
    receiver_frequency_to_channel,
)
from pilot_proxy.integration.stream_layout import InputStreamMap
from pilot_proxy.integration.weight_generation import (
    generate_weight_table_from_receiver_profile,
    target_layout,
)
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CHORD_PROFILE = CONFIGS_DIR / "receiver_profiles" / "chord_dtv_fengine.json"
PATHFINDER_PROFILE = (
    CONFIGS_DIR / "receiver_profiles" / "chord_pathfinder_dtv_fengine.json"
)
CHIME_PROFILE = CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
CORE_PROFILE = CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
CHORD_STREAM_MAP = CONFIGS_DIR / "stream_maps" / "chord_dish_pol_example.json"
PATHFINDER_STREAM_MAP = (
    CONFIGS_DIR / "stream_maps" / "chord_pathfinder_dish_pol_example.json"
)

# Exact CHORD channelizer grid: 3.2 GS/s ADC, 16384-point PFB, Nyquist zone 1.
CHORD_ADC_HZ = 3_200_000_000
CHORD_PFB_FFT_SIZE = 16_384
CHORD_NUM_COARSE_CHANNELS = CHORD_PFB_FFT_SIZE // 2
CHORD_COARSE_WIDTH = Fraction(CHORD_ADC_HZ, CHORD_PFB_FFT_SIZE)  # 195312.5 Hz
CHORD_DETECTOR_WINDOW = 128
CHORD_FINE_BIN_WIDTH = CHORD_COARSE_WIDTH / CHORD_DETECTOR_WINDOW

ATSC_PHYSICAL_CHANNELS = list(range(14, 37))

# Independently derived (documented in docs/KOTEKAN_INTERFACE_PREP.md):
# chord_freq_id = round(pilot_hz / 195312.5) for ATSC physical channels 14-36.
EXPECTED_CHORD_CHANNEL_IDS = [
    2408, 2439, 2469, 2500, 2531, 2562, 2592, 2623, 2654, 2684, 2715,
    2746, 2777, 2807, 2838, 2869, 2900, 2930, 2961, 2992, 3022, 3053,
    3084,
]


def _expected_channel_and_offset(physical_channel: int) -> tuple[int, Fraction]:
    """Nearest CHORD coarse channel and exact fine offset for one pilot."""
    pilot = Fraction(int(physical_channel_to_pilot_hz(physical_channel)))
    ratio = pilot / CHORD_COARSE_WIDTH
    # Straightforward nearest-integer; no ATSC pilot sits exactly on a
    # half-channel boundary in this grid.
    assert ratio % 1 != Fraction(1, 2)
    index = int(ratio + Fraction(1, 2))
    return index, pilot - index * CHORD_COARSE_WIDTH


@pytest.mark.parametrize(
    "profile_path,num_streams",
    [(CHORD_PROFILE, 1024), (PATHFINDER_PROFILE, 128)],
)
def test_chord_profile_channelizer_grid(profile_path, num_streams) -> None:
    profile = load_receiver_profile(profile_path)
    assert profile.spectral_sense == "normal"
    assert profile.frequency_order == FREQUENCY_ORDER_ASCENDING_RF
    assert profile.num_coarse_channels == CHORD_NUM_COARSE_CHANNELS
    assert profile.coarse_channel_width_hz == float(CHORD_COARSE_WIDTH)
    assert profile.center_offset_hz == 0.0
    assert profile.frame_size_samples == 16_384
    assert profile.num_input_streams == num_streams
    assert profile.detector_window_samples == CHORD_DETECTOR_WINDOW
    assert profile.bin_enbw_hz == float(CHORD_FINE_BIN_WIDTH)
    # Channel RF center = index * width exactly (kotekan freq_id namespace,
    # channel 1536 = 300 MHz, channel 7680 = 1500 MHz).
    assert profile.coarse_channel_center_hz(0) == 0.0
    assert profile.coarse_channel_center_hz(1536) == 300.0e6
    assert profile.coarse_channel_center_hz(7680) == 1500.0e6
    # Explicit baseband frame: coarse-channel center at DFT DC.
    assert profile.frame_center_normalized(0) == 0.0
    assert profile.frame_center_normalized(1) == 0.0
    # Channel-id map declared for runtime-bundle population.
    channel_id_map = profile.metadata["channel_id_map"]
    assert channel_id_map["namespace"] == "chord_freq_id"
    assert int(channel_id_map["offset_from_coarse_channel_index"]) == 0


@pytest.mark.parametrize("profile_path", [CHORD_PROFILE, PATHFINDER_PROFILE])
def test_chord_pilot_channel_mapping(profile_path) -> None:
    profile = load_receiver_profile(profile_path)
    seen_ids = []
    for channel in ATSC_PHYSICAL_CHANNELS:
        expected_index, expected_offset = _expected_channel_and_offset(channel)
        pilot_hz = physical_channel_to_pilot_hz(channel)
        selection = receiver_frequency_to_channel(pilot_hz, profile)
        assert selection.coarse_channel_index == expected_index
        # Upright sense: no window time reversal, fine offset = rf - center.
        assert selection.requires_time_reversal is False
        assert selection.fine_bin_offset_hz == float(expected_offset)
        # Offsets are exact multiples of 0.5 Hz, representable in binary
        # floats, so the equality above is exact.
        assert abs(selection.fine_bin_offset_hz) <= float(CHORD_COARSE_WIDTH) / 2
        seen_ids.append(selection.coarse_channel_index)
    assert seen_ids == EXPECTED_CHORD_CHANNEL_IDS
    assert len(set(seen_ids)) == len(seen_ids)


@pytest.mark.parametrize(
    "profile_path,expected_rows",
    [(CHORD_PROFILE, 131_072), (PATHFINDER_PROFILE, 16_384)],
)
def test_chord_detector_layout_matches_frozen_fine_geometry(
    profile_path, expected_rows
) -> None:
    profile = load_receiver_profile(profile_path)
    layout = DetectorInputLayout(
        samples_per_block=profile.frame_size_samples,
        num_streams=profile.num_input_streams,
        detector_window_samples=profile.detector_window_samples,
    )
    # windows_per_block must equal the frozen fine-reduction transform length
    # (FSTAT_FINE_WINDOWS_PER_STREAM = 128) for the fused fine/mask kernels.
    assert layout.windows_per_block == 128
    assert layout.detector_rows_per_block == expected_rows
    # One detector block = 2 kotekan GPU frames of 8192 samples.
    assert profile.frame_size_samples == 2 * 8192


def test_chord_reference_placement_channel14_is_only_adaptive_case() -> None:
    profile = load_receiver_profile(CHORD_PROFILE)
    core = load_detector_core_profile(CORE_PROFILE)
    for channel in ATSC_PHYSICAL_CHANNELS:
        layout = target_layout(
            physical_channel=channel, profile=profile, core=core
        )
        assert layout["baseband_frame_mode"] == "explicit_profile_frame"
        assert layout["strict_reference_offset_pass"] is True
        assert layout["lower_reference_offset_bins"] == -2
        if channel == 14:
            # The ch14 pilot sits 2.005 fine bins below the coarse-channel
            # center, so the requested +2 upper reference lands on the
            # forbidden DC bin and shifts one bin farther, wrapping through
            # the frame edge (DC is the wrap point for a center-at-DC frame).
            assert layout["reference_placement_status"] == (
                "edge_wrapped_and_dc_shifted"
            )
            assert layout["upper_reference_offset_bins"] == 3
            assert layout["upper_reference_dc_shifted"] is True
            assert layout["forbidden_tone_in_skipped_guard"] is True
        else:
            assert layout["reference_placement_status"] == "nominal"
            assert layout["upper_reference_offset_bins"] == 2
            assert layout["adaptive_reference_placement"] is False


def test_chord_weight_tables_identical_for_both_coordinate_systems() -> None:
    # Upright spectral sense: post-spectral-sense-normalization and raw
    # input-frequency coordinates coincide, so both banks must be
    # bit-identical and no window time reversal is requested.
    profile = load_receiver_profile(CHORD_PROFILE)
    core = load_detector_core_profile(CORE_PROFILE)
    table_post, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=ATSC_PHYSICAL_CHANNELS,
        weight_coordinate_system="post_spectral_sense_normalization",
    )
    table_raw, _ = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=ATSC_PHYSICAL_CHANNELS,
        weight_coordinate_system="raw_input_frequency_coordinate",
    )
    assert np.array_equal(table_post, table_raw)
    populated = sorted(np.flatnonzero(table_post.any(axis=(1, 2))).tolist())
    assert populated == EXPECTED_CHORD_CHANNEL_IDS
    assert len(layouts) == len(ATSC_PHYSICAL_CHANNELS)


@pytest.mark.parametrize(
    "profile_path,map_path,num_streams,num_dishes",
    [
        (CHORD_PROFILE, CHORD_STREAM_MAP, 1024, 512),
        (PATHFINDER_PROFILE, PATHFINDER_STREAM_MAP, 128, 64),
    ],
)
def test_chord_stream_maps_match_profiles(
    profile_path, map_path, num_streams, num_dishes
) -> None:
    profile = load_receiver_profile(profile_path)
    stream_map = InputStreamMap.from_json(map_path)
    assert stream_map.num_streams == num_streams == profile.num_input_streams
    assert stream_map.stream_unit == "dish_polarization"
    # stream_index = polarization_index * num_dishes + dish_index, matching
    # the kotekan voltage-buffer inner axes [P, D].
    entries = stream_map.streams
    assert len(entries) == num_streams
    for entry in entries:
        expected = (
            int(entry["polarization_index"]) * num_dishes
            + int(entry["dish_index"])
        )
        assert int(entry["stream_index"]) == expected


@pytest.mark.parametrize(
    "profile_path",
    [CHORD_PROFILE, PATHFINDER_PROFILE],
)
def test_chord_runtime_bundle_populates_chord_channel_ids(
    tmp_path, profile_path
) -> None:
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=profile_path,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=ATSC_PHYSICAL_CHANNELS,
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=tmp_path / "bundle",
    )
    report = validate_runtime_weight_bundle(bundle_dir=tmp_path / "bundle")
    assert report["valid"], report["errors"]

    pilot_profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    assert pilot_profiles["receiver_channel_id_namespace"] == "chord_freq_id"
    rows = pilot_profiles["profiles"]
    assert [row["chord_channel_id"] for row in rows] == (
        EXPECTED_CHORD_CHANNEL_IDS
    )
    for row in rows:
        assert row["receiver_channel_id"] == row["chord_channel_id"]
        assert row["receiver_channel_id"] == row["coarse_channel_index"]
        assert row["receiver_channel_id_namespace"] == "chord_freq_id"
        assert row["chime_channel_id"] is None
        assert row["coarse_channel_center_hz"] == (
            row["coarse_channel_index"] * float(CHORD_COARSE_WIDTH)
        )
    contract = json.loads(outputs["detector_contract"].read_text("utf-8"))
    # Upright sense: the deployed coordinate system requests no reversal.
    assert contract["input_preprocessing"][
        "time_reverse_detector_windows_before_kernel"
    ] is False


def test_chime_runtime_bundle_keeps_null_channel_ids(tmp_path) -> None:
    # CHIME keeps the legacy null channel ids until its Kotekan metadata
    # mapping is verified; the chord_freq_id map must not leak across
    # profiles.
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=CHIME_PROFILE,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=[14, 15],
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=tmp_path / "bundle",
    )
    report = validate_runtime_weight_bundle(bundle_dir=tmp_path / "bundle")
    assert report["valid"], report["errors"]
    pilot_profiles = json.loads(outputs["pilot_profiles"].read_text("utf-8"))
    assert pilot_profiles["receiver_channel_id_namespace"] is None
    for row in pilot_profiles["profiles"]:
        assert row["chime_channel_id"] is None
        assert row["chord_channel_id"] is None
        assert row["receiver_channel_id"] is None


def test_runtime_bundle_validator_rejects_duplicate_chord_channel_ids(
    tmp_path,
) -> None:
    export_runtime_weight_bundle(
        receiver_profile_path=CHORD_PROFILE,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=[14, 15],
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=tmp_path / "bundle",
    )
    profiles_path = tmp_path / "bundle" / "pilot_profiles.json"
    payload = json.loads(profiles_path.read_text("utf-8"))
    payload["profiles"][1]["chord_channel_id"] = (
        payload["profiles"][0]["chord_channel_id"]
    )
    profiles_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = validate_runtime_weight_bundle(bundle_dir=tmp_path / "bundle")
    assert not report["valid"]
    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.chord_channel_id" in checks


def test_runtime_bundle_validator_rejects_alias_mismatch(tmp_path) -> None:
    export_runtime_weight_bundle(
        receiver_profile_path=CHORD_PROFILE,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=[14],
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=tmp_path / "bundle",
    )
    profiles_path = tmp_path / "bundle" / "pilot_profiles.json"
    payload = json.loads(profiles_path.read_text("utf-8"))
    payload["profiles"][0]["chord_channel_id"] = 999
    profiles_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = validate_runtime_weight_bundle(bundle_dir=tmp_path / "bundle")
    assert not report["valid"]
    checks = {error["check"] for error in report["errors"]}
    assert "pilot_profiles.chord_channel_id" in checks
