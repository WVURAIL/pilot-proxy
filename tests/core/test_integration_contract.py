# coding=utf-8
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_contract import (
    INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
)
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED
from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.integration import (
    COMBINE_MODE_COMBINED_STREAMS,
    COMBINE_MODE_PER_STREAM_DIAGNOSTIC,
    DetectorCoreProfile,
    FREQUENCY_ORDER_DESCENDING_RF,
    InputStreamMap,
    QUANTIZATION_SCALE_MODE_PER_STREAM,
    ReceiverProfile,
    default_reference_receiver_profile,
    detector_shape_for_combined_streams,
    detector_shape_for_per_stream_diagnostics,
    generate_weight_table_from_receiver_profile,
    layout_uint64_bound_check,
    load_detector_core_profile,
    quantization_metadata,
    receiver_frequency_to_channel,
    receiver_profile_hash,
    validate_weight_manifest_profile_hash,
    write_weight_bank_from_receiver_profile,
)
from pilot_proxy.integration.packing import pack_channelized_streams_for_detector
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.stream_layout import load_stream_map
from pilot_proxy.integration.stream_layout import InputStreamLayout
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
PHYSICAL_CHANNEL_14 = 14
PHYSICAL_CHANNEL_21 = 21
CHANNEL_14_PILOT_HZ = physical_channel_to_pilot_hz(PHYSICAL_CHANNEL_14)
REFERENCE_CHANNEL_INDEX_FOR_CHANNEL_14 = 179
CHIME_CHANNEL_INDEX_FOR_CHANNEL_14 = 843
REFERENCE_CHANNEL_CENTER_HZ = 470_312_500.0
REFERENCE_COARSE_CHANNEL_WIDTH_HZ = 390_625.0
REFERENCE_FINE_OFFSET_HZ = -3_059.0
FRAME_SIZE_SAMPLES = 16_384
DETECTOR_WINDOW_SAMPLES = 128
NUM_INPUT_STREAMS = 4
WINDOWS_PER_STREAM = 128
DETECTOR_ROWS = NUM_INPUT_STREAMS * WINDOWS_PER_STREAM
INT4_COMPONENT_BITS = 4
CLIP_SIGMA = 3.0
HZ_PER_MHZ = 1.0e6
WEIGHT_TERMS = 3
SMALL_PACK_FRAME_SIZE = 8
SMALL_PACK_WINDOW = 4
SMALL_PACK_FEEDS = 2
SMALL_PACK_CHANNELS = 2


def test_receiver_profile_frequency_mapping_normal_and_inverted() -> None:
    profile = default_reference_receiver_profile()
    selection = receiver_frequency_to_channel(CHANNEL_14_PILOT_HZ, profile)

    assert selection.coarse_channel_index == REFERENCE_CHANNEL_INDEX_FOR_CHANNEL_14
    assert selection.coarse_channel_center_hz == REFERENCE_CHANNEL_CENTER_HZ
    assert selection.fine_bin_offset_hz == pytest.approx(REFERENCE_FINE_OFFSET_HZ)
    assert selection.requires_time_reversal is False

    inverted = ReceiverProfile.from_dict(
        {
            **profile.to_dict(),
            "spectral_sense": SPECTRAL_SENSE_INVERTED,
        }
    )
    inverted_selection = receiver_frequency_to_channel(CHANNEL_14_PILOT_HZ, inverted)
    assert inverted_selection.coarse_channel_index == (
        REFERENCE_CHANNEL_INDEX_FOR_CHANNEL_14
    )
    assert inverted_selection.fine_bin_offset_hz == pytest.approx(
        -REFERENCE_FINE_OFFSET_HZ
    )
    assert inverted_selection.requires_time_reversal is True


def test_receiver_profile_nested_json_roundtrip() -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
    )
    nested = profile.to_nested_dict()
    reparsed = ReceiverProfile.from_dict(nested)

    assert profile.name == "reference_800mhz_pfb_v1"
    assert profile.instrument_name == "reference"
    assert profile.frame_size_samples == FRAME_SIZE_SAMPLES
    assert profile.num_input_streams == 1
    assert nested["detector_adapter"]["compatible_detector_core_id"] == (
        "pilotproxy_cuda_fstat_v1"
    )
    assert reparsed.to_dict() == profile.to_dict()


def test_receiver_profile_flat_json_roundtrip() -> None:
    profile = default_reference_receiver_profile(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
    )
    reparsed = ReceiverProfile.from_dict(profile.to_dict())

    assert reparsed.to_dict() == profile.to_dict()
    assert reparsed.coarse_channel_width_hz == pytest.approx(390_625.0)


def test_chime_example_profile_values() -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    selection = receiver_frequency_to_channel(CHANNEL_14_PILOT_HZ, profile)

    assert profile.profile_status == "example_requires_data_product_verification"
    assert profile.num_input_streams == 2048
    assert profile.spectral_sense == SPECTRAL_SENSE_INVERTED
    assert profile.frequency_order == FREQUENCY_ORDER_DESCENDING_RF
    assert selection.coarse_channel_index == CHIME_CHANNEL_INDEX_FOR_CHANNEL_14
    assert selection.coarse_channel_center_hz == REFERENCE_CHANNEL_CENTER_HZ
    assert selection.fine_bin_offset_hz == pytest.approx(-REFERENCE_FINE_OFFSET_HZ)
    assert selection.requires_time_reversal is True


def test_receiver_profile_hash_changes_when_frequency_mapping_changes() -> None:
    data = (
        load_receiver_profile(
            CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
        )
        .to_nested_dict()
    )
    original = ReceiverProfile.from_dict(data)
    changed = dict(data)
    changed["channelizer"] = dict(data["channelizer"])
    changed["channelizer"]["frequency_axis"] = dict(
        data["channelizer"]["frequency_axis"]
    )
    changed["channelizer"]["frequency_axis"]["order"] = "ascending_rf"
    changed_profile = ReceiverProfile.from_dict(changed)

    assert receiver_profile_hash(original) != receiver_profile_hash(changed_profile)


def test_detector_core_profile_json_roundtrip() -> None:
    profile = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
    )
    reparsed = DetectorCoreProfile.from_dict(profile.to_dict())

    assert profile.detector_window_samples == DETECTOR_WINDOW_SAMPLES
    assert profile.num_weight_terms == WEIGHT_TERMS
    assert profile.skipped_guard_bins == 1
    assert profile.reference_offset_bins == 2
    assert profile.schema_version == "pilotproxy_detector_core_profile_v2"
    assert profile.host_masking_policy == "positive_excess_from_uint64_powers"
    assert profile.per_frequency_threshold is False
    assert reparsed.to_dict() == profile.to_dict()


def test_detector_core_accepts_skipped_guard_bins_as_input_source() -> None:
    profile = DetectorCoreProfile.from_dict({"kernel_contract": {"skipped_guard_bins": 2}})

    assert profile.skipped_guard_bins == 2
    assert profile.reference_offset_bins == 3


def test_detector_core_rejects_reference_offset_bins_as_input_source() -> None:
    with pytest.raises(ValueError, match="source of truth"):
        DetectorCoreProfile.from_dict({"reference_offset_bins": 3})


def test_detector_core_rejects_kernel_contract_reference_offset_bins() -> None:
    with pytest.raises(ValueError, match="source of truth"):
        DetectorCoreProfile.from_dict({"kernel_contract": {"reference_offset_bins": 3}})


def test_stream_map_json_roundtrip() -> None:
    stream_map = load_stream_map(
        CONFIGS_DIR / "stream_maps" / "chime_feed_pol_example.json"
    )

    assert isinstance(stream_map, InputStreamMap)
    assert stream_map.num_streams == 2048
    assert len(stream_map.streams) == 2048
    assert stream_map.streams[0]["polarization_label"] == "X"
    assert stream_map.streams[-1]["stream_index"] == 2047
    assert stream_map.streams[-1]["cylinder_index"] == 3
    assert stream_map.streams[-1]["feed_index_within_cylinder"] == 255
    assert stream_map.streams[-1]["polarization_label"] == "Y"


def test_make_weights_from_receiver_profile_roundtrip(tmp_path) -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "reference_800mhz_pfb.json"
    )
    core = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
    )
    output = tmp_path / "generated_weights.bin"

    manifest = write_weight_bank_from_receiver_profile(
        output_path=output,
        profile=profile,
        core=core,
        physical_channels=[PHYSICAL_CHANNEL_14],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    )
    bank = DetectorWeightBank(explicit_path=output)
    shipped = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(PHYSICAL_CHANNEL_14)
    shipped_weights, shipped_valid = shipped.get_weights_for_physical_channel(
        PHYSICAL_CHANNEL_14
    )
    layout = bank.layout_for_physical_channel(PHYSICAL_CHANNEL_14)

    assert output.exists()
    assert output.with_suffix(output.suffix + ".manifest.json").exists()
    assert manifest["receiver_profile_hash"] == receiver_profile_hash(profile)
    assert manifest["schema_version"] == "fstat_weight_manifest_v2"
    assert manifest["weight_coordinate_system"] == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    assert (
        manifest["input_coordinate_system"]
        == INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED
    )
    assert (
        manifest["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )
    assert manifest["kernel_spec"]["skipped_guard_bins"] == 1
    assert manifest["kernel_spec"]["reference_offset_bins"] == 2
    assert valid is True
    assert shipped_valid is True
    assert weights is not None
    assert shipped_weights is not None
    assert weights.shape == (WEIGHT_TERMS, DETECTOR_WINDOW_SAMPLES)
    np.testing.assert_array_equal(weights, shipped_weights)
    assert layout["coarse_channel_index"] == REFERENCE_CHANNEL_INDEX_FOR_CHANNEL_14
    assert layout["target_offset_hz"] == pytest.approx(REFERENCE_FINE_OFFSET_HZ)
    assert layout["detector_fine_bin_width_hz"] == pytest.approx(
        REFERENCE_COARSE_CHANNEL_WIDTH_HZ / DETECTOR_WINDOW_SAMPLES
    )
    assert layout["lower_reference_relative_to_target_hz"] == pytest.approx(
        -2 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / DETECTOR_WINDOW_SAMPLES
    )
    assert layout["upper_reference_relative_to_target_hz"] == pytest.approx(
        2 * REFERENCE_COARSE_CHANNEL_WIDTH_HZ / DETECTOR_WINDOW_SAMPLES
    )
    policy = manifest["forbidden_tone_policy"]
    assert policy["forbidden_tone"] == "coarse_channel_dc"
    assert policy["forbidden_tone_normalized"] == 0.5
    assert policy["forbidden_collision_rule"] == (
        "circular_normalized_distance <= 0.5 / detector_window_samples"
    )
    assert policy["forbidden_collision_half_width_bins"] == 0.5
    assert policy["forbidden_collision_half_width_normalized"] == pytest.approx(
        0.5 / DETECTOR_WINDOW_SAMPLES
    )


def test_chime_weight_generation_post_coordinate_uses_detector_coordinate() -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    core = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
    )

    _, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=[PHYSICAL_CHANNEL_14, PHYSICAL_CHANNEL_21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    )
    by_channel = {int(row["physical_channel"]): row for row in layouts}

    assert by_channel[PHYSICAL_CHANNEL_14]["target_offset_hz"] == pytest.approx(
        REFERENCE_FINE_OFFSET_HZ
    )
    assert by_channel[PHYSICAL_CHANNEL_21]["lower_reference_edge_wrapped"] is True
    assert by_channel[PHYSICAL_CHANNEL_21]["upper_reference_edge_wrapped"] is False


def test_chime_weight_generation_raw_coordinate_uses_native_inverted_coordinate() -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    core = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
    )

    _, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=[PHYSICAL_CHANNEL_14, PHYSICAL_CHANNEL_21],
        weight_coordinate_system=WEIGHT_COORDINATE_RAW_INPUT,
    )
    by_channel = {int(row["physical_channel"]): row for row in layouts}

    assert by_channel[PHYSICAL_CHANNEL_14]["target_offset_hz"] == pytest.approx(
        -REFERENCE_FINE_OFFSET_HZ
    )
    assert by_channel[PHYSICAL_CHANNEL_21]["lower_reference_edge_wrapped"] is False
    assert by_channel[PHYSICAL_CHANNEL_21]["upper_reference_edge_wrapped"] is True


def test_chime_post_coordinate_generation_matches_shipped_weight_layout() -> None:
    profile = load_receiver_profile(
        CONFIGS_DIR / "receiver_profiles" / "chime_dtv_fengine.json"
    )
    core = load_detector_core_profile(
        CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_fstat_v1.json"
    )
    table, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=[PHYSICAL_CHANNEL_14, PHYSICAL_CHANNEL_21],
        weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    )
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)

    for layout in layouts:
        channel = int(layout["physical_channel"])
        shipped_layout = bank.layout_for_physical_channel(channel)
        shipped_weights, valid = bank.get_weights_for_physical_channel(channel)
        assert valid is True
        assert shipped_weights is not None
        assert layout["target_offset_hz"] == pytest.approx(
            shipped_layout["target_offset_hz"]
        )
        assert layout["lower_reference_edge_wrapped"] == (
            shipped_layout["lower_reference_edge_wrapped"]
        )
        assert layout["upper_reference_edge_wrapped"] == (
            shipped_layout["upper_reference_edge_wrapped"]
        )
        np.testing.assert_array_equal(
            table[int(layout["coarse_channel_index"])],
            shipped_weights,
        )


def test_receiver_profile_hash_validation() -> None:
    profile = default_reference_receiver_profile(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
    )
    digest = receiver_profile_hash(profile)

    assert validate_weight_manifest_profile_hash(
        {"receiver_profile_hash": digest},
        profile,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_weight_manifest_profile_hash({"receiver_profile_hash": "bad"}, profile)


def test_stream_layout_shapes_and_uint64_bound_check() -> None:
    layout = InputStreamLayout(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
    )

    assert layout.combine_mode == COMBINE_MODE_COMBINED_STREAMS
    assert layout.detector_rows_per_frame == DETECTOR_ROWS
    assert detector_shape_for_combined_streams(layout) == (
        1,
        DETECTOR_ROWS,
        DETECTOR_WINDOW_SAMPLES,
    )
    assert detector_shape_for_per_stream_diagnostics(layout) == (
        NUM_INPUT_STREAMS,
        WINDOWS_PER_STREAM,
        DETECTOR_WINDOW_SAMPLES,
    )

    check = layout_uint64_bound_check(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
    )
    assert check["detector_rows_per_frame"] == DETECTOR_ROWS
    assert check["power_sum_fits_uint64"] is True
    assert check["recommended_batching"] == "ok"


def test_quantization_metadata_records_per_stream_scale() -> None:
    metadata = quantization_metadata(
        mode=QUANTIZATION_SCALE_MODE_PER_STREAM,
        bits_per_component=INT4_COMPONENT_BITS,
        clip_sigma=CLIP_SIGMA,
        scale_by_stream=np.asarray([1.0, 2.0]),
        clip_fraction_by_stream=np.asarray([0.0, 0.01]),
    )

    assert metadata["mode"] == QUANTIZATION_SCALE_MODE_PER_STREAM
    assert metadata["num_streams"] == 2
    assert metadata["scale_by_stream"] == [1.0, 2.0]
    assert metadata["clip_fraction_by_stream"] == [0.0, 0.01]


def test_pack_channelized_streams_combined_shape_and_row_order() -> None:
    feed_channel = np.arange(
        SMALL_PACK_FEEDS
        * SMALL_PACK_CHANNELS
        * SMALL_PACK_FRAME_SIZE,
        dtype=np.float32,
    ).reshape(SMALL_PACK_FEEDS, SMALL_PACK_CHANNELS, SMALL_PACK_FRAME_SIZE)
    feed_channel = np.asarray(np.mod(feed_channel, 7), dtype=np.float32).astype(
        np.complex64
    )

    packed = pack_channelized_streams_for_detector(
        feed_channel,
        frame_size_samples=SMALL_PACK_FRAME_SIZE,
        detector_window_samples=SMALL_PACK_WINDOW,
        quantization_scale_mode="provided",
        scale=1.0,
        selected_channel_indices=[179, 180],
        physical_channel=PHYSICAL_CHANNEL_14,
    )

    assert packed.packed.shape == (
        1,
        SMALL_PACK_FEEDS * SMALL_PACK_CHANNELS * 2,
        SMALL_PACK_WINDOW,
    )
    expected_first = quantize_complex_numpy(
        feed_channel[0, 0, 0:SMALL_PACK_WINDOW][np.newaxis, :],
        INT4_COMPONENT_BITS,
        1.0,
    )[0]
    expected_third = quantize_complex_numpy(
        feed_channel[0, 1, 0:SMALL_PACK_WINDOW][np.newaxis, :],
        INT4_COMPONENT_BITS,
        1.0,
    )[0]
    np.testing.assert_array_equal(packed.packed[0, 0], expected_first)
    np.testing.assert_array_equal(packed.packed[0, 2], expected_third)
    assert packed.input_layout["combine_mode"] == COMBINE_MODE_COMBINED_STREAMS
    assert packed.stream_map[2]["feed_index"] == 1
    assert packed.stream_map[1]["selected_channel_index"] == 180
    assert packed.quantization["scale_by_stream"] == [1.0, 1.0, 1.0, 1.0]


def test_pack_channelized_streams_per_stream_diagnostic_shape() -> None:
    feed_channel = np.ones(
        (SMALL_PACK_FEEDS, SMALL_PACK_CHANNELS, SMALL_PACK_FRAME_SIZE),
        dtype=np.complex64,
    )

    packed = pack_channelized_streams_for_detector(
        feed_channel,
        frame_size_samples=SMALL_PACK_FRAME_SIZE,
        detector_window_samples=SMALL_PACK_WINDOW,
        quantization_scale_mode="provided",
        scale=1.0,
        combine_mode=COMBINE_MODE_PER_STREAM_DIAGNOSTIC,
    )

    assert packed.packed.shape == (
        SMALL_PACK_FEEDS * SMALL_PACK_CHANNELS,
        SMALL_PACK_FRAME_SIZE // SMALL_PACK_WINDOW,
        SMALL_PACK_WINDOW,
    )
    assert packed.input_layout["combine_mode"] == COMBINE_MODE_PER_STREAM_DIAGNOSTIC
