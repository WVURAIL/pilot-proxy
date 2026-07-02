# coding=utf-8
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.provenance import file_sha256
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
# noinspection PyProtectedMember
from pilot_proxy.chime.runner import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WeightBankLike,
    _reference_placement_summary,
    _validate_detector_window_contract,
    _weight_coordinate_metadata,
    build_parser,
    run_chime_analysis,
)


def _encode_offset_binary(real: np.ndarray, imag: np.ndarray | int = 0) -> np.ndarray:
    r = (np.asarray(real, dtype=np.int16) + 8) & 0x0F
    i = (np.asarray(imag, dtype=np.int16) + 8) & 0x0F
    packed = np.asarray((r << 4) | i, dtype=np.uint8)
    return packed.astype(np.uint8, copy=False)


def _tone_streams(
    *,
    frequency_hz: float,
    sample_rate_hz: float,
    num_samples: int,
    phases: np.ndarray,
) -> np.ndarray:
    n = np.arange(int(num_samples), dtype=np.float64)
    tone = np.exp(2j * np.pi * float(frequency_hz) * n / float(sample_rate_hz))
    return np.asarray(
        [np.exp(1j * phase) * tone for phase in phases],
        dtype=np.complex64,
    )


def test_chime_runner_parser_does_not_expose_detector_window_samples() -> None:
    parser = build_parser()

    assert "--detector-window-samples" not in parser.format_help()


def _fake_kernel(detector_window_samples: int):
    specs = SimpleNamespace(
        K=detector_window_samples,
        N=3,
        bits=4,
        reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": detector_window_samples,
            "num_weight_terms": 3,
            "sample_bits_per_component": 4,
            "reference_offset_bins": 2,
        },
    )
    version = SimpleNamespace(as_string=lambda: "test")
    return SimpleNamespace(specs=specs, version=version)


def test_chime_runner_small_writes_expected_shapes(tmp_path) -> None:
    input_dir = tmp_path / "input"
    ch_dir = input_dir / "ch0844"
    ch_dir.mkdir(parents=True)
    data = _encode_offset_binary(
        np.asarray(
            [
                [-3, -2],
                [-1, 0],
                [1, 2],
                [3, 4],
                [4, 3],
                [2, 1],
                [0, -1],
                [-2, -3],
            ],
            dtype=np.int16,
        )
    )
    with h5py.File(ch_dir / "001.h5", "w") as h5:
        h5.attrs["freq"] = 470.3125
        h5.attrs["freq_id"] = 844
        ds = h5.create_dataset("baseband", data=data)
        ds.attrs["axis"] = np.asarray(["time", "input"], dtype=object)

    profile = default_reference_receiver_profile(
        frame_size_samples=4,
        num_input_streams=2,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_nested_dict()), encoding="utf-8")

    def fake_detector(**kwargs):
        packed = kwargs["packed"]
        batch = int(packed.shape[0])
        return {
            "batch": batch,
            "detector_rows_per_block": int(packed.shape[1]),
            "rational_overflow_count": 0,
            "results": [
                {
                    "block_index": index,
                    "mask": 1,
                    "p_target_u64": 30 + index,
                    "p_ref_sum_u64": 20,
                }
                for index in range(batch)
            ],
        }

    output_dir = tmp_path / "run"
    run_chime_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        receiver_profile_path=profile_path,
        stream_map_path=None,
        physical_channels=[14],
        frame_size_samples=4,
        detector_window_samples=2,
        frames_per_chunk=1,
        max_frames=2,
        kernel=_fake_kernel(2),
        detector_fn=fake_detector,
        weights_by_channel={14: np.ones((3, 2), dtype=np.int8)},
    )

    detector = np.load(output_dir / "chime_detector_outputs.npz")
    cache = np.load(output_dir / "chime_spectrogram_cache.npz")
    reductions = np.load(output_dir / "chime_reductions_10s.npz")
    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))

    assert detector["p_target_u64"].shape == (2, 1)
    assert detector["p_ref_sum_u64"].shape == (2, 1)
    assert detector["mask"].shape == (2, 1)
    assert float(detector["chime_frequency_hz"][0]) == pytest.approx(470_312_500.0)
    assert cache["baseband_power_linear"].shape == (2, 1)
    assert float(cache["chime_frequency_hz"][0]) == pytest.approx(470_312_500.0)
    assert reductions["input_power_mean"].shape == (1, 1)
    assert stats["detector_rows_per_frame"] == 4
    assert stats["chime_frequency_hz_by_pilot"] == [470_312_500.0]
    assert stats["combine_mode"] == "all_rows_summed_before_ratio"
    assert stats["weight_coordinate"]["effective_weight_coordinate_system"] == (
        "caller_supplied_weights"
    )
    assert stats["receiver_profile_sha256"] == file_sha256(profile_path)
    assert stats["input_manifest_sha256"] == file_sha256(
        output_dir / "input_manifest.json"
    )
    assert stats["stream_map_sha256"] is None
    assert stats["weights_sha256"] is None
    assert stats["weight_manifest_sha256"] is None
    assert "kernel_library_sha256" in stats
    run_config = json.loads((output_dir / "run_config.json").read_text("utf-8"))
    assert run_config["provenance"]["receiver_profile_sha256"] == (
        stats["receiver_profile_sha256"]
    )
    assert run_config["schema_version"] == "fstat_chime_run_config_v2"
    assert stats["schema_version"] == "fstat_chime_stats_v2"
    assert run_config["detector_contract"] == stats["detector_contract"]
    assert stats["detector_contract"]["schema_version"] == (
        "pilotproxy_chime_detector_contract_v1"
    )
    assert stats["detector_contract"]["per_frequency_threshold"] is False
    assert stats["detector_contract"]["mask_source"] == "positive_excess"
    assert stats["detector_contract"]["weight_coordinate_system"] == (
        WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    )
    assert stats["detector_contract"]["input_coordinate_system"] == (
        "post_spectral_sense_normalized"
    )
    assert (
        stats["detector_contract"]["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is False
    )


def test_chime_runner_rejects_detector_window_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match kernel specs"):
        _validate_detector_window_contract(
            requested_detector_window_samples=256,
            selected_physical_channels=[14],
            kernel=_fake_kernel(128),
            weight_bank=None,
            weights_by_channel={14: np.ones((3, 128), dtype=np.int8)},
        )


def test_weight_coordinate_rejects_missing_coordinate_manifest() -> None:
    weight_bank = cast(
        WeightBankLike,
        cast(
            object,
            SimpleNamespace(
                path="weights/chime_raw.bin",
                manifest={
                    "receiver_profile": {
                        "channelizer": {
                            "frequency_axis": {
                                "spectral_sense": "inverted",
                            },
                        },
                    },
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires weight_coordinate_system"):
        _weight_coordinate_metadata(
            weight_bank=weight_bank,
            input_spectral_sense="inverted",
        )


def test_weight_coordinate_rejects_raw_inverted_manifest() -> None:
    weight_bank = cast(
        WeightBankLike,
        cast(
            object,
            SimpleNamespace(
                path="weights/chime_raw.bin",
                manifest={
                    "weight_coordinate_system": "raw_input_frequency_coordinate",
                    "receiver_profile": {
                        "channelizer": {
                            "frequency_axis": {
                                "spectral_sense": "inverted",
                            },
                        },
                    },
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="detector-coordinate weights"):
        _weight_coordinate_metadata(
            weight_bank=weight_bank,
            input_spectral_sense="inverted",
        )


def test_weight_coordinate_accepts_declared_post_normalization_manifest() -> None:
    weight_bank = cast(
        WeightBankLike,
        cast(
            object,
            SimpleNamespace(
                path="weights/chime_detector_coordinate.bin",
                manifest={
                    "weight_coordinate_system": WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
                    "input_preprocessing": {
                        "time_reverse_detector_windows_before_kernel": True,
                    },
                    "receiver_profile": {
                        "channelizer": {
                            "frequency_axis": {
                                "spectral_sense": "inverted",
                            },
                        },
                    },
                },
            ),
        ),
    )

    metadata = _weight_coordinate_metadata(
        weight_bank=weight_bank,
        input_spectral_sense="inverted",
    )

    assert metadata["effective_weight_coordinate_system"] == (
        WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
    )
    assert metadata["input_requires_time_reversal"] is True


def test_reference_placement_summary_compacts_manifest() -> None:
    manifest = {
        "kernel_spec": {
            "reference_offset_bins": 2,
            "skipped_guard_bins": 1,
        },
        "forbidden_tone_policy": {
            "forbidden_tone": "coarse_channel_dc",
        },
        "target_reference_layout": [
            {
                "physical_channel": 14,
                "reference_placement_status": "nominal",
                "placement_warnings": "",
                "lower_reference_offset_bins": -2,
                "upper_reference_offset_bins": 2,
                "lower_reference_relative_to_target_hz": -6103.515625,
                "upper_reference_relative_to_target_hz": 6103.515625,
                "lower_reference_edge_wrapped": False,
                "upper_reference_edge_wrapped": False,
                "lower_reference_dc_shifted": False,
                "upper_reference_dc_shifted": False,
                "edge_reference_wrapped": False,
                "dc_reference_shifted": False,
                "forbidden_tone_in_skipped_guard": True,
            },
            {
                "physical_channel": 21,
                "reference_placement_status": "edge_wrapped",
                "placement_warnings": "lower reference wrapped across coarse-channel edge",
                "lower_reference_offset_bins": -2,
                "upper_reference_offset_bins": 2,
                "lower_reference_relative_to_target_hz": -6103.515625,
                "upper_reference_relative_to_target_hz": 6103.515625,
                "lower_reference_edge_wrapped": True,
                "upper_reference_edge_wrapped": False,
                "lower_reference_dc_shifted": False,
                "upper_reference_dc_shifted": False,
                "edge_reference_wrapped": True,
                "dc_reference_shifted": False,
                "forbidden_tone_in_skipped_guard": False,
            },
            {
                "physical_channel": 99,
                "reference_placement_status": "nominal",
            },
        ],
    }

    summary = _reference_placement_summary(
        manifest,
        np.asarray([14, 21], dtype=np.int32),
    )

    assert summary is not None
    assert summary["reference_offset_bins"] == 2
    assert summary["skipped_guard_bins"] == 1
    assert summary["reference_placement_status"] == "mixed:edge_wrapped;nominal"
    assert summary["num_channels_with_adaptive_reference"] == 1
    assert summary["channels_with_adaptive_reference"] == [21]
    assert summary["num_edge_wrapped_references"] == 1
    assert summary["channels_with_edge_wrapped_reference"] == [21]
    assert summary["num_forbidden_tone_in_skipped_guard"] == 1
    assert summary["channels_with_forbidden_tone_in_skipped_guard"] == [14]
    assert summary["forbidden_tone_policy"]["forbidden_tone"] == "coarse_channel_dc"
    assert [row["physical_channel"] for row in summary["by_channel"]] == [14, 21]
