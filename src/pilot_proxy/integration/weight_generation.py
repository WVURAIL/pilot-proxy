# coding=utf-8
"""Generate packed detector weight ROMs from receiver profiles."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
)
from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.detector_geometry import (
    SPECTRAL_SENSE_NORMAL,
    spectral_sense_requires_time_reversal,
)
from pilot_proxy.detector_weights import (
    CRC32_UNSIGNED_MASK,
    CRC_SIZE,
    HEADER_FIXED_FMT,
    HEADER_FIXED_SIZE,
    WEIGHT_MAGIC,
    WEIGHT_VERSION,
)

from .detector_core import DetectorCoreProfile
from .receiver_profile import (
    ReceiverProfile,
    receiver_frequency_to_channel,
    receiver_profile_hash,
)

DEFAULT_DOPPLER_TOL_HZ = math.nan
HZ_PER_MHZ = 1.0e6
WEIGHT_TERM_TARGET = 0
WEIGHT_TERM_REF_LOWER = 1
WEIGHT_TERM_REF_UPPER = 2
NORMALIZED_NYQUIST_CENTER = 0.5
REFERENCE_SELECTION_SCORE_NOMINAL = 0.6
REFERENCE_SELECTION_SCORE_ADAPTIVE = 0.5
REFERENCE_SELECTION_METHOD = "adaptive_circular_reference_placement_v1"
REFERENCE_PLACEMENT_STATUS_NOMINAL = "nominal"
REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED = "edge_wrapped"
REFERENCE_PLACEMENT_STATUS_DC_SHIFTED = "dc_shifted"
REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED_AND_DC_SHIFTED = (
    "edge_wrapped_and_dc_shifted"
)
REFERENCE_PLACEMENT_REASON_NOMINAL = "nominal"
REFERENCE_PLACEMENT_REASON_EDGE_WRAPPED = "edge_reference_wrapped"
REFERENCE_PLACEMENT_REASON_DC_SHIFTED = "dc_reference_shifted_away"
REFERENCE_FORBIDDEN_DC_NORMALIZED = NORMALIZED_NYQUIST_CENTER
REFERENCE_FORBIDDEN_COLLISION_RULE = (
    "circular_normalized_distance <= 0.5 / detector_window_samples"
)


@dataclass(frozen=True)
class DetectorCoreLayout:
    detector_window_samples: int
    skipped_guard_bins: int
    reference_offset_bins: int


def _physical_channels_from_range(value: str) -> list[int]:
    text = str(value).strip()
    if ":" in text:
        start, stop = [int(part.strip()) for part in text.split(":", 1)]
        step = 1 if stop >= start else -1
        return list(range(start, stop + step, step))
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(text)]


def parse_physical_channel_selection(
    *,
    physical_channels: Sequence[int] | None = None,
    physical_channel_range: str | None = None,
) -> list[int]:
    """Parse explicit and range-based ATSC physical-channel selections."""
    out: list[int] = []
    if physical_channel_range is not None:
        out.extend(_physical_channels_from_range(physical_channel_range))
    if physical_channels is not None:
        out.extend(int(channel) for channel in physical_channels)
    seen: set[int] = set()
    unique: list[int] = []
    for channel in out:
        if int(channel) in seen:
            continue
        seen.add(int(channel))
        unique.append(int(channel))
    if not unique:
        raise ValueError("at least one physical channel must be selected.")
    return unique


def _packed_weight_vector(
    normalized_frequency: float,
    *,
    detector_window_samples: int,
    bits_per_component: int,
) -> np.ndarray:
    sample_index = np.arange(int(detector_window_samples), dtype=np.float64)
    max_int = float((1 << (int(bits_per_component) - 1)) - 1)
    weights = max_int * np.exp(
        -2j * math.pi * float(normalized_frequency) * sample_index
    )
    return quantize_complex_numpy(
        weights[np.newaxis, :].astype(np.complex64),
        int(bits_per_component),
        1.0,
    )[0]


def _wrap_normalized_frequency(value: float) -> float:
    return float(float(value) % 1.0)


def _circular_normalized_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 1.0
    return float(min(delta, 1.0 - delta))


def _reference_collides_with_dc(
    normalized_frequency: float,
    *,
    detector_window_samples: int,
) -> bool:
    fine_bin = 1.0 / float(detector_window_samples)
    return bool(
        _circular_normalized_distance(
            float(normalized_frequency),
            REFERENCE_FORBIDDEN_DC_NORMALIZED,
        )
        <= 0.5 * fine_bin
    )


def _dc_in_skipped_guard(
    target_normalized_frequency: float,
    *,
    reference_offset_bins: int,
    detector_window_samples: int,
) -> bool:
    fine_bin = 1.0 / float(detector_window_samples)
    signed_bins_to_dc = (
        float(REFERENCE_FORBIDDEN_DC_NORMALIZED)
        - float(target_normalized_frequency)
    ) / fine_bin
    selected = int(reference_offset_bins)
    if selected == 0:
        return False
    same_side = math.copysign(1.0, signed_bins_to_dc) == math.copysign(
        1.0, selected
    )
    return bool(same_side and 0.0 < abs(signed_bins_to_dc) < abs(selected))


def _resolve_one_reference_offset(
    target_normalized_frequency: float,
    *,
    detector_window_samples: int,
    requested_offset_bins: int,
    direction: int,
) -> dict[str, Any]:
    if int(direction) not in {-1, 1}:
        raise ValueError("direction must be -1 or +1")
    requested = abs(int(requested_offset_bins))
    if requested <= 0:
        raise ValueError("requested_offset_bins must be positive")

    fine_bin = 1.0 / float(detector_window_samples)
    offset = int(direction) * requested
    wrapped = False
    dc_shifted = False
    requested_raw = float(target_normalized_frequency) + offset * fine_bin
    requested_wrapped = not (0.0 <= requested_raw < 1.0)
    requested_normalized = _wrap_normalized_frequency(requested_raw)
    requested_dc_collision = _reference_collides_with_dc(
        requested_normalized,
        detector_window_samples=int(detector_window_samples),
    )

    for _ in range(int(detector_window_samples)):
        raw = float(target_normalized_frequency) + offset * fine_bin
        if not (0.0 <= raw < 1.0):
            wrapped = True
        normalized = _wrap_normalized_frequency(raw)
        if not _reference_collides_with_dc(
            normalized,
            detector_window_samples=int(detector_window_samples),
        ):
            return {
                "offset_bins": int(offset),
                "normalized_frequency": float(normalized),
                "edge_wrapped": bool(wrapped),
                "dc_shifted": bool(dc_shifted),
                "requested_offset_bins": int(direction) * requested,
                "requested_normalized_frequency": float(requested_normalized),
                "requested_edge_wrapped": bool(requested_wrapped),
                "requested_dc_collision": bool(requested_dc_collision),
            }
        dc_shifted = True
        offset += int(direction)

    raise ValueError(
        "could not choose a reference bin that avoids the forbidden DC bin"
    )


def _reference_placement(
    target_normalized_frequency: float,
    *,
    detector_window_samples: int,
    reference_offset_bins: int,
) -> dict[str, Any]:
    lower = _resolve_one_reference_offset(
        target_normalized_frequency,
        detector_window_samples=int(detector_window_samples),
        requested_offset_bins=int(reference_offset_bins),
        direction=-1,
    )
    upper = _resolve_one_reference_offset(
        target_normalized_frequency,
        detector_window_samples=int(detector_window_samples),
        requested_offset_bins=int(reference_offset_bins),
        direction=1,
    )
    edge_wrapped = bool(lower["edge_wrapped"] or upper["edge_wrapped"])
    dc_shifted = bool(lower["dc_shifted"] or upper["dc_shifted"])
    requested_dc_collision = bool(
        lower["requested_dc_collision"] or upper["requested_dc_collision"]
    )
    forbidden_tone_in_skipped_guard = bool(
        _dc_in_skipped_guard(
            target_normalized_frequency,
            reference_offset_bins=int(lower["offset_bins"]),
            detector_window_samples=int(detector_window_samples),
        )
        or _dc_in_skipped_guard(
            target_normalized_frequency,
            reference_offset_bins=int(upper["offset_bins"]),
            detector_window_samples=int(detector_window_samples),
        )
    )
    if edge_wrapped and dc_shifted:
        status = REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED_AND_DC_SHIFTED
    elif edge_wrapped:
        status = REFERENCE_PLACEMENT_STATUS_EDGE_WRAPPED
    elif dc_shifted:
        status = REFERENCE_PLACEMENT_STATUS_DC_SHIFTED
    else:
        status = REFERENCE_PLACEMENT_STATUS_NOMINAL
    reasons: list[str] = []
    if edge_wrapped:
        reasons.append(REFERENCE_PLACEMENT_REASON_EDGE_WRAPPED)
    if dc_shifted:
        reasons.append(REFERENCE_PLACEMENT_REASON_DC_SHIFTED)
    warnings: list[str] = []
    if bool(lower["edge_wrapped"]):
        warnings.append("lower reference wrapped across coarse-channel edge")
    if bool(upper["edge_wrapped"]):
        warnings.append("upper reference wrapped across coarse-channel edge")
    if bool(lower["dc_shifted"]):
        warnings.append("lower reference shifted away from coarse-channel DC")
    if bool(upper["dc_shifted"]):
        warnings.append("upper reference shifted away from coarse-channel DC")

    return {
        "lower": lower,
        "upper": upper,
        "placement_status": status,
        "adaptive_reference_placement": bool(edge_wrapped or dc_shifted),
        "reference_placement_reason": (
            ",".join(reasons) if reasons else REFERENCE_PLACEMENT_REASON_NOMINAL
        ),
        "placement_warnings": "; ".join(warnings),
        "edge_reference_wrapped": bool(edge_wrapped),
        "dc_reference_collision": bool(requested_dc_collision),
        "dc_reference_shifted": bool(dc_shifted),
        "forbidden_tone_in_skipped_guard": bool(forbidden_tone_in_skipped_guard),
    }


def target_layout(
    *,
    physical_channel: int,
    profile: ReceiverProfile,
    core: DetectorCoreLayout | DetectorCoreProfile,
) -> dict[str, Any]:
    pilot_hz = physical_channel_to_pilot_hz(int(physical_channel))
    selection = receiver_frequency_to_channel(pilot_hz, profile)
    target_normalized = (
        NORMALIZED_NYQUIST_CENTER
        + float(selection.fine_bin_offset_hz) / float(profile.coarse_channel_width_hz)
    )
    if _reference_collides_with_dc(
        target_normalized,
        detector_window_samples=int(core.detector_window_samples),
    ):
        raise ValueError(
            "target pilot bin collides with the forbidden coarse-channel DC bin "
            f"for physical channel {int(physical_channel)}; target bins cannot be "
            "moved safely."
        )
    placement = _reference_placement(
        target_normalized,
        detector_window_samples=int(core.detector_window_samples),
        reference_offset_bins=int(core.reference_offset_bins),
    )
    lower = placement["lower"]
    upper = placement["upper"]
    lower_offset = int(lower["offset_bins"])
    upper_offset = int(upper["offset_bins"])
    fine_bin_width_hz = (
        float(profile.coarse_channel_width_hz)
        / float(core.detector_window_samples)
    )
    lower_normalized = float(lower["normalized_frequency"])
    upper_normalized = float(upper["normalized_frequency"])
    lower_requested_normalized = float(lower["requested_normalized_frequency"])
    upper_requested_normalized = float(upper["requested_normalized_frequency"])
    target_offset_hz = float(selection.fine_bin_offset_hz)
    lower_reference_offset_hz = (
        lower_normalized - NORMALIZED_NYQUIST_CENTER
    ) * float(profile.coarse_channel_width_hz)
    upper_reference_offset_hz = (
        upper_normalized - NORMALIZED_NYQUIST_CENTER
    ) * float(profile.coarse_channel_width_hz)
    lower_requested_reference_offset_hz = (
        lower_requested_normalized - NORMALIZED_NYQUIST_CENTER
    ) * float(profile.coarse_channel_width_hz)
    upper_requested_reference_offset_hz = (
        upper_requested_normalized - NORMALIZED_NYQUIST_CENTER
    ) * float(profile.coarse_channel_width_hz)
    if not (0.0 <= lower_normalized < 1.0 and 0.0 <= upper_normalized < 1.0):
        raise ValueError(
            "could not choose in-channel reference bins for physical channel "
            f"{int(physical_channel)}."
        )
    min_selected_offset = min(abs(lower_offset), abs(upper_offset))
    return {
        "physical_channel": int(physical_channel),
        "dtv_pilot_hz": float(pilot_hz),
        "target_frequency_mhz": float(pilot_hz / HZ_PER_MHZ),
        "coarse_channel_index": int(selection.coarse_channel_index),
        "coarse_channel_center_hz": float(selection.coarse_channel_center_hz),
        "fine_bin_offset_hz": float(selection.fine_bin_offset_hz),
        "detector_fine_bin_width_hz": float(fine_bin_width_hz),
        "target_offset_hz": float(target_offset_hz),
        "target_normalized_frequency": float(target_normalized),
        "lower_reference_normalized_frequency": float(lower_normalized),
        "upper_reference_normalized_frequency": float(upper_normalized),
        "lower_reference_offset_hz": float(lower_reference_offset_hz),
        "upper_reference_offset_hz": float(upper_reference_offset_hz),
        "lower_reference_relative_to_target_hz": float(
            lower_offset * fine_bin_width_hz
        ),
        "upper_reference_relative_to_target_hz": float(
            upper_offset * fine_bin_width_hz
        ),
        "symmetric_reference_offsets": bool(abs(lower_offset) == abs(upper_offset)),
        "reference_offset_bins_requested": int(core.reference_offset_bins),
        "skipped_guard_bins_requested": int(core.skipped_guard_bins),
        "reference_offset_bins_min_empirical": int(min_selected_offset),
        "skipped_guard_bins_min_empirical": int(max(0, min_selected_offset - 1)),
        "reference_offset_bins_selected": int(min_selected_offset),
        "skipped_guard_bins_selected": int(max(0, min_selected_offset - 1)),
        "strict_reference_offset_pass": bool(
            min_selected_offset >= int(core.reference_offset_bins)
        ),
        "lower_reference_offset_bins": int(lower_offset),
        "upper_reference_offset_bins": int(upper_offset),
        "lower_skipped_guard_bins": int(max(0, abs(lower_offset) - 1)),
        "upper_skipped_guard_bins": int(max(0, abs(upper_offset) - 1)),
        "reference_selection_score": float(
            REFERENCE_SELECTION_SCORE_ADAPTIVE
            if placement["adaptive_reference_placement"]
            else REFERENCE_SELECTION_SCORE_NOMINAL
        ),
        "adaptive_reference_placement": bool(
            placement["adaptive_reference_placement"]
        ),
        "reference_placement_reason": str(placement["reference_placement_reason"]),
        "reference_selection_method": REFERENCE_SELECTION_METHOD,
        "reference_placement_status": str(placement["placement_status"]),
        "placement_warnings": str(placement["placement_warnings"]),
        "edge_reference_wrapped": bool(placement["edge_reference_wrapped"]),
        "dc_reference_collision": bool(placement["dc_reference_collision"]),
        "dc_reference_shifted": bool(placement["dc_reference_shifted"]),
        "forbidden_tone_in_skipped_guard": bool(
            placement["forbidden_tone_in_skipped_guard"]
        ),
        "lower_reference_edge_wrapped": bool(lower["edge_wrapped"]),
        "upper_reference_edge_wrapped": bool(upper["edge_wrapped"]),
        "lower_reference_dc_shifted": bool(lower["dc_shifted"]),
        "upper_reference_dc_shifted": bool(upper["dc_shifted"]),
        "lower_reference_requested_offset_bins": int(lower["requested_offset_bins"]),
        "upper_reference_requested_offset_bins": int(upper["requested_offset_bins"]),
        "lower_reference_requested_normalized_frequency": float(
            lower_requested_normalized
        ),
        "upper_reference_requested_normalized_frequency": float(
            upper_requested_normalized
        ),
        "lower_reference_requested_offset_hz": float(
            lower_requested_reference_offset_hz
        ),
        "upper_reference_requested_offset_hz": float(
            upper_requested_reference_offset_hz
        ),
        "lower_reference_requested_relative_to_target_hz": float(
            int(lower["requested_offset_bins"]) * fine_bin_width_hz
        ),
        "upper_reference_requested_relative_to_target_hz": float(
            int(upper["requested_offset_bins"]) * fine_bin_width_hz
        ),
        "lower_reference_requested_edge_wrapped": bool(
            lower["requested_edge_wrapped"]
        ),
        "upper_reference_requested_edge_wrapped": bool(
            upper["requested_edge_wrapped"]
        ),
        "lower_reference_requested_dc_collision": bool(
            lower["requested_dc_collision"]
        ),
        "upper_reference_requested_dc_collision": bool(
            upper["requested_dc_collision"]
        ),
        "max_lower_reference_offset_bins": int(
            math.floor(target_normalized * int(core.detector_window_samples))
        ),
        "max_upper_reference_offset_bins": int(
            math.floor((1.0 - target_normalized) * int(core.detector_window_samples))
        ),
    }


_target_layout = target_layout


def generate_weight_table_from_receiver_profile(
    *,
    profile: ReceiverProfile,
    core: DetectorCoreProfile,
    physical_channels: Sequence[int],
    weight_coordinate_system: str = WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return the weight table and manifest target-layout entries."""
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    generation_profile = (
        replace(profile, spectral_sense=SPECTRAL_SENSE_NORMAL)
        if coordinate_system == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
        else profile
    )
    table = np.zeros(
        (
            int(generation_profile.num_coarse_channels),
            int(core.num_weight_terms),
            int(core.detector_window_samples),
        ),
        dtype=np.int8,
    )
    layouts: list[dict[str, Any]] = []
    for channel in physical_channels:
        layout = target_layout(
            physical_channel=int(channel),
            profile=generation_profile,
            core=core,
        )
        coarse_index = int(cast(int, layout["coarse_channel_index"]))
        if np.any(table[coarse_index]):
            raise ValueError(
                "multiple physical channels map to the same coarse channel: "
                f"coarse_channel_index={coarse_index}."
            )
        table[coarse_index, WEIGHT_TERM_TARGET] = _packed_weight_vector(
            float(layout["target_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        table[coarse_index, WEIGHT_TERM_REF_LOWER] = _packed_weight_vector(
            float(layout["lower_reference_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        table[coarse_index, WEIGHT_TERM_REF_UPPER] = _packed_weight_vector(
            float(layout["upper_reference_normalized_frequency"]),
            detector_window_samples=int(core.detector_window_samples),
            bits_per_component=int(core.sample_bits_per_component),
        )
        layouts.append(layout)
    return np.ascontiguousarray(table), layouts


def _weight_header_bytes(
    *,
    core: DetectorCoreProfile,
    profile: ReceiverProfile,
    fine_bin_width_hz: float,
    weights_bytes: bytes,
) -> bytes:
    reference_name = b"profile_generated"
    profile_name = profile.name.encode("utf-8")
    header_size = HEADER_FIXED_SIZE + len(reference_name) + len(profile_name) + CRC_SIZE
    fixed_without_crc = struct.pack(
        HEADER_FIXED_FMT,
        WEIGHT_MAGIC,
        int(WEIGHT_VERSION),
        int(header_size),
        int(core.detector_window_samples),
        int(core.num_weight_terms),
        int(core.reference_offset_bins),
        int(core.sample_bits_per_component),
        int(profile.num_coarse_channels),
        float(DEFAULT_DOPPLER_TOL_HZ),
        float(fine_bin_width_hz),
        int(len(reference_name)),
        int(len(profile_name)),
    )
    header_zero_crc = fixed_without_crc + reference_name + profile_name + struct.pack(
        "<I",
        0,
    )
    crc = zlib.crc32(header_zero_crc)
    crc = zlib.crc32(weights_bytes, crc) & CRC32_UNSIGNED_MASK
    return fixed_without_crc + reference_name + profile_name + struct.pack("<I", crc)


def write_weight_bank_from_receiver_profile(
    *,
    output_path: str | Path,
    profile: ReceiverProfile,
    core: DetectorCoreProfile,
    physical_channels: Sequence[int],
    weight_coordinate_system: str = WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
) -> dict[str, Any]:
    """Write a packed weight bank plus the adjacent manifest and return manifest."""
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=physical_channels,
        weight_coordinate_system=coordinate_system,
    )
    weights_bytes = table.tobytes(order="C")
    header = _weight_header_bytes(
        core=core,
        profile=profile,
        fine_bin_width_hz=float(profile.bin_enbw_hz),
        weights_bytes=weights_bytes,
    )
    payload = header + weights_bytes
    output.write_bytes(payload)
    manifest = {
        "schema_version": "fstat_weight_manifest_v2",
        "weight_format_version": int(WEIGHT_VERSION),
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": input_coordinate_system_for_weight_coordinate(
            coordinate_system
        ),
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": bool(
                coordinate_system == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
                and spectral_sense_requires_time_reversal(profile.spectral_sense)
            ),
        },
        "kernel_spec": {
            "detector_window_samples": int(core.detector_window_samples),
            "num_weight_terms": int(core.num_weight_terms),
            "reference_offset_bins": int(core.reference_offset_bins),
            "skipped_guard_bins": int(core.skipped_guard_bins),
            "packed_complex_bits": int(core.packed_complex_bits),
            "sample_bits_per_component": int(core.sample_bits_per_component),
        },
        "receiver_profile_hash": receiver_profile_hash(profile),
        "receiver_profile": profile.to_nested_dict(),
        "physical_channels": [int(channel) for channel in physical_channels],
        "forbidden_tone_policy": {
            "forbidden_tone": "coarse_channel_dc",
            "forbidden_tone_normalized": float(REFERENCE_FORBIDDEN_DC_NORMALIZED),
            "forbidden_collision_rule": REFERENCE_FORBIDDEN_COLLISION_RULE,
            "forbidden_collision_half_width_bins": 0.5,
            "forbidden_collision_half_width_normalized": float(
                0.5 / float(core.detector_window_samples)
            ),
        },
        "target_reference_layout": layouts,
        "artifacts": {
            "weights_path": str(output),
            "manifest_path": str(output.with_suffix(output.suffix + ".manifest.json")),
            "weights_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "DetectorCoreLayout",
    "generate_weight_table_from_receiver_profile",
    "parse_physical_channel_selection",
    "target_layout",
    "write_weight_bank_from_receiver_profile",
]
