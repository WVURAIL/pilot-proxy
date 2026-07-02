# coding=utf-8
"""Export compact runtime weight bundles for future deployment paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    build_chime_detector_contract,
    detector_contract_sha256,
    input_coordinate_system_for_weight_coordinate,
    norm_corrected_mu0,
    normalize_weight_coordinate_system,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import spectral_sense_requires_time_reversal
from pilot_proxy.integration import (
    load_detector_core_profile,
    load_receiver_profile,
    parse_physical_channel_selection,
)
from pilot_proxy.integration.receiver_profile import receiver_profile_hash
from pilot_proxy.integration.weight_generation import (
    generate_weight_table_from_receiver_profile,
)
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.provenance import file_sha256

RUNTIME_WEIGHT_MANIFEST_SCHEMA_VERSION = "fstat_runtime_weights_manifest_v1"
RUNTIME_PILOT_PROFILES_SCHEMA_VERSION = "fstat_runtime_pilot_profiles_v1"
RUNTIME_BUNDLE_VALIDATION_SCHEMA_VERSION = "fstat_runtime_bundle_validation_v1"
DEFAULT_WEIGHTS_FILENAME = "weights.bin"
DEFAULT_WEIGHTS_MANIFEST_FILENAME = "weights.manifest.json"
DEFAULT_DETECTOR_CONTRACT_FILENAME = "detector_contract.json"
DEFAULT_PILOT_PROFILES_FILENAME = "pilot_profiles.json"
DEFAULT_SHA256SUMS_FILENAME = "sha256sums.txt"
REQUIRED_RUNTIME_BUNDLE_FILES = (
    DEFAULT_DETECTOR_CONTRACT_FILENAME,
    DEFAULT_PILOT_PROFILES_FILENAME,
    DEFAULT_WEIGHTS_FILENAME,
    DEFAULT_WEIGHTS_MANIFEST_FILENAME,
    DEFAULT_SHA256SUMS_FILENAME,
)


def _reference_placement_summary_from_layouts(
    *,
    core: Any,
    layouts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    adaptive = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("adaptive_reference_placement", False))
    ]
    dc_shifted = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("dc_reference_shifted", False))
    ]
    edge_wrapped = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("edge_reference_wrapped", False))
    ]
    skipped_guard = [
        int(row["physical_channel"])
        for row in layouts
        if bool(row.get("forbidden_tone_in_skipped_guard", False))
    ]
    statuses = sorted({str(row.get("reference_placement_status", "")) for row in layouts})
    status = statuses[0] if len(statuses) == 1 else "mixed:" + ";".join(statuses)
    return {
        "reference_offset_bins": int(core.reference_offset_bins),
        "skipped_guard_bins": int(core.skipped_guard_bins),
        "reference_placement_status": status,
        "num_channels_with_adaptive_reference": int(len(adaptive)),
        "channels_with_adaptive_reference": adaptive,
        "num_dc_shifted_references": int(len(dc_shifted)),
        "channels_with_dc_shifted_reference": dc_shifted,
        "num_edge_wrapped_references": int(len(edge_wrapped)),
        "channels_with_edge_wrapped_reference": edge_wrapped,
        "num_forbidden_tone_in_skipped_guard": int(len(skipped_guard)),
        "channels_with_forbidden_tone_in_skipped_guard": skipped_guard,
        "forbidden_tone_policy": {
            "forbidden_tone": "coarse_channel_dc",
        },
    }


def _write_sha256sums(output_dir: Path, filenames: Sequence[str]) -> Path:
    path = output_dir / DEFAULT_SHA256SUMS_FILENAME
    lines = []
    for name in filenames:
        digest = file_sha256(output_dir / name)
        lines.append(f"{digest}  {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _add_error(errors: list[dict[str, str]], check: str, message: str) -> None:
    errors.append({"severity": "error", "check": str(check), "message": str(message)})


def _load_json_object(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _add_error(errors, f"required_file.{path.name}", f"missing {path}")
        return {}
    except json.JSONDecodeError as exc:
        _add_error(errors, f"json.{path.name}", f"invalid JSON: {exc}")
        return {}
    if not isinstance(data, dict):
        _add_error(errors, f"json.{path.name}", "top-level JSON value is not an object")
        return {}
    return data


def _read_sha256sums(path: Path, errors: list[dict[str, str]]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _add_error(errors, "required_file.sha256sums.txt", f"missing {path}")
        return {}
    out: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            _add_error(
                errors,
                "sha256sums.format",
                f"invalid line {line_number}: {line!r}",
            )
            continue
        digest, filename = parts
        out[filename.strip()] = digest.strip()
    return out


def _validate_expected_sha256sums(
    *,
    bundle_dir: Path,
    expected: dict[str, str],
    errors: list[dict[str, str]],
) -> None:
    for filename in REQUIRED_RUNTIME_BUNDLE_FILES:
        if filename == DEFAULT_SHA256SUMS_FILENAME:
            continue
        path = bundle_dir / filename
        if not path.exists():
            continue
        expected_digest = expected.get(filename)
        if expected_digest is None:
            _add_error(
                errors,
                f"sha256sums.{filename}",
                f"{filename} is missing from sha256sums.txt",
            )
            continue
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            _add_error(
                errors,
                f"sha256sums.{filename}",
                f"sha256 mismatch for {filename}: {actual_digest} != {expected_digest}",
            )


def _validate_runtime_profile_offsets(
    *,
    bundle_dir: Path,
    profiles: list[Any],
    expected_profile_nbytes: int,
    errors: list[dict[str, str]],
) -> None:
    weights_path = bundle_dir / DEFAULT_WEIGHTS_FILENAME
    weights_size = weights_path.stat().st_size if weights_path.exists() else 0
    ranges: list[tuple[int, int, int]] = []
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            _add_error(
                errors,
                "pilot_profiles.profiles",
                f"profile row {index} is not an object",
            )
            continue
        try:
            offset = int(row["weight_bank_offset_bytes"])
            nbytes = int(row["weight_bank_nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            _add_error(
                errors,
                "pilot_profiles.profile_offsets",
                f"profile row {index} has invalid offset/size fields: {exc}",
            )
            continue
        if expected_profile_nbytes > 0 and offset % expected_profile_nbytes != 0:
            _add_error(
                errors,
                "pilot_profiles.profile_offset_alignment",
                f"profile row {index} offset {offset} is not aligned to "
                f"{expected_profile_nbytes} bytes",
            )
        if 0 < expected_profile_nbytes != nbytes:
            _add_error(
                errors,
                "pilot_profiles.profile_nbytes",
                f"profile row {index} nbytes {nbytes} does not match "
                f"{expected_profile_nbytes}",
            )
        if offset < 0 or nbytes <= 0 or offset + nbytes > weights_size:
            _add_error(
                errors,
                "pilot_profiles.profile_bounds",
                f"profile row {index} range [{offset}, {offset + nbytes}) "
                f"is outside weights.bin size {weights_size}",
            )
        if not str(row.get("reference_placement_status", "")).strip():
            _add_error(
                errors,
                "pilot_profiles.reference_placement_status",
                f"profile row {index} is missing reference_placement_status",
            )
        # Norm-corrected threshold fields: optional (absent in legacy bundles),
        # but when declared they must match the exact integer norms recomputed
        # from the bundled weight bytes, and the half-threshold rational must
        # be nt:(nl+nu).
        if "target_norm_sq" in row or "ref_norm_sum_sq" in row:
            if 0 <= offset and offset + nbytes <= weights_size and nbytes > 0:
                raw = weights_path.read_bytes()[offset : offset + nbytes]
                packed = np.frombuffer(raw, dtype=np.int8).reshape(3, -1)
                got_nt, got_nl, got_nu = weight_term_norms_sq(packed)
                got_nrs = int(got_nl + got_nu)
                declared = {
                    "target_norm_sq": got_nt,
                    "ref_norm_sum_sq": got_nrs,
                    "positive_excess_half_threshold_num": got_nt,
                    "positive_excess_half_threshold_den": got_nrs,
                }
                for key, expected_value in declared.items():
                    if key in row and int(row[key]) != int(expected_value):
                        _add_error(
                            errors,
                            f"pilot_profiles.{key}",
                            f"profile row {index} declares {key}="
                            f"{row[key]!r} but the bundled weights give "
                            f"{expected_value}",
                        )
                if "mu0" in row and got_nrs > 0:
                    expected_mu0 = norm_corrected_mu0(got_nt, got_nrs)
                    if abs(float(row["mu0"]) - expected_mu0) > 1e-12:
                        _add_error(
                            errors,
                            "pilot_profiles.mu0",
                            f"profile row {index} declares mu0={row['mu0']!r} "
                            f"but the bundled weights give {expected_mu0!r}",
                        )
        ranges.append((offset, offset + nbytes, index))
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            _add_error(
                errors,
                "pilot_profiles.profile_overlap",
                f"profile row {previous[2]} overlaps profile row {current[2]}",
            )


def _coerce_int_metadata(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return int(value)
    raise TypeError(f"{field} must be an integer-compatible value, got {value!r}")


def _validate_chime_channel_ids(
    *,
    profiles: list[Any],
    errors: list[dict[str, str]],
) -> None:
    seen: set[int] = set()
    for index, row in enumerate(profiles):
        if not isinstance(row, dict):
            continue
        value = row.get("chime_channel_id")
        if value is None:
            continue
        try:
            channel_id = _coerce_int_metadata(
                value,
                field="chime_channel_id",
            )
        except (TypeError, ValueError):
            _add_error(
                errors,
                "pilot_profiles.chime_channel_id",
                f"invalid chime_channel_id {value!r} at profile row {index}",
            )
            continue
        if channel_id in seen:
            _add_error(
                errors,
                "pilot_profiles.chime_channel_id",
                f"duplicate chime_channel_id {channel_id} at profile row {index}",
            )
        seen.add(channel_id)


def _validate_bundle_coordinate_systems(
    *,
    detector_contract: dict[str, Any],
    pilot_profiles: dict[str, Any],
    weights_manifest: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    values = {
        "detector_contract": detector_contract.get("weight_coordinate_system"),
        "pilot_profiles": pilot_profiles.get("weight_coordinate_system"),
        "weights_manifest": weights_manifest.get("weight_coordinate_system"),
    }
    normalized: dict[str, str] = {}
    for label, value in values.items():
        if value is None:
            _add_error(
                errors,
                f"{label}.weight_coordinate_system",
                "missing weight_coordinate_system",
            )
            continue
        try:
            normalized[label] = normalize_weight_coordinate_system(value)
        except ValueError as exc:
            _add_error(errors, f"{label}.weight_coordinate_system", str(exc))
    coordinate_system = next(iter(normalized.values())) if normalized else None
    if len(set(normalized.values())) > 1:
        _add_error(
            errors,
            "weight_coordinate_system.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{values}",
        )

    input_coordinate_values = {
        "detector_contract": detector_contract.get("input_coordinate_system"),
        "pilot_profiles": pilot_profiles.get("input_coordinate_system"),
        "weights_manifest": weights_manifest.get("input_coordinate_system"),
    }
    present_input_coordinates: dict[str, str] = {}
    for label, value in input_coordinate_values.items():
        if value is None:
            _add_error(
                errors,
                f"{label}.input_coordinate_system",
                "missing input_coordinate_system",
            )
        else:
            present_input_coordinates[label] = str(value)
    if len(set(present_input_coordinates.values())) > 1:
        _add_error(
            errors,
            "input_coordinate_system.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{input_coordinate_values}",
        )
    if coordinate_system is not None:
        expected_coordinate = input_coordinate_system_for_weight_coordinate(
            coordinate_system
        )
        for label, value in present_input_coordinates.items():
            if value != expected_coordinate:
                _add_error(
                    errors,
                    f"{label}.input_coordinate_system",
                    f"input_coordinate_system {value!r} does not match "
                    f"{expected_coordinate!r} for weight_coordinate_system "
                    f"{coordinate_system!r}",
                )

    preprocessing_values = {
        "detector_contract": detector_contract.get("input_preprocessing"),
        "pilot_profiles": pilot_profiles.get("input_preprocessing"),
        "weights_manifest": weights_manifest.get("input_preprocessing"),
    }
    time_reverse_values: dict[str, bool] = {}
    for label, preprocessing in preprocessing_values.items():
        if not isinstance(preprocessing, dict):
            _add_error(
                errors,
                f"{label}.input_preprocessing",
                "input_preprocessing is missing or not an object",
            )
            continue
        if "time_reverse_detector_windows_before_kernel" not in preprocessing:
            _add_error(
                errors,
                f"{label}.input_preprocessing.time_reverse",
                "missing time_reverse_detector_windows_before_kernel",
            )
            continue
        time_reverse_values[label] = bool(
            preprocessing["time_reverse_detector_windows_before_kernel"]
        )
    if len(set(time_reverse_values.values())) > 1:
        _add_error(
            errors,
            "input_preprocessing.time_reverse.consistency",
            "detector_contract, pilot_profiles, and weights_manifest disagree: "
            f"{time_reverse_values}",
        )
    if (
        coordinate_system == WEIGHT_COORDINATE_RAW_INPUT
        and any(time_reverse_values.values())
    ):
        _add_error(
            errors,
            "input_preprocessing.time_reverse",
            "raw input-coordinate weights must not request detector-window "
            "time reversal before the kernel",
        )


def validate_runtime_weight_bundle(
    *,
    bundle_dir: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Validate a runtime weight bundle for deployment use."""
    bundle = Path(bundle_dir)
    errors: list[dict[str, str]] = []
    for filename in REQUIRED_RUNTIME_BUNDLE_FILES:
        if not (bundle / filename).exists():
            _add_error(
                errors,
                f"required_file.{filename}",
                f"missing {bundle / filename}",
            )

    detector_contract = _load_json_object(
        bundle / DEFAULT_DETECTOR_CONTRACT_FILENAME,
        errors,
    )
    pilot_profiles = _load_json_object(
        bundle / DEFAULT_PILOT_PROFILES_FILENAME,
        errors,
    )
    weights_manifest = _load_json_object(
        bundle / DEFAULT_WEIGHTS_MANIFEST_FILENAME,
        errors,
    )
    expected_sums = _read_sha256sums(bundle / DEFAULT_SHA256SUMS_FILENAME, errors)
    _validate_expected_sha256sums(
        bundle_dir=bundle,
        expected=expected_sums,
        errors=errors,
    )

    if pilot_profiles.get("schema_version") != RUNTIME_PILOT_PROFILES_SCHEMA_VERSION:
        _add_error(
            errors,
            "pilot_profiles.schema_version",
            f"schema_version {pilot_profiles.get('schema_version')!r} does not "
            f"match {RUNTIME_PILOT_PROFILES_SCHEMA_VERSION!r}",
        )
    if (
        weights_manifest.get("schema_version")
        != RUNTIME_WEIGHT_MANIFEST_SCHEMA_VERSION
    ):
        _add_error(
            errors,
            "weights_manifest.schema_version",
            f"schema_version {weights_manifest.get('schema_version')!r} does not "
            f"match {RUNTIME_WEIGHT_MANIFEST_SCHEMA_VERSION!r}",
        )

    contract_digest = detector_contract_sha256(detector_contract)
    weights_path = bundle / DEFAULT_WEIGHTS_FILENAME
    weights_digest = file_sha256(weights_path) if weights_path.exists() else ""
    for label, payload in (
        ("pilot_profiles", pilot_profiles),
        ("weights_manifest", weights_manifest),
    ):
        if payload.get("detector_contract_sha256") != contract_digest:
            _add_error(
                errors,
                f"{label}.detector_contract_sha256",
                f"detector_contract_sha256 does not match detector_contract.json "
                f"for {label}",
            )
        if payload.get("weights_sha256") != weights_digest:
            _add_error(
                errors,
                f"{label}.weights_sha256",
                f"weights_sha256 does not match weights.bin for {label}",
            )

    profiles_raw = pilot_profiles.get("profiles", [])
    profiles = profiles_raw if isinstance(profiles_raw, list) else []
    if not isinstance(profiles_raw, list):
        _add_error(errors, "pilot_profiles.profiles", "profiles is not a list")
    manifest_num_profiles = weights_manifest.get("num_profiles")
    if manifest_num_profiles is not None:
        try:
            manifest_num_profiles_int = _coerce_int_metadata(
                manifest_num_profiles,
                field="num_profiles",
            )
        except (TypeError, ValueError) as exc:
            _add_error(errors, "weights_manifest.num_profiles", str(exc))
            manifest_num_profiles_int = None
        if (
            manifest_num_profiles_int is not None
            and manifest_num_profiles_int != len(profiles)
        ):
            _add_error(
                errors,
                "weights_manifest.num_profiles",
                f"num_profiles {manifest_num_profiles!r} does not match "
                f"profile count {len(profiles)}",
            )
    manifest_channels = weights_manifest.get("physical_channels", [])
    if isinstance(manifest_channels, list) and len(manifest_channels) != len(profiles):
        _add_error(
            errors,
            "weights_manifest.physical_channels",
            "physical_channels length does not match profile count",
        )
    expected_profile_nbytes = int(weights_manifest.get("weight_profile_nbytes", 0) or 0)
    _validate_runtime_profile_offsets(
        bundle_dir=bundle,
        profiles=profiles,
        expected_profile_nbytes=expected_profile_nbytes,
        errors=errors,
    )
    _validate_chime_channel_ids(profiles=profiles, errors=errors)
    _validate_bundle_coordinate_systems(
        detector_contract=detector_contract,
        pilot_profiles=pilot_profiles,
        weights_manifest=weights_manifest,
        errors=errors,
    )

    report = {
        "schema_version": RUNTIME_BUNDLE_VALIDATION_SCHEMA_VERSION,
        "bundle_dir": str(bundle),
        "valid": len(errors) == 0,
        "num_errors": int(len(errors)),
        "errors": errors,
    }
    if output_json is not None:
        write_json_strict(Path(output_json), report, indent=2, sort_keys=True)
    return report


def export_runtime_weight_bundle(
    *,
    receiver_profile_path: Path,
    detector_core_profile_path: Path,
    physical_channels: Sequence[int],
    weight_coordinate_system: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Write a compact runtime detector bundle for selected DTV pilots."""
    profile = load_receiver_profile(receiver_profile_path)
    core = load_detector_core_profile(detector_core_profile_path)
    coordinate_system = normalize_weight_coordinate_system(weight_coordinate_system)
    channels = [int(channel) for channel in physical_channels]
    if not channels:
        raise ValueError("at least one physical channel must be selected")

    table, layouts = generate_weight_table_from_receiver_profile(
        profile=profile,
        core=core,
        physical_channels=channels,
        weight_coordinate_system=coordinate_system,
    )
    layout_by_channel = {int(row["physical_channel"]): row for row in layouts}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    profile_nbytes = int(core.num_weight_terms) * int(core.detector_window_samples)
    profile_nbytes *= np.dtype(np.int8).itemsize
    weights_path = output / DEFAULT_WEIGHTS_FILENAME
    profile_rows: list[dict[str, Any]] = []
    weight_chunks: list[bytes] = []
    offset = 0
    for index, channel in enumerate(channels):
        layout = layout_by_channel[int(channel)]
        coarse_index = int(layout["coarse_channel_index"])
        weights = np.ascontiguousarray(table[coarse_index], dtype=np.int8)
        payload = weights.tobytes(order="C")
        if len(payload) != profile_nbytes:
            raise ValueError(
                f"unexpected weight profile size for channel {channel}: "
                f"{len(payload)} != {profile_nbytes}"
            )
        weight_chunks.append(payload)
        row_nt, row_nl, row_nu = weight_term_norms_sq(weights)
        row_nrs = int(row_nl + row_nu)
        profile_rows.append(
            {
                "physical_channel": int(channel),
                "chime_channel_id": None,
                "pilot_frequency_hz": float(layout["dtv_pilot_hz"]),
                "chime_frequency_hz": float(layout["coarse_channel_center_hz"]),
                "coarse_channel_index": coarse_index,
                "weight_bank_index": int(index),
                "weight_bank_offset_bytes": int(offset),
                "weight_bank_nbytes": int(profile_nbytes),
                # Exact integer weight-norm zero-point for this channel. The
                # kernel's rational half-threshold half_num:half_den = nt:(nl+nu)
                # sets the mask threshold at F = mu0 (the norm-corrected
                # positive-excess rule) with zero kernel changes.
                "target_norm_sq": int(row_nt),
                "ref_norm_sum_sq": row_nrs,
                "mu0": float(norm_corrected_mu0(row_nt, row_nrs)),
                "positive_excess_half_threshold_num": int(row_nt),
                "positive_excess_half_threshold_den": row_nrs,
                "reference_placement_status": str(
                    layout["reference_placement_status"]
                ),
                "placement_warnings": str(layout["placement_warnings"]),
            }
        )
        offset += profile_nbytes
    weights_path.write_bytes(b"".join(weight_chunks))

    reference_placement_summary = _reference_placement_summary_from_layouts(
        core=core,
        layouts=layouts,
    )
    detector_contract = build_chime_detector_contract(
        detector_window_samples=int(core.detector_window_samples),
        skipped_guard_bins=int(core.skipped_guard_bins),
        reference_offset_bins=int(core.reference_offset_bins),
        num_weight_terms=int(core.num_weight_terms),
        sample_bits_per_component=int(core.sample_bits_per_component),
        input_format=core.input_format,
        power_accumulator=core.power_accumulator,
        weight_coordinate_system=coordinate_system,
        input_coordinate_system=input_coordinate_system_for_weight_coordinate(
            coordinate_system
        ),
        time_reverse_detector_windows_before_kernel=(
            coordinate_system == WEIGHT_COORDINATE_POST_SPECTRAL_SENSE
            and spectral_sense_requires_time_reversal(profile.spectral_sense)
        ),
        reference_placement_summary=reference_placement_summary,
    )
    contract_path = output / DEFAULT_DETECTOR_CONTRACT_FILENAME
    write_json_strict(contract_path, detector_contract, indent=2, sort_keys=True)
    contract_digest = detector_contract_sha256(detector_contract)
    weights_digest = file_sha256(weights_path)

    weights_manifest = {
        "schema_version": RUNTIME_WEIGHT_MANIFEST_SCHEMA_VERSION,
        "weight_format": "concatenated_int8_weight_profiles_v1",
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": detector_contract["input_coordinate_system"],
        "input_preprocessing": detector_contract["input_preprocessing"],
        "detector_contract_sha256": contract_digest,
        "receiver_profile_path": str(receiver_profile_path),
        "receiver_profile_hash": receiver_profile_hash(profile),
        "detector_core_profile_path": str(detector_core_profile_path),
        "physical_channels": channels,
        "num_profiles": int(len(channels)),
        "weight_profile_shape": [
            int(core.num_weight_terms),
            int(core.detector_window_samples),
        ],
        "weight_profile_dtype": "int8",
        "weight_profile_nbytes": int(profile_nbytes),
        "weights_sha256": weights_digest,
        "target_reference_layout": layouts,
    }
    weights_manifest_path = output / DEFAULT_WEIGHTS_MANIFEST_FILENAME
    write_json_strict(weights_manifest_path, weights_manifest, indent=2, sort_keys=True)

    pilot_profiles = {
        "schema_version": RUNTIME_PILOT_PROFILES_SCHEMA_VERSION,
        "weight_coordinate_system": coordinate_system,
        "input_coordinate_system": detector_contract["input_coordinate_system"],
        "input_preprocessing": detector_contract["input_preprocessing"],
        "detector_contract_sha256": contract_digest,
        "weights_sha256": weights_digest,
        "profiles": profile_rows,
    }
    pilot_profiles_path = output / DEFAULT_PILOT_PROFILES_FILENAME
    write_json_strict(pilot_profiles_path, pilot_profiles, indent=2, sort_keys=True)

    sha256sums_path = _write_sha256sums(
        output,
        [
            DEFAULT_DETECTOR_CONTRACT_FILENAME,
            DEFAULT_PILOT_PROFILES_FILENAME,
            DEFAULT_WEIGHTS_FILENAME,
            DEFAULT_WEIGHTS_MANIFEST_FILENAME,
        ],
    )
    return {
        "detector_contract": contract_path,
        "pilot_profiles": pilot_profiles_path,
        "weights": weights_path,
        "weights_manifest": weights_manifest_path,
        "sha256sums": sha256sums_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a compact runtime detector weight bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--receiver-profile", type=Path, required=True)
    parser.add_argument("--detector-core-profile", type=Path, required=True)
    parser.add_argument(
        "--weight-coordinate-system",
        choices=[
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
    )
    parser.add_argument("--physical-channel", type=int, action="append", default=None)
    parser.add_argument("--physical-channel-range", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=args.receiver_profile,
        detector_core_profile_path=args.detector_core_profile,
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
        output_dir=args.output_dir,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
