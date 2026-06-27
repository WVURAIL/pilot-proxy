# coding=utf-8
"""Validate CHIME run products for shape and metadata consistency."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import numpy as np

from pilot_proxy.detector_contract import (
    CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION,
    CHIME_RUN_CONFIG_SCHEMA_VERSION,
    CHIME_STATS_SCHEMA_VERSION,
    POSITIVE_EXCESS_EQUIVALENT_RULE,
    POSITIVE_EXCESS_MASK_RULE,
    POSITIVE_EXCESS_MASK_SOURCE,
    POSITIVE_EXCESS_VALID_RULE,
    WEIGHT_COORDINATE_RAW_INPUT,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
)
from pilot_proxy.json_utils import write_json_strict

from .frequency_offset import FREQUENCY_OFFSET_OUTPUTS_FILENAME
from .products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)
from .reductions import CHIME_REDUCTIONS_10S_FILENAME


class NpzLike(Protocol):
    files: list[str]

    def __getitem__(self, key: str) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


def _add_error(errors: list[dict[str, str]], check: str, message: str) -> None:
    errors.append({"severity": "error", "check": str(check), "message": str(message)})


def _load_json(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
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


def _load_npz(path: Path, errors: list[dict[str, str]]) -> NpzLike | None:
    try:
        return cast(NpzLike, np.load(path))
    except FileNotFoundError:
        _add_error(errors, f"required_file.{path.name}", f"missing {path}")
    except Exception as exc:  # noqa: BLE001 - validator should report file problems.
        _add_error(errors, f"npz.{path.name}", f"could not load NPZ: {exc}")
    return None


def _coerce_int_metadata(value: object, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, bytes)):
        return int(value)
    raise TypeError(f"{field} must be an integer-compatible value, got {value!r}")


def _require_arrays(
    npz: NpzLike,
    *,
    filename: str,
    names: Sequence[str],
    errors: list[dict[str, str]],
) -> None:
    files = set(npz.files)
    for name in names:
        if name not in files:
            _add_error(errors, f"{filename}.{name}", "required array is missing")


def _check_shape(
    *,
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    check: str,
    errors: list[dict[str, str]],
) -> None:
    if tuple(actual) != tuple(expected):
        _add_error(errors, check, f"shape {actual!r} does not match {expected!r}")


def _check_json_schema(
    *,
    payload: dict[str, Any],
    filename: str,
    expected: str,
    errors: list[dict[str, str]],
) -> None:
    actual = payload.get("schema_version")
    if actual != expected:
        _add_error(
            errors,
            f"{filename}.schema_version",
            f"schema_version {actual!r} does not match {expected!r}",
        )


def _is_binary_array(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    return bool(np.all((arr == 0) | (arr == 1)))


def _is_positive_excess_run(
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
) -> bool:
    policy = stats.get("mask_policy") or run_config.get("mask_policy")
    contract = stats.get("detector_contract") or run_config.get("detector_contract")
    return bool(
        (
            isinstance(policy, dict)
            and str(policy.get("mask_source")) == POSITIVE_EXCESS_MASK_SOURCE
        )
        or (
            isinstance(contract, dict)
            and str(contract.get("mask_source")) == POSITIVE_EXCESS_MASK_SOURCE
        )
    )


def _validate_detector_contract(
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    run_contract = run_config.get("detector_contract")
    stats_contract = stats.get("detector_contract")
    if not isinstance(run_contract, dict):
        _add_error(
            errors,
            "run_config.detector_contract",
            "detector_contract is missing or not an object",
        )
        return
    run_contract_typed: dict[str, Any] = dict(run_contract)
    if not isinstance(stats_contract, dict):
        _add_error(
            errors,
            "stats.detector_contract",
            "detector_contract is missing or not an object",
        )
        return
    stats_contract_typed: dict[str, Any] = dict(stats_contract)
    if run_contract_typed != stats_contract_typed:
        _add_error(
            errors,
            "detector_contract.consistency",
            "run_config and stats detector_contract objects differ",
        )
    if (
        run_contract_typed.get("schema_version")
        != CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION
    ):
        _add_error(
            errors,
            "detector_contract.schema_version",
            "detector_contract schema_version "
            f"{run_contract_typed.get('schema_version')!r} does not match "
            f"{CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION!r}",
        )
    required = {
        "detector_window_samples",
        "skipped_guard_bins",
        "reference_offset_bins",
        "num_weight_terms",
        "input_format",
        "power_accumulator",
        "statistic",
        "combine_mode",
        "weight_coordinate_system",
        "input_coordinate_system",
        "input_preprocessing",
        "mask_source",
        "valid_rule",
        "mask_rule",
        "equivalent_mask_rule",
        "per_frequency_threshold",
    }
    missing: list[str] = sorted(
        name for name in required if name not in run_contract_typed
    )
    if missing:
        _add_error(
            errors,
            "detector_contract.required_fields",
            "missing fields: " + ", ".join(missing),
        )
    if (
        "skipped_guard_bins" in run_contract_typed
        and "reference_offset_bins" in run_contract_typed
        and int(run_contract_typed["reference_offset_bins"])
        != int(run_contract_typed["skipped_guard_bins"]) + 1
    ):
        _add_error(
            errors,
            "detector_contract.reference_offset_relation",
            "reference_offset_bins must equal skipped_guard_bins + 1",
        )
    expected_policy = {
        "mask_source": POSITIVE_EXCESS_MASK_SOURCE,
        "valid_rule": POSITIVE_EXCESS_VALID_RULE,
        "mask_rule": POSITIVE_EXCESS_MASK_RULE,
        "equivalent_mask_rule": POSITIVE_EXCESS_EQUIVALENT_RULE,
    }
    for key, expected in expected_policy.items():
        if run_contract_typed.get(key) != expected:
            _add_error(
                errors,
                f"detector_contract.{key}",
                f"{key}={run_contract_typed.get(key)!r} does not match {expected!r}",
            )
    if bool(run_contract_typed.get("per_frequency_threshold")):
        _add_error(
            errors,
            "detector_contract.per_frequency_threshold",
            "CHIME positive-excess products must not declare thresholds",
        )
    weight_coordinate = run_contract_typed.get("weight_coordinate_system")
    if weight_coordinate is not None:
        try:
            normalized_weight_coordinate = normalize_weight_coordinate_system(
                weight_coordinate
            )
        except ValueError as exc:
            _add_error(
                errors,
                "detector_contract.weight_coordinate_system",
                str(exc),
            )
            normalized_weight_coordinate = None
        if normalized_weight_coordinate is not None:
            expected_input_coordinate = input_coordinate_system_for_weight_coordinate(
                normalized_weight_coordinate
            )
            input_coordinate = run_contract_typed.get("input_coordinate_system")
            if str(input_coordinate) != expected_input_coordinate:
                _add_error(
                    errors,
                    "detector_contract.input_coordinate_system",
                    f"input_coordinate_system {input_coordinate!r} does not "
                    f"match {expected_input_coordinate!r}",
                )
    preprocessing = run_contract_typed.get("input_preprocessing")
    if not isinstance(preprocessing, dict):
        _add_error(
            errors,
            "detector_contract.input_preprocessing",
            "input_preprocessing must be an object",
        )
    elif "time_reverse_detector_windows_before_kernel" not in preprocessing:
        _add_error(
            errors,
            "detector_contract.input_preprocessing.time_reverse",
            "missing time_reverse_detector_windows_before_kernel",
        )
    elif (
        weight_coordinate == WEIGHT_COORDINATE_RAW_INPUT
        and bool(preprocessing.get("time_reverse_detector_windows_before_kernel"))
    ):
        _add_error(
            errors,
            "detector_contract.input_preprocessing.time_reverse",
            "raw input-coordinate weights must not request detector-window "
            "time reversal before the kernel",
        )
    contract_window = run_contract_typed.get("detector_window_samples")
    if contract_window is not None:
        try:
            expected_window = _coerce_int_metadata(
                contract_window,
                field="detector_window_samples",
            )
        except (TypeError, ValueError) as exc:
            _add_error(
                errors,
                "detector_contract.detector_window_samples",
                str(exc),
            )
            expected_window = None
        for label, payload in (
            ("run_config", run_config),
            ("stats", stats),
        ):
            payload_window = payload.get("detector_window_samples")
            if payload_window is None or expected_window is None:
                continue
            try:
                payload_window_int = _coerce_int_metadata(
                    payload_window,
                    field=f"{label}.detector_window_samples",
                )
            except (TypeError, ValueError) as exc:
                _add_error(
                    errors,
                    f"{label}.detector_window_samples",
                    str(exc),
                )
                continue
            if payload_window_int != expected_window:
                _add_error(
                    errors,
                    f"{label}.detector_window_samples",
                    "detector_window_samples does not match detector_contract: "
                    f"{payload_window!r} != {expected_window}",
                )


def _validate_detector(
    detector: NpzLike,
    *,
    run_config: dict[str, Any],
    stats: dict[str, Any],
    errors: list[dict[str, str]],
) -> tuple[int, int] | None:
    required = [
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "frame_index",
        "p_target_u64",
        "p_ref_sum_u64",
        "fstat_raw",
        "fstat_level_db",
        "pnr_bin_db",
        "snr_shelf_db",
        "mask",
        "valid",
    ]
    _require_arrays(
        detector,
        filename=CHIME_DETECTOR_OUTPUTS_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in detector.files for name in required):
        return None

    num_frames = int(np.asarray(detector["frame_index"]).size)
    num_pilots = int(np.asarray(detector["physical_channel"]).size)
    frame_pilot_shape = (num_frames, num_pilots)

    for name in [
        "p_target_u64",
        "p_ref_sum_u64",
        "fstat_raw",
        "fstat_level_db",
        "pnr_bin_db",
        "snr_shelf_db",
        "mask",
        "valid",
    ]:
        _check_shape(
            actual=tuple(np.asarray(detector[name]).shape),
            expected=frame_pilot_shape,
            check=f"detector.{name}.shape",
            errors=errors,
        )
    for name in [
        "pilot_frequency_hz",
        "chime_frequency_hz",
    ]:
        _check_shape(
            actual=tuple(np.asarray(detector[name]).shape),
            expected=(num_pilots,),
            check=f"detector.{name}.shape",
            errors=errors,
        )

    mask = np.asarray(detector["mask"])
    valid = np.asarray(detector["valid"])
    if not _is_binary_array(mask):
        _add_error(errors, "detector.mask.binary", "mask contains values outside 0/1")
    if not _is_binary_array(valid):
        _add_error(errors, "detector.valid.binary", "valid contains values outside 0/1")
    if np.any((mask != 0) & (valid == 0)):
        _add_error(errors, "detector.mask.invalid", "invalid frames are masked")
    if np.any((np.asarray(detector["p_ref_sum_u64"]) > 0) != (valid != 0)):
        _add_error(
            errors,
            "detector.valid.denominator",
            "valid array does not match p_ref_sum_u64 > 0",
        )
    if _is_positive_excess_run(run_config=run_config, stats=stats):
        p_target = np.asarray(detector["p_target_u64"], dtype=np.uint64)
        p_ref_sum = np.asarray(detector["p_ref_sum_u64"], dtype=np.uint64)
        expected_mask = (
            (valid != 0)
            & (p_ref_sum != 0)
            & (p_target > (p_ref_sum >> 1))
        )
        if np.any((mask != 0) != expected_mask):
            _add_error(
                errors,
                "detector.mask.positive_excess_rule",
                "mask does not match valid && p_target > (p_ref_sum >> 1)",
            )

    if stats:
        for key, expected in [
            ("num_frames", num_frames),
            ("num_pilots", num_pilots),
        ]:
            if key in stats and int(stats[key]) != int(expected):
                _add_error(
                    errors,
                    f"stats.{key}",
                    f"stats {key}={stats[key]!r} does not match detector {expected}",
                )
        overflow = stats.get("rational_overflow_count_by_pilot")
        if (
            isinstance(overflow, Iterable)
            and not isinstance(overflow, (str, bytes, dict))
            and any(int(value) != 0 for value in overflow)
        ):
            _add_error(
                errors,
                "stats.rational_overflow_count_by_pilot",
                f"nonzero rational overflow counts: {overflow!r}",
            )

    return frame_pilot_shape


def _rows_from_csv(
    path: Path,
    *,
    check: str,
    errors: list[dict[str, str]],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        _add_error(errors, check, f"missing {path}")
    except Exception as exc:  # noqa: BLE001 - validator reports product issues.
        _add_error(errors, check, f"could not read {path}: {exc}")
    return []


def _validate_frequency_offset_products(
    *,
    run: Path,
    detector: NpzLike,
    shape: tuple[int, int],
    errors: list[dict[str, str]],
) -> NpzLike | None:
    products = _load_npz(run / FREQUENCY_OFFSET_OUTPUTS_FILENAME, errors)
    if products is None:
        return None
    required = [
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "coarse_channel_center_hz",
        "expected_pilot_offset_hz",
        "frame_index",
        "relative_time_s",
        "peak_offset_hz",
        "frequency_offset_hz",
        "peak_power_linear",
        "local_floor_power_linear",
        "peak_prominence_db",
        "valid",
        "fft_frequency_axis_hz",
        "time_average_spectrum_power_linear",
        "time_average_spectrum_count",
        "fft_size",
    ]
    _require_arrays(
        products,
        filename=FREQUENCY_OFFSET_OUTPUTS_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in products.files for name in required):
        return products

    num_frames, num_pilots = shape
    fft_size = int(np.asarray(products["fft_size"]).item())
    for name in [
        "peak_offset_hz",
        "frequency_offset_hz",
        "peak_power_linear",
        "local_floor_power_linear",
        "peak_prominence_db",
        "valid",
    ]:
        _check_shape(
            actual=tuple(np.asarray(products[name]).shape),
            expected=shape,
            check=f"frequency_offset.{name}.shape",
            errors=errors,
        )
    for name in [
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "coarse_channel_center_hz",
        "expected_pilot_offset_hz",
        "time_average_spectrum_count",
    ]:
        _check_shape(
            actual=tuple(np.asarray(products[name]).shape),
            expected=(num_pilots,),
            check=f"frequency_offset.{name}.shape",
            errors=errors,
        )
    for name in ["frame_index", "relative_time_s"]:
        _check_shape(
            actual=tuple(np.asarray(products[name]).shape),
            expected=(num_frames,),
            check=f"frequency_offset.{name}.shape",
            errors=errors,
        )
    _check_shape(
        actual=tuple(np.asarray(products["fft_frequency_axis_hz"]).shape),
        expected=(fft_size,),
        check="frequency_offset.fft_frequency_axis_hz.shape",
        errors=errors,
    )
    _check_shape(
        actual=tuple(
            np.asarray(products["time_average_spectrum_power_linear"]).shape
        ),
        expected=(num_pilots, fft_size),
        check="frequency_offset.time_average_spectrum_power_linear.shape",
        errors=errors,
    )
    if not np.array_equal(
        np.asarray(products["physical_channel"]),
        np.asarray(detector["physical_channel"]),
    ):
        _add_error(
            errors,
            "frequency_offset.physical_channel",
            "physical_channel does not match detector output",
        )
    if not np.array_equal(
        np.asarray(products["frame_index"]),
        np.asarray(detector["frame_index"]),
    ):
        _add_error(
            errors,
            "frequency_offset.frame_index",
            "frame_index does not match detector output",
        )
    if not _is_binary_array(np.asarray(products["valid"])):
        _add_error(
            errors,
            "frequency_offset.valid.binary",
            "valid contains values outside 0/1",
        )
    rows = _rows_from_csv(
        run / "tables" / "frequency_offset_summary_by_pilot.csv",
        check="frequency_offset_summary.exists",
        errors=errors,
    )
    if rows and len(rows) != num_pilots:
        _add_error(
            errors,
            "frequency_offset_summary.row_count",
            f"row count {len(rows)} does not match num_pilots {num_pilots}",
        )
    return products


def _validate_k_candidate_table(
    *,
    run: Path,
    errors: list[dict[str, str]],
) -> None:
    rows = _rows_from_csv(
        run / "tables" / "k_candidate_summary.csv",
        check="k_candidate_summary.exists",
        errors=errors,
    )
    if not rows:
        return
    columns = set(rows[0])
    required_columns = {
        "K",
        "reference_spacing_policy",
        "recommended",
        "reference_placement_status",
        "placement_warnings",
        "channels_with_adaptive_reference",
        "channels_with_dc_shifted_reference",
        "channels_with_edge_wrapped_reference",
        "channels_with_forbidden_tone_in_skipped_guard",
    }
    missing = sorted(required_columns.difference(columns))
    if missing:
        _add_error(
            errors,
            "k_candidate_summary.columns",
            "missing columns: " + ", ".join(missing),
        )
    k_values = {int(row["K"]) for row in rows if str(row.get("K", "")).strip()}
    for required_k in (128, 256):
        if required_k not in k_values:
            _add_error(
                errors,
                "k_candidate_summary.required_k",
                f"K={required_k} is missing from k_candidate_summary.csv",
            )


def _validate_spectrogram_cache(
    cache: NpzLike,
    detector: NpzLike,
    *,
    shape: tuple[int, int],
    errors: list[dict[str, str]],
) -> None:
    required = [
        "baseband_power_linear",
        "baseband_power_db",
        "mask",
        "valid",
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "frame_index",
        "relative_time_s",
    ]
    _require_arrays(
        cache,
        filename=CHIME_SPECTROGRAM_CACHE_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in cache.files for name in required):
        return

    for name in ["baseband_power_linear", "baseband_power_db", "mask", "valid"]:
        _check_shape(
            actual=tuple(np.asarray(cache[name]).shape),
            expected=shape,
            check=f"spectrogram.{name}.shape",
            errors=errors,
        )
    for name in ["physical_channel", "pilot_frequency_hz", "chime_frequency_hz"]:
        if not np.array_equal(np.asarray(cache[name]), np.asarray(detector[name])):
            _add_error(
                errors,
                f"spectrogram.{name}",
                f"{name} does not match detector output",
            )
    if not np.array_equal(np.asarray(cache["frame_index"]), np.asarray(detector["frame_index"])):
        _add_error(errors, "spectrogram.frame_index", "frame_index does not match")
    if not np.array_equal(np.asarray(cache["mask"]), np.asarray(detector["mask"])):
        _add_error(errors, "spectrogram.mask", "mask does not match detector output")
    if not np.array_equal(np.asarray(cache["valid"]), np.asarray(detector["valid"])):
        _add_error(errors, "spectrogram.valid", "valid does not match detector output")
    _check_shape(
        actual=tuple(np.asarray(cache["relative_time_s"]).shape),
        expected=(shape[0],),
        check="spectrogram.relative_time_s.shape",
        errors=errors,
    )


def _validate_reductions(
    reductions: NpzLike,
    detector: NpzLike,
    *,
    shape: tuple[int, int],
    errors: list[dict[str, str]],
) -> None:
    required = [
        "chunk_index",
        "chunk_start_frame",
        "chunk_stop_frame",
        "input_power_mean",
        "cleaned_power_mean",
        "valid_count",
        "invalid_count",
        "masked_count_valid",
        "unmasked_count_valid",
        "mask_fraction_valid",
        "mask_fraction_total",
    ]
    _require_arrays(
        reductions,
        filename=CHIME_REDUCTIONS_10S_FILENAME,
        names=required,
        errors=errors,
    )
    if any(name not in reductions.files for name in required):
        return

    num_chunks = int(np.asarray(reductions["chunk_index"]).size)
    num_pilots = int(shape[1])
    for name in ["chunk_start_frame", "chunk_stop_frame"]:
        _check_shape(
            actual=tuple(np.asarray(reductions[name]).shape),
            expected=(num_chunks,),
            check=f"reductions.{name}.shape",
            errors=errors,
        )
    for name in [
        "input_power_mean",
        "cleaned_power_mean",
        "valid_count",
        "invalid_count",
        "masked_count_valid",
        "unmasked_count_valid",
        "mask_fraction_valid",
        "mask_fraction_total",
    ]:
        _check_shape(
            actual=tuple(np.asarray(reductions[name]).shape),
            expected=(num_chunks, num_pilots),
            check=f"reductions.{name}.shape",
            errors=errors,
        )

    valid = np.asarray(detector["valid"]) != 0
    mask = np.asarray(detector["mask"]) != 0
    if np.any(np.sum(np.asarray(reductions["valid_count"]), axis=0) != np.sum(valid, axis=0)):
        _add_error(errors, "reductions.valid_count", "valid_count does not sum to detector valid frames")
    if np.any(
        np.sum(np.asarray(reductions["masked_count_valid"]), axis=0)
        != np.sum(mask & valid, axis=0)
    ):
        _add_error(
            errors,
            "reductions.masked_count_valid",
            "masked_count_valid does not sum to detector masked valid frames",
        )


def validate_products(
    *,
    run_dir: Path,
    output_json: Path | None = None,
    offset_only: bool = False,
) -> dict[str, Any]:
    run = Path(run_dir)
    errors: list[dict[str, str]] = []

    if offset_only:
        # Minimal structural validation for an offset-only chime-scan directory.
        products = _load_npz(run / FREQUENCY_OFFSET_OUTPUTS_FILENAME, errors)
        if products is not None:
            required = [
                "physical_channel", "pilot_frequency_hz", "chime_frequency_hz",
                "frame_index", "relative_time_s", "peak_offset_hz",
                "frequency_offset_hz", "peak_prominence_db", "valid",
                "fft_frequency_axis_hz", "time_average_spectrum_power_linear",
                "time_average_spectrum_count", "fft_size",
            ]
            _require_arrays(
                products,
                filename=FREQUENCY_OFFSET_OUTPUTS_FILENAME,
                names=required,
                errors=errors,
            )
            close = getattr(products, "close", None)
            if callable(close):
                close()
        report = {
            "schema_version": "fstat_chime_product_validation_v1",
            "run_dir": str(run),
            "valid": len(errors) == 0,
            "num_errors": int(len(errors)),
            "errors": errors,
        }
        if output_json is not None:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            write_json_strict(Path(output_json), report, indent=2, sort_keys=True)
        return report

    run_config = _load_json(run / "run_config.json", errors)
    input_manifest = _load_json(run / "input_manifest.json", errors)
    stats = _load_json(run / "stats.json", errors)
    _ = run_config, input_manifest
    if run_config:
        _check_json_schema(
            payload=run_config,
            filename="run_config.json",
            expected=CHIME_RUN_CONFIG_SCHEMA_VERSION,
            errors=errors,
        )
    if stats:
        _check_json_schema(
            payload=stats,
            filename="stats.json",
            expected=CHIME_STATS_SCHEMA_VERSION,
            errors=errors,
        )
    if run_config and stats:
        _validate_detector_contract(
            run_config=run_config,
            stats=stats,
            errors=errors,
        )

    detector = _load_npz(run / CHIME_DETECTOR_OUTPUTS_FILENAME, errors)
    cache = _load_npz(run / CHIME_SPECTROGRAM_CACHE_FILENAME, errors)
    reductions = _load_npz(run / CHIME_REDUCTIONS_10S_FILENAME, errors)
    frequency_offset = None

    try:
        if detector is not None:
            shape = _validate_detector(
                detector,
                run_config=run_config,
                stats=stats,
                errors=errors,
            )
            if shape is not None and cache is not None:
                _validate_spectrogram_cache(
                    cache,
                    detector,
                    shape=shape,
                    errors=errors,
                )
            if shape is not None and reductions is not None:
                _validate_reductions(
                    reductions,
                    detector,
                    shape=shape,
                    errors=errors,
                )
            diagnostic_requested = bool(
                run_config.get(
                    "frequency_offset_diagnostic",
                    stats.get("frequency_offset_diagnostic", False),
                )
            )
            if shape is not None and diagnostic_requested:
                frequency_offset = _validate_frequency_offset_products(
                    run=run,
                    detector=detector,
                    shape=shape,
                    errors=errors,
                )
                _validate_k_candidate_table(run=run, errors=errors)
    finally:
        for item in [detector, cache, reductions, frequency_offset]:
            if item is not None:
                close = getattr(item, "close", None)
                if callable(close):
                    close()

    report = {
        "schema_version": "fstat_chime_product_validation_v1",
        "run_dir": str(run),
        "valid": len(errors) == 0,
        "num_errors": int(len(errors)),
        "errors": errors,
    }
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        write_json_strict(Path(output_json), report, indent=2, sort_keys=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CHIME run products for internal consistency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--offset-only",
        action="store_true",
        help="Validate a frequency-offset-only run directory without requiring detector products.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_products(
        run_dir=args.run_dir,
        output_json=args.output_json,
        offset_only=args.offset_only,
    )
    print("valid, num_errors, run_dir", flush=True)
    print(
        f"{bool(report['valid'])}, {int(report['num_errors'])}, {report['run_dir']}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"ERROR {error['check']}: {error['message']}", flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
