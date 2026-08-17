# coding=utf-8
"""Current per-pilot product and decision-method identities.

Schema identity, scientific measurement, and rejection policy are separate
coordinates.  Keeping them explicit prevents a diagnostic fine spectrum or an
implemented kernel entry point from being mistaken for the active mask used by
an archive product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from pilot_proxy.atsc_channels import (
    ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
)
from pilot_proxy.detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    normalized_positive_excess,
)

PER_PILOT_PRODUCT_SCHEMA_NAME = "pilotproxy_per_pilot_product"
PER_PILOT_PRODUCT_SCHEMA_REVISION = 1
PER_PILOT_PRODUCT_SCHEMA_TOKEN = (
    f"{PER_PILOT_PRODUCT_SCHEMA_NAME}_v{PER_PILOT_PRODUCT_SCHEMA_REVISION}"
)
SOURCE_EVENT_KEY_SCHEMA_VERSION = "pilotproxy_namespaced_source_event_key_v1"

COARSE_MEASUREMENT_METHOD = "coarse_local_reference_power_ratio"
FINE_MEASUREMENT_METHOD = "fine_local_reference_power_ratio"
ACTIVE_DECISION_METHOD = "coarse_normalized_positive_excess"
ACTIVE_DECISION_IMPLEMENTATION = "host_exact_integer_comparison"
FINE_DIAGNOSTIC_METHOD = "per_frame_robust_null_bulk_threshold"
FINE_DIAGNOSTIC_ROLE = "diagnostic_only"
FINE_CANDIDATE_DECISION_METHOD = "fine_order_statistic_cfar"
FINE_CANDIDATE_CALIBRATION_STATUS = "pending_campaign"


def current_decision_contract() -> dict[str, Any]:
    """Return a fresh JSON-safe description of product decision semantics."""
    return {
        "measurements": [COARSE_MEASUREMENT_METHOD, FINE_MEASUREMENT_METHOD],
        "active_decision": {
            "method": ACTIVE_DECISION_METHOD,
            "implementation": ACTIVE_DECISION_IMPLEMENTATION,
            "output_field": "reject_mask",
        },
        "fine_diagnostic": {
            "method": FINE_DIAGNOSTIC_METHOD,
            "role": FINE_DIAGNOSTIC_ROLE,
            "ratio_field": "fine_power_ratio",
            "threshold_field": "fine_cfar_threshold",
            "exceedance_fraction_field": (
                "fine_null_bulk_exceedance_fraction"
            ),
        },
        "fine_candidate_decision": {
            "method": FINE_CANDIDATE_DECISION_METHOD,
            "calibration_status": FINE_CANDIDATE_CALIBRATION_STATUS,
            "active": False,
        },
    }


class CurrentProductContractError(ValueError):
    """A per-pilot product does not satisfy the only supported contract."""


def exact_integer_array(
    value: Any,
    *,
    field: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
    ndim: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an integer array without accepting bool/float/string coercion."""
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"} or array.dtype.kind == "b":
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have an integer dtype"
        )
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if minimum is not None and array.size and np.any(array < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum}"
        )
    if maximum is not None and array.size and np.any(array > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum}"
        )
    return array


def exact_integer_scalar(
    product: Mapping[str, Any],
    field: str,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    array = exact_integer_array(
        product.get(field, np.asarray([], dtype=dtype)),
        field=field,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
    )
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    if array.size != 1:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be scalar"
        )
    return int(array.reshape(()).item())


def _exact_string_scalar(product: Mapping[str, Any], field: str) -> str:
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    value = np.asarray(product[field])
    if value.size != 1 or value.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string scalar"
        )
    return str(value.reshape(()).item())


def _exact_float_array(
    product: Mapping[str, Any],
    field: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    allow_nan: bool = True,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    array = np.asarray(product[field])
    if array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if shape is not None and array.shape != shape:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have shape {shape}, "
            f"got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} contains non-finite values"
        )
    finite = array[np.isfinite(array)]
    if minimum is not None and finite.size and np.any(finite < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum} when finite"
        )
    if maximum is not None and finite.size and np.any(finite > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum} when finite"
        )
    return array


def _exact_string_vector(
    product: Mapping[str, Any], field: str, *, length: int
) -> list[str]:
    array = np.asarray(product[field])
    if array.ndim != 1 or array.size != length or array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string vector of "
            f"length {length}"
        )
    return [str(value) for value in array.tolist()]


def validate_current_product_identity(
    product: Mapping[str, Any], *, allow_empty_checkpoint: bool = False
) -> None:
    """Require the only supported schema and decision semantics.

    Development snapshots must be deleted and regenerated. This function
    intentionally contains no aliases, migration, or best-effort fallback.
    """
    schema_name = _exact_string_scalar(product, "schema_name")
    schema_revision = exact_integer_scalar(
        product,
        "schema_revision",
        dtype=np.int64,
        minimum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
        maximum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
    )
    schema_token = _exact_string_scalar(product, "schema_version")
    if (
        schema_name != PER_PILOT_PRODUCT_SCHEMA_NAME
        or schema_revision != PER_PILOT_PRODUCT_SCHEMA_REVISION
        or schema_token != PER_PILOT_PRODUCT_SCHEMA_TOKEN
    ):
        raise CurrentProductContractError(
            "unsupported per-pilot product identity: "
            f"schema_name={schema_name!r}, schema_revision={schema_revision!r}, "
            f"schema_version={schema_token!r}; delete the product and regenerate "
            "it with the current PilotProxy release"
        )

    event_key_schema = _exact_string_scalar(
        product, "source_event_key_schema_version"
    )
    if event_key_schema != SOURCE_EVENT_KEY_SCHEMA_VERSION:
        raise CurrentProductContractError(
            "unsupported source-event identity schema: "
            f"{event_key_schema!r}; delete the product and regenerate it so "
            "event keys retain their archive/campaign namespace"
        )

    raw_contract = _exact_string_scalar(product, "decision_contract_json")
    try:
        decision_contract = json.loads(raw_contract)
    except json.JSONDecodeError as exc:
        raise CurrentProductContractError(
            "current per-pilot decision_contract_json is invalid JSON"
        ) from exc
    if decision_contract != current_decision_contract():
        raise CurrentProductContractError(
            "per-pilot decision contract does not match the current release; "
            "delete the product and regenerate it"
        )

    required = {
        "physical_channel",
        "freq_id",
        "pilot_in_band",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "nfft",
        "sample_rate_hz",
        "detector_window_samples",
        "num_input_streams",
        "sense",
        "frame_index",
        "p_target_u64",
        "p_ref_sum_u64",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "reject_mask",
        "valid",
        "target_norm_sq",
        "reference_norm_sum_sq",
        "null_power_ratio",
        "normalized_pilot_excess",
        "baseband_power_linear",
        "integrated_spectrum_before_mask",
        "integrated_spectrum_after_mask",
        "fine_power_ratio",
        "fine_cfar_location",
        "fine_cfar_scale",
        "fine_cfar_threshold",
        "fine_cfar_mode",
        "fine_threshold_exceedance_count",
        "fine_threshold_exceedance_frame",
        "fine_threshold_exceedance_bin",
        "fine_pad_factor",
        "fine_num_bins",
        "fine_p_fa",
        "fine_guard_fine_bins",
        "fine_designated_bins",
        "fine_census_excluded_bins",
        "fine_status",
        "fine_null_bulk_exceedance_fraction",
        "source_event_keys",
        "unit_keys",
        "unit_order",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "unit_event_id",
        "unit_delta_time",
        "archive_version",
        "frame_unit_index",
        "frame_in_unit",
        "detector_contract_json",
        "max_chunks_per_file",
        "rational_overflow_count",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
        "pilot_below_data_db",
        "bin_enbw_hz",
        "dtv_bandwidth_hz",
        "pilot_capture_efficiency",
    }
    missing = sorted(required.difference(product))
    if missing:
        raise CurrentProductContractError(
            "current per-pilot product is missing required arrays: "
            + ", ".join(missing)
        )

    if "mask" in product:
        raise CurrentProductContractError(
            "per-pilot product contains the ambiguous field 'mask'; the current "
            "decision field is 'reject_mask'"
        )

    exact_integer_scalar(
        product,
        "physical_channel",
        dtype=np.int32,
        minimum=ATSC_UHF_MIN_PHYSICAL_CHANNEL,
        maximum=ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    )
    exact_integer_scalar(
        product, "freq_id", dtype=np.int64, minimum=0, maximum=1023
    )
    nfft = exact_integer_scalar(
        product, "nfft", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "detector_window_samples", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "num_input_streams", dtype=np.int64, minimum=1
    )
    sense = exact_integer_scalar(
        product, "sense", dtype=np.int64, minimum=-1, maximum=1
    )
    if sense not in {-1, 1}:
        raise CurrentProductContractError(
            "current per-pilot field 'sense' must be exactly -1 or 1"
        )
    exact_integer_scalar(
        product, "max_chunks_per_file", dtype=np.int64, minimum=-1
    )
    exact_integer_scalar(
        product, "rational_overflow_count", dtype=np.uint64, minimum=0
    )
    pilot_in_band = exact_integer_array(
        product["pilot_in_band"],
        field="pilot_in_band",
        dtype=np.uint8,
        ndim=1,
        minimum=0,
        maximum=1,
    )
    if pilot_in_band.shape != (1,):
        raise CurrentProductContractError(
            "current per-pilot field 'pilot_in_band' must have shape (1,)"
        )

    _exact_float_array(
        product, "pilot_frequency_hz", shape=(1,), allow_nan=False
    )
    _exact_float_array(
        product, "chime_frequency_hz", shape=(1,), allow_nan=False
    )
    sample_rate = _exact_float_array(
        product, "sample_rate_hz", shape=(), allow_nan=False, minimum=0.0
    )
    if float(sample_rate.reshape(()).item()) <= 0.0:
        raise CurrentProductContractError(
            "current per-pilot field 'sample_rate_hz' must be positive"
        )
    for field, minimum, maximum in (
        ("pilot_below_data_db", 0.0, None),
        ("bin_enbw_hz", 0.0, None),
        ("dtv_bandwidth_hz", 0.0, None),
        ("pilot_capture_efficiency", 0.0, 1.0),
    ):
        values = _exact_float_array(
            product,
            field,
            shape=(),
            allow_nan=False,
            minimum=minimum,
            maximum=maximum,
        )
        if field != "pilot_below_data_db" and float(values.reshape(()).item()) <= 0.0:
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must be positive"
            )
    for field in (
        "detector_contract_json",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
    ):
        _exact_string_scalar(product, field)
    if _exact_string_scalar(product, "mask_rule") != NORMALIZED_POSITIVE_EXCESS_MASK_RULE:
        raise CurrentProductContractError(
            "current per-pilot mask_rule does not declare the normalized "
            "positive-excess policy"
        )

    frame_index = exact_integer_array(
        product["frame_index"],
        field="frame_index",
        dtype=np.int64,
        ndim=1,
        minimum=0,
    )
    frame_count = int(frame_index.size)
    if (frame_count == 0 and not allow_empty_checkpoint) or not np.array_equal(
        frame_index, np.arange(frame_count, dtype=np.int64)
    ):
        raise CurrentProductContractError(
            "current per-pilot frame_index must be a non-empty contiguous "
            "zero-based int64 vector"
        )

    unit_order_array = np.asarray(product["unit_order"])
    if unit_order_array.ndim != 1 or unit_order_array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            "current per-pilot field 'unit_order' must be a 1D string vector"
        )
    unit_count = int(unit_order_array.size)
    if unit_count == 0 and not allow_empty_checkpoint:
        raise CurrentProductContractError(
            "current per-pilot unit_order must not be empty"
        )
    unit_order = _exact_string_vector(
        product, "unit_order", length=unit_count
    )
    unit_keys = _exact_string_vector(product, "unit_keys", length=unit_count)
    source_event_keys = _exact_string_vector(
        product, "source_event_keys", length=unit_count
    )
    _exact_string_vector(product, "archive_version", length=unit_count)
    if len(set(unit_order)) != unit_count or len(set(unit_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must each be unique"
        )
    if set(unit_order) != set(unit_keys):
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must contain the exact "
            "same unit identities"
        )
    if any(not value for value in unit_order + unit_keys + source_event_keys):
        raise CurrentProductContractError(
            "current per-pilot unit and source-event identities must not be empty"
        )
    if len(set(source_event_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot source_event_keys must be unique"
        )

    unit_event_id = exact_integer_array(
        product["unit_event_id"],
        field="unit_event_id",
        dtype=np.int64,
        ndim=1,
        minimum=-1,
    )
    unit_time0_fpga = exact_integer_array(
        product["unit_time0_fpga"],
        field="unit_time0_fpga",
        dtype=np.uint64,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("unit_event_id", unit_event_id),
        ("unit_time0_fpga", unit_time0_fpga),
    ):
        if values.shape != (unit_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({unit_count},)"
            )
    _exact_float_array(
        product, "unit_time0_ctime", shape=(unit_count,), allow_nan=True
    )
    unit_delta_time = _exact_float_array(
        product,
        "unit_delta_time",
        shape=(unit_count,),
        allow_nan=True,
        minimum=0.0,
    )
    if np.any(np.isfinite(unit_delta_time) & (unit_delta_time <= 0.0)):
        raise CurrentProductContractError(
            "current per-pilot finite unit_delta_time values must be positive"
        )
    finite_delta_time = unit_delta_time[np.isfinite(unit_delta_time)]
    if finite_delta_time.size and not np.allclose(
        finite_delta_time,
        1.0 / float(sample_rate.reshape(()).item()),
        rtol=1e-12,
        atol=0.0,
    ):
        raise CurrentProductContractError(
            "current per-pilot unit_delta_time disagrees with sample_rate_hz"
        )

    frame_unit_index = exact_integer_array(
        product["frame_unit_index"],
        field="frame_unit_index",
        dtype=np.int32,
        ndim=1,
        minimum=0,
        maximum=unit_count - 1,
    )
    frame_in_unit = exact_integer_array(
        product["frame_in_unit"],
        field="frame_in_unit",
        dtype=np.int32,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("frame_unit_index", frame_unit_index),
        ("frame_in_unit", frame_in_unit),
    ):
        if values.shape != (frame_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count},)"
            )
    if frame_count == 0 and unit_count != 0:
        raise CurrentProductContractError(
            "an empty current per-pilot checkpoint must not claim consumed units"
        )
    if frame_count and set(frame_unit_index.tolist()) != set(range(unit_count)):
        raise CurrentProductContractError(
            "current per-pilot frame_unit_index must reference every consumed "
            "unit at least once"
        )

    exact_powers: dict[str, np.ndarray] = {}
    for field in ("p_target_u64", "p_ref_sum_u64"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint64,
            minimum=0,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_powers[field] = values
    exact_flags: dict[str, np.ndarray] = {}
    for field in ("reject_mask", "valid"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint8,
            minimum=0,
            maximum=1,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_flags[field] = values
    for field, minimum in (
        ("coarse_power_ratio", 0.0),
        ("normalized_coarse_power_ratio_db", None),
        ("pilot_excess_db", None),
        ("estimated_data_shelf_snr_db", None),
        ("normalized_pilot_excess", -1.0),
        ("baseband_power_linear", 0.0),
        ("fine_cfar_location", 0.0),
        ("fine_cfar_scale", 0.0),
        ("fine_cfar_threshold", 0.0),
        ("fine_null_bulk_exceedance_fraction", 0.0),
    ):
        values = _exact_float_array(
            product,
            field,
            shape=(frame_count, 1),
            allow_nan=True,
            minimum=minimum,
        )
        if field == "fine_null_bulk_exceedance_fraction":
            finite = values[np.isfinite(values)]
            if finite.size and np.any(finite > 1.0):
                raise CurrentProductContractError(
                    "current per-pilot fine_null_bulk_exceedance_fraction must "
                    "be in [0, 1] when finite"
                )
    fine = _exact_float_array(
        product,
        "fine_power_ratio",
        ndim=2,
        dtype=np.float32,
        allow_nan=True,
        minimum=0.0,
    )
    if fine.shape[0] != frame_count:
        raise CurrentProductContractError(
            "current per-pilot fine_power_ratio frame axis must match frame_index"
        )

    fine_num_bins = exact_integer_scalar(
        product, "fine_num_bins", dtype=np.int64, minimum=0
    )
    if fine.shape[1] != fine_num_bins:
        raise CurrentProductContractError(
            "current per-pilot fine_power_ratio width must match fine_num_bins"
        )
    exact_integer_scalar(
        product, "fine_pad_factor", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "fine_guard_fine_bins", dtype=np.int64, minimum=0
    )
    fine_p_fa = _exact_float_array(
        product, "fine_p_fa", shape=(), allow_nan=False, minimum=0.0, maximum=1.0
    )
    if not 0.0 < float(fine_p_fa.reshape(()).item()) < 1.0:
        raise CurrentProductContractError(
            "current per-pilot fine_p_fa must be strictly between zero and one"
        )
    for field in ("fine_designated_bins", "fine_census_excluded_bins"):
        exact_integer_array(product[field], field=field, dtype=np.int64, ndim=1)
    _exact_string_scalar(product, "fine_status")
    fine_mode = exact_integer_array(
        product["fine_cfar_mode"],
        field="fine_cfar_mode",
        dtype=np.uint8,
        minimum=0,
        maximum=2,
    )
    fine_counts = exact_integer_array(
        product["fine_threshold_exceedance_count"],
        field="fine_threshold_exceedance_count",
        dtype=np.int32,
        minimum=0,
    )
    for field, values in (
        ("fine_cfar_mode", fine_mode),
        ("fine_threshold_exceedance_count", fine_counts),
    ):
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
    exceedance_frames = exact_integer_array(
        product["fine_threshold_exceedance_frame"],
        field="fine_threshold_exceedance_frame",
        dtype=np.int64,
        ndim=1,
        minimum=0,
        maximum=frame_count - 1,
    )
    exceedance_bins = exact_integer_array(
        product["fine_threshold_exceedance_bin"],
        field="fine_threshold_exceedance_bin",
        dtype=np.int64,
        ndim=1,
        minimum=0,
        maximum=max(0, fine_num_bins - 1),
    )
    if (
        exceedance_frames.shape != exceedance_bins.shape
        or exceedance_frames.size != int(np.sum(fine_counts, dtype=np.int64))
    ):
        raise CurrentProductContractError(
            "current per-pilot fine threshold exceedance rows must match the "
            "per-frame exceedance counts"
        )
    if fine_num_bins == 0 and exceedance_bins.size:
        raise CurrentProductContractError(
            "current per-pilot fine threshold exceedance bins require fine bins"
        )

    exact_norms: dict[str, int] = {}
    for field in ("target_norm_sq", "reference_norm_sum_sq"):
        values = exact_integer_array(
            product[field], field=field, dtype=np.int64, minimum=1
        )
        if values.shape != (1,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape (1,)"
            )
        exact_norms[field] = int(values[0])
    null_ratio = _exact_float_array(
        product, "null_power_ratio", shape=(1,), allow_nan=False, minimum=0.0
    )
    if float(null_ratio[0]) <= 0.0:
        raise CurrentProductContractError(
            "current per-pilot null_power_ratio must be positive"
        )
    expected_null_ratio = (
        2.0 * exact_norms["target_norm_sq"]
        / exact_norms["reference_norm_sum_sq"]
    )
    if float(null_ratio[0]) != expected_null_ratio:
        raise CurrentProductContractError(
            "current per-pilot null_power_ratio disagrees with the exact "
            "stored weight norms"
        )

    p_target = exact_powers["p_target_u64"].reshape(-1)
    p_ref_sum = exact_powers["p_ref_sum_u64"].reshape(-1)
    valid = exact_flags["valid"].reshape(-1)
    reject = exact_flags["reject_mask"].reshape(-1)
    expected_valid = (p_ref_sum != 0).astype(np.uint8)
    if not np.array_equal(valid, expected_valid):
        raise CurrentProductContractError(
            "current per-pilot valid flags disagree with p_ref_sum != 0"
        )
    expected_reject = np.asarray(
        [
            normalized_positive_excess(
                int(num),
                int(den),
                target_norm_sq=exact_norms["target_norm_sq"],
                reference_norm_sum_sq=exact_norms["reference_norm_sum_sq"],
            )
            for num, den in zip(p_target, p_ref_sum)
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(reject, expected_reject):
        raise CurrentProductContractError(
            "current per-pilot reject_mask disagrees with the exact normalized "
            "positive-excess decision over stored powers and norms"
        )
    spectrum_before = _exact_float_array(
        product,
        "integrated_spectrum_before_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    spectrum_after = _exact_float_array(
        product,
        "integrated_spectrum_after_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    if np.any(spectrum_after > spectrum_before):
        raise CurrentProductContractError(
            "current per-pilot integrated spectrum after mask exceeds before mask"
        )


def current_decision_contract_json() -> str:
    """Return the canonical stable serialization stored in products."""
    return json.dumps(current_decision_contract(), sort_keys=True, separators=(",", ":"))
