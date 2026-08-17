# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.product_contract import (
    CurrentProductContractError,
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    SOURCE_EVENT_KEY_SCHEMA_VERSION,
    current_decision_contract_json,
    validate_current_product_identity,
)


def current_product() -> dict[str, np.ndarray]:
    return {
        "schema_name": np.asarray(PER_PILOT_PRODUCT_SCHEMA_NAME),
        "schema_revision": np.asarray(PER_PILOT_PRODUCT_SCHEMA_REVISION),
        "schema_version": np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        "source_event_key_schema_version": np.asarray(
            SOURCE_EVENT_KEY_SCHEMA_VERSION
        ),
        "decision_contract_json": np.asarray(current_decision_contract_json()),
        "detector_contract_json": np.asarray("{\"geometry\":\"current\"}"),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "pilot_in_band": np.asarray([1], dtype=np.uint8),
        "pilot_frequency_hz": np.asarray([470_309_441.0]),
        "chime_frequency_hz": np.asarray([470_312_500.0]),
        "nfft": np.asarray(1, dtype=np.int64),
        "sample_rate_hz": np.asarray(390_625.0, dtype=np.float64),
        "detector_window_samples": np.asarray(1, dtype=np.int64),
        "num_input_streams": np.asarray(1, dtype=np.int64),
        "sense": np.asarray(-1, dtype=np.int64),
        "frame_index": np.asarray([0], dtype=np.int64),
        "p_target_u64": np.asarray([[1]], dtype=np.uint64),
        "p_ref_sum_u64": np.asarray([[2]], dtype=np.uint64),
        "coarse_power_ratio": np.asarray([[1.0]]),
        "normalized_coarse_power_ratio_db": np.asarray([[0.0]]),
        "pilot_excess_db": np.asarray([[np.nan]]),
        "estimated_data_shelf_snr_db": np.asarray([[np.nan]]),
        "reject_mask": np.asarray([[0]], dtype=np.uint8),
        "valid": np.asarray([[1]], dtype=np.uint8),
        "target_norm_sq": np.asarray([1], dtype=np.int64),
        "reference_norm_sum_sq": np.asarray([2], dtype=np.int64),
        "null_power_ratio": np.asarray([1.0]),
        "normalized_pilot_excess": np.asarray([[0.0]]),
        "baseband_power_linear": np.asarray([[1.0]]),
        "integrated_spectrum_before_mask": np.asarray([1.0]),
        "integrated_spectrum_after_mask": np.asarray([1.0]),
        "fine_power_ratio": np.zeros((1, 0), dtype=np.float32),
        "fine_cfar_location": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_scale": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_threshold": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_mode": np.asarray([[0]], dtype=np.uint8),
        "fine_threshold_exceedance_count": np.asarray([[0]], dtype=np.int32),
        "fine_threshold_exceedance_frame": np.asarray([], dtype=np.int64),
        "fine_threshold_exceedance_bin": np.asarray([], dtype=np.int64),
        "fine_pad_factor": np.asarray(4, dtype=np.int64),
        "fine_num_bins": np.asarray(0, dtype=np.int64),
        "fine_p_fa": np.asarray(0.001, dtype=np.float64),
        "fine_guard_fine_bins": np.asarray(1, dtype=np.int64),
        "fine_designated_bins": np.asarray([0], dtype=np.int64),
        "fine_census_excluded_bins": np.asarray([], dtype=np.int64),
        "fine_status": np.asarray("disabled"),
        "fine_null_bulk_exceedance_fraction": np.asarray([[np.nan]]),
        "source_event_keys": np.asarray(["event"], dtype=str),
        "unit_keys": np.asarray(["event"], dtype=str),
        "unit_order": np.asarray(["event"], dtype=str),
        "unit_time0_ctime": np.asarray([np.nan], dtype=np.float64),
        "unit_time0_fpga": np.asarray([0], dtype=np.uint64),
        "unit_event_id": np.asarray([-1], dtype=np.int64),
        "unit_delta_time": np.asarray([1.0 / 390_625.0], dtype=np.float64),
        "archive_version": np.asarray([""], dtype=str),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
        "max_chunks_per_file": np.asarray(-1, dtype=np.int64),
        "rational_overflow_count": np.asarray(0, dtype=np.uint64),
        "weights_hash": np.asarray("weights"),
        "weight_bank_sha256": np.asarray("bank"),
        "weight_manifest_sha256": np.asarray("manifest"),
        "detector_version": np.asarray("detector"),
        "mask_rule": np.asarray(NORMALIZED_POSITIVE_EXCESS_MASK_RULE),
        "reference_placement_json": np.asarray("{}"),
        "pilot_below_data_db": np.asarray(11.3, dtype=np.float64),
        "bin_enbw_hz": np.asarray(1.0, dtype=np.float64),
        "dtv_bandwidth_hz": np.asarray(6.0e6, dtype=np.float64),
        "pilot_capture_efficiency": np.asarray(1.0, dtype=np.float64),
    }


def test_current_product_is_accepted():
    validate_current_product_identity(current_product())


@pytest.mark.parametrize(
    "field",
    [
        "schema_name",
        "source_event_key_schema_version",
        "decision_contract_json",
        "source_event_keys",
        "unit_order",
        "reject_mask",
    ],
)
def test_missing_current_field_is_refused(field: str):
    product = current_product()
    product.pop(field)
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_alternate_field_name_is_refused():
    product = current_product()
    product["fine_nd_flag_rate"] = product.pop(
        "fine_null_bulk_exceedance_fraction"
    )
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_legacy_basename_event_identity_is_refused():
    product = current_product()
    product["source_event_key_schema_version"] = np.asarray(
        "pilotproxy_basename_source_event_key_v1"
    )
    with pytest.raises(CurrentProductContractError, match="source-event identity"):
        validate_current_product_identity(product)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("schema_revision", np.asarray(1.9, dtype=np.float64)),
        ("freq_id", np.asarray([844.9], dtype=np.float64)),
        ("physical_channel", np.asarray([14.9], dtype=np.float64)),
        ("frame_unit_index", np.asarray([0.1], dtype=np.float64)),
        ("frame_in_unit", np.asarray([False], dtype=np.bool_)),
        ("unit_event_id", np.asarray([123.1], dtype=np.float64)),
        ("unit_time0_fpga", np.asarray([123.1], dtype=np.float64)),
        ("p_target_u64", np.asarray([[1.0]], dtype=np.float64)),
        ("p_target_u64", np.asarray([[-1]], dtype=np.int64)),
        ("p_target_u64", np.asarray([[1], [2]], dtype=np.uint64)),
        ("sample_rate_hz", np.asarray(0.0, dtype=np.float64)),
        ("mask_rule", np.asarray("forged-policy")),
        ("reject_mask", np.asarray([[1]], dtype=np.uint8)),
    ],
)
def test_malformed_current_scalar_or_frame_array_is_refused(
    field: str, malformed: np.ndarray
) -> None:
    product = current_product()
    product[field] = malformed
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_empty_checkpoint_cannot_claim_a_consumed_unit() -> None:
    product = current_product()
    for field, dtype in (
        ("frame_index", np.int64),
        ("frame_unit_index", np.int32),
        ("frame_in_unit", np.int32),
        ("fine_threshold_exceedance_frame", np.int64),
        ("fine_threshold_exceedance_bin", np.int64),
    ):
        product[field] = np.asarray([], dtype=dtype)
    for field, dtype in (
        ("p_target_u64", np.uint64),
        ("p_ref_sum_u64", np.uint64),
        ("reject_mask", np.uint8),
        ("valid", np.uint8),
        ("fine_cfar_mode", np.uint8),
        ("fine_threshold_exceedance_count", np.int32),
    ):
        product[field] = np.empty((0, 1), dtype=dtype)
    for field in (
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "normalized_pilot_excess",
        "baseband_power_linear",
        "fine_cfar_location",
        "fine_cfar_scale",
        "fine_cfar_threshold",
        "fine_null_bulk_exceedance_fraction",
    ):
        product[field] = np.empty((0, 1), dtype=np.float64)
    product["fine_power_ratio"] = np.empty((0, 0), dtype=np.float32)

    with pytest.raises(CurrentProductContractError, match="must not claim"):
        validate_current_product_identity(product, allow_empty_checkpoint=True)


def test_nonempty_product_cannot_contain_an_unused_unit() -> None:
    product = current_product()
    product["unit_keys"] = np.asarray(["event", "unused"], dtype=str)
    product["unit_order"] = np.asarray(["event", "unused"], dtype=str)
    product["source_event_keys"] = np.asarray(["event", "unused"], dtype=str)
    product["archive_version"] = np.asarray(["", ""], dtype=str)
    product["unit_time0_ctime"] = np.asarray([np.nan, np.nan], dtype=np.float64)
    product["unit_time0_fpga"] = np.asarray([0, 0], dtype=np.uint64)
    product["unit_event_id"] = np.asarray([-1, -1], dtype=np.int64)
    product["unit_delta_time"] = np.asarray(
        [1.0 / 390_625.0, 1.0 / 390_625.0], dtype=np.float64
    )

    with pytest.raises(CurrentProductContractError, match="every consumed unit"):
        validate_current_product_identity(product)
