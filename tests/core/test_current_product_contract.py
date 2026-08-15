# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.product_contract import (
    CurrentProductContractError,
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    current_decision_contract_json,
    validate_current_product_identity,
)


def current_product() -> dict[str, np.ndarray]:
    return {
        "schema_name": np.asarray(PER_PILOT_PRODUCT_SCHEMA_NAME),
        "schema_revision": np.asarray(PER_PILOT_PRODUCT_SCHEMA_REVISION),
        "schema_version": np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        "decision_contract_json": np.asarray(current_decision_contract_json()),
        "detector_contract_json": np.asarray("{\"geometry\":\"current\"}"),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "pilot_frequency_hz": np.asarray([470_309_441.0]),
        "chime_frequency_hz": np.asarray([470_312_500.0]),
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
        "fine_null_bulk_exceedance_fraction": np.asarray([[np.nan]]),
        "source_event_keys": np.asarray(["event"], dtype=str),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
    }


def test_current_product_is_accepted():
    validate_current_product_identity(current_product())


@pytest.mark.parametrize(
    "field",
    ["schema_name", "decision_contract_json", "source_event_keys", "reject_mask"],
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
