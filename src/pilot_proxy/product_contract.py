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

PER_PILOT_PRODUCT_SCHEMA_NAME = "pilotproxy_per_pilot_product"
PER_PILOT_PRODUCT_SCHEMA_REVISION = 1
PER_PILOT_PRODUCT_SCHEMA_TOKEN = (
    f"{PER_PILOT_PRODUCT_SCHEMA_NAME}_v{PER_PILOT_PRODUCT_SCHEMA_REVISION}"
)

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


def _scalar(product: Mapping[str, Any], field: str) -> Any:
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    value = np.asarray(product[field])
    if value.size != 1:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be scalar"
        )
    return value.reshape(()).item()


def validate_current_product_identity(product: Mapping[str, Any]) -> None:
    """Require the only supported schema and decision semantics.

    Development snapshots must be deleted and regenerated. This function
    intentionally contains no aliases, migration, or best-effort fallback.
    """
    schema_name = str(_scalar(product, "schema_name"))
    schema_revision = int(_scalar(product, "schema_revision"))
    schema_token = str(_scalar(product, "schema_version"))
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

    raw_contract = str(_scalar(product, "decision_contract_json"))
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
        "pilot_frequency_hz",
        "chime_frequency_hz",
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
        "fine_null_bulk_exceedance_fraction",
        "source_event_keys",
        "frame_unit_index",
        "frame_in_unit",
        "detector_contract_json",
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


def current_decision_contract_json() -> str:
    """Return the canonical stable serialization stored in products."""
    return json.dumps(current_decision_contract(), sort_keys=True, separators=(",", ":"))
