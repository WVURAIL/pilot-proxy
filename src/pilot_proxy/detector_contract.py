# coding=utf-8
"""Public detector-contract objects shared by CHIME products and bundles."""

from __future__ import annotations

import hashlib
from typing import Any

from pilot_proxy.json_utils import json_dumps_strict

CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION = "pilotproxy_chime_detector_contract_v1"
CHIME_RUN_CONFIG_SCHEMA_VERSION = "fstat_chime_run_config_v2"
CHIME_STATS_SCHEMA_VERSION = "fstat_chime_stats_v2"
POSITIVE_EXCESS_MASK_SOURCE = "positive_excess"
POSITIVE_EXCESS_VALID_RULE = "p_ref_sum != 0"
POSITIVE_EXCESS_MASK_RULE = "valid && (p_target > (p_ref_sum >> 1))"
POSITIVE_EXCESS_EQUIVALENT_RULE = "2*p_target > p_ref_sum"
DETECTOR_STATISTIC = "F = 2 * P_target / (P_ref_lower + P_ref_upper)"
ALL_ROWS_DETECTOR_STATISTIC = (
    "F = 2 * sum(P_target) / (sum(P_ref_lower) + sum(P_ref_upper))"
)
COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO = "all_rows_summed_before_ratio"
WEIGHT_COORDINATE_POST_SPECTRAL_SENSE = "post_spectral_sense_normalization"
WEIGHT_COORDINATE_RAW_INPUT = "raw_input_frequency_coordinate"
INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED = (
    "post_spectral_sense_normalized"
)
INPUT_COORDINATE_RAW_INPUT = "raw_input_frequency_coordinate"
VALID_WEIGHT_COORDINATE_SYSTEMS = frozenset(
    {
        WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
        WEIGHT_COORDINATE_RAW_INPUT,
    }
)


def normalize_weight_coordinate_system(value: object) -> str:
    """Return a validated public weight-coordinate-system string."""
    normalized = str(value).strip()
    if normalized not in VALID_WEIGHT_COORDINATE_SYSTEMS:
        raise ValueError(
            "weight_coordinate_system must be one of "
            f"{sorted(VALID_WEIGHT_COORDINATE_SYSTEMS)}; got {value!r}."
        )
    return normalized


def input_coordinate_system_for_weight_coordinate(
    weight_coordinate_system: object,
) -> str:
    """Return the input coordinate needed by the selected weights."""
    normalized = normalize_weight_coordinate_system(weight_coordinate_system)
    if normalized == WEIGHT_COORDINATE_RAW_INPUT:
        return INPUT_COORDINATE_RAW_INPUT
    return INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED


def positive_excess_mask_policy() -> dict[str, Any]:
    """Return the thresholdless positive-excess masking policy."""
    return {
        "mask_source": POSITIVE_EXCESS_MASK_SOURCE,
        "valid_rule": POSITIVE_EXCESS_VALID_RULE,
        "mask_rule": POSITIVE_EXCESS_MASK_RULE,
        "equivalent_rule": POSITIVE_EXCESS_EQUIVALENT_RULE,
    }


def build_chime_detector_contract(
    *,
    detector_window_samples: int,
    skipped_guard_bins: int,
    reference_offset_bins: int,
    num_weight_terms: int,
    sample_bits_per_component: int = 4,
    input_format: str = "complex_int4_packed_int8",
    power_accumulator: str = "uint64",
    power_accumulator_bits: int = 64,
    combine_mode: str = COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    weight_coordinate_system: str,
    input_coordinate_system: str | None = None,
    time_reverse_detector_windows_before_kernel: bool = True,
    reference_placement_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public CHIME detector contract for run products."""
    mask_policy = positive_excess_mask_policy()
    weight_coordinate = normalize_weight_coordinate_system(weight_coordinate_system)
    input_coordinate = (
        input_coordinate_system_for_weight_coordinate(weight_coordinate)
        if input_coordinate_system is None
        else str(input_coordinate_system)
    )
    contract: dict[str, Any] = {
        "schema_version": CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION,
        "detector_window_samples": int(detector_window_samples),
        "skipped_guard_bins": int(skipped_guard_bins),
        "reference_offset_bins": int(reference_offset_bins),
        "num_weight_terms": int(num_weight_terms),
        "sample_bits_per_component": int(sample_bits_per_component),
        "input_format": str(input_format),
        "power_accumulator": str(power_accumulator),
        "power_accumulator_bits": int(power_accumulator_bits),
        "statistic": DETECTOR_STATISTIC,
        "all_rows_statistic": ALL_ROWS_DETECTOR_STATISTIC,
        "combine_mode": str(combine_mode),
        "weight_coordinate_system": weight_coordinate,
        "input_coordinate_system": input_coordinate,
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": bool(
                time_reverse_detector_windows_before_kernel
            ),
        },
        "mask_source": mask_policy["mask_source"],
        "valid_rule": mask_policy["valid_rule"],
        "mask_rule": mask_policy["mask_rule"],
        "equivalent_mask_rule": mask_policy["equivalent_rule"],
        "per_frequency_threshold": False,
        "threshold_mode": "none",
    }
    if reference_placement_summary is not None:
        contract["reference_placement_summary"] = reference_placement_summary
    return contract


def detector_contract_sha256(contract: dict[str, Any]) -> str:
    """Return the stable SHA256 for a detector-contract JSON object."""
    payload = json_dumps_strict(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ALL_ROWS_DETECTOR_STATISTIC",
    "CHIME_DETECTOR_CONTRACT_SCHEMA_VERSION",
    "CHIME_RUN_CONFIG_SCHEMA_VERSION",
    "CHIME_STATS_SCHEMA_VERSION",
    "COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO",
    "DETECTOR_STATISTIC",
    "INPUT_COORDINATE_POST_SPECTRAL_SENSE_NORMALIZED",
    "INPUT_COORDINATE_RAW_INPUT",
    "POSITIVE_EXCESS_EQUIVALENT_RULE",
    "POSITIVE_EXCESS_MASK_RULE",
    "POSITIVE_EXCESS_MASK_SOURCE",
    "POSITIVE_EXCESS_VALID_RULE",
    "VALID_WEIGHT_COORDINATE_SYSTEMS",
    "WEIGHT_COORDINATE_POST_SPECTRAL_SENSE",
    "WEIGHT_COORDINATE_RAW_INPUT",
    "build_chime_detector_contract",
    "detector_contract_sha256",
    "input_coordinate_system_for_weight_coordinate",
    "normalize_weight_coordinate_system",
    "positive_excess_mask_policy",
]
