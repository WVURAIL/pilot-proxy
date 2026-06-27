# coding=utf-8
"""Detector-core profile for the locked CUDA F-statistic contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import (
    DETECTOR_CORE_ID_PILOT_PROXY_CUDA_V1,
    DETECTOR_CORE_PROFILE_SCHEMA_VERSION,
)

LOCKED_DETECTOR_WINDOW_SAMPLES = 128
LOCKED_NUM_WEIGHT_TERMS = 3
LOCKED_SKIPPED_GUARD_BINS = 1
LOCKED_REFERENCE_OFFSET_BINS = LOCKED_SKIPPED_GUARD_BINS + 1
MIN_SKIPPED_GUARD_BINS = 1
MIN_REFERENCE_OFFSET_BINS = MIN_SKIPPED_GUARD_BINS + 1
LOCKED_PACKED_COMPLEX_BITS = 8
LOCKED_SAMPLE_BITS_PER_COMPONENT = 4
DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET = 16
MAG_SQUARED_ACCUMULATOR_BITS_TARGET = 32
POWER_SUM_ACCUMULATOR_BITS = 64
_REFERENCE_FIELD_PART = "reference"
_OLD_GAP_FIELD_PART = "guard"
_OFFSET_FIELD_PART = "offset"
_BINS_FIELD_PART = "bins"
_NOMINAL_FIELD_PART = "nominal"
_REQUESTED_FIELD_PART = "requested"
_SELECTED_FIELD_PART = "selected"
_MIN_EMPIRICAL_FIELD_PART = "min_empirical"
DEPRECATED_DETECTOR_SPACING_FIELDS = frozenset(
    {
        "_".join((_REFERENCE_FIELD_PART, _OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _REQUESTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _SELECTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _MIN_EMPIRICAL_FIELD_PART,
            )
        ),
        "_".join((_OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OFFSET_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
    }
)
DERIVED_DETECTOR_SPACING_INPUT_FIELDS = frozenset(
    {"_".join((_REFERENCE_FIELD_PART, _OFFSET_FIELD_PART, _BINS_FIELD_PART))}
)
DEPRECATED_THRESHOLD_CONTRACT_FIELDS = frozenset({"threshold_input_to_kernel"})


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _contract_value(
    kernel_contract: dict[str, Any],
    raw: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in kernel_contract:
        return kernel_contract[key]
    return raw.get(key, default)


@dataclass(frozen=True)
class DetectorCoreProfile:
    """File-backed description of what the CUDA kernel consumes and returns."""

    schema_version: str = DETECTOR_CORE_PROFILE_SCHEMA_VERSION
    detector_core_id: str = DETECTOR_CORE_ID_PILOT_PROXY_CUDA_V1
    detector_window_samples: int = LOCKED_DETECTOR_WINDOW_SAMPLES
    num_weight_terms: int = LOCKED_NUM_WEIGHT_TERMS
    skipped_guard_bins: int = LOCKED_SKIPPED_GUARD_BINS
    packed_complex_bits: int = LOCKED_PACKED_COMPLEX_BITS
    sample_bits_per_component: int = LOCKED_SAMPLE_BITS_PER_COMPONENT
    input_format: str = "complex_int4_packed_int8"
    power_accumulator: str = "uint64"
    statistic: str = "F = 2 * P_target / (P_ref_lower + P_ref_upper)"
    pilot_excess: str = "rho = F - 1"
    host_masking_policy: str = "positive_excess_from_uint64_powers"
    per_frequency_threshold: bool = False
    fixed_point_limits: dict[str, int] = field(
        default_factory=lambda: {
            "dot_product_component_accumulator_bits_target": (
                DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET
            ),
            "mag_squared_accumulator_bits_target": (
                MAG_SQUARED_ACCUMULATOR_BITS_TARGET
            ),
            "power_sum_accumulator_bits": POWER_SUM_ACCUMULATOR_BITS,
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != DETECTOR_CORE_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported detector core profile schema_version: "
                f"{self.schema_version!r}"
            )
        if self.detector_core_id != DETECTOR_CORE_ID_PILOT_PROXY_CUDA_V1:
            raise ValueError(f"unsupported detector_core_id: {self.detector_core_id!r}")
        if self.detector_window_samples != LOCKED_DETECTOR_WINDOW_SAMPLES:
            raise ValueError("detector_window_samples must be locked to 128.")
        if self.num_weight_terms != LOCKED_NUM_WEIGHT_TERMS:
            raise ValueError("num_weight_terms must be locked to 3.")
        if self.skipped_guard_bins < MIN_SKIPPED_GUARD_BINS:
            raise ValueError(
                "skipped_guard_bins must be at least "
                f"{MIN_SKIPPED_GUARD_BINS}."
            )
        if self.sample_bits_per_component != LOCKED_SAMPLE_BITS_PER_COMPONENT:
            raise ValueError("sample_bits_per_component must be locked to 4.")
        object.__setattr__(self, "fixed_point_limits", dict(self.fixed_point_limits))

    @property
    def reference_offset_bins(self) -> int:
        return int(self.skipped_guard_bins) + 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorCoreProfile":
        raw = dict(data)
        kernel_contract = _mapping_or_empty(raw.get("kernel_contract"))
        _reject_unsupported_spacing_input_fields(raw)
        _reject_unsupported_spacing_input_fields(kernel_contract)
        _reject_deprecated_threshold_contract_fields(raw)
        _reject_deprecated_threshold_contract_fields(kernel_contract)
        fixed_point_limits = _mapping_or_empty(raw.get("fixed_point_limits"))
        return cls(
            schema_version=str(
                raw.get(
                    "schema_version",
                    DETECTOR_CORE_PROFILE_SCHEMA_VERSION,
                )
            ),
            detector_core_id=str(
                raw.get("detector_core_id", DETECTOR_CORE_ID_PILOT_PROXY_CUDA_V1)
            ),
            detector_window_samples=int(
                _contract_value(
                    kernel_contract,
                    raw,
                    "detector_window_samples",
                    LOCKED_DETECTOR_WINDOW_SAMPLES,
                ),
            ),
            num_weight_terms=int(
                _contract_value(
                    kernel_contract,
                    raw,
                    "num_weight_terms",
                    LOCKED_NUM_WEIGHT_TERMS,
                ),
            ),
            skipped_guard_bins=int(
                _contract_value(
                    kernel_contract,
                    raw,
                    "skipped_guard_bins",
                    LOCKED_SKIPPED_GUARD_BINS,
                ),
            ),
            packed_complex_bits=int(
                _contract_value(
                    kernel_contract,
                    raw,
                    "packed_complex_bits",
                    LOCKED_PACKED_COMPLEX_BITS,
                ),
            ),
            sample_bits_per_component=int(
                _contract_value(
                    kernel_contract,
                    raw,
                    "sample_bits_per_component",
                    LOCKED_SAMPLE_BITS_PER_COMPONENT,
                ),
            ),
            input_format=str(
                kernel_contract.get("input_format", "complex_int4_packed_int8")
            ),
            power_accumulator=str(kernel_contract.get("power_accumulator", "uint64")),
            statistic=str(
                kernel_contract.get(
                    "statistic",
                    "F = 2 * P_target / (P_ref_lower + P_ref_upper)",
                )
            ),
            pilot_excess=str(kernel_contract.get("pilot_excess", "rho = F - 1")),
            host_masking_policy=str(
                kernel_contract.get(
                    "host_masking_policy",
                    "positive_excess_from_uint64_powers",
                )
            ),
            per_frequency_threshold=bool(
                kernel_contract.get("per_frequency_threshold", False)
            ),
            fixed_point_limits={
                "dot_product_component_accumulator_bits_target": int(
                    fixed_point_limits.get(
                        "dot_product_component_accumulator_bits_target",
                        DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET,
                    )
                ),
                "mag_squared_accumulator_bits_target": int(
                    fixed_point_limits.get(
                        "mag_squared_accumulator_bits_target",
                        MAG_SQUARED_ACCUMULATOR_BITS_TARGET,
                    )
                ),
                "power_sum_accumulator_bits": int(
                    fixed_point_limits.get(
                        "power_sum_accumulator_bits",
                        POWER_SUM_ACCUMULATOR_BITS,
                    )
                ),
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DetectorCoreProfile":
        return load_detector_core_profile(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_core_id": self.detector_core_id,
            "kernel_contract": {
                "detector_window_samples": int(self.detector_window_samples),
                "num_weight_terms": int(self.num_weight_terms),
                "skipped_guard_bins": int(self.skipped_guard_bins),
                "packed_complex_bits": int(self.packed_complex_bits),
                "sample_bits_per_component": int(self.sample_bits_per_component),
                "input_format": self.input_format,
                "power_accumulator": self.power_accumulator,
                "statistic": self.statistic,
                "pilot_excess": self.pilot_excess,
                "host_masking_policy": self.host_masking_policy,
                "per_frequency_threshold": bool(self.per_frequency_threshold),
            },
            "fixed_point_limits": dict(self.fixed_point_limits),
        }


def load_detector_core_profile(path: str | Path) -> DetectorCoreProfile:
    """Load a detector-core profile JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("detector core profile JSON must contain an object.")
    return DetectorCoreProfile.from_dict(data)


def _reject_unsupported_spacing_input_fields(data: dict[str, Any]) -> None:
    for key in sorted(DERIVED_DETECTOR_SPACING_INPUT_FIELDS):
        if key in data:
            raise ValueError(
                f"Deprecated or derived detector-spacing field found: {key}. "
                "Detector-core profiles use skipped_guard_bins as the source "
                "of truth. reference_offset_bins is derived as "
                "skipped_guard_bins + 1."
            )


def _reject_deprecated_threshold_contract_fields(data: dict[str, Any]) -> None:
    for key in sorted(DEPRECATED_THRESHOLD_CONTRACT_FIELDS):
        if key in data:
            raise ValueError(
                f"Deprecated detector threshold-contract field found: {key}. "
                "Detector-core profile v2 reads uint64 target/reference powers "
                "and applies host positive-excess masking."
            )
    for key in sorted(DEPRECATED_DETECTOR_SPACING_FIELDS):
        if key in data:
            raise ValueError(
                f"Deprecated detector-spacing field found: {key}. "
                "Detector-core profiles use skipped_guard_bins as the source "
                "of truth."
            )


__all__ = [
    "DEPRECATED_DETECTOR_SPACING_FIELDS",
    "DERIVED_DETECTOR_SPACING_INPUT_FIELDS",
    "DetectorCoreProfile",
    "LOCKED_DETECTOR_WINDOW_SAMPLES",
    "LOCKED_NUM_WEIGHT_TERMS",
    "LOCKED_REFERENCE_OFFSET_BINS",
    "LOCKED_SKIPPED_GUARD_BINS",
    "MIN_REFERENCE_OFFSET_BINS",
    "MIN_SKIPPED_GUARD_BINS",
    "load_detector_core_profile",
]
