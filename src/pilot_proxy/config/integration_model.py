"""Integration model: how the science analysis integrates a frame residual.

- ``frame_seconds``: the frame length the residual chain counts in;
- ``coherence_cap_seconds``: the longest coherence time booked (the cap used
  where the correlation time is refused);
- ``variance_split``: ``booked_when_tau_usable`` books the variance split where
  the correlation time is usable; ``off`` never books it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._files import check_keys, choice, number, read_yaml

VARIANCE_SPLIT_RULES = ("booked_when_tau_usable", "off")


@dataclass(frozen=True)
class IntegrationModel:
    frame_seconds: float
    coherence_cap_seconds: float
    variance_split: str


def integration_model_from_mapping(data: Mapping[str, Any], *, where: str) -> IntegrationModel:
    check_keys(data, required=("frame_seconds", "coherence_cap_seconds", "variance_split"),
               where=where)
    return IntegrationModel(
        frame_seconds=number(data["frame_seconds"], field="frame_seconds", where=where,
                             positive=True),
        coherence_cap_seconds=number(data["coherence_cap_seconds"],
                                     field="coherence_cap_seconds", where=where, positive=True),
        variance_split=choice(data["variance_split"], VARIANCE_SPLIT_RULES,
                              field="variance_split", where=where),
    )


def load_integration_model(path: Path | str) -> IntegrationModel:
    path = Path(path)
    return integration_model_from_mapping(read_yaml(path), where=str(path))


__all__ = [
    "IntegrationModel",
    "VARIANCE_SPLIT_RULES",
    "integration_model_from_mapping",
    "load_integration_model",
]
