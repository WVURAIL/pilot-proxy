# coding=utf-8
from __future__ import annotations

import pytest

from pilot_proxy.atsc_channels import (
    parse_physical_channel_range,
    physical_channel_to_center_hz,
    physical_channel_to_lower_edge_hz,
    physical_channel_to_pilot_hz,
)

CHANNEL_14 = 14
CHANNEL_14_LOWER_EDGE_HZ = 470_000_000.0
CHANNEL_14_CENTER_HZ = 473_000_000.0
CHANNEL_14_PILOT_HZ = 470_309_441.0
CHANNEL_15 = 15
CHANNEL_16 = 16
BELOW_UHF_CHANNEL_MIN = 13


def test_physical_channel_14_frequencies() -> None:
    assert physical_channel_to_lower_edge_hz(CHANNEL_14) == CHANNEL_14_LOWER_EDGE_HZ
    assert physical_channel_to_center_hz(CHANNEL_14) == CHANNEL_14_CENTER_HZ
    assert physical_channel_to_pilot_hz(CHANNEL_14) == CHANNEL_14_PILOT_HZ


def test_physical_channel_range_parsing() -> None:
    assert parse_physical_channel_range(str(CHANNEL_14)) == [CHANNEL_14]
    assert parse_physical_channel_range(f"{CHANNEL_14}:{CHANNEL_16}") == [
        CHANNEL_14,
        CHANNEL_15,
        CHANNEL_16,
    ]
    assert parse_physical_channel_range(f"{CHANNEL_16}:{CHANNEL_14}") == [
        CHANNEL_16,
        CHANNEL_15,
        CHANNEL_14,
    ]


def test_invalid_physical_channel_raises() -> None:
    with pytest.raises(ValueError):
        physical_channel_to_pilot_hz(BELOW_UHF_CHANNEL_MIN)
