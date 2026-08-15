# coding=utf-8
"""ATSC 1.0 physical-channel frequency helpers."""

from __future__ import annotations

# ATSC 1.0 UHF physical channels are 6 MHz wide. The pilot is offset by
# 309.441 kHz above the lower channel edge for the 8-VSB RF channel.
ATSC_CHANNEL_WIDTH_HZ = 6.0e6
ATSC_PILOT_OFFSET_HZ = 309_441.0
ATSC_UHF_MIN_PHYSICAL_CHANNEL = 14
ATSC_UHF_MAX_PHYSICAL_CHANNEL = 69
# UHF channel 14 starts at 470 MHz; higher UHF channels advance in 6 MHz steps.
ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ = 470.0e6
ATSC_CHANNEL_CENTER_OFFSET_HZ = ATSC_CHANNEL_WIDTH_HZ / 2.0


def validate_uhf_physical_channel(channel: int) -> int:
    """Validate and return an ATSC UHF physical channel number."""
    value = int(channel)
    if value < ATSC_UHF_MIN_PHYSICAL_CHANNEL or value > ATSC_UHF_MAX_PHYSICAL_CHANNEL:
        raise ValueError(
            "physical channel must be in the UHF range "
            f"{ATSC_UHF_MIN_PHYSICAL_CHANNEL}-{ATSC_UHF_MAX_PHYSICAL_CHANNEL}; "
            f"got {channel!r}."
        )
    return value


def physical_channel_to_lower_edge_hz(channel: int) -> float:
    """Return the lower RF edge for an ATSC UHF physical channel."""
    value = validate_uhf_physical_channel(channel)
    return float(
        ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ
        + (value - ATSC_UHF_MIN_PHYSICAL_CHANNEL) * ATSC_CHANNEL_WIDTH_HZ
    )


def physical_channel_to_center_hz(channel: int) -> float:
    """Return the RF center frequency for an ATSC UHF physical channel."""
    return float(
        physical_channel_to_lower_edge_hz(channel) + ATSC_CHANNEL_CENTER_OFFSET_HZ
    )


def physical_channel_to_pilot_hz(channel: int) -> float:
    """Return the ATSC pilot frequency for an ATSC UHF physical channel."""
    return float(physical_channel_to_lower_edge_hz(channel) + ATSC_PILOT_OFFSET_HZ)


def parse_physical_channel_range(value: str) -> list[int]:
    """Parse ``N`` or ``N:M`` into inclusive physical-channel numbers."""
    text = str(value).strip()
    if ":" not in text:
        return [validate_uhf_physical_channel(int(text))]
    left, right = text.split(":", 1)
    start = validate_uhf_physical_channel(int(left))
    stop = validate_uhf_physical_channel(int(right))
    step = 1 if stop >= start else -1
    return list(range(start, stop + step, step))


__all__ = [
    "ATSC_CHANNEL_WIDTH_HZ",
    "ATSC_CHANNEL_CENTER_OFFSET_HZ",
    "ATSC_PILOT_OFFSET_HZ",
    "ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ",
    "ATSC_UHF_MAX_PHYSICAL_CHANNEL",
    "ATSC_UHF_MIN_PHYSICAL_CHANNEL",
    "parse_physical_channel_range",
    "physical_channel_to_center_hz",
    "physical_channel_to_lower_edge_hz",
    "physical_channel_to_pilot_hz",
    "validate_uhf_physical_channel",
]
