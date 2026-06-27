# coding=utf-8
"""Detector input layout and block-to-kernel geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SPECTRAL_SENSE_NORMAL = "normal"
SPECTRAL_SENSE_INVERTED = "inverted"
SUPPORTED_SPECTRAL_SENSE = frozenset({SPECTRAL_SENSE_NORMAL, SPECTRAL_SENSE_INVERTED})
COMBINE_MODE_INCOHERENT_POWER_SUM_OVER_STREAMS = (
    "incoherent_power_sum_over_streams"
)


def normalize_spectral_sense(value: Any | None) -> str:
    """Normalize detector input spectral-sense metadata."""
    if value is None:
        return SPECTRAL_SENSE_NORMAL
    text = str(value).strip().lower()
    if text not in SUPPORTED_SPECTRAL_SENSE:
        raise ValueError(
            f"spectral_sense must be one of {sorted(SUPPORTED_SPECTRAL_SENSE)}, "
            f"got {value!r}"
        )
    return text


def spectral_sense_requires_time_reversal(value: Any | None) -> bool:
    """Return whether detector-window samples must be reversed for this sense."""
    return normalize_spectral_sense(value) == SPECTRAL_SENSE_INVERTED


@dataclass(frozen=True)
class DetectorInputLayout:
    """Derived geometry for one detector decision."""

    samples_per_block: int
    num_streams: int
    detector_window_samples: int

    def __post_init__(self) -> None:
        if self.samples_per_block <= 0:
            raise ValueError("samples_per_block must be positive.")
        if self.num_streams <= 0:
            raise ValueError("num_streams must be positive.")
        if self.detector_window_samples <= 0:
            raise ValueError("detector_window_samples must be positive.")
        if self.samples_per_block % self.detector_window_samples != 0:
            raise ValueError(
                "samples_per_block must be an integer multiple of "
                "detector_window_samples: "
                f"samples_per_block={self.samples_per_block}, "
                f"detector_window_samples={self.detector_window_samples}"
            )

    @property
    def windows_per_block(self) -> int:
        return self.samples_per_block // self.detector_window_samples

    @property
    def detector_rows_per_block(self) -> int:
        return self.num_streams * self.windows_per_block

    @property
    def samples_per_result(self) -> int:
        return self.samples_per_block * self.num_streams

    def as_dict(self) -> dict[str, int]:
        return {
            "samples_per_block": int(self.samples_per_block),
            "num_streams": int(self.num_streams),
            "detector_window_samples": int(self.detector_window_samples),
            "windows_per_block": int(self.windows_per_block),
            "detector_rows_per_block": int(self.detector_rows_per_block),
            "samples_per_result": int(self.samples_per_result),
        }


def derive_detector_input_layout(
    *,
    samples_per_block: int,
    num_streams: int,
    detector_window_samples: int,
) -> DetectorInputLayout:
    """Build and validate a detector input layout object."""
    return DetectorInputLayout(
        samples_per_block=int(samples_per_block),
        num_streams=int(num_streams),
        detector_window_samples=int(detector_window_samples),
    )


def flatten_feed_channel_streams(feed_channel_streams: np.ndarray) -> np.ndarray:
    """Flatten feed/channel/time data into stream/time data.

    The stream order is feed-major, then selected-channel-major.
    """
    arr = np.asarray(feed_channel_streams)
    if arr.ndim != 3:
        raise ValueError(
            "feed_channel_streams must have shape (feed, channel, time); "
            f"got shape={arr.shape}"
        )
    num_feeds, num_selected_channels, samples = arr.shape
    if num_feeds <= 0 or num_selected_channels <= 0 or samples <= 0:
        raise ValueError(
            "feed_channel_streams axes must all be non-empty; "
            f"got shape={arr.shape}"
        )
    return np.ascontiguousarray(arr.reshape(num_feeds * num_selected_channels, samples))


def build_stream_map(
    *,
    num_feeds: int,
    selected_channel_indices: list[int],
    physical_channel: int | None = None,
) -> list[dict[str, int | None]]:
    """Build metadata for feed/channel streams flattened into kernel rows."""
    feeds = int(num_feeds)
    if feeds <= 0:
        raise ValueError("num_feeds must be positive.")
    if not selected_channel_indices:
        raise ValueError("selected_channel_indices must not be empty.")

    stream_map: list[dict[str, int | None]] = []
    stream_index = 0
    for feed_index in range(feeds):
        for channel_index in selected_channel_indices:
            stream_map.append(
                {
                    "stream_index": int(stream_index),
                    "feed_index": int(feed_index),
                    "selected_channel_index": int(channel_index),
                    "physical_channel": (
                        None if physical_channel is None else int(physical_channel)
                    ),
                }
            )
            stream_index += 1
    return stream_map


def input_layout_metadata(
    *,
    frame_size_samples: int,
    detector_window_samples: int,
    num_feeds: int,
    num_selected_channels: int,
) -> dict[str, int | str]:
    """Return public metadata for combined-stream detector input layout."""
    if num_selected_channels <= 0:
        raise ValueError("num_selected_channels must be positive.")
    layout = DetectorInputLayout(
        samples_per_block=int(frame_size_samples),
        num_streams=int(num_feeds) * int(num_selected_channels),
        detector_window_samples=int(detector_window_samples),
    )
    return {
        "frame_size_samples": int(frame_size_samples),
        "samples_per_block": int(frame_size_samples),
        "detector_window_samples": int(detector_window_samples),
        "windows_per_stream": int(layout.windows_per_block),
        "windows_per_feed": int(layout.windows_per_block),
        "num_feeds": int(num_feeds),
        "num_selected_channels": int(num_selected_channels),
        "num_input_streams": int(layout.num_streams),
        "num_streams": int(layout.num_streams),
        "detector_rows_per_block": int(layout.detector_rows_per_block),
        "combine_mode": COMBINE_MODE_INCOHERENT_POWER_SUM_OVER_STREAMS,
    }


def block_time_stream_to_detector_matrix(
    block: np.ndarray,
    *,
    detector_window_samples: int,
    time_axis: int = 0,
    stream_axis: int = 1,
) -> np.ndarray:
    """Convert a time-by-stream block into row-major detector windows."""
    arr = np.asarray(block)
    if arr.ndim != 2:
        raise ValueError(
            f"block must be 2D with axes (time, stream); got shape={arr.shape}"
        )
    time_axis = int(time_axis)
    stream_axis = int(stream_axis)
    if time_axis < 0:
        time_axis += arr.ndim
    if stream_axis < 0:
        stream_axis += arr.ndim
    if time_axis == stream_axis:
        raise ValueError("time_axis and stream_axis must be different.")
    if time_axis not in (0, 1) or stream_axis not in (0, 1):
        raise ValueError("time_axis and stream_axis must identify the two axes.")

    time_stream = np.moveaxis(arr, (time_axis, stream_axis), (0, 1))
    layout = DetectorInputLayout(
        samples_per_block=int(time_stream.shape[0]),
        num_streams=int(time_stream.shape[1]),
        detector_window_samples=int(detector_window_samples),
    )
    windows = time_stream.reshape(
        layout.windows_per_block,
        layout.detector_window_samples,
        layout.num_streams,
    )
    detector_matrix = np.transpose(windows, (2, 0, 1)).reshape(
        layout.detector_rows_per_block,
        layout.detector_window_samples,
    )
    return np.ascontiguousarray(detector_matrix)


def stream_time_block_to_detector_matrix(
    streams: np.ndarray,
    *,
    detector_window_samples: int,
) -> np.ndarray:
    """Convert stream-by-time data into row-major detector windows."""
    return block_time_stream_to_detector_matrix(
        streams,
        detector_window_samples=detector_window_samples,
        time_axis=1,
        stream_axis=0,
    )


def stack_stream_time_blocks(
    streams: np.ndarray,
    *,
    detector_window_samples: int,
    samples_per_block: int,
    block_step_samples: int,
    num_blocks: int,
) -> np.ndarray:
    """Build a batch of detector matrices from stream-major time series."""
    streams = np.asarray(streams)
    if streams.ndim != 2:
        raise ValueError(
            "streams must be 2D with shape (stream, time); "
            f"got shape={streams.shape}"
        )
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1.")
    if block_step_samples <= 0:
        raise ValueError("block_step_samples must be positive.")

    blocks = []
    for block_index in range(int(num_blocks)):
        start = block_index * int(block_step_samples)
        stop = start + int(samples_per_block)
        if stop > streams.shape[1]:
            raise ValueError(
                "not enough time samples to build requested blocks: "
                f"block_index={block_index}, stop={stop}, available={streams.shape[1]}"
            )
        blocks.append(
            stream_time_block_to_detector_matrix(
                streams[:, start:stop],
                detector_window_samples=detector_window_samples,
            )
        )
    return np.ascontiguousarray(np.stack(blocks))


def apply_spectral_sense_to_detector_matrix(
    detector_matrix: np.ndarray,
    *,
    spectral_sense: str,
) -> np.ndarray:
    """Apply spectral-sense correction to a detector-matrix view."""
    sense = normalize_spectral_sense(spectral_sense)
    arr = np.asarray(detector_matrix)
    if spectral_sense_requires_time_reversal(sense):
        return np.ascontiguousarray(np.flip(arr, axis=-1))
    return np.ascontiguousarray(arr)


__all__ = [
    "DetectorInputLayout",
    "COMBINE_MODE_INCOHERENT_POWER_SUM_OVER_STREAMS",
    "SPECTRAL_SENSE_INVERTED",
    "SPECTRAL_SENSE_NORMAL",
    "SUPPORTED_SPECTRAL_SENSE",
    "apply_spectral_sense_to_detector_matrix",
    "block_time_stream_to_detector_matrix",
    "build_stream_map",
    "derive_detector_input_layout",
    "flatten_feed_channel_streams",
    "input_layout_metadata",
    "normalize_spectral_sense",
    "spectral_sense_requires_time_reversal",
    "stack_stream_time_blocks",
    "stream_time_block_to_detector_matrix",
]
