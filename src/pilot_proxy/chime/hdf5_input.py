# coding=utf-8
"""Segmented HDF5 discovery and window reads for CHIME pilot samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pilot_proxy.atsc_channels import (
    ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
    physical_channel_to_pilot_hz,
)

CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4 = "chime_offset_binary_complex_int4"
PACKED_TWOS_COMPLEMENT_COMPLEX_INT4 = "packed_twos_complement_complex_int4"
COMPLEX_FLOAT = "complex_float"
STRUCTURED_COMPLEX = "structured_complex"
REAL_IMAG_LAST_AXIS = "real_imag_last_axis"
UNKNOWN_ENCODING = "unknown"
DEFAULT_DATASET_PATH = "baseband"


@dataclass(frozen=True)
class ChimeSegment:
    path: Path
    physical_channel: int | None
    pilot_frequency_hz: float | None
    dataset_path: str
    num_time_samples: int
    shape: tuple[int, ...]
    dtype: str
    freq_id: int | None = None
    coarse_channel_center_hz: float | None = None
    sample_encoding: str = UNKNOWN_ENCODING


@dataclass(frozen=True)
class ChimePilotDataset:
    physical_channel: int
    pilot_frequency_hz: float
    segments: list[ChimeSegment]
    dataset_path: str
    time_axis: int
    stream_axis: int
    complex_axis: int | None = None
    sample_encoding: str = UNKNOWN_ENCODING
    freq_id: int | None = None
    coarse_channel_center_hz: float | None = None

    @property
    def total_time_samples(self) -> int:
        return int(sum(int(segment.num_time_samples) for segment in self.segments))

    @property
    def num_input_streams(self) -> int:
        if not self.segments:
            return 0
        shape = self.segments[0].shape
        return int(shape[int(self.stream_axis)])


def _walk(path: Path, filename_pattern: str) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"CHIME input directory does not exist: {root}")
    return sorted(root.rglob(filename_pattern))


def _axis_labels(attrs: Any) -> tuple[str, ...]:
    labels = attrs.get("axis", ())
    out: list[str] = []
    for label in labels:
        if isinstance(label, bytes):
            out.append(label.decode("utf-8", errors="replace"))
        else:
            out.append(str(label))
    return tuple(out)


def _find_dataset_path(h5: h5py.File, requested: str | None) -> str:
    if requested is not None:
        if requested not in h5:
            raise KeyError(f"dataset path {requested!r} not found in {h5.filename}")
        return requested
    if DEFAULT_DATASET_PATH in h5 and isinstance(h5[DEFAULT_DATASET_PATH], h5py.Dataset):
        return DEFAULT_DATASET_PATH

    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            candidates.append((int(np.prod(obj.shape)), name))

    h5.visititems(visitor)
    if not candidates:
        raise ValueError(f"no array-like HDF5 dataset found in {h5.filename}")
    return max(candidates)[1]


def _infer_axes(obj: h5py.Dataset) -> tuple[int, int, int | None]:
    labels = tuple(label.lower() for label in _axis_labels(obj.attrs))
    if "time" in labels and "input" in labels:
        time_axis = labels.index("time")
        stream_axis = labels.index("input")
    elif obj.ndim == 2:
        time_axis = 0
        stream_axis = 1
    else:
        dims = list(obj.shape)
        stream_axis = int(np.argmax(dims))
        time_axis = 0 if stream_axis != 0 else 1

    complex_axis = None
    if obj.ndim >= 3 and obj.shape[-1] == 2:
        tail_label = labels[-1] if len(labels) == obj.ndim else ""
        if tail_label in {"complex", "real_imag", "ri", ""}:
            complex_axis = obj.ndim - 1
    return int(time_axis), int(stream_axis), complex_axis


def _float_attr(attrs: Any, key: str) -> float | None:
    if key not in attrs:
        return None
    value = float(attrs[key])
    if not np.isfinite(value):
        return None
    return value


def _int_attr(attrs: Any, key: str) -> int | None:
    if key not in attrs:
        return None
    return int(attrs[key])


def _frequency_hz_from_attrs(attrs: Any) -> float | None:
    for key in ("pilot_frequency_hz", "dtv_pilot_hz", "frequency_hz"):
        value = _float_attr(attrs, key)
        if value is not None:
            return float(value)
    freq = _float_attr(attrs, "freq")
    if freq is None:
        return None
    # CHIME acquisition files store coarse-channel frequency in MHz.
    return float(freq * 1.0e6 if abs(freq) < 1.0e6 else freq)


def nearest_atsc_physical_channel(frequency_hz: float) -> int | None:
    """Return the nearest UHF ATSC physical channel for a pilot-like frequency."""
    freq = float(frequency_hz)
    candidates = range(ATSC_UHF_MIN_PHYSICAL_CHANNEL, ATSC_UHF_MAX_PHYSICAL_CHANNEL + 1)
    best = min(candidates, key=lambda ch: abs(physical_channel_to_pilot_hz(ch) - freq))
    delta = abs(physical_channel_to_pilot_hz(best) - freq)
    # A CHIME coarse channel can be up to half a 390.625 kHz bin from the pilot.
    return int(best) if delta <= 3.0e6 else None


def _physical_channel_from_attrs(attrs: Any) -> int | None:
    for key in ("physical_channel", "dtv_physical_channel"):
        channel = _int_attr(attrs, key)
        if channel is not None:
            return channel
    frequency_hz = _frequency_hz_from_attrs(attrs)
    if frequency_hz is None:
        return None
    return nearest_atsc_physical_channel(float(frequency_hz))


def _sample_encoding(obj: h5py.Dataset) -> str:
    dtype = np.dtype(obj.dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return COMPLEX_FLOAT
    if dtype.names:
        names = {name.lower() for name in dtype.names}
        if names & {"real", "re", "r"} and names & {"imag", "im", "i"}:
            return STRUCTURED_COMPLEX
    if dtype == np.dtype("uint8"):
        return CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4
    if dtype == np.dtype("int8"):
        return PACKED_TWOS_COMPLEMENT_COMPLEX_INT4
    if obj.ndim >= 3 and obj.shape[-1] == 2:
        return REAL_IMAG_LAST_AXIS
    return UNKNOWN_ENCODING


def _sort_key(path: Path) -> tuple[tuple[str, ...], int | str]:
    stem = path.stem
    numeric_stem: int | str = int(stem) if stem.isdigit() else stem
    return tuple(path.parent.parts), numeric_stem


def _read_segment(path: Path, dataset_path: str | None) -> tuple[ChimeSegment, int, int, int | None]:
    with h5py.File(path, "r") as h5:
        resolved_dataset_path = _find_dataset_path(h5, dataset_path)
        obj = h5[resolved_dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{resolved_dataset_path!r} is not a dataset in {path}")
        time_axis, stream_axis, complex_axis = _infer_axes(obj)
        frequency_hz = _frequency_hz_from_attrs(h5.attrs)
        physical_channel = _physical_channel_from_attrs(h5.attrs)
        pilot_frequency_hz = (
            None
            if physical_channel is None
            else float(physical_channel_to_pilot_hz(int(physical_channel)))
        )
        segment = ChimeSegment(
            path=Path(path),
            physical_channel=physical_channel,
            pilot_frequency_hz=pilot_frequency_hz,
            dataset_path=resolved_dataset_path,
            num_time_samples=int(obj.shape[time_axis]),
            shape=tuple(int(value) for value in obj.shape),
            dtype=str(obj.dtype),
            freq_id=_int_attr(h5.attrs, "freq_id"),
            coarse_channel_center_hz=frequency_hz,
            sample_encoding=_sample_encoding(obj),
        )
    return segment, int(time_axis), int(stream_axis), complex_axis


def discover_chime_pilot_datasets(
    root: Path,
    *,
    dataset_path: str | None,
    filename_pattern: str = "*.h5",
) -> dict[int, ChimePilotDataset]:
    """Discover segmented CHIME pilot-channel datasets below the root path."""
    grouped: dict[int, list[tuple[ChimeSegment, int, int, int | None]]] = {}
    for path in _walk(Path(root), filename_pattern):
        segment, time_axis, stream_axis, complex_axis = _read_segment(path, dataset_path)
        if segment.physical_channel is None:
            continue
        grouped.setdefault(int(segment.physical_channel), []).append(
            (segment, time_axis, stream_axis, complex_axis)
        )

    datasets: dict[int, ChimePilotDataset] = {}
    for physical_channel in sorted(grouped):
        items = sorted(grouped[physical_channel], key=lambda item: _sort_key(item[0].path))
        segments = [item[0] for item in items]
        first_segment, time_axis, stream_axis, complex_axis = items[0]
        for segment, seg_time_axis, seg_stream_axis, seg_complex_axis in items:
            if segment.dataset_path != first_segment.dataset_path:
                raise ValueError(
                    "segments for one physical channel use multiple dataset paths: "
                    f"{first_segment.dataset_path!r} and {segment.dataset_path!r}"
                )
            if (seg_time_axis, seg_stream_axis, seg_complex_axis) != (
                time_axis,
                stream_axis,
                complex_axis,
            ):
                raise ValueError(
                    "segments for one physical channel use inconsistent axes."
                )
            if segment.shape[stream_axis] != first_segment.shape[stream_axis]:
                raise ValueError(
                    "segments for one physical channel use inconsistent input counts."
                )
        datasets[physical_channel] = ChimePilotDataset(
            physical_channel=int(physical_channel),
            pilot_frequency_hz=float(physical_channel_to_pilot_hz(physical_channel)),
            segments=segments,
            dataset_path=first_segment.dataset_path,
            time_axis=int(time_axis),
            stream_axis=int(stream_axis),
            complex_axis=complex_axis,
            sample_encoding=first_segment.sample_encoding,
            freq_id=first_segment.freq_id,
            coarse_channel_center_hz=first_segment.coarse_channel_center_hz,
        )
    return datasets


def _structured_to_complex(arr: np.ndarray) -> np.ndarray:
    names = tuple(arr.dtype.names or ())
    lower = {name.lower(): name for name in names}
    real_name = next((lower[name] for name in ("real", "re", "r") if name in lower), None)
    imag_name = next((lower[name] for name in ("imag", "im", "i") if name in lower), None)
    if real_name is None or imag_name is None:
        raise ValueError(f"structured dtype lacks real/imag fields: {arr.dtype}")
    return np.asarray(arr[real_name]) + 1j * np.asarray(arr[imag_name])


def _normalize_read_array(
    raw: np.ndarray,
    *,
    time_axis: int,
    stream_axis: int,
    complex_axis: int | None,
) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.dtype.names:
        arr = _structured_to_complex(arr)
    elif complex_axis is not None:
        axis = int(complex_axis)
        arr = np.moveaxis(arr, axis, -1)
        arr = np.asarray(arr[..., 0]) + 1j * np.asarray(arr[..., 1])
        if axis < time_axis:
            time_axis -= 1
        if axis < stream_axis:
            stream_axis -= 1

    stream_time = np.moveaxis(arr, (int(stream_axis), int(time_axis)), (0, 1))
    if stream_time.ndim != 2:
        raise ValueError(
            "normalized CHIME block must have stream/time axes only; "
            f"got shape {stream_time.shape}"
        )
    return np.ascontiguousarray(stream_time[:, np.newaxis, :])


def _read_segment_window(
    segment: ChimeSegment,
    *,
    time_axis: int,
    stream_axis: int,
    complex_axis: int | None,
    start: int,
    stop: int,
) -> np.ndarray:
    with h5py.File(segment.path, "r") as h5:
        obj = h5[segment.dataset_path]
        if not isinstance(obj, h5py.Dataset):
            raise TypeError(f"{segment.dataset_path!r} is not a dataset in {segment.path}")
        selection: list[slice] = [slice(None)] * obj.ndim
        selection[int(time_axis)] = slice(int(start), int(stop))
        raw = obj[tuple(selection)]
    return _normalize_read_array(
        raw,
        time_axis=int(time_axis),
        stream_axis=int(stream_axis),
        complex_axis=complex_axis,
    )


def read_complex_window(
    dataset: ChimePilotDataset,
    *,
    start_sample: int,
    stop_sample: int,
) -> np.ndarray:
    """Return selected input-stream samples across segment boundaries.

    Native CHIME unsigned-byte samples remain packed in offset-binary int4 form.
    Floating-point and explicit real/imag datasets are returned as complex arrays.
    """
    start = int(start_sample)
    stop = int(stop_sample)
    if start < 0 or stop < start:
        raise ValueError("invalid sample window")
    if stop > dataset.total_time_samples:
        raise ValueError(
            "requested window extends past available samples: "
            f"stop={stop}, available={dataset.total_time_samples}"
        )
    if start == stop:
        return np.empty((dataset.num_input_streams, 1, 0), dtype=np.uint8)

    chunks: list[np.ndarray] = []
    logical_start = 0
    for segment in dataset.segments:
        logical_stop = logical_start + int(segment.num_time_samples)
        overlap_start = max(start, logical_start)
        overlap_stop = min(stop, logical_stop)
        if overlap_start < overlap_stop:
            chunks.append(
                _read_segment_window(
                    segment,
                    time_axis=dataset.time_axis,
                    stream_axis=dataset.stream_axis,
                    complex_axis=dataset.complex_axis,
                    start=overlap_start - logical_start,
                    stop=overlap_stop - logical_start,
                )
            )
        logical_start = logical_stop
        if logical_start >= stop:
            break
    if not chunks:
        raise ValueError("requested window did not overlap any segment")
    return np.ascontiguousarray(np.concatenate(chunks, axis=2))


def dataset_manifest(dataset: ChimePilotDataset) -> dict[str, Any]:
    """Return a JSON-safe manifest entry for one discovered pilot dataset."""
    return {
        "physical_channel": int(dataset.physical_channel),
        "pilot_frequency_hz": float(dataset.pilot_frequency_hz),
        "coarse_channel_center_hz": dataset.coarse_channel_center_hz,
        "freq_id": dataset.freq_id,
        "dataset_path": dataset.dataset_path,
        "time_axis": int(dataset.time_axis),
        "stream_axis": int(dataset.stream_axis),
        "complex_axis": dataset.complex_axis,
        "sample_encoding": dataset.sample_encoding,
        "num_input_streams": int(dataset.num_input_streams),
        "total_time_samples": int(dataset.total_time_samples),
        "segments": [
            {
                "path": str(segment.path),
                "num_time_samples": int(segment.num_time_samples),
                "shape": list(segment.shape),
                "dtype": segment.dtype,
                "freq_id": segment.freq_id,
                "coarse_channel_center_hz": segment.coarse_channel_center_hz,
                "sample_encoding": segment.sample_encoding,
            }
            for segment in dataset.segments
        ],
    }


__all__ = [
    "CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4",
    "COMPLEX_FLOAT",
    "ChimePilotDataset",
    "ChimeSegment",
    "PACKED_TWOS_COMPLEMENT_COMPLEX_INT4",
    "REAL_IMAG_LAST_AXIS",
    "STRUCTURED_COMPLEX",
    "UNKNOWN_ENCODING",
    "dataset_manifest",
    "discover_chime_pilot_datasets",
    "nearest_atsc_physical_channel",
    "read_complex_window",
]
