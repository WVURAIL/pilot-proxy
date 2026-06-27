# coding=utf-8
"""CHIME frame-FFT frequency-offset diagnostic for DTV pilot samples."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.detector_geometry import spectral_sense_requires_time_reversal
from pilot_proxy.integration import (
    DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    load_receiver_profile,
    parse_physical_channel_selection,
    receiver_frequency_to_channel,
)
from pilot_proxy.json_utils import write_json_strict

from .frame_adapter import (
    unpack_chime_offset_binary_i4_to_complex,
    unpack_twos_complement_i4_to_complex,
)
from .hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    COMPLEX_FLOAT,
    PACKED_TWOS_COMPLEMENT_COMPLEX_INT4,
    REAL_IMAG_LAST_AXIS,
    STRUCTURED_COMPLEX,
    ChimePilotDataset,
    dataset_manifest,
    discover_chime_pilot_datasets,
    read_complex_window,
)
from .products import ensure_run_dirs, relative_time_seconds

DEFAULT_RECEIVER_PROFILE = DEFAULT_CHIME_DTV_RECEIVER_PROFILE
DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_FFT_SIZE = 16_384
DEFAULT_FRAMES_PER_CHUNK = 1
DEFAULT_STREAM_BATCH_SIZE = 128
DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ = 5_000.0
DEFAULT_WINDOW_NAME = "hann"
DEFAULT_BACKEND = "auto"
COORDINATE_SYSTEM = "post_spectral_sense_normalization_rf_offset"
FREQUENCY_OFFSET_OUTPUTS_FILENAME = "frequency_offset_outputs.npz"
TIME_AVERAGE_SPECTRUM_BASELINE_WINDOW_BINS = 1001


@dataclass(frozen=True)
class FrequencyOffsetConfig:
    input_dir: Path
    output_dir: Path
    physical_channels: list[int] | None
    physical_channel_range: str | None = None
    dataset_path: str | None = "baseband"
    filename_pattern: str = "*.h5"
    receiver_profile: Path = DEFAULT_RECEIVER_PROFILE
    frame_size_samples: int = DEFAULT_FRAME_SIZE_SAMPLES
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK
    max_frames: int | None = None
    every_n_frames: int = 1
    fft_size: int = DEFAULT_FFT_SIZE
    stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE
    peak_search_half_width_hz: float = DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ
    window_name: str = DEFAULT_WINDOW_NAME
    min_peak_prominence_db: float | None = None
    backend: str = DEFAULT_BACKEND
    plot: bool = False


def _window(name: str, n: int) -> np.ndarray:
    normalized = str(name).strip().lower()
    if normalized == "hann":
        return np.hanning(int(n)).astype(np.float32)
    if normalized in {"rect", "rectangular", "none"}:
        return np.ones(int(n), dtype=np.float32)
    raise ValueError(f"unsupported window {name!r}")


def _rolling_nanmedian_baseline(
    values: np.ndarray,
    *,
    window_bins: int = TIME_AVERAGE_SPECTRUM_BASELINE_WINDOW_BINS,
) -> np.ndarray:
    """Return a broad robust baseline for spectrum flattening."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if arr.size == 0:
        return arr.copy()
    window = max(3, int(window_bins))
    if window % 2 == 0:
        window += 1
    if window > arr.size:
        window = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
    if window <= 1:
        fill = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 0.0
        return np.full_like(arr, fill, dtype=np.float64)

    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    baseline = np.nanmedian(windows, axis=-1)
    if np.any(~np.isfinite(baseline)):
        finite = np.isfinite(arr)
        fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
        baseline = np.where(np.isfinite(baseline), baseline, fill)
    return np.asarray(baseline, dtype=np.float64)


def _unpack_block(block: np.ndarray, sample_encoding: str) -> np.ndarray:
    if sample_encoding == CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4:
        return unpack_chime_offset_binary_i4_to_complex(block)
    if sample_encoding == PACKED_TWOS_COMPLEMENT_COMPLEX_INT4:
        return unpack_twos_complement_i4_to_complex(block)
    if sample_encoding in {COMPLEX_FLOAT, STRUCTURED_COMPLEX, REAL_IMAG_LAST_AXIS}:
        return np.asarray(block, dtype=np.complex64)
    return np.asarray(block).astype(np.complex64)


def _parabolic_delta_log_power(power: np.ndarray, index: int) -> float:
    if index <= 0 or index >= power.size - 1:
        return 0.0
    eps = 1.0e-300
    y0 = 10.0 * np.log10(max(float(power[index - 1]), eps))
    y1 = 10.0 * np.log10(max(float(power[index]), eps))
    y2 = 10.0 * np.log10(max(float(power[index + 1]), eps))
    denom = y0 - 2.0 * y1 + y2
    if not np.isfinite(denom) or abs(denom) < 1.0e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


def _resolve_backend(name: str):
    backend = str(name).strip().lower()
    if backend == "numpy":
        return np, "numpy"
    if backend not in {"auto", "cupy"}:
        raise ValueError("backend must be one of: auto, numpy, cupy")
    try:
        import cupy as cp
    except ImportError:
        if backend == "cupy":
            raise
        return np, "numpy"
    cuda_runtime_error = getattr(cp.cuda.runtime, "CUDARuntimeError", RuntimeError)
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            raise RuntimeError("no CUDA devices visible to CuPy")
        return cp, "cupy"
    except (AttributeError, RuntimeError, cuda_runtime_error):
        if backend == "cupy":
            raise
        return np, "numpy"


def _as_numpy(values: Any, xp) -> np.ndarray:
    if xp is np:
        return np.asarray(values)
    return xp.asnumpy(values)


def accumulate_noncoherent_fft_power(
    streams: np.ndarray,
    *,
    window: np.ndarray,
    stream_batch_size: int,
    backend: str = DEFAULT_BACKEND,
) -> tuple[np.ndarray, str]:
    """Return the noncoherent FFT power sum after applying an FFT shift."""
    arr = np.asarray(streams)
    if arr.ndim != 2:
        raise ValueError(f"streams must have shape (stream, time); got {arr.shape}")
    if int(stream_batch_size) <= 0:
        raise ValueError("stream_batch_size must be positive")
    xp, backend_used = _resolve_backend(backend)
    n_fft = int(arr.shape[1])
    power_sum = np.zeros(n_fft, dtype=np.float64)
    window_np = np.asarray(window, dtype=np.float32)

    for start in range(0, int(arr.shape[0]), int(stream_batch_size)):
        stop = min(start + int(stream_batch_size), int(arr.shape[0]))
        x_np = np.ascontiguousarray(arr[start:stop, :])
        if xp is np:
            spectrum = np.fft.fftshift(
                np.fft.fft(x_np * window_np[None, :], axis=-1),
                axes=-1,
            )
            power_sum += np.sum(np.abs(spectrum) ** 2, axis=0)
        else:
            x_gpu = xp.asarray(x_np)
            window_gpu = xp.asarray(window_np)
            spectrum = xp.fft.fftshift(
                xp.fft.fft(x_gpu * window_gpu[None, :], axis=-1),
                axes=-1,
            )
            batch_power = xp.sum(xp.abs(spectrum) ** 2, axis=0)
            power_sum += _as_numpy(batch_power, xp).astype(np.float64, copy=False)
    return power_sum, backend_used


def frame_noncoherent_fft_power(
    block: np.ndarray,
    *,
    sample_encoding: str,
    spectral_sense: str,
    fft_size: int,
    stream_batch_size: int,
    window: np.ndarray,
    backend: str = DEFAULT_BACKEND,
) -> tuple[np.ndarray, str]:
    """Return noncoherent frame-FFT power in detector frequency coordinates."""
    arr = np.asarray(block)
    if arr.ndim != 3 or arr.shape[1] != 1:
        raise ValueError(f"expected block shape (streams, 1, time), got {arr.shape}")
    if arr.shape[2] != int(fft_size):
        raise ValueError("frequency-offset diagnostic requires fft_size == frame size")

    batch_size = int(stream_batch_size)
    if batch_size <= 0:
        raise ValueError("stream_batch_size must be positive")
    reverse = spectral_sense_requires_time_reversal(spectral_sense)
    power_sum = np.zeros(int(fft_size), dtype=np.float64)
    backend_used = str(backend)
    for start in range(0, int(arr.shape[0]), batch_size):
        stop = min(start + batch_size, int(arr.shape[0]))
        streams = _unpack_block(arr[start:stop, 0, :], sample_encoding)
        if reverse:
            streams = np.ascontiguousarray(streams[:, ::-1])
        batch_power, backend_used = accumulate_noncoherent_fft_power(
            streams,
            window=window,
            stream_batch_size=batch_size,
            backend=backend,
        )
        power_sum += batch_power
    return power_sum, backend_used


def estimate_peak_offset_from_power(
    power_sum: np.ndarray,
    *,
    sample_rate_hz: float,
    expected_offset_hz: float,
    fft_size: int,
    peak_search_half_width_hz: float,
) -> dict[str, float]:
    """Estimate one pilot peak from an already-computed frame FFT power spectrum."""
    power = np.asarray(power_sum, dtype=np.float64)
    if power.shape != (int(fft_size),):
        raise ValueError(
            f"power_sum must have shape ({int(fft_size)},), got {power.shape}"
        )

    freq_axis = np.fft.fftshift(
        np.fft.fftfreq(int(fft_size), d=1.0 / float(sample_rate_hz))
    )
    search = np.abs(freq_axis - float(expected_offset_hz)) <= float(
        peak_search_half_width_hz
    )
    search_indices = np.flatnonzero(search)
    if search_indices.size < 3:
        raise ValueError("peak search window contains fewer than three FFT bins")

    peak_index = int(search_indices[np.argmax(power[search_indices])])
    delta = _parabolic_delta_log_power(power, peak_index)
    bin_width_hz = float(sample_rate_hz) / float(fft_size)
    peak_offset_hz = float(freq_axis[peak_index] + delta * bin_width_hz)
    local_floor = float(np.median(power[search_indices]))
    peak_power = float(power[peak_index])
    prominence_db = (
        10.0 * np.log10(peak_power / local_floor)
        if peak_power > 0.0 and local_floor > 0.0
        else float("nan")
    )

    return {
        "peak_offset_hz": peak_offset_hz,
        "frequency_offset_hz": peak_offset_hz - float(expected_offset_hz),
        "peak_power_linear": peak_power,
        "local_floor_power_linear": local_floor,
        "peak_prominence_db": float(prominence_db),
    }


def estimate_frame_peak_offset(
    block: np.ndarray,
    *,
    sample_encoding: str,
    spectral_sense: str,
    sample_rate_hz: float,
    expected_offset_hz: float,
    fft_size: int,
    stream_batch_size: int,
    peak_search_half_width_hz: float,
    window: np.ndarray,
    backend: str = DEFAULT_BACKEND,
) -> dict[str, float | str]:
    """Estimate the strongest narrowband peak offset in one CHIME frame."""
    power_sum, backend_used = frame_noncoherent_fft_power(
        block,
        sample_encoding=sample_encoding,
        spectral_sense=spectral_sense,
        fft_size=int(fft_size),
        stream_batch_size=int(stream_batch_size),
        window=window,
        backend=backend,
    )
    result: dict[str, float | str] = estimate_peak_offset_from_power(
        power_sum,
        sample_rate_hz=float(sample_rate_hz),
        expected_offset_hz=float(expected_offset_hz),
        fft_size=int(fft_size),
        peak_search_half_width_hz=float(peak_search_half_width_hz),
    )
    result["backend"] = backend_used
    return result


def _select_datasets(
    discovered: dict[int, ChimePilotDataset],
    *,
    physical_channels: Sequence[int] | None,
    physical_channel_range: str | None,
) -> list[ChimePilotDataset]:
    if physical_channels is None and physical_channel_range is None:
        selected_channels = sorted(discovered)
    else:
        selected_channels = parse_physical_channel_selection(
            physical_channels=physical_channels,
            physical_channel_range=physical_channel_range,
        )
    missing = [
        channel for channel in selected_channels if int(channel) not in discovered
    ]
    if missing:
        raise ValueError(f"requested CHIME channels were not found: {missing}")
    return [discovered[int(channel)] for channel in selected_channels]


def _common_frame_indices(
    datasets: Sequence[ChimePilotDataset],
    *,
    frame_size_samples: int,
    max_frames: int | None,
    every_n_frames: int,
) -> np.ndarray:
    if not datasets:
        return np.empty(0, dtype=np.int64)
    available = min(
        int(dataset.total_time_samples // int(frame_size_samples))
        for dataset in datasets
    )
    if max_frames is not None:
        available = min(available, int(max_frames))
    step = int(every_n_frames)
    if step <= 0:
        raise ValueError("every_n_frames must be positive")
    return np.arange(0, int(available), step, dtype=np.int64)


def _coarse_center_hz(dataset: ChimePilotDataset, receiver_profile) -> float:
    if dataset.coarse_channel_center_hz is not None:
        return float(dataset.coarse_channel_center_hz)
    selection = receiver_frequency_to_channel(
        float(dataset.pilot_frequency_hz),
        receiver_profile,
    )
    return float(
        receiver_profile.coarse_channel_center_hz(selection.coarse_channel_index)
    )


def _write_outputs(
    run_dir: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    coarse_channel_center_hz: np.ndarray,
    expected_pilot_offset_hz: np.ndarray,
    frame_index: np.ndarray,
    relative_time_s: np.ndarray,
    peak_offset_hz: np.ndarray,
    frequency_offset_hz: np.ndarray,
    peak_power_linear: np.ndarray,
    local_floor_power_linear: np.ndarray,
    peak_prominence_db: np.ndarray,
    valid: np.ndarray,
    fft_size: int,
    fft_bin_width_hz: float,
    sample_rate_hz: float,
    window_name: str,
    peak_search_half_width_hz: float,
    fft_frequency_axis_hz: np.ndarray | None = None,
    time_average_spectrum_power_linear: np.ndarray | None = None,
    time_average_spectrum_count: np.ndarray | None = None,
) -> Path:
    path = Path(run_dir) / FREQUENCY_OFFSET_OUTPUTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        physical_channel=np.asarray(physical_channel, dtype=np.int32),
        pilot_frequency_hz=np.asarray(pilot_frequency_hz, dtype=np.float64),
        chime_frequency_hz=np.asarray(chime_frequency_hz, dtype=np.float64),
        coarse_channel_center_hz=np.asarray(coarse_channel_center_hz, dtype=np.float64),
        expected_pilot_offset_hz=np.asarray(expected_pilot_offset_hz, dtype=np.float64),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        relative_time_s=np.asarray(relative_time_s, dtype=np.float64),
        peak_offset_hz=np.asarray(peak_offset_hz, dtype=np.float64),
        frequency_offset_hz=np.asarray(frequency_offset_hz, dtype=np.float64),
        peak_power_linear=np.asarray(peak_power_linear, dtype=np.float64),
        local_floor_power_linear=np.asarray(local_floor_power_linear, dtype=np.float64),
        peak_prominence_db=np.asarray(peak_prominence_db, dtype=np.float64),
        valid=np.asarray(valid, dtype=np.uint8),
        fft_frequency_axis_hz=(
            np.fft.fftshift(
                np.fft.fftfreq(int(fft_size), d=1.0 / float(sample_rate_hz))
            )
            if fft_frequency_axis_hz is None
            else np.asarray(fft_frequency_axis_hz, dtype=np.float64)
        ),
        time_average_spectrum_power_linear=(
            np.empty((0, 0), dtype=np.float64)
            if time_average_spectrum_power_linear is None
            else np.asarray(time_average_spectrum_power_linear, dtype=np.float64)
        ),
        time_average_spectrum_count=(
            np.zeros(np.asarray(physical_channel).shape, dtype=np.uint64)
            if time_average_spectrum_count is None
            else np.asarray(time_average_spectrum_count, dtype=np.uint64)
        ),
        fft_size=np.asarray(int(fft_size), dtype=np.int64),
        fft_bin_width_hz=np.asarray(float(fft_bin_width_hz), dtype=np.float64),
        sample_rate_hz=np.asarray(float(sample_rate_hz), dtype=np.float64),
        window_name=np.asarray(str(window_name)),
        peak_search_half_width_hz=np.asarray(
            float(peak_search_half_width_hz),
            dtype=np.float64,
        ),
        coordinate_system=np.asarray(COORDINATE_SYSTEM),
    )
    return path


def _finite_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    flags = np.asarray(valid) != 0
    return arr[flags & np.isfinite(arr)]


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _write_summary_table(
    run_dir: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    coarse_channel_center_hz: np.ndarray,
    expected_pilot_offset_hz: np.ndarray,
    frequency_offset_hz: np.ndarray,
    peak_prominence_db: np.ndarray,
    valid: np.ndarray,
) -> Path:
    path = Path(run_dir) / "tables" / "frequency_offset_summary_by_pilot.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "physical_channel",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "coarse_channel_center_hz",
        "expected_pilot_offset_hz",
        "num_valid_frames",
        "median_frequency_offset_hz",
        "mean_frequency_offset_hz",
        "std_frequency_offset_hz",
        "mad_frequency_offset_hz",
        "p05_frequency_offset_hz",
        "p95_frequency_offset_hz",
        "median_abs_frequency_offset_hz",
        "p95_abs_frequency_offset_hz",
        "median_peak_prominence_db",
        "p05_peak_prominence_db",
        "p95_peak_prominence_db",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            offsets = _finite_valid(frequency_offset_hz[:, index], valid[:, index])
            prominence = _finite_valid(peak_prominence_db[:, index], valid[:, index])
            median = float(np.median(offsets)) if offsets.size else float("nan")
            mad = (
                float(np.median(np.abs(offsets - median)))
                if offsets.size
                else float("nan")
            )
            abs_offsets = np.abs(offsets)
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(chime_frequency_hz[index]),
                    "coarse_channel_center_hz": float(coarse_channel_center_hz[index]),
                    "expected_pilot_offset_hz": float(expected_pilot_offset_hz[index]),
                    "num_valid_frames": int(offsets.size),
                    "median_frequency_offset_hz": median,
                    "mean_frequency_offset_hz": (
                        float(np.mean(offsets)) if offsets.size else float("nan")
                    ),
                    "std_frequency_offset_hz": (
                        float(np.std(offsets)) if offsets.size else float("nan")
                    ),
                    "mad_frequency_offset_hz": mad,
                    "p05_frequency_offset_hz": _percentile(offsets, 5),
                    "p95_frequency_offset_hz": _percentile(offsets, 95),
                    "median_abs_frequency_offset_hz": (
                        float(np.median(abs_offsets))
                        if abs_offsets.size
                        else float("nan")
                    ),
                    "p95_abs_frequency_offset_hz": _percentile(abs_offsets, 95),
                    "median_peak_prominence_db": (
                        float(np.median(prominence))
                        if prominence.size
                        else float("nan")
                    ),
                    "p05_peak_prominence_db": _percentile(prominence, 5),
                    "p95_peak_prominence_db": _percentile(prominence, 95),
                }
            )
    return path


def _plot_histogram(run_dir: Path, data: np.lib.npyio.NpzFile) -> Path:
    from .plots import _setup_matplotlib

    plt = _setup_matplotlib()
    physical_channel = data["physical_channel"]
    chime_frequency_hz = data["chime_frequency_hz"]
    offsets = data["frequency_offset_hz"]
    valid = data["valid"]
    search_half_width_hz = float(np.asarray(data["peak_search_half_width_hz"]).item())
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )

    def _bins_for_xlim(xlim: tuple[float, float]) -> np.ndarray:
        bin_width_hz = 50.0
        num_bins = int(round((float(xlim[1]) - float(xlim[0])) / bin_width_hz))
        return np.linspace(float(xlim[0]), float(xlim[1]), num_bins + 1)

    def _plot_panel(
        ax,
        *,
        xlim: tuple[float, float],
        title: str,
        include_legend: bool = False,
    ) -> None:
        bins = _bins_for_xlim(xlim)
        channel_histograms: list[np.ndarray] = []
        for index, channel in enumerate(physical_channel):
            values = _finite_valid(offsets[:, index], valid[:, index])
            values = values[(values >= float(xlim[0])) & (values <= float(xlim[1]))]
            if values.size == 0:
                continue
            hist, _ = np.histogram(values, bins=bins, density=True)
            channel_histograms.append(hist)
            ax.stairs(
                hist,
                bins,
                color=colors[int(index)],
                label=(
                    rf"DTV {int(channel)}, "
                    rf"$f_{{\mathrm{{CHIME}}}}="
                    rf"{chime_frequency_hz[index] / 1.0e6:.1f}"
                    rf"\,\mathrm{{MHz}}$"
                ),
            )
        if channel_histograms:
            mean_hist = np.mean(np.vstack(channel_histograms), axis=0)
            ax.stairs(
                mean_hist,
                bins,
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean over plotted pilots",
                zorder=20,
            )
        ax.set_xlim(*xlim)
        ax.set_ylabel(r"Probability density")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if include_legend:
            ax.legend(ncol=2, fontsize="x-small")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 9.2),
        constrained_layout=True,
    )
    search_half_width_khz = search_half_width_hz / 1.0e3
    _plot_panel(
        axes[0],
        xlim=(-search_half_width_hz, search_half_width_hz),
        title=(
            rf"All estimates in $\pm{search_half_width_khz:g}"
            rf"\,\mathrm{{kHz}}$ search range"
        ),
        include_legend=True,
    )
    _plot_panel(
        axes[1],
        xlim=(-1_500.0, 1_500.0),
        title=r"Core range; estimates outside $\pm1.5\,\mathrm{kHz}$ excluded",
    )
    axes[1].set_xlabel(r"Frequency offset, measured peak $-$ nominal pilot [Hz]")
    fig.suptitle(r"CHIME DTV pilot frequency-offset diagnostic")

    path = Path(run_dir) / "figures" / "frequency_offset_histogram_by_pilot.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_time_average_spectrum(
    run_dir: Path,
    data: np.lib.npyio.NpzFile,
) -> Path | None:
    if (
        "time_average_spectrum_power_linear" not in data.files
        or "fft_frequency_axis_hz" not in data.files
    ):
        return None
    power = np.asarray(data["time_average_spectrum_power_linear"], dtype=np.float64)
    if power.ndim != 2 or power.size == 0:
        return None

    from .plots import _setup_matplotlib

    plt = _setup_matplotlib()
    physical_channel = np.asarray(data["physical_channel"], dtype=np.int32)
    chime_frequency_hz = np.asarray(data["chime_frequency_hz"], dtype=np.float64)
    expected_offset_hz = np.asarray(data["expected_pilot_offset_hz"], dtype=np.float64)
    fft_frequency_axis_hz = np.asarray(data["fft_frequency_axis_hz"], dtype=np.float64)
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )
    shifted_min = float(np.min(fft_frequency_axis_hz[:, None] - expected_offset_hz))
    shifted_max = float(np.max(fft_frequency_axis_hz[:, None] - expected_offset_hz))
    grid = np.linspace(shifted_min, shifted_max, int(fft_frequency_axis_hz.size))
    bin_width_hz = float(np.median(np.diff(fft_frequency_axis_hz)))
    dc_gap_half_width_hz = 0.5 * abs(bin_width_hz)
    dc_bin = np.abs(fft_frequency_axis_hz) <= dc_gap_half_width_hz
    interpolated_curves: list[np.ndarray] = []
    zoom_xlim = (-10_000.0, 10_000.0)
    zoom_values: list[np.ndarray] = []

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 9.0),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    fft_axis_hz = np.asarray(fft_frequency_axis_hz, dtype=np.float64)
    for index, channel in enumerate(physical_channel):
        rel = np.asarray(fft_axis_hz - float(expected_offset_hz[index]), dtype=np.float64)
        values = np.asarray(power[index, :], dtype=np.float64)
        good = np.isfinite(rel) & np.isfinite(values) & (values > 0.0)
        if np.count_nonzero(good) < 2:
            continue
        rel = rel[good]
        values = values[good]
        dc_in_curve = dc_bin[good]
        level_db = 10.0 * np.log10(values)
        level_db[dc_in_curve] = np.nan
        baseline_db = _rolling_nanmedian_baseline(level_db)
        residual_db = level_db - baseline_db
        order = np.argsort(rel)
        rel = rel[order]
        residual_db = residual_db[order]
        label = (
            rf"DTV {int(channel)}, "
            rf"$f_{{\mathrm{{CHIME}}}}="
            rf"{chime_frequency_hz[index] / 1.0e6:.1f}"
            rf"\,\mathrm{{MHz}}$"
        )
        for axis_index, axis in enumerate(axes):
            axis.plot(
                rel,
                residual_db,
                color=colors[int(index)],
                linewidth=1.0,
                alpha=0.85,
                label=label if axis_index == 0 else None,
            )
        zoom = residual_db[
            (rel >= zoom_xlim[0]) & (rel <= zoom_xlim[1]) & np.isfinite(residual_db)
        ]
        if zoom.size:
            zoom_values.append(zoom)
        interp_ok = np.isfinite(residual_db)
        if np.count_nonzero(interp_ok) >= 2:
            interpolated_curves.append(
                np.interp(
                    grid,
                    rel[interp_ok],
                    residual_db[interp_ok],
                    left=np.nan,
                    right=np.nan,
                )
            )

    if interpolated_curves:
        stack = np.vstack(interpolated_curves)
        finite_stack = np.isfinite(stack)
        counts = np.sum(finite_stack, axis=0)
        sums = np.sum(np.where(finite_stack, stack, 0.0), axis=0)
        mean_level_db = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        for dc_position_hz in -expected_offset_hz:
            mean_level_db[
                np.abs(grid - float(dc_position_hz)) <= dc_gap_half_width_hz
            ] = np.nan
        for axis_index, axis in enumerate(axes):
            axis.plot(
                grid,
                mean_level_db,
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean over plotted pilots" if axis_index == 0 else None,
                zorder=20,
            )
        mean_zoom = mean_level_db[
            (grid >= zoom_xlim[0]) & (grid <= zoom_xlim[1]) & np.isfinite(mean_level_db)
        ]
        if mean_zoom.size:
            zoom_values.append(mean_zoom)

    baseline_window_khz = (
        TIME_AVERAGE_SPECTRUM_BASELINE_WINDOW_BINS * abs(bin_width_hz) / 1.0e3
    )
    for axis in axes:
        axis.axvline(0.0, color="0.25", linewidth=1.0, linestyle=":", alpha=0.7)
        axis.set_ylabel(r"Power above local rolling-median baseline [dB]")
        axis.grid(True, alpha=0.25)
    axes[0].set_xlim(shifted_min, shifted_max)
    axes[0].set_title(
        rf"Time-averaged {int(fft_frequency_axis_hz.size)}-point FFT spectrum, "
        rf"pilot aligned at $0$ Hz; DC removed; "
        rf"{baseline_window_khz:.1f} kHz baseline subtracted"
    )
    axes[0].legend(ncol=3, fontsize="x-small")
    axes[1].set_xlim(*zoom_xlim)
    axes[1].set_title(r"Zoom: $\pm10\,\mathrm{kHz}$ around nominal pilot")
    axes[1].set_xlabel(r"FFT bin offset from nominal DTV pilot [Hz]")
    if zoom_values:
        zoom_all = np.concatenate(zoom_values)
        if zoom_all.size and np.any(np.isfinite(zoom_all)):
            ymin = float(np.nanmin(zoom_all))
            ymax = float(np.nanmax(zoom_all))
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = max(0.5, 0.05 * (ymax - ymin))
                axes[1].set_ylim(ymin - pad, ymax + pad)

    path = (
        Path(run_dir)
        / "figures"
        / "frequency_offset_time_average_spectrum_by_pilot.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_spectrogram(
    run_dir: Path,
    data: np.lib.npyio.NpzFile,
    *,
    key: str,
    title: str,
    colorbar_label: str,
    basename: str,
    cmap: str,
    symmetric_half_width: float | None = None,
) -> Path:
    from .plots import (
        _add_dtv_channel_axis,
        _add_frame_index_axis,
        _coordinate_edges,
        _set_spectrogram_time_axis,
        _setup_matplotlib,
    )

    plt = _setup_matplotlib()
    values = np.asarray(data[key], dtype=np.float64)
    valid = np.asarray(data["valid"]) != 0
    plot_values = np.where(valid, values, np.nan)
    time_s = np.asarray(data["relative_time_s"], dtype=np.float64)
    frame_index = np.asarray(data["frame_index"], dtype=np.int64)
    freq_hz = np.asarray(data["chime_frequency_hz"], dtype=np.float64)
    freq_mhz = np.asarray(freq_hz / 1.0e6, dtype=np.float64).reshape(-1)
    physical_channel = np.asarray(data["physical_channel"], dtype=np.int32).reshape(-1)
    x_edges = _coordinate_edges(time_s)
    y_edges = _coordinate_edges(freq_mhz)
    vmin = vmax = None
    if symmetric_half_width is not None:
        vmin = -float(symmetric_half_width)
        vmax = float(symmetric_half_width)
    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        plot_values.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"Relative time, $t_{\mathrm{rel}}\;[\mathrm{s}]$")
    ax.set_ylabel(r"CHIME frequency, $f_{\mathrm{CHIME}}\;[\mathrm{MHz}]$")
    ax.set_title(title)
    ax.set_yticks(freq_mhz)
    ax.set_yticklabels([f"{value:.1f}" for value in freq_mhz])
    _set_spectrogram_time_axis(ax, x_edges)
    _add_frame_index_axis(ax, relative_time_s=time_s, frame_index=frame_index)
    _add_dtv_channel_axis(
        ax,
        frequency_mhz=freq_mhz,
        physical_channel=physical_channel,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    path = Path(run_dir) / "figures" / f"{basename}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_frequency_offset_products(run_dir: Path) -> list[Path]:
    data = np.load(Path(run_dir) / FREQUENCY_OFFSET_OUTPUTS_FILENAME)
    half_width = float(np.asarray(data["peak_search_half_width_hz"]).reshape(()))
    outputs = [
        _plot_histogram(Path(run_dir), data),
        _plot_spectrogram(
            Path(run_dir),
            data,
            key="frequency_offset_hz",
            title=r"CHIME DTV pilot frequency-offset spectrogram",
            colorbar_label=r"$\Delta f$ [Hz]",
            basename="frequency_offset_spectrogram",
            cmap="coolwarm",
            symmetric_half_width=half_width,
        ),
        _plot_spectrogram(
            Path(run_dir),
            data,
            key="peak_prominence_db",
            title=r"CHIME DTV pilot peak-prominence spectrogram",
            colorbar_label=r"Peak prominence [dB]",
            basename="peak_prominence_spectrogram",
            cmap="viridis",
        ),
    ]
    spectrum = _plot_time_average_spectrum(Path(run_dir), data)
    if spectrum is not None:
        outputs.append(spectrum)
    data.close()
    return outputs


def run_frequency_offset_diagnostic(config: FrequencyOffsetConfig) -> dict[str, Path]:
    if int(config.frames_per_chunk) != 1:
        raise ValueError(
            "frequency-offset diagnostic currently requires frames_per_chunk=1"
        )
    if int(config.frame_size_samples) != int(config.fft_size):
        raise ValueError(
            "frequency-offset diagnostic currently requires fft_size == frame_size_samples"
        )

    run_dir = Path(config.output_dir)
    ensure_run_dirs(run_dir)
    receiver_profile = load_receiver_profile(config.receiver_profile)
    sample_rate_hz = float(receiver_profile.coarse_channel_width_hz)
    fft_bin_width_hz = sample_rate_hz / float(config.fft_size)
    window = _window(config.window_name, int(config.fft_size))
    discovered = discover_chime_pilot_datasets(
        Path(config.input_dir),
        dataset_path=config.dataset_path,
        filename_pattern=config.filename_pattern,
    )
    selected = _select_datasets(
        discovered,
        physical_channels=config.physical_channels,
        physical_channel_range=config.physical_channel_range,
    )
    frame_index = _common_frame_indices(
        selected,
        frame_size_samples=int(config.frame_size_samples),
        max_frames=config.max_frames,
        every_n_frames=int(config.every_n_frames),
    )
    shape = (int(frame_index.size), int(len(selected)))
    physical_channel = np.asarray(
        [dataset.physical_channel for dataset in selected], dtype=np.int32
    )
    pilot_frequency_hz = np.asarray(
        [dataset.pilot_frequency_hz for dataset in selected], dtype=np.float64
    )
    coarse_channel_center_hz = np.asarray(
        [_coarse_center_hz(dataset, receiver_profile) for dataset in selected],
        dtype=np.float64,
    )
    chime_frequency_hz = coarse_channel_center_hz.copy()
    expected_pilot_offset_hz = pilot_frequency_hz - coarse_channel_center_hz
    relative_time_s = relative_time_seconds(
        frame_index,
        frame_size_samples=int(config.frame_size_samples),
        sample_rate_hz=sample_rate_hz,
    )

    peak_offset_hz = np.full(shape, np.nan, dtype=np.float64)
    frequency_offset_hz = np.full(shape, np.nan, dtype=np.float64)
    peak_power_linear = np.full(shape, np.nan, dtype=np.float64)
    local_floor_power_linear = np.full(shape, np.nan, dtype=np.float64)
    peak_prominence_db = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=np.uint8)
    fft_frequency_axis_hz = np.fft.fftshift(
        np.fft.fftfreq(int(config.fft_size), d=1.0 / sample_rate_hz)
    )
    time_average_spectrum_sum = np.zeros(
        (int(len(selected)), int(config.fft_size)),
        dtype=np.float64,
    )
    time_average_spectrum_count = np.zeros(int(len(selected)), dtype=np.uint64)
    backend_used = str(config.backend)

    for pilot_index, dataset in enumerate(selected):
        print(
            "[frequency-offset] "
            f"channel {int(dataset.physical_channel)} "
            f"({pilot_index + 1}/{len(selected)}), "
            f"{int(frame_index.size)} frames",
            flush=True,
        )
        for output_frame, source_frame in enumerate(frame_index):
            if output_frame == 0 or (output_frame + 1) % 10 == 0:
                print(
                    "[frequency-offset] "
                    f"channel {int(dataset.physical_channel)} "
                    f"frame {output_frame + 1}/{int(frame_index.size)}",
                    flush=True,
                )
            start_sample = int(source_frame) * int(config.frame_size_samples)
            stop_sample = start_sample + int(config.frame_size_samples)
            block = read_complex_window(
                dataset,
                start_sample=start_sample,
                stop_sample=stop_sample,
            )
            power_sum, backend_used = frame_noncoherent_fft_power(
                block,
                sample_encoding=dataset.sample_encoding,
                spectral_sense=receiver_profile.spectral_sense,
                fft_size=int(config.fft_size),
                stream_batch_size=int(config.stream_batch_size),
                window=window,
                backend=config.backend,
            )
            time_average_spectrum_sum[pilot_index, :] += power_sum
            time_average_spectrum_count[pilot_index] += np.uint64(1)
            estimate = estimate_peak_offset_from_power(
                power_sum,
                sample_rate_hz=sample_rate_hz,
                expected_offset_hz=float(expected_pilot_offset_hz[pilot_index]),
                fft_size=int(config.fft_size),
                peak_search_half_width_hz=float(config.peak_search_half_width_hz),
            )
            peak_offset_hz[output_frame, pilot_index] = float(
                estimate["peak_offset_hz"]
            )
            frequency_offset_hz[output_frame, pilot_index] = float(
                estimate["frequency_offset_hz"]
            )
            peak_power_linear[output_frame, pilot_index] = float(
                estimate["peak_power_linear"]
            )
            local_floor_power_linear[output_frame, pilot_index] = float(
                estimate["local_floor_power_linear"]
            )
            prominence = float(estimate["peak_prominence_db"])
            peak_prominence_db[output_frame, pilot_index] = prominence
            valid_estimate = np.isfinite(frequency_offset_hz[output_frame, pilot_index])
            if config.min_peak_prominence_db is not None:
                valid_estimate = valid_estimate and prominence >= float(
                    config.min_peak_prominence_db
                )
            valid[output_frame, pilot_index] = 1 if valid_estimate else 0

    time_average_spectrum_power_linear = np.full_like(
        time_average_spectrum_sum,
        np.nan,
        dtype=np.float64,
    )
    nonzero_spectrum_counts = time_average_spectrum_count > 0
    if np.any(nonzero_spectrum_counts):
        time_average_spectrum_power_linear[nonzero_spectrum_counts, :] = (
            time_average_spectrum_sum[nonzero_spectrum_counts, :]
            / time_average_spectrum_count[nonzero_spectrum_counts, np.newaxis]
        )

    outputs: dict[str, Path] = {"frequency_offset_outputs": _write_outputs(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        coarse_channel_center_hz=coarse_channel_center_hz,
        expected_pilot_offset_hz=expected_pilot_offset_hz,
        frame_index=frame_index,
        relative_time_s=relative_time_s,
        peak_offset_hz=peak_offset_hz,
        frequency_offset_hz=frequency_offset_hz,
        peak_power_linear=peak_power_linear,
        local_floor_power_linear=local_floor_power_linear,
        peak_prominence_db=peak_prominence_db,
        valid=valid,
        fft_frequency_axis_hz=fft_frequency_axis_hz,
        time_average_spectrum_power_linear=time_average_spectrum_power_linear,
        time_average_spectrum_count=time_average_spectrum_count,
        fft_size=int(config.fft_size),
        fft_bin_width_hz=fft_bin_width_hz,
        sample_rate_hz=sample_rate_hz,
        window_name=config.window_name,
        peak_search_half_width_hz=float(config.peak_search_half_width_hz),
    ), "summary_table": _write_summary_table(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        coarse_channel_center_hz=coarse_channel_center_hz,
        expected_pilot_offset_hz=expected_pilot_offset_hz,
        frequency_offset_hz=frequency_offset_hz,
        peak_prominence_db=peak_prominence_db,
        valid=valid,
    )}
    input_manifest_path = run_dir / "input_manifest.json"
    write_json_strict(
        input_manifest_path,
        {
            "schema_version": "fstat_chime_frequency_offset_input_manifest_v1",
            "input_dir": str(config.input_dir),
            "absolute_time_used": False,
            "datasets": [dataset_manifest(dataset) for dataset in selected],
        },
        indent=2,
        sort_keys=True,
    )
    outputs["input_manifest"] = input_manifest_path
    stats_path = run_dir / "stats.json"
    write_json_strict(
        stats_path,
        {
            "schema_version": "fstat_chime_frequency_offset_stats_v1",
            "coordinate_system": COORDINATE_SYSTEM,
            "fft_size": int(config.fft_size),
            "fft_bin_width_hz": float(fft_bin_width_hz),
            "sample_rate_hz": float(sample_rate_hz),
            "frame_size_samples": int(config.frame_size_samples),
            "frames_per_chunk": int(config.frames_per_chunk),
            "every_n_frames": int(config.every_n_frames),
            "stream_batch_size": int(config.stream_batch_size),
            "peak_search_half_width_hz": float(config.peak_search_half_width_hz),
            "window_name": str(config.window_name),
            "backend_requested": str(config.backend),
            "backend_used": backend_used,
            "input_spectral_sense": str(receiver_profile.spectral_sense),
            "input_requires_time_reversal": bool(
                spectral_sense_requires_time_reversal(receiver_profile.spectral_sense)
            ),
            "min_peak_prominence_db": config.min_peak_prominence_db,
            "num_frames": int(frame_index.size),
            "num_pilots": int(len(selected)),
        },
        indent=2,
        sort_keys=True,
    )
    outputs["stats"] = stats_path
    if config.plot:
        for index, path in enumerate(plot_frequency_offset_products(run_dir)):
            outputs[f"plot_{index}"] = path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate CHIME DTV pilot frequency offsets with frame FFTs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--receiver-profile", type=Path, default=DEFAULT_RECEIVER_PROFILE
    )
    parser.add_argument("--dataset-path", default="baseband")
    parser.add_argument("--filename-pattern", default="*.h5")
    parser.add_argument("--physical-channel", type=int, action="append", default=None)
    parser.add_argument("--physical-channel-range", default=None)
    parser.add_argument(
        "--frame-size-samples", type=int, default=DEFAULT_FRAME_SIZE_SAMPLES
    )
    parser.add_argument(
        "--frames-per-chunk", type=int, default=DEFAULT_FRAMES_PER_CHUNK
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE)
    parser.add_argument(
        "--stream-batch-size", type=int, default=DEFAULT_STREAM_BATCH_SIZE
    )
    parser.add_argument(
        "--peak-search-half-width-hz",
        type=float,
        default=DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ,
    )
    parser.add_argument("--window", dest="window_name", default=DEFAULT_WINDOW_NAME)
    parser.add_argument("--min-peak-prominence-db", type=float, default=None)
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "cupy"],
        default=DEFAULT_BACKEND,
    )
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FrequencyOffsetConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
        dataset_path=args.dataset_path,
        filename_pattern=args.filename_pattern,
        receiver_profile=args.receiver_profile,
        frame_size_samples=int(args.frame_size_samples),
        frames_per_chunk=int(args.frames_per_chunk),
        max_frames=args.max_frames,
        every_n_frames=int(args.every_n_frames),
        fft_size=int(args.fft_size),
        stream_batch_size=int(args.stream_batch_size),
        peak_search_half_width_hz=float(args.peak_search_half_width_hz),
        window_name=str(args.window_name),
        min_peak_prominence_db=args.min_peak_prominence_db,
        backend=str(args.backend),
        plot=bool(args.plot),
    )
    outputs = run_frequency_offset_diagnostic(config)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
