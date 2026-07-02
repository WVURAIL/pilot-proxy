# coding=utf-8
"""Plot products for CHIME real-data detector runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy import dtype, float64, ndarray

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    NO_PILOT_EXCESS_FSTAT,
    fstat_raw_to_fstat_level_db,
    pnr_bin_db_to_snr_shelf_db,
    snr_shelf_db_to_pnr_bin_db,
)
from pilot_proxy.plot_style import setup_matplotlib

from .products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
    spectrum_before_after,
    write_spectrum_table,
)

FIGURE_DPI = 300
OUTLIER_PHYSICAL_CHANNEL = 30
FSTAT_TOP_TICKS_DB = np.asarray(
    [1.0e-5, 1.0e-4, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 20.0]
)
SNR_SHELF_TOP_TICKS_DB = np.asarray([-60.0, -30.0, -25.0, -20.0, -15.0, -10.0, 0.0])

KNOWN_CHIME_FIGURES = frozenset({
    "snr_shelf_histogram_by_pilot.png",
    "fstat_survival_by_pilot.png",
    "fstat_level_spectrogram.png",
    "baseband_spectrogram.png",
    "baseband_spectrum_before_after_mask.png",
    "mask_spectrogram.png",
})

def _setup_matplotlib():
    return setup_matplotlib(force_agg=True)


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _finite_values(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if valid is not None:
        arr = arr[np.asarray(valid) != 0]
    arr = arr[np.isfinite(arr)]
    return arr.astype(np.float64, copy=False)


def _set_snr_shelf_ticks(ax, *, xmin: float, xmax: float) -> None:
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    span = float(xmax - xmin)
    if not np.isfinite(span) or span <= 0.0:
        return
    step = 2.0 if span <= 20.0 else 10.0 if span >= 60.0 else 5.0
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))


def _format_db_tick(value: float) -> str:
    if abs(float(value)) >= 1.0 or float(value) == 0.0:
        return f"{value:g}"
    return f"{value:.3g}"


def _snr_shelf_db_to_fstat_level_db(values) -> np.ndarray:
    snr = np.asarray(values, dtype=np.float64)
    pnr = snr_shelf_db_to_pnr_bin_db(snr)
    raw = NO_PILOT_EXCESS_FSTAT + DB_LINEAR_BASE ** (pnr / DB_POWER_FACTOR)
    return np.asarray(fstat_raw_to_fstat_level_db(raw), dtype=np.float64)


def _fstat_level_db_to_snr_shelf_db(values) -> np.ndarray:
    rf = np.asarray(values, dtype=np.float64)
    raw = DB_LINEAR_BASE ** (rf / DB_POWER_FACTOR)
    out = np.full(rf.shape, np.nan, dtype=np.float64)
    valid = raw > NO_PILOT_EXCESS_FSTAT
    pnr = np.full(rf.shape, np.nan, dtype=np.float64)
    pnr[valid] = DB_POWER_FACTOR * np.log10(raw[valid] - NO_PILOT_EXCESS_FSTAT)
    out[valid] = pnr_bin_db_to_snr_shelf_db(pnr[valid])
    return out


def _add_fstat_level_top_axis_for_snr(ax, *, xmin: float, xmax: float) -> None:
    positions = _fstat_level_db_to_snr_shelf_db(FSTAT_TOP_TICKS_DB)
    keep = np.isfinite(positions) & (positions >= xmin) & (positions <= xmax)
    if not np.any(keep):
        return
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(positions[keep])
    top.set_xticklabels([_format_db_tick(value) for value in FSTAT_TOP_TICKS_DB[keep]])
    top.tick_params(axis="x", labelsize="small")
    top.set_xlabel(r"$F$-statistic, $10\log_{10}F\;[\mathrm{dB}]$")


def _add_snr_shelf_top_axis_for_fstat(ax) -> None:
    xmin, xmax = ax.get_xlim()
    positions = _snr_shelf_db_to_fstat_level_db(SNR_SHELF_TOP_TICKS_DB)
    keep = np.isfinite(positions) & (positions >= xmin) & (positions <= xmax)
    if not np.any(keep):
        return
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(positions[keep])
    top.set_xticklabels(
        [_format_db_tick(value) for value in SNR_SHELF_TOP_TICKS_DB[keep]]
    )
    top.tick_params(axis="x", labelsize="small")
    top.set_xlabel(r"$\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$")


def _robust_color_limits(
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float | None, float | None]:
    finite = _finite_values(values)
    if finite.size == 0:
        return None, None
    lo = float(np.percentile(finite, lower_percentile))
    hi = float(np.percentile(finite, upper_percentile))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def _product_frequency_hz(product: np.lib.npyio.NpzFile) -> np.ndarray:
    if "chime_frequency_hz" in product.files:
        return np.asarray(product["chime_frequency_hz"], dtype=np.float64)
    return np.asarray(product["pilot_frequency_hz"], dtype=np.float64)


def _manifest_frequency_hz(
    run_dir: Path, physical_channel: np.ndarray
) -> np.ndarray | None:
    manifest_path = Path(run_dir) / "input_manifest.json"
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("datasets", [])
    if not isinstance(rows, list):
        return None
    by_channel = {
        int(row["physical_channel"]): float(row["coarse_channel_center_hz"])
        for row in rows
        if isinstance(row, dict)
        and row.get("physical_channel") is not None
        and row.get("coarse_channel_center_hz") is not None
    }
    values: list[float] = []
    for channel in physical_channel:
        if int(channel) not in by_channel:
            return None
        values.append(by_channel[int(channel)])
    return np.asarray(values, dtype=np.float64)


def _run_frequency_hz(run_dir: Path, product: np.lib.npyio.NpzFile) -> np.ndarray:
    if "chime_frequency_hz" in product.files:
        return np.asarray(product["chime_frequency_hz"], dtype=np.float64)
    from_manifest = _manifest_frequency_hz(Path(run_dir), product["physical_channel"])
    if from_manifest is not None:
        return from_manifest
    return np.asarray(product["pilot_frequency_hz"], dtype=np.float64)


def _coordinate_edges(centers: np.ndarray, *, default_step: float = 1.0) -> np.ndarray:
    coords = np.asarray(centers, dtype=np.float64)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("coordinate centers must be a non-empty 1D array")
    if coords.size == 1:
        half = 0.5 * float(default_step)
        return np.asarray([coords[0] - half, coords[0] + half], dtype=np.float64)
    mid = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mid[0] - coords[0])
    last = coords[-1] + (coords[-1] - mid[-1])
    return np.concatenate(([first], mid, [last])).astype(np.float64)


def _sparse_tick_indices(size: int, *, max_ticks: int = 8) -> np.ndarray:
    if size <= 0:
        return np.asarray([], dtype=np.int64)
    count = min(int(size), int(max_ticks))
    return np.unique(np.round(np.linspace(0, size - 1, count)).astype(np.int64))


def _add_frame_index_axis(
        ax, *, relative_time_s: np.ndarray, frame_index: np.ndarray
) -> None:
    times = np.asarray(relative_time_s, dtype=np.float64)
    frames = np.asarray(frame_index, dtype=np.float64)
    if times.size != frames.size or times.size == 0:
        return
    if times.size == 1:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([float(times[0])])
        sec.set_xticklabels([str(int(frames[0]))])
        sec.set_xlabel(r"Frame index, $n_{\mathrm{frame}}$")
        return
    dt = float((times[-1] - times[0]) / (frames[-1] - frames[0]))
    if not np.isfinite(dt) or dt == 0.0:
        return

    def time_to_frame(value):
        arr = np.asarray(value, dtype=np.float64)
        return (arr - times[0]) / dt + frames[0]

    def frame_to_time(value):
        arr = np.asarray(value, dtype=np.float64)
        return (arr - frames[0]) * dt + times[0]

    sec = ax.secondary_xaxis("top", functions=(time_to_frame, frame_to_time))
    tick_indices = _sparse_tick_indices(times.size)
    sec.set_xticks(frames[tick_indices])
    sec.set_xticklabels([str(int(value)) for value in frames[tick_indices]])
    sec.set_xlabel(r"Frame index, $n_{\mathrm{frame}}$")


def _add_dtv_channel_axis(
    ax,
    *,
    frequency_mhz: np.ndarray,
    physical_channel: np.ndarray,
    tick_labels: Sequence[str] | None = None,
    ylabel: str = r"DTV physical channel",
) -> None:
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(np.asarray(frequency_mhz, dtype=np.float64))
    right.set_yticklabels(
        list(tick_labels)
        if tick_labels is not None
        else [str(int(ch)) for ch in physical_channel]
    )
    right.set_ylabel(ylabel)


def _set_spectrogram_time_axis(ax, x_edges: np.ndarray) -> None:
    from matplotlib.ticker import MultipleLocator

    edges = np.asarray(x_edges, dtype=np.float64)
    if edges.size < 2:
        return
    left = max(0.0, float(edges[0]))
    right = float(edges[-1])
    if 9.0 <= right < 10.0:
        right = 10.0
    ax.set_xlim(left, right)
    if right >= 9.0:
        ax.xaxis.set_major_locator(MultipleLocator(2.0))


def _without_outlier_channel(physical_channel: np.ndarray) -> np.ndarray:
    keep = np.asarray(physical_channel, dtype=np.int64) != OUTLIER_PHYSICAL_CHANNEL
    if not np.any(keep):
        return np.ones_like(keep, dtype=bool)
    return keep.astype(bool, copy=False)


def _write_histogram_summary(
    path: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    snr_shelf_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "num_detector_valid_frames",
                "num_positive_excess_frames",
                "positive_excess_fraction",
                "mean_snr_shelf_db",
                "max_snr_shelf_db",
                "mask_fraction",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            values = np.asarray(snr_shelf_db[:, index], dtype=np.float64)
            finite = values[np.isfinite(values)]
            detector_valid = np.asarray(valid[:, index]) != 0
            detector_valid_count = int(np.sum(detector_valid))
            positive_count = int(finite.size)
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(chime_frequency_hz[index]),
                    "num_detector_valid_frames": detector_valid_count,
                    "num_positive_excess_frames": positive_count,
                    "positive_excess_fraction": (
                        float(positive_count / detector_valid_count)
                        if detector_valid_count
                        else float("nan")
                    ),
                    "mean_snr_shelf_db": (
                        float(np.mean(finite)) if finite.size else float("nan")
                    ),
                    "max_snr_shelf_db": (
                        float(np.max(finite)) if finite.size else float("nan")
                    ),
                    "mask_fraction": float(np.mean(mask[:, index] != 0)),
                }
            )


def _write_fstat_summary(
    path: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    fstat_level_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "num_detector_valid_frames",
                "mean_fstat_level_db",
                "max_fstat_level_db",
                "mask_fraction",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            detector_valid = np.asarray(valid[:, index]) != 0
            values = _finite_values(fstat_level_db[:, index], detector_valid)
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(chime_frequency_hz[index]),
                    "num_detector_valid_frames": int(np.sum(detector_valid)),
                    "mean_fstat_level_db": (
                        float(np.mean(values)) if values.size else float("nan")
                    ),
                    "max_fstat_level_db": (
                        float(np.max(values)) if values.size else float("nan")
                    ),
                    "mask_fraction": float(np.mean(mask[:, index] != 0)),
                }
            )


def _histogram_probability_density(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(int(np.asarray(bins).size) - 1, dtype=np.float64)
    counts, _ = np.histogram(finite, bins=bins)
    bin_width = np.diff(np.asarray(bins, dtype=np.float64))
    return counts.astype(np.float64) / (float(finite.size) * bin_width)


def _survival_probability_on_grid(values: np.ndarray, grid: np.ndarray) -> ndarray[tuple[()], dtype[float64]] | float:
    """Evaluate P(values >= grid) for finite values on a common plot grid."""
    finite = np.sort(_finite_values(values))
    grid_values = np.asarray(grid, dtype=np.float64)
    if finite.size == 0:
        return np.full(grid_values.shape, np.nan, dtype=np.float64)
    count_below = np.searchsorted(finite, grid_values, side="left")
    return (float(finite.size) - count_below.astype(np.float64)) / float(finite.size)


def _fstat_survival_plot_bounds(
    panel_values: Sequence[np.ndarray],
    *,
    include_comparison_range: bool,
) -> tuple[float, float]:
    finite_all = np.concatenate([_finite_values(values) for values in panel_values])
    xmin = float(np.min(finite_all))
    xmax = float(np.max(finite_all))
    span = xmax - xmin
    pad = max(0.05, 0.03 * span) if span > 0.0 else 0.5
    left = xmin - pad
    right = xmax + pad
    if include_comparison_range:
        left = min(left, -1.0)
        right = max(right, 1.0)
    return left, right


def plot_snr_shelf_histogram(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    physical_channel = detector["physical_channel"]
    pilot_frequency_hz = detector["pilot_frequency_hz"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), detector)
    snr_shelf_db = detector["snr_shelf_db"]
    fstat_level_db = detector["fstat_level_db"]
    mask = detector["mask"]
    valid = (
        detector["valid"]
        if "valid" in detector.files
        else (detector["p_ref_sum_u64"] > 0).astype(np.uint8)
    )

    panels = [
        (r"$\mathrm{SNR}_{\mathrm{shelf}}$ distribution", -90.0, 0.0),
        (
            r"Low range: $\mathrm{SNR}_{\mathrm{shelf}}$ from $-90$ to $-25$ dB",
            -90.0,
            -25.0,
        ),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.5, 8.4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )

    for panel_index, (title, xmin, xmax) in enumerate(panels):
        ax = axes[panel_index]
        bins = np.linspace(xmin, xmax, 100)
        channel_histograms: list[np.ndarray] = []
        for index, channel in enumerate(physical_channel):
            detector_valid = np.asarray(valid[:, index]) != 0
            values = np.asarray(snr_shelf_db[:, index], dtype=np.float64)
            hist = _histogram_probability_density(values[detector_valid], bins)
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
        ax.set_xlim(xmin, xmax)
        _set_snr_shelf_ticks(ax, xmin=xmin, xmax=xmax)
        ax.set_ylabel(r"Probability density [1/dB]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if panel_index == len(panels) - 1:
            ax.set_xlabel(
                r"DTV pilot shelf strength, "
                r"$\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
            )
        if panel_index == 0:
            ax.legend(ncol=3, fontsize="x-small")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "snr_shelf_histogram_by_pilot.png",
    ]
    for path in outputs:
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    _write_fstat_summary(
        Path(run_dir) / "tables" / "fstat_summary_by_pilot.csv",
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        fstat_level_db=fstat_level_db,
        mask=mask,
        valid=valid,
    )
    _write_histogram_summary(
        Path(run_dir) / "tables" / "snr_shelf_histogram_summary.csv",
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        snr_shelf_db=snr_shelf_db,
        mask=mask,
        valid=valid,
    )
    return outputs


def plot_fstat_survival(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    physical_channel = detector["physical_channel"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), detector)
    fstat_level_db = detector["fstat_level_db"]
    valid = (
        detector["valid"]
        if "valid" in detector.files
        else (detector["p_ref_sum_u64"] > 0).astype(np.uint8)
    )

    panels = [
        (r"All DTV pilot channels", np.ones_like(physical_channel, dtype=bool)),
        (
            rf"Excluding DTV {OUTLIER_PHYSICAL_CHANNEL}",
            _without_outlier_channel(physical_channel),
        ),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.5, 8.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )
    for panel_index, (title, keep) in enumerate(panels):
        ax = axes[panel_index]
        channel_values: list[tuple[int, int, np.ndarray]] = []
        for index, channel in enumerate(physical_channel):
            if not keep[index]:
                continue
            values = np.sort(_finite_values(fstat_level_db[:, index], valid[:, index]))
            if values.size == 0:
                continue
            channel_values.append((index, int(channel), values))

        panel_values = [values for _, _, values in channel_values]
        if panel_values:
            xlim = _fstat_survival_plot_bounds(
                panel_values,
                include_comparison_range=panel_index == 1,
            )
            max_count = max(values.size for values in panel_values)
            survival_floor = 0.5 / float(max_count)
            grid = np.linspace(xlim[0], xlim[1], 900)

            for index, channel, values in channel_values:
                survival = np.asarray(
                    _survival_probability_on_grid(values, grid),
                    dtype=np.float64,
                )
                ax.step(
                    grid,
                    np.maximum(survival, survival_floor),
                    where="post",
                    color=colors[int(index)],
                    label=(
                        rf"DTV {channel}, "
                        rf"$f_{{\mathrm{{CHIME}}}}="
                        rf"{chime_frequency_hz[index] / 1.0e6:.1f}"
                        rf"\,\mathrm{{MHz}}$"
                    ),
                )

            mean_survival = np.nanmean(
                np.vstack(
                    [
                        _survival_probability_on_grid(values, grid)
                        for values in panel_values
                    ]
                ),
                axis=0,
            )
            ax.plot(
                grid,
                np.maximum(mean_survival, survival_floor),
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean over plotted pilots",
                zorder=20,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(survival_floor, 1.15)
        ax.set_ylabel(r"$P(R_F \geq T)$")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        _add_snr_shelf_top_axis_for_fstat(ax)
        if panel_index == 0:
            ax.legend(ncol=3, fontsize="x-small")
        if panel_index == len(panels) - 1:
            ax.set_xlabel(r"$R_F=10\log_{10}F\;[\mathrm{dB}]$")
    fig.suptitle(r"CHIME DTV pilot $F$-statistic survival curves")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "fstat_survival_by_pilot.png",
    ]
    for path in outputs:
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return outputs


def _load_json_object(path: Path) -> dict[str, object]:
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _mask_label_from_policy(policy: object) -> str:
    if not isinstance(policy, dict):
        return ""
    source = policy.get("mask_source")
    if source is None:
        return ""
    source_text = str(source)
    if source_text == "positive_excess":
        return (
            r"Mask rule: positive excess; valid if $P_{\mathrm{ref}}\ne0$, "
            r"mask if $P_t > (P_{\mathrm{ref}}\gg1)$"
        )
    return f"Mask rule: {source_text}"


def _mask_label_for_run(run_dir: Path) -> str:
    run = Path(run_dir)
    stats = _load_json_object(run / "stats.json")
    run_config = _load_json_object(run / "run_config.json")
    return _mask_label_from_policy(
        stats.get("mask_policy") or run_config.get("mask_policy")
    )


def plot_baseband_spectrum(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    physical_channel = cache["physical_channel"]
    pilot_frequency_hz = cache["pilot_frequency_hz"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), cache)
    baseband_power_linear = cache["baseband_power_linear"]
    mask = cache["mask"]
    valid = (
        cache["valid"]
        if "valid" in cache.files
        else np.ones_like(baseband_power_linear, dtype=np.uint8)
    )
    before_db, after_db = spectrum_before_after(baseband_power_linear, mask, valid)

    x = chime_frequency_hz / 1.0e6
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(x, before_db, marker="o", label=r"before mask")
    ax.plot(x, after_db, marker="o", label=r"after mask")
    ax.set_xlabel(r"CHIME frequency, $f_{\mathrm{CHIME}}\;[\mathrm{MHz}]$")
    ax.set_ylabel(r"$10\log_{10}P_{\mathrm{bb}}\;[\mathrm{dB}]$")
    title = r"Integrated baseband power at ATSC pilot coarse channels"
    mask_label = _mask_label_for_run(Path(run_dir))
    if mask_label:
        title += "\n" + mask_label
    if not np.any(mask != 0):
        title += (
            "\n" + r"No mask applied; before and after are identical by construction."
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(x)
    sec.set_xticklabels([str(int(ch)) for ch in physical_channel], rotation=90)
    sec.set_xlabel(r"DTV physical channel")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "baseband_spectrum_before_after_mask.png",
    ]
    for path in outputs:
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    write_spectrum_table(
        Path(run_dir),
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    return outputs


def _plot_spectrogram(
    *,
    run_dir: Path,
    values: np.ndarray,
    physical_channel: np.ndarray,
    frequency_hz: np.ndarray,
    frame_index: np.ndarray,
    relative_time_s: np.ndarray,
    title: str,
    colorbar_label: str,
    basename: str,
    cmap: str,
    robust_limits: bool = False,
    exclude_outlier_panel: bool = False,
    discrete_mask_colorbar: bool = False,
) -> list[Path]:
    plt = _setup_matplotlib()
    if discrete_mask_colorbar:
        from matplotlib.colors import BoundaryNorm, ListedColormap

        plot_cmap = ListedColormap(["white", "black"])
    else:
        BoundaryNorm = None
        plot_cmap = cmap
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    freq_hz = np.asarray(frequency_hz, dtype=np.float64)
    freq_mhz = np.asarray(freq_hz / 1.0e6, dtype=np.float64).reshape(-1)
    channel_array = np.asarray(physical_channel).reshape(-1)
    x_edges = _coordinate_edges(time_s)
    value_array = np.asarray(values)
    panels: list[tuple[str, np.ndarray]] = [
        (title, np.ones_like(channel_array, dtype=bool)),
    ]
    if exclude_outlier_panel:
        panels.append(
            (
                rf"{title}, excluding DTV {OUTLIER_PHYSICAL_CHANNEL}",
                _without_outlier_channel(channel_array),
            )
        )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.8, 5.2 if len(panels) == 1 else 8.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for panel_index, (ax, (panel_title, keep)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        panel_freq_mhz = freq_mhz[keep]
        panel_channels = channel_array[keep]
        panel_values = value_array[:, keep]
        y_edges = _coordinate_edges(panel_freq_mhz)
        vmin, vmax = (
            _robust_color_limits(panel_values) if robust_limits else (None, None)
        )
        norm = (
            BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)
            if discrete_mask_colorbar and BoundaryNorm is not None
            else None
        )
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            panel_values.T,
            shading="auto",
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        if panel_index == len(panels) - 1:
            ax.set_xlabel(r"Relative time, $t_{\mathrm{rel}}\;[\mathrm{s}]$")
        ax.set_ylabel(r"CHIME frequency, $f_{\mathrm{CHIME}}\;[\mathrm{MHz}]$")
        ax.set_title(panel_title)
        ax.set_yticks(panel_freq_mhz)
        ax.set_yticklabels([f"{value:.1f}" for value in panel_freq_mhz])
        _set_spectrogram_time_axis(ax, x_edges)
        _add_frame_index_axis(ax, relative_time_s=time_s, frame_index=frame_index)
        _add_dtv_channel_axis(
            ax,
            frequency_mhz=panel_freq_mhz,
            physical_channel=panel_channels,
        )
        cbar = fig.colorbar(
            image,
            ax=ax,
            ticks=[0, 1] if discrete_mask_colorbar else None,
        )
        cbar.set_label(colorbar_label)

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [figures / f"{basename}.png"]
    for path in outputs:
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    return outputs


def plot_baseband_spectrogram(run_dir: Path) -> list[Path]:
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=cache["baseband_power_db"],
        physical_channel=cache["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), cache),
        frame_index=cache["frame_index"],
        relative_time_s=cache["relative_time_s"],
        title=r"Baseband power spectrogram",
        colorbar_label=r"$10\log_{10}P_{\mathrm{bb}}\;[\mathrm{dB}]$",
        basename="baseband_spectrogram",
        cmap="viridis",
    )


def plot_fstat_level_spectrogram(run_dir: Path) -> list[Path]:
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=detector["fstat_level_db"],
        physical_channel=detector["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), detector),
        frame_index=detector["frame_index"],
        relative_time_s=cache["relative_time_s"],
        title=r"$F$-statistic level spectrogram",
        colorbar_label=r"$R_F=10\log_{10}F\;[\mathrm{dB}]$",
        basename="fstat_level_spectrogram",
        cmap="magma",
        robust_limits=True,
        exclude_outlier_panel=True,
    )


def plot_mask_spectrogram(run_dir: Path) -> list[Path]:
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    title = r"Detector mask spectrogram"
    mask_label = _mask_label_for_run(Path(run_dir))
    if mask_label:
        title += "\n" + mask_label
    if not np.any(cache["mask"] != 0):
        title += "\n" + r"No mask applied; all mask samples are zero."
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=cache["mask"],
        physical_channel=cache["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), cache),
        frame_index=cache["frame_index"],
        relative_time_s=cache["relative_time_s"],
        title=title,
        colorbar_label=r"Mask $M$",
        basename="mask_spectrogram",
        cmap="gray_r",
        discrete_mask_colorbar=True,
    )


def clean_known_figures(run_dir: Path) -> None:
    """Delete all known CHIME figure files from the figures directory."""
    figures = Path(run_dir) / "figures"
    for name in KNOWN_CHIME_FIGURES:
        p = figures / name
        if p.exists():
            p.unlink()


def generate_chime_plots(run_dir: Path) -> list[Path]:
    run = Path(run_dir)
    outputs: list[Path] = []
    # Without detector products there is nothing to plot; return instead of
    # failing while trying to load chime_detector_outputs.npz.
    if not (run / CHIME_DETECTOR_OUTPUTS_FILENAME).exists():
        return outputs
    outputs.extend(plot_snr_shelf_histogram(run))
    outputs.extend(plot_fstat_survival(run))
    outputs.extend(plot_fstat_level_spectrogram(run))
    outputs.extend(plot_baseband_spectrogram(run))
    outputs.extend(plot_baseband_spectrum(run))
    outputs.extend(plot_mask_spectrogram(run))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CHIME real-data PilotProxy figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-figures",
        action="store_true",
        help="Delete known CHIME figures before generating current products.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.clean_figures:
        clean_known_figures(args.run_dir)
    outputs = generate_chime_plots(args.run_dir)
    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
