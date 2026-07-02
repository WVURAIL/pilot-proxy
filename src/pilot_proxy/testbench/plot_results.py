#!/usr/bin/env python3
# coding=utf-8
"""Plot DTV shelf-SNR sweep summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    NO_PILOT_EXCESS_FSTAT,
    fstat_raw_to_fstat_level_db,
    pnr_bin_db_to_snr_shelf_db,
    pnr_bin_db_to_fstat_raw_threshold,
    snr_shelf_db_to_pnr_bin_db,
)
from pilot_proxy.plot_style import setup_matplotlib

DEFAULT_INPUT_CSV = Path("generated/dtv_snr_eval/dtv_snr_summary.csv")
DEFAULT_OUTPUT_PNG = Path("generated/dtv_snr_eval/dtv_snr_sweep.png")
DEFAULT_PLOT_DPI = 300  # publication-grade raster; match chime FIGURE_DPI
FIGURE_WIDTH_IN = 8.0
FIGURE_HEIGHT_IN = 5.5
MARKER_SIZE = 4.0
IDENTITY_LINE_WIDTH = 1.4
RESULT_LINE_WIDTH = 1.6
HZ_PER_KHZ = 1_000.0
DEFAULT_SMOOTH_WINDOW = 1
MIN_SMOOTH_WINDOW = 1

REQUESTED_SNR_COLUMN = "requested_snr_shelf_db"
FREQUENCY_OFFSET_COLUMN = "frequency_offset_hz"
CPU_FLOAT_SNR_COLUMN = "cpu_float_estimated_snr_shelf_db_mean"
GPU_SNR_COLUMN = "estimated_snr_shelf_db_mean"
FSTAT_LEVEL_TICKS_DB = np.asarray([0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 20.0])


def _read_summary_rows(path: Path) -> list[dict[str, float]]:
    """Read numeric rows from an evaluator summary CSV."""
    rows: list[dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Summary CSV has no header: {path}")
        required = {
            REQUESTED_SNR_COLUMN,
            FREQUENCY_OFFSET_COLUMN,
            CPU_FLOAT_SNR_COLUMN,
            GPU_SNR_COLUMN,
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Summary CSV is missing required plot columns: "
                + ", ".join(sorted(missing))
            )
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            rows.append(row)
    if not rows:
        raise ValueError(f"Summary CSV has no data rows: {path}")
    return rows


def _finite_pairs(
    rows: list[dict[str, float]],
    *,
    x_column: str,
    y_column: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = float(row.get(x_column, math.nan))
        y_value = float(row.get(y_column, math.nan))
        if math.isfinite(x_value) and math.isfinite(y_value):
            x_values.append(x_value)
            y_values.append(y_value)
    return x_values, y_values


def _frequency_offsets(rows: list[dict[str, float]]) -> list[float]:
    values = sorted({float(row[FREQUENCY_OFFSET_COLUMN]) for row in rows})
    return values


def _rows_for_offset(
    rows: list[dict[str, float]],
    frequency_offset_hz: float,
) -> list[dict[str, float]]:
    return sorted(
        [
            row
            for row in rows
            if float(row[FREQUENCY_OFFSET_COLUMN]) == float(frequency_offset_hz)
        ],
        key=lambda row: float(row[REQUESTED_SNR_COLUMN]),
    )


def _offset_label(prefix: str, frequency_offset_hz: float) -> str:
    if frequency_offset_hz == 0.0:
        return f"{prefix}, 0 Hz"
    if abs(frequency_offset_hz) >= HZ_PER_KHZ:
        return f"{prefix}, {frequency_offset_hz / HZ_PER_KHZ:+.1f} kHz"
    return f"{prefix}, {frequency_offset_hz:+.1f} Hz"


def _centered_moving_average(values: list[float], window: int) -> list[float]:
    if window <= MIN_SMOOTH_WINDOW:
        return list(values)
    smoothed: list[float] = []
    radius = int(window) // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        finite = [value for value in values[start:stop] if math.isfinite(value)]
        if finite:
            smoothed.append(sum(finite) / float(len(finite)))
        else:
            smoothed.append(math.nan)
    return smoothed


def _curve_label(prefix: str, frequency_offset_hz: float, smooth_window: int) -> str:
    label = _offset_label(prefix, frequency_offset_hz)
    if smooth_window > MIN_SMOOTH_WINDOW:
        label += f", {smooth_window}-pt smooth"
    return label


def _snr_shelf_db_to_fstat_level_db(values) -> np.ndarray:
    """Map shelf-SNR display coordinates to F-statistic level coordinates."""
    snr = np.asarray(values, dtype=np.float64)
    pnr = snr_shelf_db_to_pnr_bin_db(snr)
    raw = pnr_bin_db_to_fstat_raw_threshold(pnr)
    return np.asarray(fstat_raw_to_fstat_level_db(raw), dtype=np.float64)


def _fstat_level_db_to_snr_shelf_db(values) -> np.ndarray:
    """Map F-statistic level display coordinates back to shelf-SNR coordinates."""
    fstat_level = np.asarray(values, dtype=np.float64)
    raw = DB_LINEAR_BASE ** (fstat_level / DB_POWER_FACTOR)
    excess = np.maximum(raw - NO_PILOT_EXCESS_FSTAT, np.finfo(np.float64).tiny)
    pnr = DB_POWER_FACTOR * np.log10(excess)
    return np.asarray(pnr_bin_db_to_snr_shelf_db(pnr), dtype=np.float64)


def _format_fstat_tick(value: float) -> str:
    if abs(float(value)) >= 1.0 or float(value) == 0.0:
        return f"{value:g}"
    return f"{value:.3g}"


def _add_fstat_level_axis(axis, *, orientation: str):
    """Add an F-statistic dB secondary axis matching a shelf-SNR axis."""
    if orientation == "x":
        secondary = axis.secondary_xaxis(
            "top",
            functions=(
                _snr_shelf_db_to_fstat_level_db,
                _fstat_level_db_to_snr_shelf_db,
            ),
        )
        xmin, xmax = axis.get_xlim()
        secondary.set_xlabel(r"Known pilot strength, $10\log_{10}F\;[\mathrm{dB}]$")
        set_ticks = secondary.set_xticks
        set_ticklabels = secondary.set_xticklabels
    elif orientation == "y":
        secondary = axis.secondary_yaxis(
            "right",
            functions=(
                _snr_shelf_db_to_fstat_level_db,
                _fstat_level_db_to_snr_shelf_db,
            ),
        )
        ymin, ymax = axis.get_ylim()
        xmin, xmax = ymin, ymax
        secondary.set_ylabel(r"Measured pilot statistic, $10\log_{10}F\;[\mathrm{dB}]$")
        set_ticks = secondary.set_yticks
        set_ticklabels = secondary.set_yticklabels
    else:
        raise ValueError(f"unknown secondary-axis orientation: {orientation!r}")

    tick_positions = _fstat_level_db_to_snr_shelf_db(FSTAT_LEVEL_TICKS_DB)
    keep = (
        np.isfinite(tick_positions)
        & (tick_positions >= xmin)
        & (tick_positions <= xmax)
    )
    if np.any(keep):
        ticks = FSTAT_LEVEL_TICKS_DB[keep]
        set_ticks(ticks)
        set_ticklabels([_format_fstat_tick(value) for value in ticks])
    return secondary


def plot_summary(
    *,
    input_csv: Path,
    output_png: Path,
    title: str,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    show: bool = False,
) -> Path:
    """Render an SNR sweep comparison plot."""
    if int(smooth_window) < MIN_SMOOTH_WINDOW:
        raise ValueError(
            f"smooth_window must be >= {MIN_SMOOTH_WINDOW}; got {smooth_window}."
        )
    smooth_window = int(smooth_window)

    try:
        plt = setup_matplotlib(force_agg=not bool(show))
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install the optional plot "
            "dependency, for example: python -m pip install matplotlib"
        ) from exc

    rows = _read_summary_rows(input_csv)
    requested_values = [
        float(row[REQUESTED_SNR_COLUMN])
        for row in rows
        if math.isfinite(float(row[REQUESTED_SNR_COLUMN]))
    ]
    if not requested_values:
        raise ValueError("No finite requested SNR values found in summary CSV.")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    x_min = min(requested_values)
    x_max = max(requested_values)
    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        linestyle="--",
        color="black",
        linewidth=IDENTITY_LINE_WIDTH,
        label="Ideal input = output",
    )

    for offset in _frequency_offsets(rows):
        group = _rows_for_offset(rows, offset)
        x_cpu, y_cpu = _finite_pairs(
            group,
            x_column=REQUESTED_SNR_COLUMN,
            y_column=CPU_FLOAT_SNR_COLUMN,
        )
        x_gpu, y_gpu = _finite_pairs(
            group,
            x_column=REQUESTED_SNR_COLUMN,
            y_column=GPU_SNR_COLUMN,
        )
        if x_cpu:
            ax.plot(
                x_cpu,
                _centered_moving_average(y_cpu, smooth_window),
                linestyle=":",
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=RESULT_LINE_WIDTH,
                label=_curve_label("CPU float", offset, smooth_window),
            )
        if x_gpu:
            ax.plot(
                x_gpu,
                _centered_moving_average(y_gpu, smooth_window),
                linestyle="-",
                marker="s",
                markersize=MARKER_SIZE,
                linewidth=RESULT_LINE_WIDTH,
                label=_curve_label("GPU fixed-point", offset, smooth_window),
            )

    ax.set_title(title)
    ax.set_xlabel(
        r"Known DTV pilot shelf strength, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
    )
    ax.set_ylabel(
        r"Measured DTV pilot shelf strength, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    _add_fstat_level_axis(ax, orientation="x")
    _add_fstat_level_axis(ax, orientation="y")
    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=DEFAULT_PLOT_DPI)
    if show:
        plt.show()
    plt.close(fig)
    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluator summary CSV with ideal, CPU float, and GPU "
            "fixed-point SNR curves."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument(
        "--title",
        default="PilotProxy SNR sweep",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=(
            "Centered moving-average window for plotted CPU/GPU curves. "
            "The CSV/JSON data are not modified."
        ),
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = plot_summary(
        input_csv=args.input_csv,
        output_png=args.output_png,
        title=str(args.title),
        smooth_window=int(args.smooth_window),
        show=bool(args.show),
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
