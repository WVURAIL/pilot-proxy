# coding=utf-8
"""Post-hoc cleaning-tradeoff analysis over stored detector products.

The product schema keeps the raw ``p_target_u64``/``p_ref_sum_u64`` verbatim
and records the per-pilot weight norms precisely so alternative thresholds are
a recompute, never a re-run. This module sweeps a mask threshold
``tau = mu0 * 10^(x/10)`` over a run (or combined survey) directory and
reports, per channel and per ``x``:

* masked / kept fraction over valid frames,
* mean cleaned baseband power (mask applied) and, when a pilot-free control
  run is given, the residual above the control floor in dB,
* the recovered-bandwidth headline at the shipped operating point ``x = 0``.

The ``x = 0`` point must reproduce the stored mask exactly --- the stored mask
was computed with exact integer cross-multiplication, so this anchor proves
the float recompute path against the deployed one before any other threshold
is trusted. A mismatch aborts the analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)

# CHIME coarse channel width: 400 MHz over 1024 channels. Matches the
# instrument geometry datatrawl loads from chime.yaml; kept as a constant so
# this analysis stays importable without datatrawl.
CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ = 400.0 / 1024.0

TRADEOFF_CSV_FILENAME = "cleaning_tradeoff.csv"
TRADEOFF_SUMMARY_FILENAME = "cleaning_tradeoff_summary.json"
OPERATING_CURVE_FIGURE = "cleaning_tradeoff_operating_curve.png"
RECOVERED_BANDWIDTH_FIGURE = "recovered_bandwidth_vs_threshold.png"


def _load_run(run_dir: Path) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    detector_path = run_dir / CHIME_DETECTOR_OUTPUTS_FILENAME
    cache_path = run_dir / CHIME_SPECTROGRAM_CACHE_FILENAME
    for path in (detector_path, cache_path):
        if not path.exists():
            raise SystemExit(f"missing product: {path}")
    detector = np.load(detector_path)
    cache = np.load(cache_path)
    required = ("p_target_u64", "p_ref_sum_u64", "valid", "mask",
                "target_norm_sq", "ref_norm_sum_sq", "mu0", "physical_channel")
    missing = [key for key in required if key not in detector]
    if missing:
        raise SystemExit(
            f"{detector_path} lacks {missing}; the tradeoff sweep needs "
            "norm-corrected products (legacy products predate the recorded "
            "norms and cannot anchor the recompute)"
        )
    if cache["baseband_power_linear"].shape != detector["valid"].shape:
        raise SystemExit(
            "detector outputs and spectrogram cache disagree on (frames, "
            "pilots) shape; run validate-products on this directory"
        )
    return {
        "p_target": detector["p_target_u64"].astype(np.float64),
        "p_ref": detector["p_ref_sum_u64"].astype(np.float64),
        "valid": detector["valid"].astype(bool),
        "stored_mask": detector["mask"].astype(bool),
        "mu0": detector["mu0"].astype(np.float64),
        "physical_channel": detector["physical_channel"].astype(int),
        "power": cache["baseband_power_linear"].astype(np.float64),
    }


def _fstat(run: dict[str, np.ndarray]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * run["p_target"] / run["p_ref"]


def control_floor_db(control_run_dir: Path) -> float:
    """Mean valid-frame baseband power of a pilot-free control run, in dB."""
    run = _load_run(Path(control_run_dir))
    values = run["power"][run["valid"]]
    if values.size == 0:
        raise SystemExit("control run has no valid frames")
    return float(10.0 * np.log10(values.mean()))


def sweep_cleaning_tradeoff(
    run_dir: Path,
    *,
    excess_db_grid: Sequence[float],
    control_run_dir: Path | None = None,
    survey_hours: float | None = None,
) -> dict[str, Any]:
    """Threshold sweep; returns rows plus the operating-point summary."""
    run = _load_run(Path(run_dir))
    fstat = _fstat(run)
    grid = np.unique(np.round(np.asarray(list(excess_db_grid), dtype=np.float64), 6))
    if 0.0 not in grid:
        grid = np.unique(np.concatenate([[0.0], grid]))

    # Exact anchor: at x = 0 the float recompute must reproduce the stored
    # (exact-integer) mask on every valid frame.
    anchor = run["valid"] & (fstat > run["mu0"][np.newaxis, :])
    if not np.array_equal(anchor, run["stored_mask"] & run["valid"]):
        mismatches = int(np.count_nonzero(anchor != (run["stored_mask"] & run["valid"])))
        raise SystemExit(
            f"x=0 recompute disagrees with the stored mask on {mismatches} "
            "frame(s); the float path cannot anchor this product (validate-"
            "products it, and check for boundary-tied frames)"
        )

    floor_db = (
        control_floor_db(control_run_dir) if control_run_dir is not None else None
    )
    channels = run["physical_channel"]
    rows: list[dict[str, Any]] = []
    recovered_mhz_by_x: dict[float, float] = {}
    for x in grid:
        factor = 10.0 ** (float(x) / 10.0)
        mask_x = run["valid"] & (fstat > run["mu0"][np.newaxis, :] * factor)
        recovered = 0.0
        for pilot_index, channel in enumerate(channels):
            valid = run["valid"][:, pilot_index]
            n_valid = int(np.count_nonzero(valid))
            masked = mask_x[:, pilot_index]
            kept = valid & ~masked
            masked_fraction = (
                float(np.count_nonzero(masked)) / n_valid if n_valid else float("nan")
            )
            kept_fraction = 1.0 - masked_fraction if n_valid else float("nan")
            power = run["power"][:, pilot_index]
            input_db = (
                float(10.0 * np.log10(power[valid].mean())) if n_valid else float("nan")
            )
            cleaned_db = (
                float(10.0 * np.log10(power[kept].mean()))
                if np.count_nonzero(kept)
                else float("nan")
            )
            row: dict[str, Any] = {
                "physical_channel": int(channel),
                "excess_db": float(x),
                "n_valid": n_valid,
                "masked_fraction": masked_fraction,
                "kept_fraction": kept_fraction,
                "input_power_db": input_db,
                "cleaned_power_db": cleaned_db,
            }
            if floor_db is not None:
                row["residual_db"] = (
                    cleaned_db - floor_db if np.isfinite(cleaned_db) else float("nan")
                )
            rows.append(row)
            if np.isfinite(kept_fraction):
                recovered += kept_fraction * CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ
        recovered_mhz_by_x[float(x)] = recovered

    operating = {
        "excess_db": 0.0,
        "recovered_mhz": recovered_mhz_by_x[0.0],
        "total_affected_mhz": float(len(channels)) * CHIME_COARSE_CHANNEL_BANDWIDTH_MHZ,
    }
    if survey_hours is not None:
        operating["survey_hours"] = float(survey_hours)
        operating["recovered_mhz_hours"] = recovered_mhz_by_x[0.0] * float(survey_hours)
    return {
        "schema_version": "pilot_proxy_cleaning_tradeoff_v1",
        "run_dir": str(run_dir),
        "control_run_dir": str(control_run_dir) if control_run_dir else None,
        "control_floor_db": floor_db,
        "excess_db_grid": [float(x) for x in grid],
        "rows": rows,
        "recovered_mhz_by_excess_db": recovered_mhz_by_x,
        "operating_point": operating,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    csv_path = output_dir / TRADEOFF_CSV_FILENAME
    fieldnames = list(report["rows"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])
    written.append(csv_path)

    json_path = output_dir / TRADEOFF_SUMMARY_FILENAME
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append(json_path)

    written.extend(_write_figures(report, output_dir))
    return written


def _write_figures(report: dict[str, Any], output_dir: Path) -> list[Path]:
    from pilot_proxy.chime.plots import FIGURE_DPI, _save_figure, _setup_matplotlib

    plt = _setup_matplotlib()
    del FIGURE_DPI  # dpi applied inside _save_figure
    rows = report["rows"]
    channels = sorted({row["physical_channel"] for row in rows})
    has_control = report["control_floor_db"] is not None
    y_key = "residual_db" if has_control else "cleaned_power_db"

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for channel in channels:
        points = sorted(
            (row for row in rows if row["physical_channel"] == channel),
            key=lambda row: row["excess_db"],
        )
        ax.plot(
            [row["masked_fraction"] for row in points],
            [row[y_key] for row in points],
            marker="o", markersize=3,
            label=rf"DTV {channel}",
        )
        operating = next(row for row in points if row["excess_db"] == 0.0)
        ax.plot(operating["masked_fraction"], operating[y_key],
                marker="*", markersize=12, color="black", zorder=5)
    ax.set_xlabel(r"Masked fraction of valid frames")
    ax.set_ylabel(
        r"Residual above control floor [dB]" if has_control
        else r"Cleaned baseband power [dB, arb.]"
    )
    ax.set_title(
        r"Cleaning tradeoff (star: shipped operating point $\tau = \mu_0$)"
    )
    if len(channels) <= 12:
        ax.legend(fontsize="x-small", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = Path(output_dir) / OPERATING_CURVE_FIGURE
    _save_figure(fig, curve_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    grid = report["excess_db_grid"]
    recovered = [report["recovered_mhz_by_excess_db"][float(x)] for x in grid]
    ax.plot(grid, recovered, marker="o", markersize=3)
    ax.axhline(report["operating_point"]["total_affected_mhz"],
               linestyle="--", color="gray",
               label=r"All affected channels kept")
    ax.axvline(0.0, linestyle=":", color="black",
               label=r"Operating point $\tau = \mu_0$")
    ax.set_xlabel(r"Mask threshold above $\mu_0$ [dB]")
    ax.set_ylabel(r"Recovered bandwidth [MHz]")
    ax.set_title(r"Recovered bandwidth vs.\ mask threshold")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    bandwidth_path = Path(output_dir) / RECOVERED_BANDWIDTH_FIGURE
    _save_figure(fig, bandwidth_path)
    plt.close(fig)
    return [curve_path, bandwidth_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the positive-excess mask threshold over stored products "
            "(exact x=0 anchor against the shipped mask) and write the "
            "masked-fraction/residual operating curve plus the recovered-"
            "bandwidth headline."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A chime run or combined-survey directory.")
    parser.add_argument("--control-run-dir", type=Path, default=None,
                        help="Pilot-free control run; enables residual_db.")
    parser.add_argument("--excess-db-start", type=float, default=0.0)
    parser.add_argument("--excess-db-stop", type=float, default=12.0)
    parser.add_argument("--excess-db-step", type=float, default=0.5)
    parser.add_argument("--survey-hours", type=float, default=None,
                        help="Scales the headline to recovered MHz-hours.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <run-dir>/cleaning_tradeoff")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.excess_db_step <= 0:
        raise SystemExit("--excess-db-step must be positive")
    grid = np.arange(
        args.excess_db_start,
        args.excess_db_stop + args.excess_db_step / 2.0,
        args.excess_db_step,
    )
    report = sweep_cleaning_tradeoff(
        args.run_dir,
        excess_db_grid=grid,
        control_run_dir=args.control_run_dir,
        survey_hours=args.survey_hours,
    )
    output_dir = args.output_dir or (Path(args.run_dir) / "cleaning_tradeoff")
    written = write_outputs(report, output_dir)
    operating = report["operating_point"]
    headline = (
        f"{operating['recovered_mhz']:.3f} of "
        f"{operating['total_affected_mhz']:.3f} MHz recovered at tau = mu0"
    )
    if "recovered_mhz_hours" in operating:
        headline += f" ({operating['recovered_mhz_hours']:.1f} MHz-hours)"
    print(headline)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
