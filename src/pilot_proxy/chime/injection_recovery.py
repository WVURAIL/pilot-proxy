# coding=utf-8
"""Analyze an injection ladder: recovery linearity and the radiometer baseline.

Consumes the products of ``pilot-proxy chime-scan`` runs over
``inject-pilot-tone`` output trees (one run per amplitude, each run directory
also holding its ladder point's ``injection_manifest.json``) and produces the
two referee-facing results:

* **Recovery linearity** --- mean corrected pilot excess
  ``rho_hat = mean(pilot_excess_corrected[valid])`` per point, with standard
  errors, and the weighted linear fit ``rho_hat = floor + gain * a^2`` against
  injected tone power. Slope-one behaviour in the signal-dominated regime is
  the "recovered tracks injected" claim; the intercept is the channel's
  ambient floor, anchored by the mandatory ``a = 0`` control point (which the
  injection harness guarantees is byte-identical to the source data).
* **Radiometer baseline** --- on identical frames, detection rates for the
  F-statistic (``fstat_raw``) and the classical total-power detector
  (``baseband_power_linear``) at matched false-alarm rates, thresholds taken
  as empirical quantiles of the ``a = 0`` control, with Wilson 95% intervals
  from ``evaluate_snr.wilson_interval``. The horizontal gap between the two
  detection curves is the measured sensitivity advantage.

Everything is post-hoc over stored products; no detector re-runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.chime.injection import INJECTION_MANIFEST_FILENAME
from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
)
from pilot_proxy.testbench.evaluate_snr import wilson_interval

RECOVERY_CSV_FILENAME = "injection_recovery.csv"
RECOVERY_SUMMARY_FILENAME = "injection_recovery_summary.json"
RECOVERY_FIGURE = "injection_recovery_linearity.png"
BASELINE_FIGURE = "detector_vs_radiometer_pd.png"
DEFAULT_FALSE_ALARM_RATES = (1e-2,)
# An empirical quantile at P_fa needs enough H0 frames to be meaningful.
MIN_FRAMES_PER_FALSE_ALARM = 10.0


def _load_point(point_dir: Path) -> dict[str, Any]:
    point_dir = Path(point_dir)
    detector_path = point_dir / CHIME_DETECTOR_OUTPUTS_FILENAME
    cache_path = point_dir / CHIME_SPECTROGRAM_CACHE_FILENAME
    manifest_path = point_dir / INJECTION_MANIFEST_FILENAME
    for path in (detector_path, cache_path, manifest_path):
        if not path.exists():
            raise SystemExit(
                f"ladder point {point_dir} is missing {path.name}; each "
                "--point directory must hold the run products plus its "
                "injection_manifest.json (copy the manifest from the "
                "injected-input directory)"
            )
    detector = np.load(detector_path)
    for key in ("fstat_raw", "pilot_excess_corrected", "valid"):
        if key not in detector:
            raise SystemExit(
                f"{detector_path} lacks {key!r}; the recovery analysis needs "
                "norm-corrected products"
            )
    cache = np.load(cache_path)
    manifest = json.loads(manifest_path.read_text())
    amplitudes = {float(entry["amplitude_lsb"]) for entry in manifest["files"]}
    frequencies = {
        round(float(entry["baseband_frequency_hz"]), 6)
        for entry in manifest["files"]
    }
    if len(amplitudes) != 1 or len(frequencies) != 1:
        raise SystemExit(
            f"{manifest_path} mixes amplitudes {sorted(amplitudes)} or "
            f"frequencies {sorted(frequencies)}; one ladder point must be a "
            "single (amplitude, frequency) setting"
        )
    valid = detector["valid"].astype(bool)
    rho = detector["pilot_excess_corrected"].astype(np.float64)[valid]
    rho = rho[np.isfinite(rho)]
    if rho.size == 0:
        raise SystemExit(f"ladder point {point_dir} has no valid frames")
    power = cache["baseband_power_linear"].astype(np.float64)[valid]
    fstat = detector["fstat_raw"].astype(np.float64)[valid]
    amplitude = amplitudes.pop()
    return {
        "point_dir": str(point_dir),
        "amplitude_lsb": amplitude,
        "injected_power_lsb2": amplitude * amplitude,
        "baseband_frequency_hz": frequencies.pop(),
        "n_valid": int(rho.size),
        "rho_hat": float(rho.mean()),
        "rho_sem": float(rho.std(ddof=1) / np.sqrt(rho.size)) if rho.size > 1
        else float("nan"),
        "fstat": fstat,
        "power": power,
        "total_clip_count": int(
            sum(int(entry["clip_count"]) for entry in manifest["files"])
        ),
    }


def _weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, sem: np.ndarray
) -> dict[str, float]:
    """Weighted least squares y = floor + gain * x with 1/sem^2 weights."""
    weights = 1.0 / np.square(sem)
    design = np.stack([np.ones_like(x), x], axis=1)
    wd = design * weights[:, np.newaxis]
    normal = design.T @ wd
    covariance = np.linalg.inv(normal)
    beta = covariance @ (wd.T @ y)
    return {
        "floor": float(beta[0]),
        "gain_per_lsb2": float(beta[1]),
        "floor_err": float(np.sqrt(covariance[0, 0])),
        "gain_err": float(np.sqrt(covariance[1, 1])),
    }


def analyze_injection_recovery(
    point_dirs: Sequence[Path],
    *,
    false_alarm_rates: Sequence[float] = DEFAULT_FALSE_ALARM_RATES,
) -> dict[str, Any]:
    points = sorted(
        (_load_point(p) for p in point_dirs),
        key=lambda point: point["amplitude_lsb"],
    )
    controls = [p for p in points if p["amplitude_lsb"] == 0.0]
    if len(controls) != 1:
        raise SystemExit(
            "exactly one ladder point must be the a = 0 control (it anchors "
            f"the floor and every threshold); got {len(controls)}"
        )
    control = controls[0]

    # Matched-P_fa thresholds from the control's empirical quantiles.
    thresholds: dict[str, dict[str, float]] = {"fstat": {}, "radiometer": {}}
    usable_pfa: list[float] = []
    for pfa in sorted(set(float(p) for p in false_alarm_rates), reverse=True):
        if control["n_valid"] < MIN_FRAMES_PER_FALSE_ALARM / pfa:
            continue
        usable_pfa.append(pfa)
        quantile = 1.0 - pfa
        thresholds["fstat"][f"{pfa:g}"] = float(
            np.quantile(control["fstat"], quantile)
        )
        thresholds["radiometer"][f"{pfa:g}"] = float(
            np.quantile(control["power"], quantile)
        )
    if not usable_pfa:
        raise SystemExit(
            f"the control point's {control['n_valid']} valid frames cannot "
            "support any requested false-alarm rate (need >= "
            f"{MIN_FRAMES_PER_FALSE_ALARM}/P_fa frames); request a larger "
            "P_fa or scan more control files"
        )

    rows: list[dict[str, Any]] = []
    for point in points:
        row: dict[str, Any] = {
            "amplitude_lsb": point["amplitude_lsb"],
            "injected_power_lsb2": point["injected_power_lsb2"],
            "n_valid": point["n_valid"],
            "rho_hat": point["rho_hat"],
            "rho_sem": point["rho_sem"],
            "total_clip_count": point["total_clip_count"],
        }
        for pfa in usable_pfa:
            key = f"{pfa:g}"
            for name, stat in (("fstat", point["fstat"]),
                               ("radiometer", point["power"])):
                detected = int(np.count_nonzero(stat > thresholds[name][key]))
                lo, hi = wilson_interval(detected, point["n_valid"])
                row[f"pd_{name}_pfa{key}"] = detected / point["n_valid"]
                row[f"pd_{name}_pfa{key}_wilson95_lo"] = lo
                row[f"pd_{name}_pfa{key}_wilson95_hi"] = hi
        rows.append(row)

    x = np.asarray([row["injected_power_lsb2"] for row in rows])
    y = np.asarray([row["rho_hat"] for row in rows])
    sem = np.asarray([row["rho_sem"] for row in rows])
    if not np.all(np.isfinite(sem)) or np.any(sem <= 0):
        raise SystemExit("every ladder point needs >= 2 valid frames for a SEM")
    fit = _weighted_linear_fit(x, y, sem)

    # Slope-one check in log space over signal-dominated points.
    dominated = x * fit["gain_per_lsb2"] > 3.0 * abs(fit["floor"])
    log_slope = None
    if np.count_nonzero(dominated) >= 2:
        lx = np.log10(x[dominated])
        ly = np.log10(y[dominated] - fit["floor"])
        log_slope = float(np.polyfit(lx, ly, 1)[0])

    return {
        "schema_version": "pilot_proxy_injection_recovery_v1",
        "points": [
            {k: v for k, v in row.items()} for row in rows
        ],
        "control_point_dir": control["point_dir"],
        "false_alarm_rates": usable_pfa,
        "thresholds": thresholds,
        "fit": fit,
        "signal_dominated_log_slope": log_slope,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    csv_path = output_dir / RECOVERY_CSV_FILENAME
    fieldnames = list(report["points"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["points"])
    written.append(csv_path)

    json_path = output_dir / RECOVERY_SUMMARY_FILENAME
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append(json_path)

    written.extend(_write_figures(report, output_dir))
    return written


def _write_figures(report: dict[str, Any], output_dir: Path) -> list[Path]:
    from pilot_proxy.chime.plots import _save_figure, _setup_matplotlib

    plt = _setup_matplotlib()
    rows = report["points"]
    fit = report["fit"]
    x = np.asarray([row["injected_power_lsb2"] for row in rows])
    y = np.asarray([row["rho_hat"] for row in rows])
    sem = np.asarray([row["rho_sem"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(x, y, yerr=1.959963984540054 * sem, fmt="o", markersize=4,
                capsize=3, label=r"Recovered $\hat{\rho}$ (95\% CI)")
    grid = np.linspace(0.0, float(x.max()) * 1.05, 200)
    ax.plot(grid, fit["floor"] + fit["gain_per_lsb2"] * grid, "-",
            label=(r"Fit: $\hat{\rho} = \mathrm{floor} + g\,a^2$"))
    ax.axhline(fit["floor"], linestyle=":", color="gray",
               label=r"Ambient floor ($a = 0$)")
    ax.set_xlabel(r"Injected tone power $a^2$ [LSB$^2$]")
    ax.set_ylabel(r"Recovered pilot excess $\hat{\rho} = F/\mu_0 - 1$")
    ax.set_title(r"Injection--recovery linearity on real baseband")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    linearity_path = Path(output_dir) / RECOVERY_FIGURE
    _save_figure(fig, linearity_path)
    plt.close(fig)

    pfa = f"{report['false_alarm_rates'][0]:g}"
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, label in (("fstat", r"F-statistic"),
                        ("radiometer", r"Radiometer (total power)")):
        pd = np.asarray([row[f"pd_{name}_pfa{pfa}"] for row in rows])
        lo = np.asarray([row[f"pd_{name}_pfa{pfa}_wilson95_lo"] for row in rows])
        hi = np.asarray([row[f"pd_{name}_pfa{pfa}_wilson95_hi"] for row in rows])
        ax.errorbar(x, pd, yerr=np.stack([pd - lo, hi - pd]), fmt="o-",
                    markersize=4, capsize=3, label=label)
    ax.set_xlabel(r"Injected tone power $a^2$ [LSB$^2$]")
    ax.set_ylabel(rf"Detection rate at $P_\mathrm{{fa}} = {pfa}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r"Matched-$P_\mathrm{fa}$ detection: F-statistic vs.\ radiometer")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    baseline_path = Path(output_dir) / BASELINE_FIGURE
    _save_figure(fig, baseline_path)
    plt.close(fig)
    return [linearity_path, baseline_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an injection ladder's run products: recovery linearity "
            "(weighted fit of recovered pilot excess vs injected tone power) "
            "and the F-statistic vs radiometer detection comparison at "
            "matched false-alarm rates from the a = 0 control."
        ),
    )
    parser.add_argument("--point", type=Path, action="append", required=True,
                        dest="points",
                        help=(
                            "A ladder-point run directory (repeat per "
                            "amplitude). Must contain the run products and "
                            "that point's injection_manifest.json; exactly "
                            "one point must be the a = 0 control."
                        ))
    parser.add_argument("--false-alarm-rate", type=float, action="append",
                        dest="false_alarm_rates", default=None,
                        help="Matched P_fa (repeatable; default 1e-2).")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_injection_recovery(
        args.points,
        false_alarm_rates=args.false_alarm_rates or DEFAULT_FALSE_ALARM_RATES,
    )
    written = write_outputs(report, args.output_dir)
    fit = report["fit"]
    slope = report["signal_dominated_log_slope"]
    print(
        f"gain {fit['gain_per_lsb2']:.4g} +/- {fit['gain_err']:.2g} per LSB^2, "
        f"floor {fit['floor']:.4g} +/- {fit['floor_err']:.2g}"
        + (f", signal-dominated log-log slope {slope:.3f}" if slope is not None
           else "")
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
