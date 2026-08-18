#!/usr/bin/env python3
"""Generate the optional scientific tables for the dissertation export.

This tool assembles the CSV inputs that ``pilot_proxy.dissertation_exports``
accepts through its ``--census-psd`` and ``--bao-time-vs-masking`` options.
It reads per-pilot survey products directly and, for the forecast table,
uses the released ``baonoise`` package (bao-noise-tolerance) --- which is why
this lives in ``tools/`` rather than inside :mod:`pilot_proxy`: the package
itself stays free of the cross-repository dependency, and the export module
records whatever inputs this tool produced.

Tables NOT generated here, deliberately:

* ``worked_example_spectra.csv`` --- the worked-example figure annotates two
  specific archived frames whose identities are not recorded in this tool's
  inputs; regenerating it requires naming those frames explicitly rather
  than guessing.
* ``bao_convergence.csv`` and ``bao_two_walls.csv`` --- these depend on the
  ``_Pres`` bias-response bank and the dissertation draft's fine-credit and
  floor-basis conventions; they remain bridges until that calculation is
  reproduced under its own conventions.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

SPECTRUM_KEY = "integrated_spectrum_before_mask"
NFFT = 16384
WINDOW_KHZ = 15.0


def _write(path: Path, columns, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def census_psd_rows(products_dir: Path) -> list[dict]:
    """(channel, offset_khz, db_rel_median) around each synthesized pilot.

    The spectrum is the product's integrated before-mask spectrum in natural
    FFT order (DC at the coarse-channel center, receiver spectral frame).
    Offsets are relative to the *synthesized* pilot position and are reported
    in the RF (transmitted-frequency) sense --- the receiver's spectral frame
    is inverted relative to RF, and the inversion is undone here --- so
    off-nominal carriers appear at their measured RF displacement rather than
    being re-centered.  Values are dB relative to the channel's median
    spectral power.
    """
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(products_dir, "*.npz"))):
        z = np.load(path)
        if SPECTRUM_KEY not in z:
            z.close()
            continue
        ch = int(z["physical_channel"][0])
        spec = np.asarray(z[SPECTRUM_KEY], dtype=float).reshape(-1)
        pilot = float(z["pilot_frequency_hz"][0])
        center = float(z["chime_frequency_hz"][0])
        fs = 390625.0
        z.close()
        if spec.size != NFFT or not np.any(spec > 0):
            continue
        bin_hz = fs / NFFT
        # receiver frame: spectral sense inverted relative to RF
        offset_hz = -(pilot - center)
        pilot_bin = int(round(offset_hz / bin_hz)) % NFFT
        median = float(np.median(spec[spec > 0]))
        half = int(round(WINDOW_KHZ * 1000.0 / bin_hz))
        for k in range(-half, half + 1):
            # receiver-frame bin at +k sits at RF offset -k relative to the
            # synthesized pilot; report the RF sense
            value = spec[(pilot_bin - k) % NFFT]
            if value <= 0:
                continue
            rows.append({
                "channel": ch,
                "offset_khz": f"{k * bin_hz / 1000.0:.6g}",
                "db_rel_median": f"{10.0 * np.log10(value / median):.6g}",
            })
    if not rows:
        raise SystemExit("no products with integrated spectra found")
    return rows


def bao_time_vs_masking_rows() -> list[dict]:
    """(series, masked_fraction, time_year) from the released forecast bank.

    series: ``survey_amplitude`` (survey-level BAO amplitude S/N = 5),
    ``bin_amplitude`` (the same target in the z = 1.40--1.50 bin), and
    ``dilation`` (sigma(alpha_perp) = 2% in that bin).  Uniform masking of
    the DTV band, retained (no excision), masking cost only --- the same
    convention as the dissertation's cost-side figure.  Times are on-sky
    years at 8,760 h/yr.
    """
    from baonoise import api, scenarios

    fc = api.load()
    onsky_hours = 8760.0
    worst_bin = 6

    def req_hours(metric, target, increasing):
        lo, hi = 1.0, 1.0e6
        flo = metric(lo)
        for _ in range(200):
            mid = (lo * hi) ** 0.5
            value = metric(mid)
            reached = value >= target if increasing else value <= target
            if reached:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.0005:
                break
        return hi

    fracs = [round(0.05 * i, 2) for i in range(20)] + [0.97]
    rows: list[dict] = []
    for frac in fracs:
        sc = (scenarios.clean() if frac == 0.0
              else scenarios.uniform(frac, scenarios.DTV_BAND))
        h_survey = req_hours(lambda t: fc.significance(sc, t), 5.0, True)
        h_bin = req_hours(
            lambda t: fc.significance(sc, t, bins=[worst_bin]), 5.0, True)
        h_dil = req_hours(
            lambda t: fc.sigma_param_bin(sc, t, worst_bin, "aperp0"),
            0.02, False)
        for series, hours in (("survey_amplitude", h_survey),
                              ("bin_amplitude", h_bin),
                              ("dilation", h_dil)):
            rows.append({
                "series": series,
                "masked_fraction": f"{frac:g}",
                "time_year": f"{hours / onsky_hours:.6g}",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, required=True,
                        help="directory of per-pilot survey products (*.npz)")
    parser.add_argument("--out", type=Path,
                        default=Path("exports/dissertation/inputs"),
                        help="directory for the generated CSV tables")
    parser.add_argument("--skip-forecast", action="store_true",
                        help="skip the baonoise-dependent forecast table")
    args = parser.parse_args()

    count = _write(args.out / "census_psd.csv",
                   ("channel", "offset_khz", "db_rel_median"),
                   census_psd_rows(args.products))
    print(f"census_psd.csv: {count} rows")

    if not args.skip_forecast:
        count = _write(args.out / "bao_time_vs_masking.csv",
                       ("series", "masked_fraction", "time_year"),
                       bao_time_vs_masking_rows())
        print(f"bao_time_vs_masking.csv: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
