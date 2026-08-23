#!/usr/bin/env python3
"""Generate the optional scientific tables for the dissertation export.

This tool assembles the CSV inputs that ``pilot_proxy.dissertation_exports``
accepts through its ``--census-psd``, ``--worked-example-spectra``, and
``--bao-time-vs-masking`` options. It reads per-pilot survey products
directly and, for the forecast table, uses the released ``baonoise``
package (bao-noise-tolerance) --- which is why this lives in ``tools/``
rather than inside :mod:`pilot_proxy`: the package itself stays free of the
cross-repository dependency, and the export module records whatever inputs
this tool produced.

``worked_example_spectra.csv`` is generated only behind ``--worked-example``:
the two archived frames it contains are named explicitly (UTC day and
F/mu0 ratio, WORKED_EXAMPLE_PANELS below) and no row is written unless the
digits the dissertation quotes --- T[60..64] of the exemplar frame and the
weakest frame's designated-window maximum --- reproduce from the product.
The self-test is the provenance.

Tables NOT generated here, deliberately:

* ``bao_convergence.csv`` and ``bao_two_walls.csv`` --- these depend on the
  ``_Pres`` bias-response bank and the dissertation draft's fine-credit and
  floor-basis conventions; they remain bridges until that calculation is
  reproduced under its own conventions.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
from pathlib import Path

import numpy as np

from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO, ARCHIVED_FINE_POWER_RATIO)
from pilot_proxy.archive_health import (
    evaluate_frame_health,
    health_correct_integrated_spectra,
)

SPECTRUM_KEY = "integrated_spectrum_before_mask"
NFFT = 16384
WINDOW_KHZ = 15.0

# Worked example (pipeline chapter): the two archived channel-36 frames,
# named by UTC day and F/mu0 ratio and verified digit-for-digit against
# what the dissertation quotes before any row is emitted.
WORKED_EXAMPLE_CHANNEL = 36
WORKED_EXAMPLE_PANELS = (
    # (panel, utc day, F/mu0 target)
    ("a", "2025-07-31", 1.258),   # exemplar masked frame (15:52:22 UT)
    ("b", "2025-05-16", 0.897),   # cohort's weakest valid frame
)
WORKED_RATIO_TOL = 0.0006
WORKED_A_T60_64 = (1.066, 8.730, 18.615, 7.598, 1.083)   # published, tol 0.002
WORKED_B_T62 = 2.59                                      # published, tol 0.01


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
        with np.load(path, allow_pickle=False) as z:
            if SPECTRUM_KEY not in z:
                continue
            health = evaluate_frame_health(z)
            corrected = health_correct_integrated_spectra(z, health)
            if not corrected.exact:
                raise SystemExit(
                    f"{Path(path).name}: exact health correction is unavailable: "
                    f"{corrected.unavailable_reason}"
                )
            ch = int(z["physical_channel"][0])
            spec = np.asarray(corrected.before, dtype=float).reshape(-1)
            pilot = float(z["pilot_frequency_hz"][0])
            center = float(z["chime_frequency_hz"][0])
            fs = 390625.0
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


def census_centre_rows(products_dir: Path) -> list[dict]:
    """Per channel, where the coarse channel's own centre bin falls.

    The bin at DC in the receiver frame -- the physical centre of the CHIME
    coarse channel -- carries a line in every one of the twenty-three surveyed
    channels, including channels whose nearest emitter is hundreds of kilohertz
    away, and the reference-placement contract designates it a forbidden tone.
    It is instrumental, not a member of the transmitter population, and the
    census plate's +/-15 kHz window is narrow enough that on some channels it
    lands inside the panel where it reads as a companion carrier.

    This table states, for every channel, the RF offset of that bin from the
    synthesized pilot and its level before and after the health repair, so
    the identification is data-backed rather than remembered.
    ``inside_census_window`` marks the channels where the two share a panel.
    """
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(products_dir, "*.npz"))):
        with np.load(path, allow_pickle=False) as z:
            if SPECTRUM_KEY not in z:
                continue
            health = evaluate_frame_health(z)
            corrected = health_correct_integrated_spectra(z, health)
            if not corrected.exact:
                raise SystemExit(
                    f"{Path(path).name}: exact health correction is "
                    f"unavailable: {corrected.unavailable_reason}")
            ch = int(z["physical_channel"][0])
            spec = np.asarray(corrected.before, dtype=float).reshape(-1)
            raw = np.asarray(z[SPECTRUM_KEY], dtype=float).reshape(-1)
            pilot = float(z["pilot_frequency_hz"][0])
            centre = float(z["chime_frequency_hz"][0])
        if spec.size != NFFT or not np.any(spec > 0):
            continue
        median = float(np.median(spec[spec > 0]))
        raw_median = float(np.median(raw[raw > 0]))
        # Bin 0 of the stored array is the coarse-channel centre in either
        # spectral sense. The pilot sits at -(pilot - centre) in the receiver
        # frame, so in the RF sense the centre stands that far from the pilot.
        offset_hz = -(pilot - centre)
        rows.append({
            "channel": ch,
            "offset_from_pilot_khz": f"{offset_hz / 1000.0:.6g}",
            "db_rel_median": f"{10.0 * np.log10(spec[0] / median):.6g}",
            "db_rel_median_uncorrected":
                f"{10.0 * np.log10(raw[0] / raw_median):.6g}",
            "inside_census_window":
                "1" if abs(offset_hz) <= WINDOW_KHZ * 1000.0 else "0",
        })
    if not rows:
        raise SystemExit("no products with integrated spectra found")
    if not any(r["inside_census_window"] == "1" for r in rows):
        raise SystemExit("no channel places its centre bin inside the census "
                         "window; the caption's claim needs rechecking")
    return rows


def worked_example_rows(products_dir: Path) -> list[dict]:
    """(panel, fine_bin, T) for the worked example's two archived frames.

    Frames are located by UTC day and F/mu0 ratio (WORKED_EXAMPLE_PANELS)
    and accepted only when the published digits reproduce: T[60..64] of the
    exemplar frame, the designated-window maximum of the weakest frame.
    Exactly one frame per panel must verify; anything else aborts rather
    than guessing.
    """
    path = None
    for candidate in sorted(glob.glob(os.path.join(products_dir, "*.npz"))):
        with np.load(candidate, allow_pickle=False) as archive:
            if int(archive["physical_channel"][0]) == WORKED_EXAMPLE_CHANNEL:
                path = candidate
                break
    if path is None:
        raise SystemExit(f"no product for channel {WORKED_EXAMPLE_CHANNEL} "
                         f"under {products_dir}")
    with np.load(path, allow_pickle=False) as archive:
        fstat = np.asarray(archive[ARCHIVED_COARSE_POWER_RATIO],
                           dtype=float).reshape(-1)
        mu0 = float(np.ravel(archive["mu0"])[0])
        fine = np.asarray(archive[ARCHIVED_FINE_POWER_RATIO], dtype=float)
        unit_index = np.asarray(archive["frame_unit_index"],
                                dtype=int).reshape(-1)
        unit_t0 = np.asarray(archive["unit_time0_ctime"],
                             dtype=float).reshape(-1)
        health = evaluate_frame_health(archive)
    frame_t = unit_t0[unit_index]
    days = np.array([datetime.datetime.fromtimestamp(
        x, datetime.timezone.utc).strftime("%Y-%m-%d") for x in frame_t])
    ratio = fstat / mu0

    rows: list[dict] = []
    for panel, day, target in WORKED_EXAMPLE_PANELS:
        verified = []
        for i in np.where(health.include & (days == day)
                          & (np.abs(ratio - target) < WORKED_RATIO_TOL))[0]:
            T = fine[i]
            if panel == "a":
                ok = all(abs(float(T[60 + j]) - WORKED_A_T60_64[j]) < 0.002
                         for j in range(5))
            else:
                ok = abs(float(T[62]) - WORKED_B_T62) < 0.01
            if ok:
                verified.append(int(i))
        if len(verified) != 1:
            raise SystemExit(
                f"worked-example panel {panel}: {len(verified)} frames on "
                f"{day} reproduce the published digits (need exactly 1); "
                "refusing to guess")
        T = fine[verified[0]]
        rows += [{"panel": panel, "fine_bin": b, "T": f"{float(T[b]):.6g}"}
                 for b in range(256)]
        print(f"worked-example panel {panel}: frame {verified[0]} "
              f"({day}, F/mu0={ratio[verified[0]]:.4f}) verified")
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
    parser.add_argument("--worked-example", action="store_true",
                        help="also generate worked_example_spectra.csv "
                             "(verifies the published digits first)")
    args = parser.parse_args()

    count = _write(args.out / "census_psd.csv",
                   ("channel", "offset_khz", "db_rel_median"),
                   census_psd_rows(args.products))
    print(f"census_psd.csv: {count} rows")

    count = _write(args.out / "census_centre.csv",
                   ("channel", "offset_from_pilot_khz", "db_rel_median",
                    "db_rel_median_uncorrected", "inside_census_window"),
                   census_centre_rows(args.products))
    print(f"census_centre.csv: {count} rows")

    if args.worked_example:
        count = _write(args.out / "worked_example_spectra.csv",
                       ("panel", "fine_bin", "T"),
                       worked_example_rows(args.products))
        print(f"worked_example_spectra.csv: {count} rows")

    if not args.skip_forecast:
        count = _write(args.out / "bao_time_vs_masking.csv",
                       ("series", "masked_fraction", "time_year"),
                       bao_time_vs_masking_rows())
        print(f"bao_time_vs_masking.csv: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
