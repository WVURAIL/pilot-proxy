#!/usr/bin/env python3
"""Characterise the line at every channel's centre.

The channel-centre bin carries a strong line in all 23 channels, including
channels with no transmitter anywhere near it.  This script separates the two
contributions:

* the exactly repairable part -- frames whose every decoded complex int4
  sample is (-8, -8), which have a reconstructible DC-only spectrum and are
  subtracted by ``archive_health.health_correct_integrated_spectra``;
* whatever survives that repair.

It also reports which channels have their pilot close enough to the centre
for the two features to fall inside the same +/-15 kHz census window.
"""
from __future__ import annotations

import _calibration_paths as P  # noqa: F401

import numpy as np  # noqa: E402

from ppcal.products import SAMPLE_RATE_HZ, load_all  # noqa: E402

D = str(P.PER_PILOT)


def dc_db(rf, s):
    pos = s[s > 0]
    j = int(np.argmin(np.abs(rf)))
    return 10.0 * np.log10(s[j] / np.median(pos))


def main():
    print("%3s %10s %10s %9s %9s %10s  %s"
          % ("ch", "raw_dB", "repaired_dB", "removed", "ceil_frm",
             "pilot_kHz", "note"))
    rows = []
    for c in sorted(load_all(D), key=lambda c: c.ch):
        z = c._z
        f_stored = np.fft.fftfreq(c.nfft, d=1.0 / SAMPLE_RATE_HZ)
        rf_raw = c.sense * f_stored
        o = np.argsort(rf_raw)
        raw = dc_db(rf_raw[o],
                    np.asarray(z["integrated_spectrum_before_mask"])[o])
        rf, s = c.spectrum
        rep = dc_db(rf, s)
        ceil = int(c.health_reasons.get(
            "baseband_power_at_negative_full_scale_ceiling", 0))
        poff = c.pilot_offset_hz / 1e3
        note = ("DC inside the +/-15 kHz census window about the pilot"
                if abs(poff) <= 15.0 else "")
        print("%3d %10.1f %10.1f %9.1f %9d %10.2f  %s"
              % (c.ch, raw, rep, raw - rep, ceil, poff, note))
        rows.append((c.ch, raw, rep, ceil, poff))

    arr = np.array([[r[1], r[2]] for r in rows])
    print("\nraw      : median %.1f dB, range %.1f..%.1f over all %d channels"
          % (np.median(arr[:, 0]), arr[:, 0].min(), arr[:, 0].max(), len(rows)))
    print("repaired : median %.1f dB, range %.1f..%.1f"
          % (np.median(arr[:, 1]), arr[:, 1].min(), arr[:, 1].max()))
    near = [r for r in rows if abs(r[4]) <= 15.0]
    print("channels whose pilot is within 15 kHz of the centre:",
          ", ".join("ch%d (%+.2f kHz)" % (r[0], r[4]) for r in near))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
