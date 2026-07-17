#!/usr/bin/env python3
"""Trim the pilot tone of a generated ATSC capture to a target direct ratio.

The gr-dtv modulator's pilot amplitude is not a tunable flag, and the golden
capture's pilot sits 0.72 dB below the nominal 11.3 dB (by direct
integration; see audit v2). This utility coherently rescales the pilot line
in an existing capture: it estimates the pilot's complex amplitude by
coherent projection, then adds (g-1) * A * exp(+2j*pi*f*t) so the pilot
power lands exactly at --target-below-data-db relative to the directly
integrated data power. Everything else in the capture is untouched.

Verify afterwards with the extended audit:

    pilot-proxy audit-atsc --input-iq <output> --output-json audit_v3.json
    # expect measured_pilot_below_data_direct_db = target +/- 0.05

Usage:
    python scripts/trim_pilot_amplitude.py \
        --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
        --output-iq generated/atsc/atsc_8vsb_pilot_trimmed.cfile \
        [--target-below-data-db 11.3] [--sample-rate-hz 10762237.762237763] \
        [--expected-pilot-hz -2690559.0] [--channel-width-hz 6e6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DB = 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-iq", type=Path, required=True)
    ap.add_argument("--output-iq", type=Path, required=True)
    ap.add_argument("--target-below-data-db", type=float, default=11.3)
    ap.add_argument("--sample-rate-hz", type=float, default=10762237.762237763)
    ap.add_argument("--expected-pilot-hz", type=float, default=-2690559.0)
    ap.add_argument("--channel-width-hz", type=float, default=6.0e6)
    ap.add_argument("--search-half-width-hz", type=float, default=1000.0)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    iq = np.fromfile(str(args.input_iq), dtype=np.complex64)
    if iq.size == 0:
        print("empty input", file=sys.stderr)
        return 2
    n = iq.size
    fs = float(args.sample_rate_hz)
    t = np.arange(n, dtype=np.float64) / fs

    # locate the pilot precisely: FFT peak near the expected frequency,
    # then refine with three coherent projections (parabolic on power).
    spec = np.fft.fft(iq.astype(np.complex128))
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    near = np.abs(freqs - args.expected_pilot_hz) <= args.search_half_width_hz
    if not near.any():
        print("pilot search window off the grid", file=sys.stderr)
        return 2
    k0 = np.flatnonzero(near)[int(np.argmax(np.abs(spec[near])))]
    df = fs / n

    def proj(f):
        return complex(np.mean(iq * np.exp(-2j * np.pi * f * t)))

    f_c = float(freqs[k0])
    ps = [abs(proj(f_c + d)) ** 2 for d in (-df / 2, 0.0, df / 2)]
    denom = ps[0] - 2 * ps[1] + ps[2]
    shift = 0.5 * (ps[0] - ps[2]) / denom * (df / 2) if denom != 0 else 0.0
    f_pilot = f_c + float(np.clip(shift, -df, df))
    A = proj(f_pilot)
    p_pilot = abs(A) ** 2

    # direct in-band data power (band centred on the allocation implied by
    # the pilot placement, matching the audit's convention)
    band_lower = args.expected_pilot_hz - 309441.0
    band_upper = band_lower + args.channel_width_hz
    psd_mask = (freqs >= band_lower) & (freqs <= band_upper)
    band_power = float(np.sum(np.abs(spec[psd_mask]) ** 2) / n**2)
    p_data = band_power - p_pilot
    if p_data <= 0 or p_pilot <= 0:
        print("degenerate power split", file=sys.stderr)
        return 2

    ratio_now = DB * np.log10(p_data / p_pilot)
    g = float(np.sqrt(p_data * 10 ** (-args.target_below_data_db / DB) / p_pilot))
    out = iq.astype(np.complex128) + (g - 1.0) * A * np.exp(
        2j * np.pi * f_pilot * t
    )
    out.astype(np.complex64).tofile(str(args.output_iq))

    report = {
        "input_iq": str(args.input_iq),
        "output_iq": str(args.output_iq),
        "n_samples": int(n),
        "pilot_frequency_hz": f_pilot,
        "pilot_power_before": p_pilot,
        "data_power_direct": p_data,
        "pilot_below_data_direct_db_before": ratio_now,
        "target_below_data_db": float(args.target_below_data_db),
        "amplitude_gain_applied": g,
        # power change of the pilot line: 20*log10(amplitude gain)
        "gain_db": 2 * DB * np.log10(g),
    }
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\npilot trimmed: {ratio_now:.4f} -> {args.target_below_data_db:.2f} dB "
          f"(amplitude x{g:.5f}); verify with pilot-proxy audit-atsc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
