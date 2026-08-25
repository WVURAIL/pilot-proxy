#!/usr/bin/env python3
"""Real-data validation of the pilot-proxy sense/reversal contract (CHIME).

Reads a real CHIME baseband capture (``baseband_<event>_<freqid>.h5`` event
dumps or ``chNNNN/<event>.h5`` pilot captures), finds the ATSC pilot
expected in that coarse channel, and answers two questions with hardware
data instead of code comments:

1. NATIVE-FRAME SENSE: CHIME declares ``spectral_sense: inverted`` (second
   Nyquist zone), so a pilot BELOW the channel's RF center must appear at a
   POSITIVE baseband frequency -(pilot_RF - center) under the e^{+}
   convention (np.fft.fft analysis sign). This check confirms the declared
   sense against the physical emission.

2. DEPLOYED PAIRING: running the exact detector arithmetic (packed int4
   dot products, exact integer power sums) with the shipped CHIME weight
   bank, the real pilot must detect WITH the per-window time reversal and
   not without it -- the same reversal-on pairing the production CHIME
   runner uses and that corrected CHORD bundles now request.

Why this matters for CHORD: with the reversal on, detection requires the
bank's stored frequency to equal the pilot's NATIVE-frame frequency. CHIME
(native inverted) stores the mirrored value; CHORD (native upright per the
kotekan chord upchannelizer contract) stores the true-sense value. Passing
both checks on real emissions validates the contract structure end to end;
CHORD then differs only by its declared native frame.

Usage (any working directory, pilot-proxy venv active for numpy/h5py):

    python3 tools/chime_pilot_sense_check.py \
        --file /path/to/baseband_1153713684_506.h5

Options: --freq-id (CHIME channel id when the path does not carry it),
--max-samples (default 262144), --max-feeds (default 32),
--weights (default <repo>/weights/chime_dtv_weights_k128.bin).
Exit code 0 = both checks confirmed; 2 = pilot line not found;
3 = a check contradicted the declared convention (read the verdict lines).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Resolve the repo's src/ from this file's location (tools/) so the script
# runs from any working directory, installed package or not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

K = 128
CHIME_TOP_HZ = 800e6
CHIME_DF_HZ = 390625.0
DC_GUARD_HZ = 300.0
LINE_TOL_BINS = 3.0


def decode_components(packed: np.ndarray) -> np.ndarray:
    """Offset-binary 4+4 uint8 -> complex (real high nibble, imag low)."""
    packed = np.asarray(packed, dtype=np.uint8)
    real = (packed >> 4).astype(np.float64) - 8.0
    imag = (packed & 0x0F).astype(np.float64) - 8.0
    return real + 1j * imag


def channel_center_hz(path: Path, freq_id_arg: int | None) -> tuple[float, float, str]:
    """(center_hz, sample_rate_hz, source): CLI override, file metadata, filename
    ``*_<id>.h5``, a ``chNNNN`` ancestor directory, or an h5 center attr."""
    if freq_id_arg is not None:
        return (
            CHIME_TOP_HZ - freq_id_arg * CHIME_DF_HZ,
            CHIME_DF_HZ,
            f"--freq-id {freq_id_arg}",
        )
    try:
        from pilot_proxy.chime import baseband_format as fmt

        return float(fmt.channel_center_hz(str(path))), float(fmt.FS), "file-metadata"
    except Exception:
        pass
    match = re.search(r"_(\d{1,4})\.h5$", path.name)
    if match and 0 <= int(match.group(1)) <= 1023:
        freq_id = int(match.group(1))
        return (
            CHIME_TOP_HZ - freq_id * CHIME_DF_HZ,
            CHIME_DF_HZ,
            f"filename freq_id {freq_id}",
        )
    for parent in path.resolve().parents:
        dir_match = re.fullmatch(r"ch0*(\d{1,4})", parent.name)
        if dir_match and 0 <= int(dir_match.group(1)) <= 1023:
            freq_id = int(dir_match.group(1))
            return (
                CHIME_TOP_HZ - freq_id * CHIME_DF_HZ,
                CHIME_DF_HZ,
                f"directory {parent.name} -> freq_id {freq_id}",
            )
    import h5py

    with h5py.File(str(path), "r") as handle:
        pools = [("/", dict(handle.attrs))]
        if "baseband" in handle:
            pools.append(("baseband", dict(handle["baseband"].attrs)))
        for where, attrs in pools:
            for key, value in attrs.items():
                if "freq" not in key.lower() and "centre" not in key.lower() \
                        and "center" not in key.lower():
                    continue
                try:
                    hz = float(np.ravel(value)[0])
                except (TypeError, ValueError):
                    continue
                if 4e8 <= hz <= 8e8:
                    return hz, CHIME_DF_HZ, f"h5 attr {where}:{key}"
                if 400.0 <= hz <= 800.0:
                    return hz * 1e6, CHIME_DF_HZ, f"h5 attr {where}:{key} (MHz)"
    raise SystemExit(
        f"cannot determine channel center for {path.name}; rerun with "
        "--freq-id <chime channel id> (e.g. --freq-id 506 for the ch0506 "
        "captures)"
    )


def nearest_pilot(center_hz: float, fs_hz: float):
    from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz

    best = None
    for channel in range(14, 37):
        pilot = float(physical_channel_to_pilot_hz(channel))
        offset = pilot - center_hz
        if best is None or abs(offset) < abs(best[2]):
            best = (channel, pilot, offset)
    channel, pilot, offset = best
    if abs(offset) > 0.45 * fs_hz:
        raise SystemExit(
            f"no ATSC pilot inside this coarse channel (nearest: ch {channel} "
            f"at {offset:+.1f} Hz from center); pick a pilot-channel file"
        )
    return channel, pilot, offset


def integer_powers(samples: np.ndarray, weights_packed: np.ndarray) -> list[int]:
    """Exact integer P[target, ref_lower, ref_upper]; samples [rows, K]."""
    w = weights_packed.astype(np.uint8)
    w_re = ((w >> 4).astype(np.int64) ^ 8) - 8  # two's-complement nibbles
    w_im = ((w & 0x0F).astype(np.int64) ^ 8) - 8
    x_re = samples.real.astype(np.int64)
    x_im = samples.imag.astype(np.int64)
    powers = []
    for term in range(3):
        z_re = (x_re * w_re[term] + x_im * w_im[term]).sum(axis=1)
        z_im = (x_im * w_re[term] - x_re * w_im[term]).sum(axis=1)
        powers.append(int((z_re * z_re + z_im * z_im).sum()))
    return powers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--freq-id", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=262144)
    parser.add_argument("--max-feeds", type=int, default=32)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    import h5py

    center_hz, fs_hz, source = channel_center_hz(args.file, args.freq_id)
    print(f"file: {args.file}")
    print(f"channel center {center_hz / 1e6:.6f} MHz, fs {fs_hz:.1f} Hz ({source})")

    channel, pilot_hz, rf_offset_hz = nearest_pilot(center_hz, fs_hz)
    predicted_hz = -rf_offset_hz  # CHIME native frame is spectrally inverted
    print(
        f"expected pilot: ATSC ch {channel} at {pilot_hz / 1e6:.6f} MHz, "
        f"RF offset {rf_offset_hz:+.1f} Hz from center -> predicted native "
        f"line {predicted_hz:+.1f} Hz (declared sense: inverted)"
    )

    with h5py.File(str(args.file), "r") as handle:
        dataset = handle["baseband"]
        n_time, n_feeds = dataset.shape
        n = min(args.max_samples, (n_time // K) * K)
        feeds = min(args.max_feeds, n_feeds)
        packed = dataset[:n, :feeds]
    print(f"read {n} of {n_time} samples x {feeds} of {n_feeds} feeds")
    x = decode_components(packed)  # [time, feed]

    # --- Check 1: raw-frame spectral sense --------------------------------
    spec = np.zeros(n)
    for feed in range(feeds):
        column = x[:, feed] - x[:, feed].mean()
        spec += np.abs(np.fft.fft(column)) ** 2
    freqs = np.fft.fftfreq(n, d=1.0 / fs_hz)
    bin_hz = fs_hz / n
    search = spec.copy()
    search[np.abs(freqs) < DC_GUARD_HZ] = 0.0
    peak = int(np.argmax(search))
    peak_hz = float(freqs[peak])
    floor = float(np.median(search[search > 0]))
    line_snr = search[peak] / floor
    print(
        f"strongest line: {peak_hz:+.1f} Hz (bin {bin_hz:.2f} Hz, "
        f"power {line_snr:.1f}x median floor)"
    )
    # Real transmitters carry per-station pilot offsets (up to ~1 kHz from
    # the nominal ATSC pilot) and several co-channel stations can be visible
    # at once, so judge the sense by comparing line power in a window around
    # the predicted native offset against the mirrored window -- not by
    # demanding the global peak land exactly on the nominal frequency.
    window_hz = 2000.0

    def window_peak(center_freq_hz):
        mask = np.abs(freqs - center_freq_hz) <= window_hz
        if not np.any(mask):
            return 0.0, 0.0
        idx = int(np.argmax(np.where(mask, search, 0.0)))
        return float(search[idx] / floor), float(freqs[idx])

    native_snr, native_hz = window_peak(predicted_hz)
    mirror_snr, mirror_hz = window_peak(-predicted_hz)
    print(
        f"window +/-{window_hz:.0f} Hz around predicted native "
        f"{predicted_hz:+.0f} Hz: peak {native_snr:.1f}x floor at "
        f"{native_hz:+.1f} Hz (station offset {native_hz - predicted_hz:+.1f} Hz)"
    )
    print(
        f"window +/-{window_hz:.0f} Hz around mirrored "
        f"{-predicted_hz:+.0f} Hz: peak {mirror_snr:.1f}x floor at "
        f"{mirror_hz:+.1f} Hz"
    )
    if native_snr >= 20.0 and native_snr >= 100.0 * max(mirror_snr, 1.0):
        sense_ok = True
        print(
            "CHECK 1 CONFIRMED: pilot line(s) on the predicted NATIVE "
            "(inverted-sense) side with an empty mirror -- CHIME's declared "
            "spectral_sense matches the physical emission."
        )
        station_offset_hz = native_hz - predicted_hz
        if abs(station_offset_hz) > 100.0:
            loss = float(np.sinc(station_offset_hz * K / fs_hz) ** 2)
            print(
                f"  note: dominant station runs {station_offset_hz:+.1f} Hz "
                f"from nominal; the deployed nominal-frequency template "
                f"still captures {100.0 * loss:.1f}% of its power per window."
            )
    elif mirror_snr >= 20.0 and mirror_snr >= 100.0 * max(native_snr, 1.0):
        sense_ok = False
        print(
            "CHECK 1 CONTRADICTED: pilot line(s) only on the mirror of the "
            "predicted native side -- the data appears TRUE-SENSE, "
            "contradicting CHIME's declared inverted sense. STOP: re-examine "
            "the sense model before trusting any reversal conclusions."
        )
    else:
        print(
            "CHECK 1 INCONCLUSIVE: no decisive line asymmetry between the "
            "predicted native window and its mirror; the pilot may be "
            "off-air in this capture or the file may not be a pilot channel."
        )
        top = np.argsort(search)[-5:][::-1]
        for idx in top:
            print(f"  candidate line {freqs[int(idx)]:+.1f} Hz, {search[int(idx)] / floor:.1f}x floor")
        return 2

    # --- Check 2: deployed pairing (reversal on vs off) -------------------
    from pilot_proxy.detector_weights import DetectorWeightBank

    weights_path = args.weights or (
        Path(__file__).resolve().parents[1] / "weights" / "chime_dtv_weights_k128.bin"
    )
    bank = DetectorWeightBank(explicit_path=str(weights_path))
    packed_weights, valid = bank.get_weights_for_physical_channel(channel)
    if packed_weights is None or not valid:
        raise SystemExit(f"weight bank has no valid profile for ch {channel}")
    layout = bank.layout_for_physical_channel(channel)
    null_power_ratio = float(layout["null_power_ratio"])

    rows = np.ascontiguousarray(
        x.T.reshape(feeds, n // K, K).reshape(-1, K)
    )  # per-feed K-sample windows, stream-major
    p_forward = integer_powers(rows, np.asarray(packed_weights))
    p_reversed = integer_powers(rows[:, ::-1], np.asarray(packed_weights))
    f_forward = 2.0 * p_forward[0] / max(p_forward[1] + p_forward[2], 1)
    f_reversed = 2.0 * p_reversed[0] / max(p_reversed[1] + p_reversed[2], 1)
    print(
        f"detector F (null_power_ratio zero-point {null_power_ratio:.4f}): "
        f"no-reversal {f_forward:.3f}, with-reversal {f_reversed:.3f}"
    )
    pairing_ok = f_reversed > 2.0 * null_power_ratio and f_reversed > 5.0 * f_forward
    if pairing_ok:
        print(
            "CHECK 2 CONFIRMED: the real pilot detects WITH per-window time "
            "reversal and not without it -- the deployed pairing the "
            "corrected CHORD bundles request."
        )
    else:
        print(
            "CHECK 2 NOT CONFIRMED on this capture: with-reversal F is not "
            "decisively above both the zero-point and the no-reversal F. "
            "Weak/absent pilot in this data, or a convention problem -- try "
            "a canfar_pilots_10s capture before drawing conclusions."
        )

    if sense_ok and pairing_ok:
        print("VERDICT: real-data validation PASSED (both checks confirmed)")
        return 0
    return 3


if __name__ == "__main__":
    sys.exit(main())
