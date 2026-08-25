#!/usr/bin/env python3
# coding=utf-8
"""Pre-flight gate: prove a scan writes the exact integer accumulators.

Run this before committing a machine to the multi-week archive re-run. It
streams one capture through the real archive entry point (``chime-scan``, the
archive analyzer path) at the real 2048-stream geometry, then checks that the
emitted per-pilot product carries what an offline threshold replay needs:

  * ``fine_power_u64`` -- the three fine-power accumulators per bin, exact
    ``uint64``, not a float ratio;
  * ``p_ref_lower_u64`` / ``p_ref_upper_u64`` -- the reference split, unsummed;
  * a working replay of the frozen fine decision at several Q16 multipliers,
    straight from the stored product.

The last check is the point of the whole exercise: the deployed rule compares
exact integer cross products, so with these terms on disk a new operating point
is a re-read, not a re-scan.

Test captures are never committed. With no ``--input-dir`` this synthesises one
deterministically; point ``--input-dir`` at a fetched real capture (for example
a staged Channel 36 exemplar) to run the same gate against real bytes.

Exit status is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Physical channel -> the CHIME coarse channel its pilot lands in.
_DEFAULT_FREQ_ID = {36: 506, 14: 844}


def _synthesise(dest: Path, physical_channel: int, freq_id: int,
                streams: int, frames: int) -> Path:
    """Write one deterministic baseband file at the real stream count."""
    from pilot_proxy.chime.baseband_format import make_synth_file
    from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
    from pilot_proxy.archive.chime_coarse import (
        CHIME_BAND_TOP_HZ,
        CHIME_COARSE_WIDTH_HZ,
    )

    centre_hz = CHIME_BAND_TOP_HZ - freq_id * CHIME_COARSE_WIDTH_HZ
    pilot_hz = float(physical_channel_to_pilot_hz(physical_channel))
    # Baseband offset of the pilot within its coarse channel. The reader's
    # spectral sense is applied downstream; this only has to land in band.
    tone_bb = pilot_hz - centre_hz

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"baseband_preflight_{freq_id}.h5"
    make_synth_file(
        str(out),
        n_time=16384 * frames,
        n_feeds=streams,
        f_center_mhz=centre_hz / 1e6,
        f_tone_bb=tone_bb,
        seed=physical_channel,
    )
    return out


def _check(product: Path) -> bool:
    from pilot_proxy.fine_decision import FINE_BINS, fine_mask_decision
    from pilot_proxy.product_contract import validate_current_product_identity

    z = dict(np.load(product, allow_pickle=True))
    ok = True

    def report(passed: bool, message: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {message}")

    print(f"\nproduct : {product}")
    print(f"schema  : {np.asarray(z['schema_version'])}")
    print(f"channel : {int(np.asarray(z['physical_channel']).reshape(-1)[0])}"
          f"  streams: {int(np.asarray(z['num_input_streams']).reshape(()))}"
          f"  frames: {int(np.asarray(z['frame_index']).size)}\n")

    try:
        validate_current_product_identity(z)
        report(True, "product satisfies the current per-pilot contract")
    except Exception as exc:  # noqa: BLE001
        report(False, f"contract: {exc}")

    S = np.asarray(z.get("fine_power_u64", np.zeros(0)))
    report(S.dtype == np.dtype(np.uint64),
           f"fine_power_u64 dtype is {S.dtype} (want uint64)")
    report(S.ndim == 3 and S.shape[1] == 3 and S.shape[2] == FINE_BINS,
           f"fine_power_u64 shape {S.shape} (want [N, 3, {FINE_BINS}])")
    report(S.size > 0 and int(S.sum()) > 0,
           "fine_power_u64 holds a real reduction, not a placeholder")
    if S.size:
        print(f"         max accumulator {int(S.max())} "
              f"({int(S.max()).bit_length()} bits)")

    codes = np.asarray(z.get("psd_frame_db_i16", np.zeros(0)))
    report(codes.dtype == np.dtype(np.int16),
           f"psd_frame_db_i16 dtype is {codes.dtype} (want int16)")
    report(codes.ndim == 2 and codes.shape[1] > 0,
           f"psd_frame_db_i16 shape {codes.shape} carries a spectrum per frame")
    if codes.size:
        invalid = int(np.asarray(z["psd_db_invalid_code"]).reshape(()))
        refs = np.asarray(z["psd_db_reference"], dtype=np.float64).reshape(-1)
        real = codes[codes != invalid]
        finite_refs = refs[np.isfinite(refs) & (refs > 0.0)]
        report(real.size > 0 and finite_refs.size > 0,
               "psd_frame_db_i16 holds decodable spectra, not all-invalid")
        report(refs.size == codes.shape[0],
               "psd_db_reference carries one level per frame")
        if real.size and finite_refs.size:
            span = 0.01 * (int(real.max()) - int(real.min()))
            print(f"         reference median {np.median(finite_refs):.4e},"
                  f" span {span:.1f} dB")

    lo = np.asarray(z["p_ref_lower_u64"], dtype=object)
    up = np.asarray(z["p_ref_upper_u64"], dtype=object)
    report(bool(np.all(lo + up == np.asarray(z["p_ref_sum_u64"], dtype=object))),
           "coarse reference split retained and reconciles with its sum")

    report("weight_coefficients_sha256" in z,
           "weight_coefficients_sha256 recorded")

    # The capability the retention exists for.
    try:
        anchor = int(np.asarray(z["fine_designated_bins"]).reshape(-1)[0])
        masks = [
            fine_mask_decision(
                np.ascontiguousarray(S[0]),
                anchor_bin=anchor,
                designated_half_width=1,
                bulk_mask=[True] * FINE_BINS,
                cfar_rank=FINE_BINS // 2,
                multiplier_q16=q,
            ).mask
            for q in (1 << 14, 1 << 16, 1 << 18, 1 << 20)
        ]
        report(True, f"offline threshold replay from disk: masks {masks}")
    except Exception as exc:  # noqa: BLE001
        report(False, f"offline threshold replay: {exc}")

    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   help="directory of staged baseband .h5 (default: synthesise)")
    p.add_argument("--physical-channel", type=int, default=36)
    p.add_argument("--freq-id", type=int,
                   help="coarse channel to select (default: derived)")
    p.add_argument("--streams", type=int, default=2048,
                   help="feeds to synthesise; the real geometry is 2048")
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--weights-path", type=Path,
                   default=REPO / "weights" / "chime_dtv_weights_k128.bin")
    p.add_argument("--lib-path", type=Path,
                   help="libfstatistic .so; omit to use the CPU reference")
    p.add_argument("--output-dir", type=Path,
                   help="scan output (default: a temporary directory)")
    p.add_argument("--keep", action="store_true",
                   help="keep synthesised input and scan output")
    args = p.parse_args(argv)

    freq_id = args.freq_id or _DEFAULT_FREQ_ID.get(args.physical_channel)
    if freq_id is None:
        p.error("pass --freq-id for this physical channel")

    work = Path(tempfile.mkdtemp(prefix="preflight-"))
    input_dir = args.input_dir
    try:
        if input_dir is None:
            print(f"synthesising {args.streams}-stream capture for channel "
                  f"{args.physical_channel} (freq_id {freq_id}) ...")
            _synthesise(work / "in", args.physical_channel, freq_id,
                        args.streams, args.frames)
            input_dir = work / "in"
        else:
            print(f"using staged capture directory {input_dir}")

        out_dir = args.output_dir or (work / "scan")
        cmd = [
            sys.executable, "-m", "pilot_proxy.cli", "chime-scan",
            "--source", "local",
            "--input-dir", str(input_dir),
            "--output-dir", str(out_dir),
            "--select", str(freq_id),
            "--weights-path", str(args.weights_path),
        ]
        if args.lib_path:
            cmd += ["--lib-path", str(args.lib_path)]

        print("running the archive entry point:\n  " + " ".join(cmd))
        run = subprocess.run(cmd, cwd=str(REPO))
        if run.returncode != 0:
            print("\nRESULT: FAIL - the scan itself did not complete")
            return 1

        product = out_dir / "_per_pilot" / f"{freq_id}.npz"
        if not product.exists():
            print(f"\nRESULT: FAIL - no per-pilot product at {product}")
            return 1

        ok = _check(product)
        print("\nRESULT: " + ("PASS - the scan writes the exact integers; "
                              "safe to launch the archive re-run"
                              if ok else
                              "FAIL - do not launch the archive re-run"))
        return 0 if ok else 1
    finally:
        if args.keep:
            print(f"\nkept working directory {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
