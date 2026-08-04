#!/usr/bin/env python3
"""Preliminary pilot-free control floor from staged raw CHIME baseband files.

CPU-only. Runs the exact float64 reference correlator (the same
detector_reference path the Fig-3 sweep validated) over every frame of every
staged file, using a real ATSC channel's weight vectors applied to a
pilot-free coarse channel. Produces the provisional H0 floor for the paper's
control slot: measured zero point, gap to analytic mu0, tail fractions, and
the per-frame F series.

    python floor_from_raw.py --input-dir ~/control_staging \
        --output-dir ~/pilot_proxy_runs/control_floor_cpu [--like-channel 31]
        [--stream-stride 4] [--max-files 0] [--repo ~/pilot-proxy]

Notes
-----
* --like-channel picks whose weight geometry to apply (target/ref fine-bin
  steering vectors + norms). Any trusted channel works; 31 is the default.
* --stream-stride N correlates every Nth input stream. Floor position is
  unbiased under H0; core width grows ~sqrt(N). stride 4 is ~4x faster and
  still leaves sigma_core well inside the +/-12e-3 band. Use stride 1 for
  the final number.
* Expect ~0.5-1 s/frame at stride 1 (full 2048 streams), ~40 frames/event.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--like-channel", type=int, default=31)
    ap.add_argument("--frame-size", type=int, default=16384)
    ap.add_argument("--stream-stride", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-frames-per-file", type=int, default=0)
    ap.add_argument("--repo", type=Path,
                    default=Path.home() / "pilot-proxy")
    ap.add_argument("--pattern", default="*.h5")
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    import h5py
    from pilot_proxy.detector_weights import DetectorWeightBank
    from pilot_proxy.detector_reference import (
        fstat_cpu_reference,
        unpack_packed_complex,
        REFERENCE_TARGET_TERM_INDEX as IT,
        REFERENCE_LOWER_TERM_INDEX as IL,
        REFERENCE_UPPER_TERM_INDEX as IU,
    )
    from pilot_proxy.chime.hdf5_input import _find_dataset_path, _infer_axes
    from pilot_proxy.chime.frame_adapter import (
        unpack_chime_offset_binary_i4_to_complex as unpack_ob,
        unpack_twos_complement_i4_to_complex as unpack_tc,
    )

    bank = DetectorWeightBank(
        explicit_path=str(args.repo / "weights/chime_dtv_weights_k128.bin"))
    K = int(bank.K)
    packed, ok = bank.get_weights_for_physical_channel(args.like_channel)
    if packed is None or not np.size(packed):
        raise SystemExit(f"no weights for channel {args.like_channel}")
    bits = 4
    for attr in ("component_bits", "weight_component_bits"):
        h = getattr(bank, "header", None)
        if h is not None and hasattr(h, attr):
            bits = int(getattr(h, attr))
            break
    w = np.asarray(unpack_packed_complex(np.asarray(packed), bits),
                   dtype=np.complex128).reshape(3, K)
    nq = lambda v: float(np.sum(np.abs(v) ** 2))
    tns, lns, uns = nq(w[IT]), nq(w[IL]), nq(w[IU])
    mu0 = 2.0 * tns / (lns + uns)
    print(f"like-channel {args.like_channel}: K={K} bits={bits} "
          f"tns={tns:.0f} rnss={lns+uns:.0f} mu0={mu0:.6f}")

    files = sorted(args.input_dir.rglob(args.pattern))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"no files matching {args.pattern} under "
                         f"{args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    F, PT, PR, FIDX = [], [], [], []
    t0 = time.time()
    for fi, path in enumerate(files):
        with h5py.File(path, "r") as h5:
            fid = h5.attrs.get("freq_id", None)
            dpath = _find_dataset_path(h5, None)
            obj = h5[dpath]
            time_axis, stream_axis, complex_axis = _infer_axes(obj)
            nt = obj.shape[time_axis]
            n_frames = nt // args.frame_size
            if args.max_frames_per_file:
                n_frames = min(n_frames, args.max_frames_per_file)
            print(f"[{fi+1}/{len(files)}] {path.name}: freq_id={fid} "
                  f"shape={obj.shape} dtype={obj.dtype} frames={n_frames}")
            for k in range(n_frames):
                sl = [slice(None)] * obj.ndim
                sl[time_axis] = slice(k * args.frame_size,
                                      (k + 1) * args.frame_size)
                arr = obj[tuple(sl)]
                arr = np.moveaxis(arr, (time_axis, stream_axis), (0, 1))
                if arr.dtype == np.uint8:
                    x = unpack_ob(arr)
                elif arr.dtype == np.int8:
                    x = unpack_tc(arr.view(np.uint8))
                elif arr.dtype.names:
                    names = {n.lower(): n for n in arr.dtype.names}
                    x = (arr[names.get("r", names.get("real"))]
                         .astype(np.float32)
                         + 1j * arr[names.get("i", names.get("imag"))]
                         .astype(np.float32))
                elif np.issubdtype(arr.dtype, np.complexfloating):
                    x = arr
                else:
                    raise SystemExit(f"unhandled dtype {arr.dtype}")
                if x.ndim == 3:          # residual complex axis
                    x = x[..., 0] + 1j * x[..., 1]
                x = x[:, :: args.stream_stride]
                S = x.shape[1]
                W = args.frame_size // K
                rows = np.ascontiguousarray(x.T).reshape(S * W, K)
                f, sums = fstat_cpu_reference(rows, w)
                F.append(f)
                PT.append(sums[IT])
                PR.append(sums[IL] + sums[IU])
                FIDX.append(fi)
                if len(F) % 50 == 0:
                    r = len(F) / (time.time() - t0)
                    print(f"    {len(F)} frames  ({r:.2f} fr/s)")

    F = np.asarray(F)
    PT = np.asarray(PT)
    PR = np.asarray(PR)
    ok_f = np.isfinite(F) & (PR > 0)
    fv = F[ok_f]

    # mode-anchored core zero point (same recipe as the production study)
    lo_q, hi_q = np.percentile(fv, [1, 99])
    hist, edges = np.histogram(fv, bins=200, range=(lo_q, hi_q))
    mu_hat = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    for _ in range(3):
        core = fv[np.abs(fv - mu_hat) <= 6e-3 * mu0]
        if core.size < 50:
            break
        mu_hat = float(np.median(core))
    core = fv[np.abs(fv - mu_hat) <= 6e-3 * mu0]
    gap = 1e3 * (mu_hat - mu0) / mu0
    lo_t = float((fv < mu_hat - 12e-3 * mu0).mean())
    hi_t = float((fv > mu_hat + 12e-3 * mu0).mean())
    mf_an = float((fv > mu0).mean())
    mf_em = float((fv > mu_hat).mean())
    err = (1.2533 * np.std(core) / max(np.sqrt(core.size), 1)
           if core.size else float("nan"))

    np.savez_compressed(
        args.output_dir / "control_floor_frames.npz",
        f=F, p_target=PT, p_ref_sum=PR, file_index=np.asarray(FIDX),
        files=np.asarray([str(p) for p in files]),
        like_channel=args.like_channel, mu0_like=mu0,
        stream_stride=args.stream_stride)
    with open(args.output_dir / "control_floor_summary.csv", "w",
              newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["like_channel", "n_files", "n_frames", "stream_stride",
                     "mu0_like", "mu_hat_control", "mu_hat_err",
                     "gap_1e3", "core_frac", "low_tail_frac",
                     "high_tail_frac", "mask_frac_analytic",
                     "mask_frac_empirical"])
        cw.writerow([args.like_channel, len(files), int(fv.size),
                     args.stream_stride, f"{mu0:.6f}", f"{mu_hat:.6f}",
                     f"{err:.6f}", f"{gap:+.3f}",
                     f"{core.size / max(fv.size, 1):.3f}", f"{lo_t:.4f}",
                     f"{hi_t:.4f}", f"{mf_an:.4f}", f"{mf_em:.4f}"])
    print(f"\ncontrol floor: n={fv.size} frames from {len(files)} files")
    print(f"  mu0(like ch{args.like_channel}) = {mu0:.6f}")
    print(f"  mu_hat(control)  = {mu_hat:.6f} +/- {err:.6f} "
          f"(gap {gap:+.3f} x10^-3)")
    print(f"  tails beyond +/-12e-3: low {100*lo_t:.2f}%  high {100*hi_t:.2f}%")
    print(f"  mask fraction: analytic {mf_an:.3f}  empirical {mf_em:.3f} "
          "(H0 target ~0.5, item-1 window 0.45-0.55)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        span = 40e-3 * mu0
        ax.hist(fv, bins=240, range=(mu_hat - span, mu_hat + span),
                color="#0072B2", alpha=0.85)
        ax.axvline(mu0, color="#D55E00", lw=1.2, label=r"analytic $\mu_0$")
        ax.axvline(mu_hat, color="k", lw=1.2, ls="--",
                   label=r"measured $\hat\mu_0$ (control)")
        for s in (-1, 1):
            ax.axvline(mu_hat + s * 12e-3 * mu0, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("F")
        ax.set_ylabel("frames")
        ax.set_title(f"Pilot-free control floor "
                     f"(geometry of ch{args.like_channel}, "
                     f"{fv.size} frames)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "control_floor_hist.png", dpi=200)
        print(f"  wrote {args.output_dir}/control_floor_hist.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (no histogram: {e})")


if __name__ == "__main__":
    main()
