#!/usr/bin/env python3
# coding=utf-8
"""Measured coherent-gain Monte Carlo: coarse vs fine reduction Pd curves.

The publication claim ("the v2 fine reduction recovers up to
10 log10(sqrt(W)) ~ 10.5 dB of deflection sensitivity") is measured here
rather than asserted. The Monte Carlo operates at the row-sum level ---
the exact field both reductions consume (the bit-exact marginal identity
guarantees the coarse statistic is a pure function of it) --- at the
deployed geometry (2048 streams x 128 windows), with integer row sums and
the deployed statistics:

  coarse:  F = 2 sum|z_t|^2 / (sum|z_r1|^2 + sum|z_r2|^2), exact int64
           sums (null_power_ratio = 1: simulated weight norms are equal by
           construction, stated in the output).
  fine:    fine_power_ratio[b] = 2 S_t[b] / (S_l[b] + S_u[b]) from the x2
           zero-padded window-axis FFT, then the deployed designated-set
           statistic max over the anchor +/- 2 window.

The batched reduction used for speed is verified against
``pilot_proxy.fine_reduction.fine_reduce`` on random trials by
``--verify`` (and by ``tests/core/test_measure_fine_gain.py``); the MC
runs only the verified-equal path.

Signal model (matches the documented zoom identity): the target-term row
sum of stream n, window m is A e^{i(2 pi b0 m / 256 + theta_n)} + noise,
theta_n uniform per stream (feeds add incoherently), b0 the injected fine
bin (bin-centered, plus a half-bin case measuring the scalloping worst
band). Per-row-sum SNR = A^2 / (2 sigma^2).

Thresholds are empirical H0 quantiles at fixed Pfa per statistic --- the
fixed-false-alarm policy the survey adopts. Stages are resumable shards:

  --stage h0    --trials N --seed K        (append H0 shard)
  --stage sweep --snr-db X --trials N --seed K
  --stage report                            (thresholds, curves, figure)
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

STREAMS = 2048
WINDOWS = 128
BINS = 256
SIGMA = 120.0
ANCHOR = 62          # injected fine bin (bin-centered case)
HALF_BIN = 62.5      # scalloping worst case
WINDOW = np.arange(ANCHOR - 2, ANCHOR + 3) % BINS
PFA_LIST = (1e-2, 1e-3)


def gen_rows(rng, batch, streams, amp=0.0, b0=0.0):
    """Integer row sums [batch, 3, streams, W, 2]; target may carry a line."""
    z = rng.standard_normal((batch, 3, streams, WINDOWS, 2), dtype=np.float32)
    z *= SIGMA
    if amp > 0.0:
        m = np.arange(WINDOWS, dtype=np.float32)
        theta = rng.uniform(0, 2 * np.pi, size=(batch, streams, 1)).astype(np.float32)
        ph = 2.0 * np.pi * (b0 / BINS) * m + theta
        z[:, 0, :, :, 0] += amp * np.cos(ph)
        z[:, 0, :, :, 1] += amp * np.sin(ph)
    return np.round(z).astype(np.int32)


def reduce_batch(zi):
    """Batched deployed reductions. zi: int32 [batch, 3, streams, W, 2].

    Returns (coarse F, fine window-max) per trial. Identical math to the
    packaged pipeline: exact int64 marginals; complex128 x2-padded FFT,
    incoherent stream sum, fine_power_ratio ratio, designated-window max.
    """
    wide = zi.astype(np.int64)
    powers = (wide[..., 0] ** 2 + wide[..., 1] ** 2).sum(axis=(2, 3))
    den = powers[:, 1] + powers[:, 2]
    coarse = 2.0 * powers[:, 0] / den

    zc = zi[..., 0].astype(np.complex128) + 1j * zi[..., 1].astype(np.complex128)
    spec = np.fft.fft(zc, n=BINS, axis=-1)
    p = (spec.real ** 2 + spec.imag ** 2).sum(axis=2)      # [batch, 3, BINS]
    f2 = 2.0 * p[:, 0] / (p[:, 1] + p[:, 2])
    fine = f2[:, WINDOW].max(axis=1)
    return coarse, fine


def verify(seed=0, trials=3, streams=64):
    """Gate: the batched reduction equals the packaged pipeline.

    Exact integer marginals must match bit-for-bit; the fine spectrum
    must match at the product's own precision (``fine_reduce`` emits
    float32; the batched path carries the same float64 operations, so
    the float32 casts must be identical).
    """
    from pilot_proxy import fine_reduction
    rng = np.random.default_rng(seed)
    zi = gen_rows(rng, trials, streams, amp=300.0, b0=ANCHOR)
    wide = zi.astype(np.int64)
    powers = (wide[..., 0] ** 2 + wide[..., 1] ** 2).sum(axis=(2, 3))
    zc = zi[..., 0].astype(np.complex128) + 1j * zi[..., 1].astype(np.complex128)
    spec = np.fft.fft(zc, n=BINS, axis=-1)
    p = (spec.real ** 2 + spec.imag ** 2).sum(axis=2)
    f2 = 2.0 * p[:, 0] / (p[:, 1] + p[:, 2])
    for t in range(trials):
        rows = zi[t].reshape(3, streams * WINDOWS, 2)
        res = fine_reduction.fine_reduce(
            rows, num_streams=streams, windows_per_stream=WINDOWS)
        assert np.array_equal(res.coarse_power_by_term.astype(np.int64), powers[t]), \
            "integer marginals diverge"
        ref32 = np.asarray(res.fine_power_ratio, dtype=np.float32)
        got32 = f2[t].astype(np.float32)
        assert np.array_equal(ref32, got32), (
            "fine spectrum diverges at product precision: max|d|="
            f"{np.abs(ref32.astype(np.float64) - got32.astype(np.float64)).max()}")
    print(f"verify: batched reduction == fine_reduce on {trials} trials "
          f"(streams={streams}; exact marginals, float32-identical spectra)")


def run_stage(out, stage, trials, seed, snr_db=None, b0=ANCHOR, streams=STREAMS,
              batch=8, tag=""):
    rng = np.random.default_rng(seed)
    amp = 0.0
    if stage == "sweep":
        amp = float(np.sqrt(2.0 * SIGMA * SIGMA * 10 ** (snr_db / 10.0)))
    co, fi = [], []
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        zi = gen_rows(rng, b, streams, amp=amp, b0=b0)
        c, f = reduce_batch(zi)
        co.append(c)
        fi.append(f)
        done += b
    co = np.concatenate(co)
    fi = np.concatenate(fi)
    name = (f"h0_s{seed}.npz" if stage == "h0"
            else f"h1{tag}_{snr_db:+06.2f}dB_s{seed}.npz")
    np.savez_compressed(os.path.join(out, name), coarse=co, fine=fi,
                        snr_db=(np.nan if snr_db is None else snr_db),
                        b0=b0, streams=streams, sigma=SIGMA, seed=seed)
    print(f"wrote {name}: {trials} trials "
          f"(coarse med {np.median(co):.5f}, fine med {np.median(fi):.4f})")


def collect(out, pattern):
    cs, fs, meta = [], [], []
    for p in sorted(glob.glob(os.path.join(out, pattern))):
        z = np.load(p)
        cs.append(z["coarse"])
        fs.append(z["fine"])
        meta.append(float(z["snr_db"]))
    return (np.concatenate(cs) if cs else np.zeros(0),
            np.concatenate(fs) if fs else np.zeros(0), meta)


def report(out, make_figure=True):
    c0, f0, _ = collect(out, "h0_s*.npz")
    if c0.size == 0:
        print("no H0 shards", file=sys.stderr)
        return 1
    print(f"H0 trials: {c0.size}")
    thr = {}
    for pfa in PFA_LIST:
        thr[pfa] = (float(np.quantile(c0, 1 - pfa)),
                    float(np.quantile(f0, 1 - pfa)))
        print(f"  Pfa={pfa:g}: coarse thr {thr[pfa][0]:.6f}, "
              f"fine thr {thr[pfa][1]:.4f}")

    curves = {}
    for tag in ("", "_half"):
        pts = {}
        for p in sorted(glob.glob(os.path.join(out, f"h1{tag}_*dB_s*.npz"))):
            z = np.load(p)
            s = float(z["snr_db"])
            pts.setdefault(s, [[], []])
            pts[s][0].append(z["coarse"])
            pts[s][1].append(z["fine"])
        if not pts:
            continue
        snrs = np.array(sorted(pts))
        rows = []
        for s in snrs:
            c = np.concatenate(pts[s][0])
            f = np.concatenate(pts[s][1])
            row = {"snr_db": s, "n": c.size}
            for pfa in PFA_LIST:
                row[f"pd_coarse_{pfa:g}"] = float((c > thr[pfa][0]).mean())
                row[f"pd_fine_{pfa:g}"] = float((f > thr[pfa][1]).mean())
            rows.append(row)
        curves[tag or "centered"] = rows

    def snr_at(rows, key, target=0.5):
        xs = [r["snr_db"] for r in rows]
        ys = [r[key] for r in rows]
        for i in range(1, len(xs)):
            if ys[i - 1] < target <= ys[i]:
                a, b = ys[i - 1], ys[i]
                return xs[i - 1] + (xs[i] - xs[i - 1]) * (target - a) / (b - a)
        return float("nan")

    gains = {}
    for pfa in PFA_LIST:
        rows = curves.get("centered", [])
        if rows:
            s_c = snr_at(rows, f"pd_coarse_{pfa:g}")
            s_f = snr_at(rows, f"pd_fine_{pfa:g}")
            gains[pfa] = (s_c, s_f, s_c - s_f)
            print(f"Pfa={pfa:g}: SNR@Pd=0.5 coarse {s_c:+.2f} dB, "
                  f"fine {s_f:+.2f} dB  ->  measured gain {s_c - s_f:.2f} dB")
    np.savez(os.path.join(out, "gain_report.npz"),
             thresholds=str(thr), curves=str(curves), gains=str(gains),
             h0_trials=c0.size)

    if make_figure and curves:
        make_fig(out, thr, curves, gains)
    return 0


def make_fig(out, thr, curves, gains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID, BLUE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                      "#e5e4e0", "#2a78d6")
    pfa = 1e-2
    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=300, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    rows = curves["centered"]
    xs = [r["snr_db"] for r in rows]
    ax.plot(xs, [r[f"pd_coarse_{pfa:g}"] for r in rows], "--", color=INK2,
            lw=2.0, label="coarse (v1 reduction)")
    ax.plot(xs, [r[f"pd_fine_{pfa:g}"] for r in rows], "-", color=BLUE,
            lw=2.2, marker="o", ms=4.5, mec=SURFACE, mew=0.8,
            label="fine, designated window (v2)")
    if "_half" in curves or "half" in curves:
        h = curves.get("_half") or curves.get("half")
        ax.plot([r["snr_db"] for r in h], [r[f"pd_fine_{pfa:g}"] for r in h],
                ":", color=BLUE, lw=1.8, alpha=0.75,
                label="fine, half-bin offset (scalloping)")
    if gains.get(pfa):
        s_c, s_f, g = gains[pfa]
        ax.annotate("", xy=(s_f, 0.5), xytext=(s_c, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
        ax.annotate(f"measured gain {g:.1f} dB",
                    xy=((s_c + s_f) / 2, 0.53), ha="center",
                    fontsize=10, color=INK)
    ax.axhline(0.5, color=GRID, lw=0.8)
    ax.set_xlabel("per-row-sum pilot SNR (dB)", fontsize=10, color=INK)
    ax.set_ylabel("detection probability", fontsize=10, color=INK)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK)
    ax.set_title(
        f"Coarse vs fine reduction at matched false-alarm rate "
        f"(Pfa = {pfa:g}, empirical H0 thresholds)\n"
        f"deployed geometry: {STREAMS} streams × {WINDOWS} windows, "
        f"integer row sums, exact deployed statistics",
        fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "measured_fine_gain.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="generated/fine_gain_mc")
    ap.add_argument("--stage", choices=["h0", "sweep", "report", "verify"],
                    required=True)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--half-bin", action="store_true",
                    help="inject at the half-bin (scalloping) offset")
    ap.add_argument("--streams", type=int, default=STREAMS)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "verify":
        verify()
        return 0
    if args.stage == "report":
        return report(args.out)
    if args.stage == "sweep":
        if args.snr_db is None:
            ap.error("--snr-db required for sweep")
        run_stage(args.out, "sweep", args.trials, args.seed,
                  snr_db=args.snr_db,
                  b0=(HALF_BIN if args.half_bin else ANCHOR),
                  streams=args.streams, batch=args.batch,
                  tag=("_half" if args.half_bin else ""))
        return 0
    run_stage(args.out, "h0", args.trials, args.seed, streams=args.streams,
              batch=args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
