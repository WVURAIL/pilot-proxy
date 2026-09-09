#!/usr/bin/env python3
"""Illustrative ideal F(2,4): additive complex signal versus switched populations.

This contains no hardware data and does not calibrate or validate a receiver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import f, ncf


def curves(x):
    """Return exact CDFs; endpoint mixture has E[lambda]=2, like steady lambda2."""
    central = f.cdf(x, 2, 4)
    moderate = ncf.cdf(x, 2, 4, 2)
    strong = ncf.cdf(x, 2, 4, 20)
    return central, moderate, strong, .9 * central + .1 * strong


def render(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    x = np.geomspace(.001, 300, 2400)
    central, moderate, strong, mixture = curves(x)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")
    labels = [r"Noise only: $F_{2,4}$ ($\lambda=0$)",
              r"Steady signal + noise: $F_{2,4}(\lambda=2)$",
              r"Strong steady signal + noise: $F_{2,4}(\lambda=20)$",
              r"Switching: 90% noise + 10% strong"]
    colors = ["#61717c", "#007f86", "#b45916", "#8b43a8"]
    for ax in axes:
        for y, label, color, style in zip((central, moderate, strong, mixture), labels, colors, ("-", "-", "-", "--")):
            ax.plot(x, y, color=color, lw=2.3, ls=style, label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel(r"CDF: probability that the statistic is at most $r$")
        ax.set_xlabel(r"Single-row ratio $r=2|z_0|^2/(|z_-|^2+|z_+|^2)$")
        ax.grid(alpha=.2)
    axes[0].set_xlim(0, 12)
    axes[0].set_title("Same mean, different distributions")
    axes[1].set_xscale("log")
    axes[1].set_xlim(.001, 300)
    axes[1].set_title("The full CDF across scales")
    axes[0].legend(loc="lower right", fontsize=8.6, framealpha=.96)
    fig.suptitle("Adding a steady signal is different from switching signal on and off", fontsize=14)
    fig.supxlabel("Illustrative theory only: proper complex Gaussian noise; three independent equal-variance projections.\nSteady λ=2 and the 90/10 mixture both have mean 4. All these distributions have infinite variance.", fontsize=9.5)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"additive-vs-mixture.{extension}", dpi=180)
    plt.close(fig)
    delta = mixture - moderate
    index = int(np.argmax(np.abs(delta)))
    anchors = np.array([.5, 1., 2., 4., 10.])
    a = curves(anchors)
    report = {
        "schema": "additive-versus-mixture-theory-v1",
        "scope": "Illustrative analytic model only; no measured data or receiver validation",
        "statistic": "R=2*abs(z0)**2/(abs(zminus)**2+abs(zplus)**2)",
        "model": "z0=mu+n0, zminus=nminus, zplus=nplus; independent proper complex Gaussian n with common E|n|^2=v",
        "lambda": "2*abs(mu)**2/v; additive amplitudes are combined before taking power and forming the ratio",
        "mixture": "Select noise population with probability0.9 or strong lambda20 population with probability0.1",
        "mean_steady_lambda2": 4., "mean_mixture": 4., "variance": "infinite for all displayed curves",
        "grid_max_abs_cdf_difference": float(abs(delta[index])), "grid_location": float(x[index]),
        "anchors": [{"ratio": float(t), "steady_lambda2_cdf": float(a[1][i]), "mixture_cdf": float(a[3][i]), "mixture_minus_steady": float(a[3][i]-a[1][i])} for i,t in enumerate(anchors)],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html"],
    }
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(render(parser.parse_args().output), indent=2))
