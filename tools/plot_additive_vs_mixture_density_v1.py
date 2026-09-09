#!/usr/bin/env python3
"""Ideal probability densities for additive signals and switched populations.

This contains no hardware data. Historical CDF artifacts remain unchanged.
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
from scipy.integrate import quad
from scipy.stats import f, ncf


def density(r, noncentrality):
    """Exact noncentral-F(2,4) density, including its right-hand value at zero."""
    r = np.asarray(r, dtype=float)
    lam = float(noncentrality)
    if np.any(~np.isfinite(r)) or np.any(r < 0) or not np.isfinite(lam) or lam < 0:
        raise ValueError("Require finite nonnegative ratios and noncentrality")
    d = r + 2
    return np.exp(-lam / d) * (8 / d**3 + 8 * lam * r / d**4 + lam**2 * r**2 / d**5)


def curves(r):
    central = density(r, 0)
    moderate = density(r, 2)
    strong = density(r, 20)
    return central, moderate, strong, .9 * central + .1 * strong


def verify():
    points = np.geomspace(1e-7, 1e5, 4000)
    checks = []
    for lam in (0, 2, 20):
        expected = f.pdf(points, 2, 4) if lam == 0 else ncf.pdf(points, 2, 4, lam)
        error = float(np.max(np.abs(density(points, lam) - expected)))
        assert np.allclose(density(points, lam), expected, rtol=2e-11, atol=2e-14)
        area, integration_error = quad(lambda r: float(density(r, lam)), 0, np.inf, epsabs=1e-11)
        assert abs(area - 1) < 1e-9
        assert float(density(0, lam)) == float(np.exp(-lam / 2))
        checks.append({"lambda": lam, "positive_grid_max_abs_difference_from_scipy": error,
                       "integral_0_infinity": area, "quadrature_error_estimate": integration_error,
                       "right_hand_density_at_zero": float(density(0, lam))})
    area, error = quad(lambda r: float(curves(r)[3]), 0, np.inf, epsabs=1e-11)
    assert abs(area - 1) < 1e-9
    return {"passed": True, "component_checks": checks, "mixture_integral": area,
            "mixture_quadrature_error_estimate": error,
            "endpoint_note": "Use the analytic right-hand density at zero; scipy.stats.ncf.pdf in the recorded runtime reports zero at that endpoint for positive noncentrality."}


def render(output):
    checks = verify()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    labels = [r"Noise only: $F_{2,4}$ ($\lambda=0$)",
              r"Steady signal + noise: $F_{2,4}(\lambda=2)$",
              r"Strong steady signal + noise: $F_{2,4}(\lambda=20)$",
              "Switching: 90% noise + 10% strong"]
    colors = ["#61717c", "#007f86", "#b45916", "#8b43a8"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    grids = [np.linspace(0, 12, 2401), np.geomspace(.001, 300, 3000)]
    for ax, x in zip(axes, grids):
        for y, label, color, style in zip(curves(x), labels, colors, ("-", "-", "-", "--")):
            ax.plot(x, y, color=color, lw=2.3, ls=style, label=label)
        ax.set_ylabel("Probability density per unit ratio")
        ax.set_xlabel(r"Single-row ratio $r=2|z_0|^2/(|z_-|^2+|z_+|^2)$")
        ax.grid(alpha=.2)
    axes[0].set(xlim=(0, 12), ylim=(0, 1.03), title="Density on linear axes")
    axes[1].set(xscale="log", yscale="log", xlim=(.001, 300), ylim=(1e-7, 1.1), title="Density and tails on log axes")
    axes[0].legend(loc="upper right", fontsize=8.6, framealpha=.96)
    fig.suptitle("Probability densities: steady signal versus on/off switching", fontsize=14)
    fig.supxlabel("Illustrative theory only; no SDR data. Independent, equal-variance complex Gaussian noise projections.\nTeal and purple have the same mean. Densities are not renormalized over the displayed range.", fontsize=9.5)
    artifacts = {}
    for extension in ("png", "pdf"):
        path = output / f"additive-vs-mixture-density.{extension}"
        fig.savefig(path, dpi=180)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    report = {"schema": "additive-versus-mixture-density-v1", "scope": "Illustrative theory only, no measured data",
              "statistic": "R=2*abs(z0)**2/(abs(zminus)**2+abs(zplus)**2)",
              "formula": "p(r;lambda)=exp(-lambda/(r+2))*(8/(r+2)^3+8*lambda*r/(r+2)^4+lambda^2*r^2/(r+2)^5), r>=0",
              "mixture": "0.9*p(r;0)+0.1*p(r;20)", "mean_steady_lambda2": 4., "mean_mixture": 4.,
              "density_measure": "Per unit linear ratio r, including on logarithmic plotting axes",
              "histogram_comparison": "Normalize each histogram bin as count/(total observations * bin width); use all observations in the denominator even when displayed tails are omitted.",
              "verification": checks, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
              "artifacts": artifacts,
              "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html"]}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.output), indent=2))
