"""Shared figure style for the survey-plate scripts.

The palette and quiet-chart specs match the RFIsher manuscript figures (this
module began as a subset of ``rfisher.plots``), so plates
generated from either repository sit together without restyling: 2px lines,
hairline solid gridlines, recessive axes, no dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def setup_style() -> None:
    """Quiet-chart styling with Computer Modern text to match the LaTeX
    manuscript (matplotlib's bundled cmr10 + 'cm' mathtext; no external
    TeX required, so the figures render identically for every tool user).
    cmr10 lacks some unicode glyphs: use ASCII hyphens and $\\times$ in
    labels, and keep unicode_minus off."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10.5, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2.0, "lines.solid_joinstyle": "round",
        "lines.solid_capstyle": "round", "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path
