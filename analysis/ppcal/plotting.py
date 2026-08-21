# coding=utf-8
"""Figure style and shared drawing helpers.

Palette and quiet-chart rules are taken from ``analysis/_style.py`` so these
plates sit alongside the survey and manuscript figures without restyling.
cmr10 lacks some unicode glyphs: ASCII hyphens only, and
``axes.unicode_minus`` stays off.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# sequential ramp for occupancy and spectrograms: paper white -> ink blue ->
# warning orange, so "no excess" is invisible and a live carrier is loud.
OCC_CMAP = LinearSegmentedColormap.from_list(
    "ppcal_occ",
    ["#f7f7f4", "#cfe0f2", "#7fb0e0", "#2a78d6", "#1f4f96",
     "#7a3d9c", "#c9452f", "#eda100"])

KEEP_COLOR = SERIES[2]
MASK_COLOR = SERIES[3]
EXCISE_COLOR = CRITICAL


def setup_style():
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


def save_fig(fig, path, pdf=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def year_ticks(ax, months, axis="x"):
    """Label January of each year on a survey-month axis."""
    from .products import M0
    pos, lab = [], []
    for j, m in enumerate(months):
        if (M0 + m) % 12 == 0:
            pos.append(j)
            lab.append(str((M0 + m) // 12))
    if axis == "x":
        ax.set_xticks(pos)
        ax.set_xticklabels(lab)
    else:
        ax.set_yticks(pos)
        ax.set_yticklabels(lab)


def annotate_channel(ax, text, loc=(0.015, 0.86), size=10, weight="semibold"):
    ax.annotate(text, xy=loc, xycoords="axes fraction", fontsize=size,
                color=INK, fontweight=weight)


def disposition_color(verdict):
    return EXCISE_COLOR if verdict == "excise" else KEEP_COLOR


def db(x, floor=1e-6):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))
