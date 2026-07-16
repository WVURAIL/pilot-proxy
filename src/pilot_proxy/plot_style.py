# coding=utf-8
"""Shared Matplotlib style for PilotProxy figures."""

from __future__ import annotations

import os
from pathlib import Path


def _command_available(command: str) -> bool:
    command_text = str(command)
    has_path_separator = any(
        separator and separator in command_text for separator in (os.sep, os.altsep)
    )
    if not command_text or has_path_separator:
        return False
    return any(
        (Path(directory) / command_text).exists()
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
    )


def setup_matplotlib(*, force_agg: bool = True):
    """Configure Matplotlib for LaTeX-style PilotProxy plots.

    External TeX rendering is opt-in through the PILOT_PROXY_USE_TEX environment
    variable and only used when the TeX helper commands are available. Otherwise, Matplotlib's
    Computer Modern mathtext renderer gives the same visual language without a
    TeX runtime dependency.
    """
    import matplotlib

    if force_agg:
        matplotlib.use("Agg", force=True)
    use_tex = (
        os.environ.get("PILOT_PROXY_USE_TEX", "0") == "1"
        and _command_available("latex")
        and _command_available("dvipng")
    )
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": bool(use_tex),
        }
    )
    if use_tex:
        # Match the journal build (mnras/rasti classes load newtxtext/newtxmath)
        # when the newtx fonts are installed; otherwise fall back to Computer
        # Modern, which matches the article-class draft build. An unconditional
        # newtx preamble crashes savefig on hosts without newtxtext.sty.
        import subprocess

        try:
            has_newtx = (
                subprocess.run(
                    ["kpsewhich", "newtxtext.sty"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                != ""
            )
        except OSError:
            has_newtx = False
        preamble = r"\usepackage{amsmath}"
        if has_newtx:
            preamble += r"\usepackage{newtxtext,newtxmath}"
        matplotlib.rcParams["text.latex.preamble"] = preamble

    import matplotlib.pyplot as plt

    return plt
