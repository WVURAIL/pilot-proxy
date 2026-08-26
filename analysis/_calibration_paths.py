"""Locations for the calibration-suite scripts, following ``_paths.py``.

  PP_PER_PILOT  directory of per-pilot survey products (*.npz)
  PP_CALIB_OUT  output directory for the calibration suite
                (default: $PP_OUT/calibration)
  PP_THRESHOLD_TABLE  science-priced per-channel thresholds

Importing this module puts ``<repo>/src`` on ``sys.path`` via ``_paths``, so
``from pilot_proxy...`` works without installing the package.
"""
from __future__ import annotations

import os
from pathlib import Path

import _paths  # noqa: F401  -- puts <repo>/src on sys.path
import _products

PER_PILOT = _products.PER_PILOT
OUT = Path(os.environ.get("PP_CALIB_OUT",
                          str(_paths.OUT / "calibration"))).expanduser()
THRESHOLD_TABLE = Path(os.environ.get(
    "PP_THRESHOLD_TABLE",
    str(OUT / "tables" / "thresholds.csv"))).expanduser()
