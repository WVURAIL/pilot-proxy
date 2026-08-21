"""Locations for the calibration-suite scripts, following ``_paths.py``.

  PP_PER_PILOT  directory of per-pilot survey products (*.npz)
  PP_CALIB_OUT  output directory for the calibration suite
                (default: $PP_OUT/calibration)
  PP_ETA_BAO    the BAO-priced per-channel thresholds produced by
                bao-noise-tolerance (default: $PP_CALIB_OUT/tables/eta_bao.csv)

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
ETA_BAO = Path(os.environ.get("PP_ETA_BAO",
                              str(OUT / "tables" / "eta_bao.csv"))).expanduser()
