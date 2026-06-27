# coding=utf-8
"""Filesystem paths for source checkouts and installed package data."""

from __future__ import annotations

import os
import re
import sysconfig
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
_SOURCE_DATA_ROOT = REPO_ROOT
_INSTALLED_DATA_ROOT = Path(sysconfig.get_path("data")) / "share" / "pilot-proxy"


def _data_root() -> Path:
    source_weight = _SOURCE_DATA_ROOT / "weights" / "chime_dtv_weights_k128.bin"
    if source_weight.is_file():
        return _SOURCE_DATA_ROOT
    if _INSTALLED_DATA_ROOT.is_dir():
        return _INSTALLED_DATA_ROOT
    # Preserve actionable source-tree paths when neither location is populated.
    return _SOURCE_DATA_ROOT


DATA_ROOT = _data_root()
CUDA_DIR = REPO_ROOT / "cuda"
CONFIGS_DIR = DATA_ROOT / "configs"
GENERATED_DIR = REPO_ROOT / "generated"
WEIGHTS_DIR = DATA_ROOT / "weights"

DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "chime_dtv_weights_k128.bin"
DEFAULT_CONFIG_H = CUDA_DIR / "config.h"


def _default_lib_path() -> Path:
    candidate = CUDA_DIR / "libfstatistic.so"
    cached = Path.home() / ".cache" / "pilot_proxy" / "libfstatistic.so"
    # Prefer a locally rebuilt source-tree library except on WSL-mounted paths.
    # Wheel installs have no source-tree library and therefore use the staged cache.
    if candidate.exists() and not str(candidate).startswith("/mnt/"):
        return candidate
    if cached.exists():
        return cached
    return candidate


DEFAULT_LIB_PATH = _default_lib_path()

_MNT_PATH_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")
_WIN_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def normalize_user_path(value: str | os.PathLike[str]) -> Path:
    """Normalize common Windows / WSL path forms into the local Path format."""
    raw = str(value).strip()
    if not raw:
        return Path(raw)

    if os.name == "nt":
        match = _MNT_PATH_RE.match(raw)
        if match:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{tail}")
    else:
        match = _WIN_DRIVE_RE.match(raw)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}")
    return Path(raw)


__all__ = [
    "CUDA_DIR",
    "CONFIGS_DIR",
    "DATA_ROOT",
    "DEFAULT_CONFIG_H",
    "DEFAULT_LIB_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "GENERATED_DIR",
    "REPO_ROOT",
    "SRC_ROOT",
    "WEIGHTS_DIR",
    "normalize_user_path",
]
