# coding=utf-8
"""Compatibility import for archive channel helpers."""
from __future__ import annotations

import sys

from pilot_proxy.archive import chime_coarse as _implementation

sys.modules[__name__] = _implementation
