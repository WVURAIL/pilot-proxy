# coding=utf-8
"""Compatibility import for the archive scan workflow."""
from __future__ import annotations

import sys

from pilot_proxy.archive import scan as _implementation

sys.modules[__name__] = _implementation
