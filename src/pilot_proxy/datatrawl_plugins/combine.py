# coding=utf-8
"""Compatibility import for archive product combination."""
from __future__ import annotations

import sys

from pilot_proxy.archive import combine as _implementation

sys.modules[__name__] = _implementation
