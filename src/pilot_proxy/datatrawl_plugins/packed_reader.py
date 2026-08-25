# coding=utf-8
"""Compatibility import for the packed baseband reader."""
from __future__ import annotations

import sys

from pilot_proxy.archive import packed_reader as _implementation

sys.modules[__name__] = _implementation
