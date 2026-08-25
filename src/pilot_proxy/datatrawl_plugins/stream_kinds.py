# coding=utf-8
"""Compatibility import for archive stream kinds."""
from __future__ import annotations

import sys

from pilot_proxy.archive import stream_kinds as _implementation

sys.modules[__name__] = _implementation
