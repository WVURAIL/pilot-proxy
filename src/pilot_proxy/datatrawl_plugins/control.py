# coding=utf-8
"""Compatibility import for the control analyzer."""
from __future__ import annotations

import sys

from pilot_proxy.archive import control as _implementation

sys.modules[__name__] = _implementation
