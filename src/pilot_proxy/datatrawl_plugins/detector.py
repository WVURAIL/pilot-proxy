# coding=utf-8
"""Compatibility import for the detector analyzer."""
from __future__ import annotations

import sys

from pilot_proxy.archive import detector as _implementation

sys.modules[__name__] = _implementation
