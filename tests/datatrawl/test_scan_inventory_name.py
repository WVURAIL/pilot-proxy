# coding=utf-8
"""Named-inventory path handling for pilot-proxy chime-scan."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pilot_proxy.datatrawl_plugins.scan import _named_inventory_path


def test_named_inventory_path_uses_source_root(tmp_path: Path) -> None:
    assert _named_inventory_path("chime-pilots", tmp_path) == (
        tmp_path / "data" / "chime-pilots" / "inventory.jsonl"
    )


def test_named_inventory_path_defaults_to_cwd(tmp_path: Path, monkeypatch) -> None:
    # Exercise the documented legacy fallback: no source root and no
    # datatrawl.invpaths resolver installed. Forcing the ImportError branch
    # keeps this test meaningful on hosts whose datatrawl ships invpaths.
    monkeypatch.setitem(sys.modules, "datatrawl.invpaths", None)
    monkeypatch.chdir(tmp_path)
    assert _named_inventory_path("chime-ch614-706", None) == (
        tmp_path / "data" / "chime-ch614-706" / "inventory.jsonl"
    )


def test_named_inventory_path_delegates_to_datatrawl_invpaths() -> None:
    # When datatrawl ships invpaths, it is the single source of truth for
    # named-inventory locations (canonical root + legacy fallbacks).
    invpaths = pytest.importorskip("datatrawl.invpaths")
    expected = Path(invpaths.resolve_inventory("chime-ch614-706"))
    assert _named_inventory_path("chime-ch614-706", None) == expected
