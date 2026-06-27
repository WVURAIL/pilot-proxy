# coding=utf-8
"""Named-inventory path handling for pilot-proxy chime-scan."""

from __future__ import annotations

from pathlib import Path

from pilot_proxy.datatrawl_plugins.scan import _named_inventory_path


def test_named_inventory_path_uses_source_root(tmp_path: Path) -> None:
    assert _named_inventory_path("chime-pilots", tmp_path) == (
        tmp_path / "data" / "chime-pilots" / "inventory.jsonl"
    )


def test_named_inventory_path_defaults_to_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _named_inventory_path("chime-ch614-706", None) == (
        tmp_path / "data" / "chime-ch614-706" / "inventory.jsonl"
    )
