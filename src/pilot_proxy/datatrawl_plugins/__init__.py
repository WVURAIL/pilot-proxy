# coding=utf-8
"""datatrawl plugins exported by pilot-proxy.

These modules make pilot-proxy's CHIME analyses runnable through the datatrawl
streaming engine (storage-safe fetch -> read -> analyze -> evict), so the *same*
analyzer runs on a local 10 s chunk (``--source local``) and storage-safely
across the full CADC/Datatrail archive (``--source cadc-datatrail``).

datatrawl stays a separate, shareable dependency; only the analysis (the
Analyzer) lives here. The concrete plugin modules are advertised through the
``datatrawl.plugins`` entry-point group in ``pyproject.toml`` and are imported by
datatrawl when datatrawl is installed. This package initializer intentionally
does not import those modules, so lightweight helpers such as
``pilot_proxy.datatrawl_plugins.scan._named_inventory_path`` remain importable in
environments that do not install datatrawl.
"""
from __future__ import annotations

__all__ = ["scan"]
