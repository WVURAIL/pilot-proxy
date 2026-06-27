# coding=utf-8
"""Small provenance helpers for reproducible PilotProxy products."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path | str | None) -> str | None:
    """Return the SHA256 hex digest for an existing file, or None otherwise."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path | str | None) -> str | None:
    """Return Git's SHA-1 object id for an existing file's blob contents."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    payload = file_path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object id


def package_source_sha256(package_root: Path | str | None = None) -> str:
    """Hash the installed Python implementation, independent of absolute paths.

    Development versions can span many commits while retaining the same package
    version.  This digest makes checkpoint compatibility depend on the actual
    implementation that produced the product, not only ``__version__``.
    """
    root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).resolve()
    )
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sidecar_manifest_path(path: Path | str | None) -> Path | None:
    """Return the conventional manifest sidecar path."""
    if path is None:
        return None
    return Path(f"{Path(path)}.manifest.json")


__all__ = [
    "file_sha256",
    "git_blob_sha1",
    "package_source_sha256",
    "sidecar_manifest_path",
]
