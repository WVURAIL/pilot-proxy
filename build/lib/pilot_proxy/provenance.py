# coding=utf-8
"""Small provenance helpers for reproducible PilotProxy products."""

from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=8)
def package_source_sha256(package_root: Path | str | None = None) -> str:
    """Hash the installed Python implementation, independent of absolute paths.

    Development versions can span many commits while retaining the same package
    version.  This digest makes checkpoint compatibility depend on the actual
    implementation that produced the product rather than only ``__version__``.

    The digest is memoized per process: a long-running scan stamps every
    product with the tree as first observed, even if source files change on
    disk mid-run (for example, patches applied while a survey is processing).
    The imported code cannot change within a process, so the first observation
    is the closest available proxy for the implementation actually running; a
    relaunched process observes the new tree.
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


# -- detector_version token policy -----------------------------------------
#
# A detector_version string is
#
#     pilot-proxy/<version> source=<tree hash> kernel=<v> kernel_sha256=<hash>
#     <schema tag> K=<K>
#
# and its tokens fall into two classes. ``pilot-proxy/<version>`` and
# ``source=<tree hash>`` both name the BUILD that produced a product: the
# release label a maintainer typed, and the digest of the implementation that
# actually ran. Neither constrains the numbers. A version bump made for a
# release (0.3.0.dev0 -> 1.0.0) changes both tokens without touching detector
# math, and conversely a tree can move many commits while staying at one
# version. Thus the label is the weaker of the two, and the source digest is
# what genuinely identifies an implementation.
#
# The remaining tokens (kernel version, kernel binary hash, schema tag and K)
# are geometry: they are what resume and cross-pilot stacking actually
# depend on, alongside the separately compared weights hashes, detector
# contract JSON, mask rule and reference placement.
#
# Both classes are recorded in full in every product, and combine reports the
# distinct build strings it stacked, so relaxing the gate loses no provenance.
DETECTOR_VERSION_BUILD_TOKEN_PREFIXES = ("pilot-proxy/", "source=")


def detector_version_geometry(version: object) -> tuple[str, ...]:
    """Return the geometry-bearing tokens of a ``detector_version`` string.

    Build-identity tokens (see ``DETECTOR_VERSION_BUILD_TOKEN_PREFIXES``) are
    dropped; everything else is preserved in order. Two products whose
    geometries compare equal were produced by the same detector configuration
    and may be resumed into or stacked together.
    """
    return tuple(
        token
        for token in str(version).split()
        if not token.startswith(DETECTOR_VERSION_BUILD_TOKEN_PREFIXES)
    )


def detector_version_build_id(version: object, digest_chars: int = 12) -> str:
    """Return a short human-readable build identity, ``<version>@<source>``.

    Used in the operator-facing notes printed when products from different
    builds are resumed into or stacked. Missing tokens render as ``?``.
    """
    label = "?"
    source = "?"
    for token in str(version).split():
        if token.startswith("pilot-proxy/"):
            label = token[len("pilot-proxy/"):] or "?"
        elif token.startswith("source="):
            source = token[len("source="):][:digest_chars] or "?"
    return f"{label}@{source}"


__all__ = [
    "DETECTOR_VERSION_BUILD_TOKEN_PREFIXES",
    "detector_version_build_id",
    "detector_version_geometry",
    "file_sha256",
    "git_blob_sha1",
    "package_source_sha256",
    "sidecar_manifest_path",
]
