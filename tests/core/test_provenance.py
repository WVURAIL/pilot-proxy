# coding=utf-8
from __future__ import annotations

import hashlib

from pilot_proxy.provenance import file_sha256, sidecar_manifest_path


def test_file_sha256_returns_digest_for_existing_file(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"fstat provenance\n")

    assert file_sha256(path) == hashlib.sha256(b"fstat provenance\n").hexdigest()


def test_file_sha256_returns_none_for_missing_file(tmp_path) -> None:
    assert file_sha256(tmp_path / "missing.bin") is None
    assert file_sha256(None) is None


def test_sidecar_manifest_path_uses_weight_manifest_convention(tmp_path) -> None:
    path = tmp_path / "weights.bin"
    assert sidecar_manifest_path(path) == tmp_path / "weights.bin.manifest.json"


def test_package_source_sha256_is_memoized_per_process(tmp_path) -> None:
    from pilot_proxy.provenance import package_source_sha256

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_bytes(b"x = 1\n")
    package_source_sha256.cache_clear()
    first = package_source_sha256(pkg)

    # A mid-run change to the tree must not change the stamp of this process.
    (pkg / "a.py").write_bytes(b"x = 2\n")
    assert package_source_sha256(pkg) == first

    # A fresh process (cache cleared) observes the new tree.
    package_source_sha256.cache_clear()
    assert package_source_sha256(pkg) != first
