"""Resumable file-header inspection."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .interfaces import DataSource, Reader, RunContext, Unit

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeaderIndexResult:
    """Summary of one inspection pass."""

    total: int
    cached: int
    inspected: int
    failed: int


def _fingerprint(unit: Unit) -> str:
    metadata = unit.meta or {}
    identity = {
        "key": unit.key,
        "name": unit.name,
        "size_bytes": metadata.get("size_bytes"),
        "checksum": metadata.get("checksum") or metadata.get("md5"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"invalid header index record at line {line_number}")
            key = record.get("unit_key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"missing unit key at line {line_number}")
            records[key] = record
    return records


def _write(path: Path, records: Mapping[str, Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(records):
                handle.write(json.dumps(records[key], sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _stage_name(unit: Unit) -> str:
    digest = hashlib.sha256(unit.key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{Path(unit.name).name or 'file'}"


def inspect_headers(
    *,
    source: DataSource,
    reader: Reader,
    units: Iterable[Unit],
    ctx: RunContext,
    output: str | Path,
    scratch: str | Path,
) -> HeaderIndexResult:
    """Inspect one staged file at a time and cache successful results."""
    output_path = Path(output).expanduser()
    scratch_path = Path(scratch).expanduser()
    scratch_path.mkdir(parents=True, exist_ok=True)
    records = _load(output_path)
    selected = list(units)
    cached = 0
    inspected = 0
    failed = 0

    for unit in selected:
        fingerprint = _fingerprint(unit)
        previous = records.get(unit.key)
        if (
            previous is not None
            and previous.get("status") == "ok"
            and previous.get("fingerprint") == fingerprint
        ):
            cached += 1
            continue

        destination = scratch_path / _stage_name(unit)
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "unit_key": unit.key,
            "unit_name": unit.name,
            "fingerprint": fingerprint,
        }
        try:
            fetch_result = source.fetch(unit, str(destination))
            if (
                not isinstance(fetch_result, tuple)
                or len(fetch_result) != 2
                or not isinstance(fetch_result[0], bool)
                or not isinstance(fetch_result[1], str)
            ):
                raise TypeError("source.fetch must return (bool, str)")
            ok, detail = fetch_result
            if not ok:
                raise RuntimeError(detail or "fetch failed")
            metadata = reader.probe(str(destination))
            if not isinstance(metadata, Mapping):
                raise TypeError("reader.probe must return a mapping")
            record.update(status="ok", metadata=dict(metadata))
            inspected += 1
        except Exception as exc:  # noqa: BLE001
            record.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
        finally:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        records[unit.key] = record
        _write(output_path, records)

    return HeaderIndexResult(
        total=len(selected),
        cached=cached,
        inspected=inspected,
        failed=failed,
    )
