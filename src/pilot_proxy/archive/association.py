"""Associate primary files with required companion files."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AssociationResult:
    """Summary of one association pass."""

    total: int
    matched: int
    missing: int
    ambiguous: int


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _field(row: Mapping[str, object], path: str) -> object:
    value: object = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _write(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def associate_by_key(
    primary: Iterable[Mapping[str, object]],
    companions: Iterable[Mapping[str, object]],
    *,
    primary_field: str,
    companion_field: str,
) -> tuple[list[dict[str, object]], AssociationResult]:
    """Match rows only when a key identifies exactly one companion."""
    companion_map: dict[object, list[Mapping[str, object]]] = {}
    for row in companions:
        companion_map.setdefault(_field(row, companion_field), []).append(row)
    output = []
    matched = missing = ambiguous = 0
    for row in primary:
        key = _field(row, primary_field)
        matches = companion_map.get(key, [])
        if len(matches) == 1:
            status = "matched"
            companion: Mapping[str, object] | None = matches[0]
            matched += 1
        elif matches:
            status = "ambiguous"
            companion = None
            ambiguous += 1
        else:
            status = "missing"
            companion = None
            missing += 1
        output.append(
            {
                "status": status,
                "match_value": key,
                "primary": dict(row),
                "companion": dict(companion) if companion is not None else None,
            }
        )
    result = AssociationResult(len(output), matched, missing, ambiguous)
    return output, result


def associate_by_time(
    primary: Iterable[Mapping[str, object]],
    companions: Iterable[Mapping[str, object]],
    *,
    primary_time_field: str,
    companion_start_field: str,
    companion_end_field: str,
) -> tuple[list[dict[str, object]], AssociationResult]:
    """Match a timestamp only when exactly one companion interval contains it."""
    intervals = [
        (
            float(_field(row, companion_start_field)),
            float(_field(row, companion_end_field)),
            row,
        )
        for row in companions
    ]
    output = []
    matched = missing = ambiguous = 0
    for row in primary:
        timestamp = float(_field(row, primary_time_field))
        matches = [entry for entry in intervals if entry[0] <= timestamp <= entry[1]]
        if len(matches) == 1:
            status = "matched"
            companion: Mapping[str, object] | None = matches[0][2]
            matched += 1
        elif matches:
            status = "ambiguous"
            companion = None
            ambiguous += 1
        else:
            status = "missing"
            companion = None
            missing += 1
        output.append(
            {
                "status": status,
                "match_value": timestamp,
                "primary": dict(row),
                "companion": dict(companion) if companion is not None else None,
            }
        )
    result = AssociationResult(len(output), matched, missing, ambiguous)
    return output, result


def build_association(
    *,
    primary: str | Path,
    companion: str | Path,
    output: str | Path,
    mode: str,
    primary_field: str,
    companion_field: str | None = None,
    companion_start_field: str | None = None,
    companion_end_field: str | None = None,
    allow_unmatched: bool = False,
) -> AssociationResult:
    """Build one required-companion manifest from two JSONL inputs."""
    primary_rows = _read_jsonl(primary)
    companion_rows = _read_jsonl(companion)
    if mode == "key":
        if companion_field is None:
            raise ValueError("key matching needs a companion field")
        rows, result = associate_by_key(
            primary_rows,
            companion_rows,
            primary_field=primary_field,
            companion_field=companion_field,
        )
    elif mode == "time":
        if companion_start_field is None or companion_end_field is None:
            raise ValueError("time matching needs companion start and end fields")
        rows, result = associate_by_time(
            primary_rows,
            companion_rows,
            primary_time_field=primary_field,
            companion_start_field=companion_start_field,
            companion_end_field=companion_end_field,
        )
    else:
        raise ValueError(f"unknown association mode: {mode}")
    _write(output, rows)
    if (result.missing or result.ambiguous) and not allow_unmatched:
        raise SystemExit(
            f"association incomplete: {result.missing} missing, "
            f"{result.ambiguous} ambiguous"
        )
    return result
