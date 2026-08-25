"""Tests for required file associations."""
import json

import pytest

from pilot_proxy.archive.association import (
    associate_by_key,
    associate_by_time,
    build_association,
)


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_key_association_requires_one_companion() -> None:
    rows, result = associate_by_key(
        [{"event": "one"}, {"event": "two"}, {"event": "three"}],
        [{"event": "one", "name": "a"}, {"event": "three", "name": "a"},
         {"event": "three", "name": "b"}],
        primary_field="event",
        companion_field="event",
    )
    assert [row["status"] for row in rows] == ["matched", "missing", "ambiguous"]
    assert (result.matched, result.missing, result.ambiguous) == (1, 1, 1)


def test_time_association_uses_header_interval() -> None:
    rows, result = associate_by_time(
        [{"metadata": {"ctime": 150.0}}, {"metadata": {"ctime": 250.0}}],
        [{"metadata": {"ctime_min": 100.0, "ctime_max": 200.0}, "name": "n2"}],
        primary_time_field="metadata.ctime",
        companion_start_field="metadata.ctime_min",
        companion_end_field="metadata.ctime_max",
    )
    assert rows[0]["companion"]["name"] == "n2"
    assert rows[1]["status"] == "missing"
    assert (result.matched, result.missing) == (1, 1)


def test_manifest_is_written_before_incomplete_exit(tmp_path) -> None:
    primary = tmp_path / "primary.jsonl"
    companion = tmp_path / "companion.jsonl"
    output = tmp_path / "associated.jsonl"
    _write(primary, [{"event": "one"}, {"event": "two"}])
    _write(companion, [{"event": "one", "name": "n2"}])
    with pytest.raises(SystemExit, match="1 missing"):
        build_association(
            primary=primary,
            companion=companion,
            output=output,
            mode="key",
            primary_field="event",
            companion_field="event",
        )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["matched", "missing"]
