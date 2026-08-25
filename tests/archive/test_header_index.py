"""Tests for resumable header inspection."""
import json
from pathlib import Path

from pilot_proxy.archive.header_index import inspect_headers
from pilot_proxy.archive.interfaces import DataSource, Reader, RunContext, Unit


class _Source(DataSource):
    def __init__(self) -> None:
        self.fetches = 0

    def fetch(self, unit: Unit, dest: str) -> tuple[bool, str]:
        self.fetches += 1
        Path(dest).write_text(str(unit.meta.get("header", "ok")), encoding="utf-8")
        return True, ""


class _Reader(Reader):
    def probe(self, path: str):
        value = Path(path).read_text(encoding="utf-8")
        if value == "bad":
            raise ValueError("bad header")
        return {"value": value}


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_header_index_resumes_success_and_retries_failure(tmp_path) -> None:
    output = tmp_path / "headers.jsonl"
    scratch = tmp_path / "scratch"
    source = _Source()
    units = [
        Unit("one", "one.h5", {"header": "first", "size_bytes": 10}),
        Unit("two", "two.h5", {"header": "bad", "size_bytes": 20}),
    ]
    result = inspect_headers(
        source=source,
        reader=_Reader(),
        units=units,
        ctx=RunContext(None),
        output=output,
        scratch=scratch,
    )
    assert (result.inspected, result.failed, result.cached) == (1, 1, 0)
    assert source.fetches == 2
    assert not list(scratch.iterdir())

    units[1] = Unit("two", "two.h5", {"header": "second", "size_bytes": 20})
    result = inspect_headers(
        source=source,
        reader=_Reader(),
        units=units,
        ctx=RunContext(None),
        output=output,
        scratch=scratch,
    )
    assert (result.inspected, result.failed, result.cached) == (1, 0, 1)
    assert source.fetches == 3
    rows = {row["unit_key"]: row for row in _rows(output)}
    assert rows["one"]["metadata"] == {"value": "first"}
    assert rows["two"]["metadata"] == {"value": "second"}


def test_header_index_rechecks_changed_identity(tmp_path) -> None:
    output = tmp_path / "headers.jsonl"
    source = _Source()
    first = Unit("one", "one.h5", {"header": "first", "size_bytes": 10})
    changed = Unit("one", "one.h5", {"header": "second", "size_bytes": 11})
    for unit in (first, changed):
        inspect_headers(
            source=source,
            reader=_Reader(),
            units=[unit],
            ctx=RunContext(None),
            output=output,
            scratch=tmp_path / "scratch",
        )
    assert source.fetches == 2
    assert _rows(output)[0]["metadata"] == {"value": "second"}
