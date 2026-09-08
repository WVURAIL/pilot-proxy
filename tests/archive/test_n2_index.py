"""N2 index entry-point tests using real local HDF5 files."""

import json

import h5py
import numpy as np
import pytest

from pilot_proxy.outrigger.index import build_n2_index
from pilot_proxy.outrigger import n2_reader


def write_n2(path, times=(100.0, 200.0)):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vis", shape=(1024, 3, 2), dtype=np.complex64)
        index = np.zeros(2, dtype=[("ctime", "f8")])
        index["ctime"] = times
        handle.create_dataset("index_map/time", data=index)


@pytest.fixture
def local_index(tmp_path, monkeypatch):
    # These fixtures are uncompressed; compressed-file availability is tested separately.
    monkeypatch.setattr(n2_reader, "hdf5plugin", object())
    source = tmp_path / "input"
    source.mkdir()
    write_n2(source / "n2.h5")
    return source, tmp_path / "headers.jsonl"


def test_index_resumes_and_invalidates_same_size_changed_input(local_index, capsys):
    source, output = local_index
    assert build_n2_index(input_dir=source, output=output) == output
    assert "1 inspected, 0 cached, 0 failed" in capsys.readouterr().out
    before = output.read_bytes()
    build_n2_index(input_dir=source, output=output)
    assert "0 inspected, 1 cached, 0 failed" in capsys.readouterr().out
    assert output.read_bytes() == before
    path = source / "n2.h5"
    size = path.stat().st_size
    with h5py.File(path, "r+") as handle:
        times = handle["index_map/time"][:]
        times["ctime"] = [300, 400]
        handle["index_map/time"][:] = times
    assert path.stat().st_size == size
    build_n2_index(input_dir=source, output=output)
    assert "1 inspected, 0 cached, 0 failed" in capsys.readouterr().out
    assert json.loads(output.read_text())["metadata"]["ctime_min"] == 300
    assert path.exists()
    assert list((output.parent / "_n2_staging").iterdir()) == []


def test_index_partial_failure_and_repair(local_index):
    source, output = local_index
    bad = source / "broken.h5"
    bad.write_bytes(b"not HDF5")
    with pytest.raises(SystemExit, match="incomplete"):
        build_n2_index(input_dir=source, output=output)
    assert {
        row["status"] for row in map(json.loads, output.read_text().splitlines())
    } == {"ok", "failed"}
    assert build_n2_index(input_dir=source, output=output, allow_partial=True) == output
    write_n2(bad)
    build_n2_index(input_dir=source, output=output)
    assert all(
        row["status"] == "ok"
        for row in map(json.loads, output.read_text().splitlines())
    )
    assert bad.exists()


@pytest.mark.parametrize(
    "options", [{}, {"input_dir": ".", "inventory": "inventory.jsonl"}]
)
def test_index_requires_exactly_one_source(tmp_path, options):
    with pytest.raises(SystemExit, match="exactly one"):
        build_n2_index(output=tmp_path / "index.jsonl", **options)


def test_index_reports_source_and_reader_preflight_failures(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="n2-index source"):
        build_n2_index(input_dir=tmp_path / "missing", output=tmp_path / "index.jsonl")
    monkeypatch.setattr(n2_reader, "hdf5plugin", None)
    with pytest.raises(SystemExit, match="n2-index reader.*hdf5plugin"):
        build_n2_index(input_dir=tmp_path, output=tmp_path / "index.jsonl")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_index_refuses_nonfinite_time_coverage(local_index, invalid):
    source, output = local_index
    write_n2(source / "n2.h5", (100.0, invalid))
    with pytest.raises(SystemExit, match="incomplete"):
        build_n2_index(input_dir=source, output=output)
    assert json.loads(output.read_text())["status"] == "failed"
