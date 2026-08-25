"""Tests for the outrigger N2 reader."""
from datetime import timezone

import h5py
import numpy as np
import pytest

from pilot_proxy.archive.interfaces import RunContext, UnreadableUnitError
from pilot_proxy.outrigger import OutriggerN2Reader


def _write_n2(path, *, n_freq: int = 1024) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vis", shape=(n_freq, 3, 2), dtype=np.complex64)
        index = handle.create_group("index_map")
        times = np.zeros(2, dtype=[("ctime", "f8")])
        times["ctime"] = [100.0, 200.0]
        index.create_dataset("time", data=times)


def test_probe_reports_time_and_frequency_coverage(tmp_path) -> None:
    path = tmp_path / "n2.h5"
    _write_n2(path)
    result = OutriggerN2Reader().probe(str(path))
    assert result == {
        "shape": (1024, 3, 2),
        "ctime_min": 100.0,
        "ctime_max": 200.0,
        "freq_start": 0,
        "freq_end": 1024,
    }


def test_visibility_iteration_is_frequency_bounded(tmp_path) -> None:
    path = tmp_path / "n2.h5"
    _write_n2(path)
    reader = OutriggerN2Reader()
    chunks = list(reader.iter_arrays(str(path), RunContext(None), freq_chunk=300))
    assert [(row["freq_start"], row["freq_end"]) for row in chunks] == [
        (0, 300),
        (300, 600),
        (600, 900),
        (900, 1024),
    ]
    assert all(np.asarray(row["vis"]).shape[0] <= 300 for row in chunks)


def test_probe_rejects_wrong_frequency_shape(tmp_path) -> None:
    path = tmp_path / "n2.h5"
    _write_n2(path, n_freq=8)
    with pytest.raises(UnreadableUnitError, match="expected 1024"):
        OutriggerN2Reader().probe(str(path))


def test_folder_name_parsing() -> None:
    result = OutriggerN2Reader.parse_folder_name("20260102T030405Z_gbostack_corr")
    assert result["site"] == "gbo"
    assert result["variant"] == "stack"
    assert result["timestamp"].tzinfo is timezone.utc
