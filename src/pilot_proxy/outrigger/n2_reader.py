"""Reader for outrigger N2 visibility files."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

import h5py

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

from pilot_proxy.archive.interfaces import (
    READY,
    STREAM_VISIBILITY_CHUNK,
    PluginInfo,
    Reader,
    RunContext,
)
from pilot_proxy.chime.unreadable import unreadable_file

_FOLDER_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6})Z_(?P<site>gbo|hco|kko)"
    r"(?P<variant>stack|rfi|cal|subband)?_corr$"
)
_EXPECTED_N_FREQ = 1024
_DEFAULT_FREQ_CHUNK = 128


def _visibility_dataset(handle: h5py.File, path: str):
    if "vis" not in handle:
        raise KeyError(f"expected dataset 'vis' in {path}")
    vis = handle["vis"]
    if vis.ndim != 3:
        raise ValueError(f"expected a 3-D vis dataset in {path}, got {vis.shape}")
    if vis.dtype.kind != "c":
        raise ValueError(f"expected a complex vis dataset in {path}, got {vis.dtype}")
    if vis.shape[0] != _EXPECTED_N_FREQ:
        raise ValueError(
            f"expected {_EXPECTED_N_FREQ} frequency channels in {path}, "
            f"got {vis.shape[0]}"
        )
    return vis


class OutriggerN2Reader(Reader):
    """Read GBO, HCO, and KKO N2 visibility data."""

    survey_schema = 1
    info = PluginInfo(
        name="outrigger-n2",
        kind="reader",
        instruments=("gbo", "hco", "kko"),
        summary="Read outrigger N2 visibility files and their time coverage.",
        status=READY,
        requires=("h5py", "hdf5plugin"),
        stream_kind=STREAM_VISIBILITY_CHUNK,
    )

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        if hdf5plugin is None:
            return False, [
                "hdf5plugin is required for compressed N2 files; "
                "install the n2 extra"
            ]
        return True, []

    @staticmethod
    def parse_folder_name(name: str) -> Mapping[str, object]:
        """Return the site, variant, and start time encoded in a folder name."""
        match = _FOLDER_RE.match(Path(name).name)
        if match is None:
            raise ValueError(f"unrecognized N2 folder name: {name}")
        timestamp = datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%S")
        return {
            "timestamp": timestamp.replace(tzinfo=timezone.utc),
            "site": match.group("site"),
            "variant": match.group("variant") or "plain",
        }

    def probe(self, path: str) -> Mapping[str, object]:
        """Read file shape and time coverage without loading visibility data."""
        with unreadable_file():
            with h5py.File(path, "r") as handle:
                shape = _visibility_dataset(handle, path).shape
                times = handle["index_map/time"]["ctime"]
                if times.ndim != 1 or times.shape[0] != shape[2] or not times.size:
                    raise ValueError(
                        "index_map/time must be non-empty and match the vis time axis"
                    )
                return {
                    "shape": tuple(int(value) for value in shape),
                    "ctime_min": float(times.min()),
                    "ctime_max": float(times.max()),
                    "freq_start": 0,
                    "freq_end": int(shape[0]),
                }

    def iter_arrays(
        self,
        path: str,
        ctx: RunContext,
        freq_chunk: int = _DEFAULT_FREQ_CHUNK,
    ) -> Iterator[Mapping[str, object]]:
        """Yield visibility data in bounded frequency chunks."""
        if isinstance(freq_chunk, bool) or not isinstance(freq_chunk, int):
            raise TypeError("freq_chunk must be an integer")
        if freq_chunk < 1:
            raise ValueError("freq_chunk must be positive")
        with unreadable_file():
            with h5py.File(path, "r") as handle:
                vis = _visibility_dataset(handle, path)
                for start in range(0, vis.shape[0], freq_chunk):
                    end = min(start + freq_chunk, vis.shape[0])
                    yield {
                        "vis": vis[start:end],
                        "freq_start": start,
                        "freq_end": end,
                    }
