"""End-to-end survey() coverage for the bare-minoc-replica patch, plus the
files() contract branches that no test in tests/archive reaches.

Two things measured coverage showed were missing (stdlib sys.monitoring line
trace over the whole tests/archive run, 2026-08-28):

1. Every survey()-level test fakes ``DATATRAIL.common_path`` itself
   (tests/archive/test_survey_contract_refusal.py:106,
   tests/archive/test_survey_empty_events.py, ...), i.e. they stub out the
   exact function the patch changed. Nothing drives survey() through the real
   ``files()`` normalization, so no test can observe what a restored bare
   replica does to the rows survey writes. The tests here fake only the one
   real seam -- ``dt.subprocess.run`` -- so ``_restore_collection`` runs for
   real.

2. datatrail_client.files() lines 388, 390-391, 397-398, 401 and 403-404 --
   the file_replica_locations/minoc shape branches -- were never executed.

Offline: no network, no live datatrail, no CADC.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.datatrail_client import DatatrailContractError
from pilot_proxy.archive.interfaces import RunContext
from pilot_proxy.archive.sources import cadc as cadc_datatrail
from pilot_proxy.chime.baseband_reader import (MINIMUM_ARCHIVE_BYTES,
                                               baseband_filename)

SCOPE = "chime.event.baseband.raw"
EVENT = "100260502"
NEIGHBOUR = "100260503"
DAY = "data/chime/baseband/raw/2020/07/15"
EVENT_DIR = f"{DAY}/astro_{EVENT}"
FREQ_IDS = [0, 1]

# What actually exists in CADC for this event, as cadcinfo would answer.
ARCHIVE = {f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, fid)}"
           for fid in FREQ_IDS}


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_ps(monkeypatch, minoc):
    """Answer `datatrail ps ... --json` with this minoc replica list.

    Patches the adapter's ONE subprocess boundary, so files(),
    _restore_collection() and the common-path split all run for real.
    """
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))
    payload = json.dumps({
        "dataset": EVENT, "scope": SCOPE, "policies": {},
        "files": {"file_replica_locations": {"minoc": list(minoc)}},
    })
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))


def _fake_archive(monkeypatch):
    """cadcinfo: a real size for objects that exist, a definitive absence
    (size None, err None -- an answer, not an outage) for everything else."""
    seen = []

    def fake_size(self, uri, *a, **k):
        seen.append(uri)
        if uri in ARCHIVE:
            return MINIMUM_ARCHIVE_BYTES + 1, None
        return None, None

    monkeypatch.setattr(cadc_datatrail.CadcDatatrailSource,
                        "_cadc_size", fake_size)
    monkeypatch.setattr(cadc_datatrail, "_enumerate_events",
                        lambda *a, **k: {(SCOPE, EVENT): ["ds"]})
    return seen


def _survey(out_dir):
    ctx = RunContext(instrument=None, selection=None,
                     options={"scope": SCOPE, "freq_ids": list(FREQ_IDS)})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        source = cadc_datatrail.CadcDatatrailSource()
        source.survey(ctx, str(out_dir))
    return buf.getvalue()


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ==========================================================================
# The documented real case: every replica bare, all in the event directory.
# survey() must produce exactly the rows it produced before Datatrail
# dropped the prefix -- same common_path, same URIs.
# ==========================================================================
def test_all_bare_replicas_survey_to_the_same_rows(monkeypatch, tmp_path):
    _fake_ps(monkeypatch, [f"{EVENT_DIR}/{baseband_filename(EVENT, fid)}"
                           for fid in FREQ_IDS])
    probed = _fake_archive(monkeypatch)
    text = _survey(tmp_path)

    rows = _rows(tmp_path / "inventory.jsonl")
    assert [r["name"] for r in rows] == [baseband_filename(EVENT, fid)
                                         for fid in FREQ_IDS]
    assert {r["common_path"] for r in rows} == {f"cadc:CHIMEFRB/{EVENT_DIR}"}
    assert {r["obs_date"] for r in rows} == {"2020-07-15"}
    # every probe went to a URI that exists -- the restored prefix is the one
    # cadcinfo resolves, not merely one that passes the collection check
    assert set(probed) == ARCHIVE
    assert "0 contract-refused" in text
    assert "accepted-empty" in text and "0 accepted-empty" in text


# ==========================================================================
# GAP the suite does not cover: restoration widens the population reaching
# os.path.commonprefix, which is CHARACTER-wise, not path-wise. A bare
# replica from a NEIGHBOURING event directory no longer refuses -- it
# silently shortens the common path from the event directory to the day
# directory. survey() then probes .../2020/07/15/baseband_<ev>_<fid>.h5,
# gets a definitive absence for every freq_id, and books 0 rows / 0 errors.
# The bytes are there; only the derived path is not.
#
# Pre-patch this input was a refusal, which is loud and ledgered with the
# offending URI. Post-patch it is a silent empty that burns _MAX_ATTEMPTS
# resumes and then writes the event off. xfail(strict) so the suite stays
# green while the hazard is pinned: fixing it (path-wise common prefix, or
# refusing a restored replica that changes the common-path depth) flips
# this to pass.
# ==========================================================================
@pytest.mark.xfail(strict=True, reason=(
    "restored bare replica from a sibling directory shortens common_path to "
    "the day directory; survey probes one level too high and writes 0 rows "
    "for an event whose bytes are in CADC"))
def test_bare_replica_from_a_sibling_directory_does_not_silently_empty(
        monkeypatch, tmp_path):
    _fake_ps(monkeypatch, [
        f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, 0)}",
        f"{DAY}/astro_{NEIGHBOUR}/{baseband_filename(NEIGHBOUR, 1)}",
    ])
    _fake_archive(monkeypatch)
    _survey(tmp_path)
    assert _rows(tmp_path / "inventory.jsonl"), (
        "survey wrote no rows for an event whose bytes exist in CADC")


def test_sibling_directory_case_churns_then_is_written_off(
        monkeypatch, tmp_path):
    """Characterization of what actually happens today, so the write-off is
    visible in the suite rather than only in a production ledger.

    The shortened path also defeats _DATE_RE (cadc.py:111), which needs a
    trailing '/' after the day and therefore cannot match a common path that
    ENDS at the day directory. obs_date is 'unknown', so the aged-out
    fast path never fires and the event burns every retry first.
    """
    _fake_ps(monkeypatch, [
        f"cadc:CHIMEFRB/{EVENT_DIR}/{baseband_filename(EVENT, 0)}",
        f"{DAY}/astro_{NEIGHBOUR}/{baseband_filename(NEIGHBOUR, 1)}",
    ])
    probed = _fake_archive(monkeypatch)

    text = _survey(tmp_path)
    assert _rows(tmp_path / "inventory.jsonl") == []
    assert "0 contract-refused" in text          # the loud path is gone
    assert "1 resolved-but-empty (retry next run)" in text
    assert _rows(tmp_path / "no_files_events.jsonl") == []
    # the probes went one directory too high, so none of them could resolve
    assert set(probed).isdisjoint(ARCHIVE)
    assert set(probed) == {f"cadc:CHIMEFRB/{DAY}/"
                           f"{baseband_filename(EVENT, fid)}"
                           for fid in FREQ_IDS}

    # ... and it is undatable, so it churns to _MAX_ATTEMPTS before the
    # event is written off as never having been in CADC storage.
    for _ in range(cadc_datatrail._MAX_ATTEMPTS - 1):
        text = _survey(tmp_path)
    ledger = _rows(tmp_path / "no_files_events.jsonl")
    assert [r["reason"] for r in ledger] == ["max-attempts"]
    assert ledger[0]["obs_date"] == "unknown"
    assert ledger[0]["common_path"] == f"cadc:CHIMEFRB/{DAY}"


# ==========================================================================
# files(): the file_replica_locations / minoc shape branches. Measured as
# never executed by tests/archive before this file
# (datatrail_client.py:388, 390-391, 397-398, 401, 403-404).
# ==========================================================================
def _files(monkeypatch, files_value):
    monkeypatch.setitem(sys.modules, "dtcli", types.ModuleType("dtcli"))
    monkeypatch.setattr(dt, "_cli_version", lambda: (0, 11, 0))
    payload = json.dumps({"dataset": "d", "scope": "s", "policies": {},
                          "files": files_value})
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))
    return dt.DATATRAIL.files("s", "d", retries=0)


def test_absent_replica_locations_is_no_data_not_a_refusal(monkeypatch):
    # line 388: "file_replica_locations": null -> queried OK, no bytes
    assert _files(monkeypatch, {"file_replica_locations": None}) == (
        None, [], True)


def test_empty_minoc_list_is_no_data_not_a_refusal(monkeypatch):
    # line 401: an empty list is an answer, never an outage
    assert _files(monkeypatch, {"file_replica_locations": {"minoc": []}}) == (
        None, [], True)


@pytest.mark.parametrize("locations,fragment", [
    (["cadc:CHIMEFRB/data/x/y.h5"], "non-object 'file_replica_locations'"),
    ("cadc:CHIMEFRB/data/x/y.h5", "non-object 'file_replica_locations'"),
])
def test_non_object_replica_locations_refuses(monkeypatch, locations,
                                              fragment):
    # lines 390-391
    with pytest.raises(DatatrailContractError, match=fragment):
        _files(monkeypatch, {"file_replica_locations": locations})


def test_non_list_minoc_refuses(monkeypatch):
    # lines 397-398: a bare string would otherwise be iterated per-character
    with pytest.raises(DatatrailContractError, match="non-list 'minoc'"):
        _files(monkeypatch, {"file_replica_locations":
                             {"minoc": "cadc:CHIMEFRB/data/x/y.h5"}})


@pytest.mark.parametrize("bad", [None, "", "   ", 7, ["nested"], {}])
def test_malformed_minoc_entry_refuses(monkeypatch, bad):
    # lines 403-404: one bad entry poisons the list; never a partial answer
    with pytest.raises(DatatrailContractError, match="malformed 'minoc'"):
        _files(monkeypatch, {"file_replica_locations": {"minoc": [
            "cadc:CHIMEFRB/data/x/y.h5", bad]}})
