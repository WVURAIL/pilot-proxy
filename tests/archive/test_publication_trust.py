#!/usr/bin/env python3
"""Publication-trust gates for the cadc-datatrail survey.

These are the checks an independent referee would demand before trusting a
published inventory, split into three kinds:

  * PASSING tests lock a property the code already has but never asserted
    (view determinism, the schema-2 migration arithmetic, the frozen bundle's
    internal accounting).

  * ``xfail(strict=True)`` tests are REAL, MEASURED GAPS. They are not
    aspirational: each one fails today for the reason in its docstring, and
    flips to XPASS -- a suite failure -- the moment the gap is closed. That
    makes every gap machine-tracked instead of prose in a report.

  * Bundle tests skip when the referenced artifact is not on this host, so the
    file is runnable from a clean checkout.

No network: the datatrail subprocess boundary is faked exactly as in
tests/archive/test_datatrail_adapter.py.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from pilot_proxy.archive import datatrail_client as dt
from pilot_proxy.archive.sources import cadc as src
from pilot_proxy.archive.sources.cadc_inventory import parse_row
from pilot_proxy.archive.survey_state import SurveyStore
from pilot_proxy.chime.baseband_reader import baseband_filename


# CHIME production geometry: one packed byte per feed per sample.
NFFT, N_FEEDS = 16384, 2048
BYTES_PER_FRAME = NFFT * N_FEEDS

FROZEN_BUNDLE = Path("/home/djg/rail/archive_inputs/chime-pilots-v5")
JULY_REFERENCES = Path(
    "/mnt/c/Users/dylan/Downloads/Datatrawl-Inventories/Datatrawl-Inventories")

# A schema-1 ("July") pilots row, verbatim in shape: no "name", float
# "n_frames", insertion-ordered keys.
SCHEMA1_ROW = {
    "scope": "chime.event.baseband.raw", "event": "100058001",
    "freq_id": 506, "size_bytes": 91311880,
    "common_path": "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
                   "astro_100058001",
    "obs_date": "2020-07-15", "datasets": ["backlog.pulsar.B0355+54"],
    "freq_mhz": 602.3438, "n_frames": 2.7213,
}


def _row(event: str, freq_id: int, size: int = 91311880) -> dict:
    """One current-schema (schema 2) inventory row."""
    return {
        "scope": "chime.event.baseband.raw", "event": event,
        "name": baseband_filename(event, freq_id), "size_bytes": size,
        "common_path": f"cadc:CHIMEFRB/data/chime/baseband/raw/x/astro_{event}",
        "obs_date": "2020-07-15", "datasets": ["ds"], "freq_id": freq_id,
        "freq_mhz": 602.3438,
        "n_frames_estimate": size // BYTES_PER_FRAME,
    }


class _Proc:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _fake_ps(monkeypatch, uris):
    """Answer every `datatrail ps --json` with this minoc replica list."""
    payload = json.dumps({
        "dataset": "d", "scope": "s",
        "files": {"file_replica_locations": {"minoc": list(uris)}},
        "policies": {},
    })
    monkeypatch.setattr(dt.subprocess, "run",
                        lambda cmd, **kw: _Proc(0, payload, ""))


# ==========================================================================
# 1. The legacy-schema migration story
#
# Measured on the real references: the July chime-pilots inventory has no
# "name" on any of its 161,872 rows, so parse_row rejects 161,872/161,872.
# The July chime-controls inventory does carry "name" and parses 32,614/32,614.
# ==========================================================================
def test_parse_row_rejects_every_schema1_pilots_row():
    """A schema-1 row is refused, and the message names the missing field."""
    with pytest.raises(SystemExit) as excinfo:
        parse_row(json.dumps(SCHEMA1_ROW), "inventory.jsonl", 1)
    assert "missing required field(s) ['name']" in str(excinfo.value)


def test_schema1_row_carries_everything_the_upgrade_needs():
    """The migration is a pure function of the row -- no archive access.

    ``name`` is recoverable from (event, freq_id) through THE naming
    definition, and schema 2's ``n_frames_estimate`` is recoverable from
    ``size_bytes`` and the instrument geometry. Nothing else is needed, which
    is why the absence of a shipped migration is a packaging gap and not a
    data-loss problem.
    """
    upgraded = dict(SCHEMA1_ROW)
    upgraded.pop("n_frames")
    upgraded["name"] = baseband_filename(
        upgraded["event"], upgraded["freq_id"])
    upgraded["n_frames_estimate"] = upgraded["size_bytes"] // BYTES_PER_FRAME

    parsed = parse_row(json.dumps(upgraded), "inventory.jsonl", 1)
    assert parsed["name"] == "baseband_100058001_506.h5"
    assert parsed["n_frames_estimate"] == 2
    # and the upgraded row addresses the same archive object as before
    assert (f"{parsed['common_path']}/{parsed['name']}"
            == "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/"
               "astro_100058001/baseband_100058001_506.h5")


@pytest.mark.parametrize("size", [
    1, BYTES_PER_FRAME - 1, BYTES_PER_FRAME, BYTES_PER_FRAME + 1,
    2 * BYTES_PER_FRAME - 1, 91311880, 32890880, 10 * BYTES_PER_FRAME,
])
def test_low_estimate_rule_survives_the_n_frames_semantics_change(size):
    """schema 1 stored a float; schema 2 stores floor() as an int.

    The frozen bundle's 4,693 low-estimate exclusions were selected with the
    float column. This is the identity that makes that ledger re-derivable
    against the int column: for a positive size, ``floor(x) < 1`` iff
    ``x < 1``, so the exclusion SET is unchanged even though the recorded
    VALUE is not.
    """
    assert (size // BYTES_PER_FRAME < 1) == (size / BYTES_PER_FRAME < 1.0)


@pytest.mark.xfail(strict=True, reason=(
    "GAP: README.md:659 and INTEGRATION.md:213 both claim 'Completed "
    "inventory.jsonl files remain readable.' Measured against the real July "
    "reference, parse_row rejects 161,872 of 161,872 chime-pilots rows. No "
    "compatibility shim and no migration command ships. Either the claim or "
    "the code has to change."))
def test_completed_legacy_inventory_remains_readable():
    parse_row(json.dumps(SCHEMA1_ROW), "inventory.jsonl", 1)


@pytest.mark.skipif(not JULY_REFERENCES.is_dir(),
                    reason="July reference inventories not on this host")
@pytest.mark.parametrize("name,expect_readable", [
    ("chime-pilots", False),    # schema 1: no "name" column at all
    ("chime-controls", True),   # already carries "name"
])
def test_july_reference_readability_is_exactly_as_measured(
        name, expect_readable):
    """Pin the real, per-inventory readability split so it cannot drift."""
    path = JULY_REFERENCES / name / "inventory.jsonl"
    accepted = rejected = 0
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                parse_row(line, str(path), number)
                accepted += 1
            except SystemExit:
                rejected += 1
    total = accepted + rejected
    assert total > 0
    assert (rejected == 0) is expect_readable, (
        f"{name}: {accepted} accepted / {rejected} rejected of {total}")


# ==========================================================================
# 2. Determinism of the durable views
#
# survey_state.py:10 claims inventory.jsonl and the text ledgers are
# "deterministic views of the database". Nothing asserted it.
# ==========================================================================
def _render(order, tmp_path, tag):
    out = tmp_path / tag
    out.mkdir()
    store = SurveyStore(out / "survey_state.sqlite3")
    for scope, event, rows in order:
        store.commit(f"{scope}|{event}", scope, event, "complete", rows)
    store.render_views(out)
    store.close()
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir()) if p.suffix in (".jsonl", ".txt")}


def test_render_views_is_independent_of_commit_order(tmp_path):
    """Two runs that visit the same events in different orders -- because the
    archive answered in a different order, or because one run resumed -- must
    produce byte-identical views. This is the only thing that makes an
    inventory comparable across runs at all."""
    events = [("chime.event.baseband.raw", ev,
               [_row(ev, f) for f in (844, 506, 721)])
              for ev in ("100058001", "1111116060", "999", "100260502")]

    baseline = _render(events, tmp_path, "a")
    for seed in (7, 99, 1234):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert _render(shuffled, tmp_path, f"s{seed}") == baseline


def test_view_order_is_lexicographic_on_the_event_key(tmp_path):
    """The ordering contract is SQLite BINARY collation on "scope|event", not
    numeric order on the event id. A reader diffing two inventories must sort
    the same way; state it once, here."""
    out = tmp_path / "lex"
    out.mkdir()
    store = SurveyStore(out / "survey_state.sqlite3")
    scope = "chime.event.baseband.raw"
    for event in ("999", "1000", "10"):
        store.commit(f"{scope}|{event}", scope, event, "complete",
                     [_row(event, 506)])
    store.render_views(out)
    store.close()
    events = [json.loads(line)["event"]
              for line in (out / "inventory.jsonl").read_text().splitlines()]
    assert events == ["10", "1000", "999"]


def test_a_flaky_probe_silently_changes_inventory_content(tmp_path):
    """The reproducibility hazard, made explicit.

    _commit_decision accepts a PARTIAL event once the retry allowance is
    exhausted (cadc.py:270 -> ``True, True, True``). So an event whose probes
    were clean writes every row, while the same event under transient CADC
    errors writes a strict subset and is committed as done. The difference is
    recorded in incomplete_events.txt, but inventory.jsonl itself is smaller
    with no in-band marker -- two runs against an unchanged archive can
    therefore disagree on CONTENT, not just presentation.
    """
    # clean: 0 errored, 3 present
    assert src._commit_decision(0, 0, 3) == (True, True, False)
    # 1 of 3 keeps erroring; the third sighting accepts the partial event
    assert src._commit_decision(1, 0, 2) == (False, False, False)
    assert src._commit_decision(1, 1, 2) == (False, False, False)
    assert src._commit_decision(1, 2, 2) == (True, True, True)

    scope, event = "chime.event.baseband.raw", "100058001"
    clean = _render([(scope, event, [_row(event, f) for f in (506, 721, 844)])],
                    tmp_path, "clean")
    partial = _render([(scope, event, [_row(event, f) for f in (506, 721)])],
                      tmp_path, "partial")
    assert clean["inventory.jsonl"] != partial["inventory.jsonl"]


# ==========================================================================
# 3. Provenance: can a third party reconstruct how the inventory was made?
# ==========================================================================
REQUIRED_PROVENANCE = frozenset({
    "source_revision",          # which commit of this tree ran
    "package_source_sha256",    # provenance.package_source_sha256()
    "datatrail_cli_version",    # the CLI whose payloads were parsed
    "cadcdata_version",
    "certificate_not_after",    # the cert identity/expiry that authorized it
    "inventory_sha256",         # self-identification of the rows
    "row_count",
})


@pytest.mark.xfail(strict=True, reason=(
    "GAP: inventory.py:118-129 writes only schema/name/telescope/source/"
    "reader/scope(s)/scope_request/freq_ids/created. None of the toolchain "
    "identity is recorded, so an inventory cannot be tied to the code, the "
    "datatrail-cli, or the certificate that produced it. The project already "
    "records exactly these facts elsewhere -- see the 'software' and "
    "'certificate_not_after' keys in chime-pilots-v5/pending_resolution.json "
    "-- so this is an omission in the sidecar, not a missing capability."))
def test_inventory_meta_records_enough_to_reproduce_the_survey(tmp_path):
    from pilot_proxy.archive.inventory import write_inventory_meta

    class _Instrument:
        name = "chime"
        scopes = ("chime.event.baseband.raw",)

    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(_row("100058001", 506)) + "\n")
    meta_path = write_inventory_meta(
        inventory, _Instrument(), source="cadc-datatrail",
        reader="chime-baseband", freq_ids="506", name="t")
    meta = json.loads(meta_path.read_text())
    assert REQUIRED_PROVENANCE.issubset(meta), sorted(
        REQUIRED_PROVENANCE - set(meta))


@pytest.mark.xfail(strict=True, reason=(
    "GAP: survey_manifest.json (survey_state.py:233) records the "
    "configuration fingerprint -- what was REQUESTED -- but nothing about "
    "what RAN: no source revision, no datatrail-cli version, no cert "
    "identity, and no start/end timestamp. Two surveys months apart against a "
    "drifted archive produce the same fingerprint, so the manifest cannot "
    "distinguish them."))
def test_survey_manifest_records_the_toolchain_that_ran():
    from pilot_proxy.archive.survey_state import build_configuration

    class _Instrument:
        name = "chime"
        scopes = ("chime.event.baseband.raw",)
        f0_mhz = 800.0
        bandwidth_mhz = 400.0
        n_channels = 1024
        descending = True
        nyquist_zone = 2
        n_feeds = N_FEEDS
        nfft = NFFT
        reader = "chime-baseband"

    class _Ctx:
        instrument = _Instrument()
        selection = None
        options: dict = {}

    class _Shape:
        survey_schema = 2

    configuration = build_configuration(
        _Ctx(), ("chime.event.baseband.raw",), [506], _Shape(),
        False, 30, 1048576, 2)
    assert REQUIRED_PROVENANCE.issubset(configuration), sorted(
        REQUIRED_PROVENANCE - set(configuration))


# ==========================================================================
# 4. Blast radius of the uncommitted collection-restore patch
#
# datatrail_client.py:157 _restore_collection() prefixes any replica path
# beginning "data/" with "cadc:CHIMEFRB/". That converts a class of inputs
# that USED to be a deterministic refusal into an accepted common path.
# ==========================================================================
def test_restore_collection_recovers_the_documented_bare_replica(monkeypatch):
    """The case the patch exists for: every replica returned bare."""
    _fake_ps(monkeypatch, [
        "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
        "baseband_1111116060_506.h5",
        "data/chime/baseband/raw/2025/03/31/astro_1111116060/"
        "baseband_1111116060_844.h5",
    ])
    common, names, ok = dt.DATATRAIL.files(
        "chime.event.baseband.raw", "1111116060", retries=0)
    assert ok and common == ("cadc:CHIMEFRB/data/chime/baseband/raw/2025/03/31/"
                             "astro_1111116060")
    assert sorted(names) == ["baseband_1111116060_506.h5",
                             "baseband_1111116060_844.h5"]


@pytest.mark.parametrize("uri", [
    "database/chime/x/astro_1/f.h5",   # lookalike root, must not be restored
    "/data/chime/x/astro_1/f.h5",      # absolute, not archive-relative
    "cadc:SOMEONE_ELSE/data/chime/x/astro_1/f.h5",
])
def test_restore_collection_stays_narrow(monkeypatch, uri):
    """Restoration must not become a blanket accept."""
    _fake_ps(monkeypatch, [uri, uri.replace("f.h5", "g.h5")])
    with pytest.raises(dt.DatatrailContractError):
        dt.DATATRAIL.files("chime.event.baseband.raw", "1", retries=0)


@pytest.mark.xfail(strict=True, reason=(
    "GAP: _restore_collection() prefixes on a literal 'data/' startswith test "
    "with no canonicalization, so 'data/../../../etc/x/f.h5' is accepted and "
    "resolves to common_path 'cadc:CHIMEFRB/data/../../../etc/x'. Measured: "
    "identical input is REFUSED with the patch reverted. common_path is never "
    "canonicalized downstream -- cadc_inventory.py:108 checks only that it is "
    "a non-empty unpadded string, while :114 does apply _safe_archive_name to "
    "'name' -- and it flows into join_uri() for cadcinfo and cadcget. This is "
    "a fail-open weakening of a contract whose module docstring says the "
    "refusal exists so 'the decision is made on evidence, not on silence.'"))
def test_restore_collection_refuses_paths_that_escape_the_collection_root(
        monkeypatch):
    _fake_ps(monkeypatch, ["data/../../../etc/x/f.h5",
                           "data/../../../etc/x/g.h5"])
    with pytest.raises(dt.DatatrailContractError):
        dt.DATATRAIL.files("chime.event.baseband.raw", "1", retries=0)


def test_a_restored_replica_is_indistinguishable_from_a_native_one(monkeypatch):
    """The audit gap: nothing in the output records that the heuristic fired.

    A bare reply and a prefixed reply produce byte-identical results, so no
    field of the published inventory tells a reviewer which rows depended on
    the reconstruction. Auditing the patch's real blast radius against the
    frozen bundle is therefore impossible after the fact.
    """
    bare = ["data/chime/x/astro_1/f.h5", "data/chime/x/astro_1/g.h5"]
    prefixed = ["cadc:CHIMEFRB/" + u for u in bare]

    _fake_ps(monkeypatch, bare)
    from_bare = dt.DATATRAIL.files("s", "1", retries=0)
    _fake_ps(monkeypatch, prefixed)
    from_prefixed = dt.DATATRAIL.files("s", "1", retries=0)
    assert from_bare == from_prefixed


# ==========================================================================
# 5. The published bundle's own accounting, re-checked rather than asserted
#    by the one-off script invocation recorded inside it.
# ==========================================================================
requires_bundle = pytest.mark.skipif(
    not (FROZEN_BUNDLE / "inventory_manifest.json").is_file(),
    reason="chime-pilots-v5 frozen bundle not on this host")


def _unit_keys(path: Path) -> set:
    keys = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["scope"], row["event"], row["name"]))
    return keys


@requires_bundle
def test_frozen_inventory_is_an_exact_partition_of_the_source():
    """source == frozen + exclusions, with no overlap and nothing invented."""
    source = _unit_keys(FROZEN_BUNDLE / "inventory.source.jsonl")
    frozen = _unit_keys(FROZEN_BUNDLE / "inventory.jsonl")
    excluded = _unit_keys(FROZEN_BUNDLE / "exclusions.jsonl")
    assert len(source) == 170377
    assert len(frozen) == 165682
    assert len(excluded) == 4695
    assert frozen & excluded == set()
    assert frozen | excluded == source


@requires_bundle
def test_frozen_bundle_files_match_the_digests_it_publishes():
    manifest = json.loads(
        (FROZEN_BUNDLE / "inventory_manifest.json").read_text())
    checked = 0
    for name, entry in manifest["outputs"].items():
        path = FROZEN_BUNDLE / name
        if not path.is_file() or entry["bytes"] > 128 * 1024 * 1024:
            continue                      # skip the 728 MB product archive
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
        assert path.stat().st_size == entry["bytes"], name
        checked += 1
    assert checked >= 10


@requires_bundle
def test_only_three_frozen_exclusions_need_evidence_outside_the_archive():
    """The re-derivability boundary, stated exactly.

    4,692 exclusions carry ``prior_product_zero_frames`` and every one of them
    is ALSO caught by the pure size rule, so the prior detector product is
    confirmatory, not load-bearing: a third party can re-derive them from the
    archive alone. The 3 ``historical_quarantine`` units cannot be re-derived
    -- they came from direct HDF5 reads that hit truncated files -- and 2 of
    the 3 are invisible to any size rule. That 4,692 / 3 split is the honest
    reproducibility claim and it belongs in a test, not in prose.
    """
    size_derivable, external, invisible = set(), set(), 0
    with (FROZEN_BUNDLE / "exclusions.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["scope"], row["event"], row["name"])
            caught = row["size_bytes"] // BYTES_PER_FRAME < 1
            if row["reasons"] == ["prior_product_zero_frames"]:
                assert caught, f"not size-derivable: {key}"
                size_derivable.add(key)
            else:
                assert row["reasons"] == ["historical_quarantine"], row["reasons"]
                external.add(key)
                invisible += not row["below_one_frame_estimate"]
    assert len(size_derivable) == 4692
    assert len(external) == 3
    assert invisible == 2


@requires_bundle
def test_frozen_inventory_cannot_be_regenerated_by_todays_code():
    """The pinned production input is schema 1.5: it has "name" (so it parses)
    but still carries the float ``n_frames`` column, not schema 2's integer
    ``n_frames_estimate``. Today's code cannot regenerate it byte-for-byte --
    JSON key order, column name, and column type all differ -- so the locked
    digest b2cfef75... is not reproducible by re-running the survey.
    """
    with (FROZEN_BUNDLE / "inventory.jsonl").open() as handle:
        first = json.loads(handle.readline())
    parse_row(json.dumps(first), "inventory.jsonl", 1)   # parses: has "name"
    assert "n_frames" in first and "n_frames_estimate" not in first
    assert list(first) != sorted(first)                  # insertion-ordered
