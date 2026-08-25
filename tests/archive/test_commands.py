from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy import cli
from pilot_proxy.archive import commands, datatrail_client, pipeline
from pilot_proxy.archive.datatrail_client import Datatrail
from pilot_proxy.archive.inventory import inventory_meta_path
from pilot_proxy.archive.sources import CadcDatatrailSource
from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
from pilot_proxy.archive.control import ControlBandAnalyzer


def _inventory_row(freq_id: int, event: str = "123456") -> dict[str, object]:
    return {
        "scope": "chime.event.baseband.raw",
        "event": event,
        "name": f"baseband_{event}_{freq_id}.h5",
        "size_bytes": 2048,
        "common_path": "cadc:CHIMEFRB/raw/2026/08/24/123456",
        "obs_date": "2026-08-24",
        "freq_id": freq_id,
    }


def test_chime_survey_writes_compatible_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "survey"

    monkeypatch.setattr(Datatrail, "installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        Datatrail,
        "api_available",
        staticmethod(lambda: (True, "")),
    )
    monkeypatch.setattr(
        CadcDatatrailSource,
        "preflight",
        lambda self, ctx: (True, [], []),
    )

    def fake_survey(self, ctx, out):
        assert isinstance(ctx.reader, ChimeBasebandReader)
        path = Path(out) / "inventory.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_inventory_row(591)) + "\n", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(CadcDatatrailSource, "survey", fake_survey)

    assert (
        cli.main(
            [
                "chime-survey",
                "--out",
                str(out_dir),
                "--name",
                "controls",
                "--freq-ids",
                "591",
            ]
        )
        == 0
    )

    meta = json.loads(
        inventory_meta_path(out_dir / "inventory.jsonl").read_text(encoding="utf-8")
    )
    assert meta["datatrawl_inventory"] == 1
    assert meta["name"] == "controls"
    assert meta["telescope"] == "chime"
    assert meta["source"] == "cadc-datatrail"
    assert meta["reader"] == "chime-baseband"


def test_chime_survey_rejects_two_output_roots(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="either --out or --root"):
        cli.main(
            [
                "chime-survey",
                "--out",
                str(tmp_path / "out"),
                "--root",
                str(tmp_path / "root"),
                "--dry-run",
            ]
        )


def test_chime_inventory_summarizes_archive_rows(
    tmp_path: Path,
    capsys,
) -> None:
    inventory = tmp_path / "inventory.jsonl"
    rows = [_inventory_row(591), _inventory_row(591, "123457"), _inventory_row(745)]
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert cli.main(["chime-inventory", "--inventory", str(inventory)]) == 0

    output = capsys.readouterr().out
    assert "files          : 3" in output
    assert "2 present (591..745)" in output
    assert "591      2" in output
    assert "745      1" in output
    assert "pilot-proxy chime-scan" in output


def test_chime_control_scan_wires_complex_reader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "baseband_123456_591.h5").write_bytes(b"x")
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return pipeline.RunResult(
            out_path=kwargs["out_path"],
            n_total=1,
            n_done=1,
            n_new=1,
            n_failed=0,
            product_available=True,
            product_written=True,
        )

    monkeypatch.setattr(commands.pipeline, "run", fake_run)

    output_dir = tmp_path / "products"
    assert (
        cli.main(
            [
                "chime-control-scan",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--select",
                "591",
                "--max-frames-per-file",
                "2",
                "--gpu",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["reader"], ChimeBasebandReader)
    assert isinstance(call["analyzer"], ControlBandAnalyzer)
    assert call["ctx"].selection == [591]
    assert call["ctx"].options["max_frames_per_file"] == 2
    assert call["ctx"].options["gpu"] is True
    assert call["out_path"] == str(output_dir / "591.npz")
    assert call["download_workers"] == 1
    assert call["max_staged_files"] == 1


def test_chime_control_scan_requires_explicit_partial_acceptance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "baseband_123456_591.h5").write_bytes(b"x")
    monkeypatch.setattr(
        commands.pipeline,
        "run",
        lambda **kwargs: pipeline.RunResult(
            out_path=kwargs["out_path"],
            n_total=2,
            n_done=1,
            n_new=1,
            n_failed=0,
            product_available=True,
            product_written=True,
        ),
    )
    args = [
        "chime-control-scan",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "products"),
        "--select",
        "591",
    ]
    with pytest.raises(SystemExit, match="--allow-partial"):
        cli.main(args)
    assert cli.main([*args, "--allow-partial"]) == 0


def test_chime_control_scan_rejects_stale_saved_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "baseband_123456_591.h5").write_bytes(b"x")
    output_dir = tmp_path / "products"
    output_dir.mkdir()
    np.savez(output_dir / "591.npz", unit_keys=np.asarray(["retired-scope"]))

    def reject_run(**kwargs):
        raise AssertionError("pipeline must not modify a stale product")

    monkeypatch.setattr(commands.pipeline, "run", reject_run)

    with pytest.raises(SystemExit, match="outside the current source scope"):
        cli.main(
            [
                "chime-control-scan",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--select",
                "591",
                "--allow-partial",
            ]
        )


def test_chime_control_scan_uses_fetch_only_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(_inventory_row(591)) + "\n",
        encoding="utf-8",
    )
    checks = []

    def fetch_preflight(self, ctx):
        checks.append(ctx)
        return True, [], []

    def survey_preflight(self, ctx):
        raise AssertionError("survey preflight must not run")

    def query_archive_map(*args, **kwargs):
        raise AssertionError("inventory scans must not query the survey service")

    monkeypatch.setattr(CadcDatatrailSource, "fetch_preflight", fetch_preflight)
    monkeypatch.setattr(CadcDatatrailSource, "preflight", survey_preflight)
    monkeypatch.setattr(datatrail_client, "_run_json", query_archive_map)
    monkeypatch.setattr(
        commands.pipeline,
        "run",
        lambda **kwargs: pipeline.RunResult(
            out_path=kwargs["out_path"],
            n_total=1,
            n_done=1,
            n_new=1,
            n_failed=0,
            product_available=True,
            product_written=True,
        ),
    )

    assert (
        cli.main(
            [
                "chime-control-scan",
                "--inventory",
                str(inventory),
                "--output-dir",
                str(tmp_path / "products"),
                "--select",
                "591",
            ]
        )
        == 0
    )
    assert len(checks) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("telescope", "gbo"),
        ("source", "local"),
        ("reader", "other-reader"),
    ],
)
def test_chime_control_scan_rejects_incompatible_inventory_metadata(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(_inventory_row(591)) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "datatrawl_inventory": 1,
        "telescope": "chime",
        "source": "cadc-datatrail",
        "reader": "chime-baseband",
    }
    metadata[field] = value
    inventory_meta_path(inventory).write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match=field):
        cli.main(
            [
                "chime-control-scan",
                "--inventory",
                str(inventory),
                "--output-dir",
                str(tmp_path / "products"),
                "--select",
                "591",
            ]
        )
