from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pilot_proxy.dissertation_exports import (
    ExportError,
    create_export,
    verify_export,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_id": "test-snapshot",
        "epoch_operating_points": [
            {
                "channel": 32,
                "epoch_key": "pre",
                "epoch_group": 0,
                "epoch_label": "pre",
                "survey_mask_fraction": 0.9,
                "fine_mask_fraction": None,
                "residual_ratio": None,
                "retained_frames": 7,
                "status": "failure",
                "evidence_state": "measured",
                "note": "test",
            }
        ],
        "channel_status_groups": [
            {
                "channels": list(range(14, 37)),
                "status": "unmeasured",
                "secondary_status": "",
                "epoch_scope": "all",
                "evidence_state": "pending",
                "note": "test",
            }
        ],
        "bao_policy_case": {
            "channel": 33,
            "residual_tolerance": 0.0015,
            "correlation_time_limit_minutes": 5.0,
            "policies": [
                {
                    "policy_key": "pilot_proxy",
                    "label": "pilot proxy",
                    "residual_multiple": 24,
                    "time_multiple": 5.3,
                    "evidence_state": "measured+modeled",
                    "note": "test",
                }
            ],
        },
    }


def _mock_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    census = repo / "data/census/census.csv"
    fields = [
        "rf_channel",
        "callsign",
        "service_class",
        "detectability_db",
        "distance_km",
        "bearing_deg",
        "frequency_tolerance",
        "chime_ch_index",
        "nominal_pilot_mhz",
        "city",
        "state_prov",
    ]
    _write_csv(
        census,
        fields,
        [
            {
                "rf_channel": 30,
                "callsign": "NEAR",
                "service_class": "Relay",
                "detectability_db": 70,
                "distance_km": 100.0,
                "bearing_deg": 10.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 598,
                "nominal_pilot_mhz": 566.3,
                "city": "Near",
                "state_prov": "BC",
            },
            {
                "rf_channel": 31,
                "callsign": "FAR",
                "service_class": "Relay",
                "detectability_db": 60,
                "distance_km": 250.0,
                "bearing_deg": 20.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 583,
                "nominal_pilot_mhz": 572.3,
                "city": "Far",
                "state_prov": "BC",
            },
            {
                "rf_channel": 29,
                "callsign": "NEAREST",
                "service_class": "Full-power",
                "detectability_db": 80,
                "distance_km": 50.0,
                "bearing_deg": 5.0,
                "frequency_tolerance": "±1 kHz",
                "chime_ch_index": 614,
                "nominal_pilot_mhz": 560.3,
                "city": "Nearest",
                "state_prov": "BC",
            },
        ],
    )
    summary = repo / "data/provenance/dissertation_summary_v1.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    return repo, summary


def test_partial_export_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    output = tmp_path / "out"
    manifest = create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit="abc123",
    )

    assert manifest["complete"] is False
    assert manifest["source"]["commit"] == "abc123"
    assert verify_export(output)["schema"]["version"] == 1

    with (output / "census_inner_120mi.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["callsign"] for row in rows] == ["NEAREST", "NEAR"]

    statuses = {
        record["path"]: record["status"] for record in manifest["artifacts"]
    }
    assert statuses["census_psd.csv"] == "pending"
    assert statuses["channel_status.csv"] == "available"


def test_optional_table_is_normalised_and_hashed(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    supplied = tmp_path / "census_psd_input.csv"
    _write_csv(
        supplied,
        ["channel", "offset_khz", "db_rel_median", "ignored"],
        [
            {"channel": 28, "offset_khz": 1, "db_rel_median": 2, "ignored": "x"},
            {"channel": 27, "offset_khz": -1, "db_rel_median": 3, "ignored": "y"},
        ],
    )
    output = tmp_path / "out"
    manifest = create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit="abc123",
        optional_inputs={"census_psd": supplied},
    )
    record = next(r for r in manifest["artifacts"] if r["path"] == "census_psd.csv")
    assert record["status"] == "available"
    with (output / "census_psd.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == ["channel", "offset_khz", "db_rel_median"]
    assert [row["channel"] for row in rows] == ["27", "28"]


def test_verify_detects_tampering(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    output = tmp_path / "out"
    create_export(
        repo_root=repo,
        output_dir=output,
        summary_path=summary,
        source_commit="abc123",
    )
    with (output / "channel_status.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ExportError, match="SHA-256 mismatch"):
        verify_export(output)


def test_require_complete_refuses_pending_tables(tmp_path: Path) -> None:
    repo, summary = _mock_repo(tmp_path)
    with pytest.raises(ExportError, match="complete export requested"):
        create_export(
            repo_root=repo,
            output_dir=tmp_path / "out",
            summary_path=summary,
            source_commit="abc123",
            require_complete=True,
        )
