# coding=utf-8
"""Tests for the portable health-repaired dissertation status export."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from pilot_proxy.archive_health import FRAME_HEALTH_GATE_SCHEMA_VERSION


_SCRIPT = Path(__file__).parents[2] / "tools" / "make_dissertation_status_v4.py"
_SPEC = importlib.util.spec_from_file_location("make_dissertation_status_v4", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_status = _MODULE.build_status
write_status_exports = _MODULE.write_status_exports


def _base_summary() -> dict[str, object]:
    return {
        "channel_status_groups": [
            {
                "channels": [20],
                "status": "resolved",
                "secondary_status": "provisional",
                "epoch_scope": "full",
                "evidence_state": "legacy",
                "note": "Legacy channel 20 note.",
            },
            {
                "channels": [21],
                "status": "trace_occupancy",
                "secondary_status": "provisional",
                "epoch_scope": "full",
                "evidence_state": "legacy",
                "note": "Channel 21 has the largest null population.",
            },
            {
                "channels": [27],
                "status": "measurement_bound",
                "secondary_status": "provisional",
                "epoch_scope": "off_epoch",
                "evidence_state": "legacy",
                "note": "Chain re-evaluation pending.",
            },
        ],
        "epoch_operating_points": [
            {
                "channel": 27,
                "epoch_key": "ch27_off_from_2022_10",
                "evidence_state": "measured+chain_pending",
                "note": "Chain re-evaluation pending.",
                "threshold": 2.5,
            }
        ],
        "bao_policy_case": {"state": "legacy"},
    }


def _health_channel(channel: int) -> dict[str, object]:
    return {
        "physical_channel": channel,
        "frame_counts": {
            "stored": 100,
            "included_by_health_gate": 99,
            "excluded_unique": 1,
        },
        "fine_anchor_audit": {
            "measured_epoch_line_anchor": {
                "epoch_definition": "retrospective_provisional_utc_quarters",
                "epoch_status_counts": {"measured": 1},
                "health_included_frames_using_measured_anchor": 90,
                "health_included_frames_falling_back_to_predicted": 9,
                "epochs": [
                    {
                        "epoch_key": "2026Q3",
                        "status": "measured",
                        "candidate_anchor_bin": 1024,
                        "candidate_minus_predicted_circular_bins": 2,
                        "refusal_reasons": [],
                    }
                ],
            }
        },
    }


def _chain_row(channel: int, epoch: str, n_null: int) -> dict[str, str]:
    return {
        "channel": str(channel),
        "epoch": epoch,
        "n_null": str(n_null),
        "floor_db": "-24.44" if n_null else "",
        "tau_quality": "refused" if n_null else "unavailable",
        "health_floor_status": "available" if n_null else "unavailable",
        "health_gate_schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION,
    }


def test_v4_separates_null_domains_and_withdraws_stale_ch27_claim(tmp_path) -> None:
    health = {
        "health_gate": {"schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION},
        "totals": {"stored_frames": 200, "health_included_frames": 198},
        "channels": [_health_channel(20), _health_channel(21), _health_channel(27)],
    }
    policy = {
        "health_gate": {"schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION},
        "channels": {
            str(channel): {
                "recommendation": {"action": "monitor", "why": "fixture"},
                "chosen": ["stored"],
            }
            for channel in (20, 21, 27)
        },
    }
    chain = [
        _chain_row(20, "full", 23_666),
        _chain_row(21, "full", 14_647),
        _chain_row(27, "full", 0),
        _chain_row(27, "off-epoch", 21_437),
    ]
    status = build_status(
        _base_summary(),
        health,
        policy,
        chain,
        provenance={"source_commit": "fixture"},
    )

    assert status["null_population_extrema"] == {
        "largest_full_archive": {"channel": 20, "epoch": "full", "n_null": 23_666},
        "largest_epoch_specific": {
            "channel": 27,
            "epoch": "off-epoch",
            "n_null": 21_437,
        },
        "interpretation": (
            "The two domains are reported separately. Channel 20 has the largest "
            "full-archive null population; channel 27 off-epoch has the largest "
            "epoch-specific null population. The old unqualified channel-21 "
            "superlative is withdrawn."
        ),
    }
    by_channel = {row["channel"]: row for row in status["channel_statuses"]}
    assert "old unqualified" in by_channel[21]["note"]
    assert "former 'chain pending' wording is withdrawn" in by_channel[27]["note"]
    epoch = status["epoch_operating_points"][0]
    assert epoch["legacy_numeric_fields_health_recomputed"] is False
    assert "21,437" in epoch["note"]
    assert "chain re-evaluation pending" in epoch["v4_note"]

    paths = write_status_exports(status, tmp_path)
    assert [path.name for path in paths] == [
        "dissertation_summary_v4.json",
        "channel_status_v4.csv",
        "epoch_operating_points_v4.csv",
    ]
    written = json.loads(paths[0].read_text(encoding="utf-8"))
    assert written["schema_version"] == "pilotproxy_dissertation_status_v4"
    assert written["legacy_artifacts_not_regenerated"][0].endswith(
        "dissertation_summary_v3.json"
    )
