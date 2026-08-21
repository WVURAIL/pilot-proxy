#!/usr/bin/env python3
# coding=utf-8
"""Build a portable provisional-v4 status export from health-repaired evidence.

The curated v3 dissertation snapshot is immutable.  This tool expands its
grouped channel classifications, attaches the v1 frame-health, policy, chain,
and provisional-quarter fine-anchor evidence, and corrects claims invalidated
by the completed repair.  It does not pretend to recompute the legacy numeric
epoch operating points; every such field remains explicitly provisional.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pilot_proxy.archive_health import FRAME_HEALTH_GATE_SCHEMA_VERSION
from pilot_proxy.atomic_io import atomic_write_json
from pilot_proxy.provenance import file_sha256


STATUS_SCHEMA_VERSION = "pilotproxy_dissertation_status_v4"


def _input_identity(path: Path) -> dict[str, Any]:
    return {
        "basename": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _base_status_by_channel(base: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for group in base.get("channel_status_groups", []):
        for channel in group["channels"]:
            value = int(channel)
            if value in rows:
                raise ValueError(f"base status repeats channel {value}")
            rows[value] = {
                "status": str(group["status"]),
                "secondary_status": str(group.get("secondary_status", "")),
                "epoch_scope": str(group.get("epoch_scope", "")),
                "base_evidence_state": str(group.get("evidence_state", "")),
                "base_note": str(group.get("note", "")),
            }
    if not rows:
        raise ValueError("base summary contains no channel status groups")
    return rows


def _as_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    return 0 if value in (None, "") else int(value)


def _as_float_or_none(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value in (None, "") else float(value)


def _fine_anchor_status(channel: Mapping[str, Any]) -> dict[str, Any]:
    audit = channel["fine_anchor_audit"]["measured_epoch_line_anchor"]
    epochs = list(audit["epochs"])
    refusal_counts: Counter[str] = Counter()
    for row in epochs:
        refusal_counts.update(str(value) for value in row.get("refusal_reasons", []))
    dated = [row for row in epochs if row["epoch_key"] != "timestamp_unavailable"]
    latest = dated[-1] if dated else None
    return {
        "epoch_partition": str(audit["epoch_definition"]),
        "epoch_status_counts": dict(audit["epoch_status_counts"]),
        "refusal_reason_counts": dict(refusal_counts),
        "health_included_frames_using_measured_anchor": int(
            audit["health_included_frames_using_measured_anchor"]
        ),
        "health_included_frames_falling_back_to_predicted": int(
            audit["health_included_frames_falling_back_to_predicted"]
        ),
        "latest_provisional_epoch": (
            {
                "epoch_key": str(latest["epoch_key"]),
                "status": str(latest["status"]),
                "candidate_anchor_bin": latest.get("candidate_anchor_bin"),
                "candidate_minus_predicted_circular_bins": latest.get(
                    "candidate_minus_predicted_circular_bins"
                ),
                "refusal_reasons": list(latest.get("refusal_reasons", [])),
            }
            if latest is not None
            else None
        ),
    }


def _corrected_note(
    channel: int,
    base_note: str,
    *,
    full_chain: Mapping[str, Any] | None,
    off_chain: Mapping[str, Any] | None,
    largest_full: Mapping[str, Any],
    largest_epoch: Mapping[str, Any],
) -> str:
    if channel in {18, 21, 23, 25}:
        return (
            "Trace-occupancy classification remains provisional. The v1 chain "
            "has full-archive kept/null floors for channels 18 and 21, but not "
            "for 23 or 25; correlation/floor limitations remain explicit. "
            f"Channel {largest_full['channel']} has the largest health-filtered "
            f"full-archive null population ({largest_full['n_null']:,} frames). "
            f"Channel {largest_epoch['channel']} {largest_epoch['epoch']} is the "
            f"largest epoch-specific null population ({largest_epoch['n_null']:,} "
            "frames), so the old unqualified 'channel 21 is largest' wording is "
            "withdrawn."
        )
    if channel == 27:
        if off_chain is None:
            raise ValueError(
                "channel 27 requires a health-filtered off-epoch chain row"
            )
        return (
            "Sign-off remains bounded inside the 2021-22 collection gap. The "
            "v1 health-filtered off-epoch chain is now evaluated: "
            f"n_null={_as_int(off_chain, 'n_null'):,}, p90 floor "
            f"{_as_float_or_none(off_chain, 'floor_db'):.2f} dB, and correlation "
            f"quality {off_chain['tau_quality']}; the former 'chain pending' "
            "wording is withdrawn. The full-era kept/null floor remains "
            "unavailable and the verdict remains measurement-bound."
        )
    if channel in {22, 24}:
        if full_chain is None or _as_int(full_chain, "n_null") != 0:
            raise ValueError(f"channel {channel} must expose the zero-null repair")
        return (
            "Under the v1 gate, no health-included frame survives the stored "
            "mask as a kept/null calibration row. The conservative policy is "
            "excision pending new clean/off data or an independently justified "
            "synthetic floor; old language implying an empirical chain floor is "
            "withdrawn."
        )
    return (
        f"{base_note} Legacy classification retained provisionally; v4 attaches "
        "health-filtered counts, policy action, chain status, and fine-anchor "
        "evidence without claiming a new blinded science verdict."
    )


def build_status(
    base: Mapping[str, Any],
    health: Mapping[str, Any],
    policy: Mapping[str, Any],
    chain_rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose the deterministic v4 status object from loaded inputs."""

    if health["health_gate"]["schema_version"] != FRAME_HEALTH_GATE_SCHEMA_VERSION:
        raise ValueError("health summary uses an unexpected frame gate")
    if policy["health_gate"]["schema_version"] != FRAME_HEALTH_GATE_SCHEMA_VERSION:
        raise ValueError("policy data does not use the v1 frame gate")
    if any(
        row.get("health_gate_schema_version") != FRAME_HEALTH_GATE_SCHEMA_VERSION
        for row in chain_rows
    ):
        raise ValueError("one or more chain rows do not use the v1 frame gate")
    base_by_channel = _base_status_by_channel(base)
    health_by_channel = {
        int(row["physical_channel"]): row for row in health["channels"]
    }
    policy_by_channel = {int(key): value for key, value in policy["channels"].items()}
    chain_by_channel: dict[int, list[Mapping[str, Any]]] = {}
    for row in chain_rows:
        chain_by_channel.setdefault(int(row["channel"]), []).append(row)
    expected = set(base_by_channel)
    for name, values in (
        ("health", set(health_by_channel)),
        ("policy", set(policy_by_channel)),
        ("chain", set(chain_by_channel)),
    ):
        if values != expected:
            raise ValueError(
                f"{name} channel set differs from base status: "
                f"missing={sorted(expected - values)}, extra={sorted(values - expected)}"
            )
    full_rows = [row for row in chain_rows if row["epoch"] == "full"]
    available_epoch_rows = [
        row
        for row in chain_rows
        if row["epoch"] != "full" and _as_int(row, "n_null") > 0
    ]
    if not available_epoch_rows:
        raise ValueError("chain table has no available epoch-specific null row")
    largest_full_row = max(full_rows, key=lambda row: _as_int(row, "n_null"))
    largest_epoch_row = max(
        available_epoch_rows, key=lambda row: _as_int(row, "n_null")
    )
    largest_full = {
        "channel": int(largest_full_row["channel"]),
        "epoch": str(largest_full_row["epoch"]),
        "n_null": _as_int(largest_full_row, "n_null"),
    }
    largest_epoch = {
        "channel": int(largest_epoch_row["channel"]),
        "epoch": str(largest_epoch_row["epoch"]),
        "n_null": _as_int(largest_epoch_row, "n_null"),
    }
    statuses: list[dict[str, Any]] = []
    for channel in sorted(expected):
        base_row = base_by_channel[channel]
        health_row = health_by_channel[channel]
        policy_row = policy_by_channel[channel]
        channel_chain = chain_by_channel[channel]
        full_chain = next(
            (row for row in channel_chain if row["epoch"] == "full"), None
        )
        off_chain = next(
            (row for row in channel_chain if row["epoch"] == "off-epoch"), None
        )
        if full_chain is None:
            raise ValueError(f"channel {channel} has no full chain row")
        statuses.append(
            {
                "channel": channel,
                "status": base_row["status"],
                "secondary_status": base_row["secondary_status"],
                "classification_status": "provisional_not_blinded_verdict",
                "epoch_scope": base_row["epoch_scope"],
                "evidence_state": "v1_health_repaired_inputs+legacy_classification",
                "note": _corrected_note(
                    channel,
                    base_row["base_note"],
                    full_chain=full_chain,
                    off_chain=off_chain,
                    largest_full=largest_full,
                    largest_epoch=largest_epoch,
                ),
                "health": {
                    "stored_frames": int(health_row["frame_counts"]["stored"]),
                    "included_frames": int(
                        health_row["frame_counts"]["included_by_health_gate"]
                    ),
                    "excluded_frames": int(
                        health_row["frame_counts"]["excluded_unique"]
                    ),
                },
                "policy": {
                    "action": str(policy_row["recommendation"]["action"]),
                    "why": str(policy_row["recommendation"]["why"]),
                    "chosen": list(policy_row["chosen"]),
                },
                "chain_rows": [dict(row) for row in channel_chain],
                "fine_anchor": _fine_anchor_status(health_row),
            }
        )
    epoch_points: list[dict[str, Any]] = []
    ch27_off = (
        next(row for row in chain_by_channel[27] if row["epoch"] == "off-epoch")
        if 27 in chain_by_channel
        else None
    )
    for source in base.get("epoch_operating_points", []):
        row = dict(source)
        row["legacy_numeric_fields_health_recomputed"] = False
        row["v4_status"] = "provisional_legacy_operating_point"
        row["v4_note"] = (
            "The survey/fine/residual numeric fields are copied from immutable "
            "v3 and were not regenerated by the archive-health command."
        )
        if int(row["channel"]) == 27 and row["epoch_key"] == "ch27_off_from_2022_10":
            if ch27_off is None:
                raise ValueError("channel 27 off-epoch chain row is missing")
            row["evidence_state"] = "measured+v1_health_repaired_chain"
            row["note"] = (
                "Trace-level from October 2022 onward. The v1 off-epoch chain "
                f"is evaluated with n_null={_as_int(ch27_off, 'n_null'):,}, "
                f"p90 floor {_as_float_or_none(ch27_off, 'floor_db'):.2f} dB, "
                f"and correlation quality {ch27_off['tau_quality']}; the "
                "classification remains conditional and measurement-bound."
            )
            row["v4_note"] += (
                " The former 'chain re-evaluation pending' claim is withdrawn."
            )
        epoch_points.append(row)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "snapshot_id": "dissertation-status-v4-health-repair",
        "classification_scope": (
            "provisional dissertation status integration; not a new blinded "
            "BAO verdict and not a rewrite of the immutable v3 snapshot"
        ),
        "health_gate_schema_version": FRAME_HEALTH_GATE_SCHEMA_VERSION,
        "provenance": dict(provenance),
        "health_totals": dict(health["totals"]),
        "null_population_extrema": {
            "largest_full_archive": largest_full,
            "largest_epoch_specific": largest_epoch,
            "interpretation": (
                "The two domains are reported separately. Channel "
                f"{largest_full['channel']} has the largest full-archive null "
                f"population; channel {largest_epoch['channel']} "
                f"{largest_epoch['epoch']} has the largest epoch-specific null "
                "population. The old unqualified channel-21 superlative is "
                "withdrawn."
            ),
        },
        "claim_corrections": [
            (
                "The old unqualified claim that channel 21 has the largest null "
                "population is withdrawn; the full-archive and epoch-specific "
                "extrema are derived and reported separately."
            ),
            (
                "Channel 27's v1 off-epoch chain is evaluated; it is no longer "
                "chain-pending."
            ),
            "Channels 22 and 24 have no health-included stored-mask kept/null calibration frame.",
            "Legacy v3 epoch operating-point numbers and their figures are not health-recomputed.",
        ],
        "channel_statuses": statuses,
        "epoch_operating_points": epoch_points,
        "bao_policy_case": base.get("bao_policy_case"),
        "legacy_artifacts_not_regenerated": [
            "data/provenance/dissertation_summary_v3.json",
            "analysis/dissertation/data/channel_status.csv",
            "analysis/dissertation/data/epoch_operating_points.csv",
            "status and epoch figures derived from those legacy CSV files",
        ],
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_status_exports(status: Mapping[str, Any], output_dir: Path) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "dissertation_summary_v4.json"
    atomic_write_json(json_path, dict(status))
    channel_rows = []
    for row in status["channel_statuses"]:
        chain = next(value for value in row["chain_rows"] if value["epoch"] == "full")
        latest = row["fine_anchor"]["latest_provisional_epoch"] or {}
        channel_rows.append(
            {
                "channel": row["channel"],
                "status": row["status"],
                "secondary_status": row["secondary_status"],
                "classification_status": row["classification_status"],
                "epoch_scope": row["epoch_scope"],
                "evidence_state": row["evidence_state"],
                "health_included_frames": row["health"]["included_frames"],
                "health_excluded_frames": row["health"]["excluded_frames"],
                "policy_action": row["policy"]["action"],
                "full_chain_n_null": chain["n_null"],
                "full_chain_floor_db": chain["floor_db"],
                "full_chain_floor_status": chain["health_floor_status"],
                "latest_fine_epoch": latest.get("epoch_key", ""),
                "latest_fine_anchor_status": latest.get("status", ""),
                "note": row["note"],
            }
        )
    channel_path = root / "channel_status_v4.csv"
    _write_csv(channel_path, channel_rows, list(channel_rows[0]))
    epoch_path = root / "epoch_operating_points_v4.csv"
    epoch_rows = list(status["epoch_operating_points"])
    fields = list(epoch_rows[0])
    _write_csv(epoch_path, epoch_rows, fields)
    return [json_path, channel_path, epoch_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--health-summary", type=Path, required=True)
    parser.add_argument("--policy-data", type=Path, required=True)
    parser.add_argument("--chain-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.base_summary.open(encoding="utf-8") as handle:
        base = json.load(handle)
    with args.health_summary.open(encoding="utf-8") as handle:
        health = json.load(handle)
    with args.policy_data.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    with args.chain_table.open(newline="", encoding="utf-8") as handle:
        chain = list(csv.DictReader(handle))
    provenance = {
        "base_summary": _input_identity(args.base_summary),
        "health_summary": _input_identity(args.health_summary),
        "policy_data": _input_identity(args.policy_data),
        "chain_table": _input_identity(args.chain_table),
        "generator_source": {
            "basename": Path(__file__).name,
            "sha256": file_sha256(Path(__file__)),
        },
        "source_commit": health["provenance"]["audit_implementation"].get("git_commit"),
    }
    status = build_status(base, health, policy, chain, provenance=provenance)
    for path in write_status_exports(status, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
