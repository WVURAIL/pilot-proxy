# coding=utf-8
"""Choose CHIME detector K candidates from frequency-offset diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pilot_proxy.integration import (
    DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    DEFAULT_REFERENCE_RECEIVER_PROFILE,
)
from pilot_proxy.integration.receiver_profile import load_receiver_profile
from pilot_proxy.integration.weight_generation import DetectorCoreLayout, target_layout

DEFAULT_CANDIDATE_K = (128, 256)
DEFAULT_REFERENCE_SPACING_POLICIES = (
    "fixed_skipped_guard",
    "fixed_hz_reference_spacing",
)
DEFAULT_REFERENCE_K = 128
DEFAULT_SKIPPED_GUARD_BINS = 1
DEFAULT_REFERENCE_OFFSET_BINS = DEFAULT_SKIPPED_GUARD_BINS + 1
MIN_SKIPPED_GUARD_BINS = 1
MIN_REFERENCE_OFFSET_BINS = 2
DEFAULT_MIN_PEAK_PROMINENCE_DB = 25.0
DEFAULT_MAX_CAPTURE_LOSS_DB = 1.0
# choose-detector-k is normally driven by CHIME frequency-offset products. Use the
# CHIME real-data receiver profile by default; the reference profile remains
# available by passing --receiver-profile explicitly.
DEFAULT_RECEIVER_PROFILE = DEFAULT_CHIME_DTV_RECEIVER_PROFILE
REFERENCE_PLACEMENT_STATUS_NOMINAL = "nominal"


@dataclass(frozen=True)
class ChannelOffsetMetric:
    physical_channel: int
    valid_count: int
    median_peak_prominence_db: float
    median_frequency_offset_hz: float
    p95_abs_frequency_offset_hz: float
    p95_abs_residual_hz: float
    reliable: bool


def capture_loss_allowed_bin_fraction(max_capture_loss_db: float) -> float:
    """Return the fractional-bin offset allowed by a rectangular-bin loss model."""
    loss_db = float(max_capture_loss_db)
    if loss_db <= 0.0 or not np.isfinite(loss_db):
        raise ValueError("max_capture_loss_db must be positive and finite")

    def loss_at_fraction(frac: float) -> float:
        if frac == 0.0:
            return 0.0
        amp = math.sin(math.pi * frac) / (math.pi * frac)
        return float(-20.0 * math.log10(abs(amp)))

    half_bin_loss = loss_at_fraction(0.5)
    if loss_db >= half_bin_loss:
        return 0.5

    lo = 0.0
    hi = 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if loss_at_fraction(mid) <= loss_db:
            lo = mid
        else:
            hi = mid
    return float(lo)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:
        return float(np.percentile(values, q, **{"interpolation": "linear"}))


def _channel_metrics(
    *,
    physical_channel: np.ndarray,
    frequency_offset_hz: np.ndarray,
    peak_prominence_db: np.ndarray,
    valid: np.ndarray,
    min_peak_prominence_db: float,
    recenter: bool,
) -> list[ChannelOffsetMetric]:
    metrics: list[ChannelOffsetMetric] = []
    for index, channel in enumerate(physical_channel):
        include = (
            (valid[:, index] != 0)
            & np.isfinite(frequency_offset_hz[:, index])
            & np.isfinite(peak_prominence_db[:, index])
        )
        offsets = np.asarray(frequency_offset_hz[:, index][include], dtype=np.float64)
        prominence = np.asarray(peak_prominence_db[:, index][include], dtype=np.float64)
        valid_count = int(offsets.size)
        median_prominence = (
            float(np.median(prominence)) if prominence.size else float("nan")
        )
        reliable = bool(
            prominence.size
            and np.isfinite(median_prominence)
            and median_prominence >= float(min_peak_prominence_db)
        )
        median_offset = float(np.median(offsets)) if offsets.size else float("nan")
        residual = offsets - median_offset if recenter else offsets
        metrics.append(
            ChannelOffsetMetric(
                physical_channel=int(channel),
                valid_count=valid_count,
                median_peak_prominence_db=median_prominence,
                median_frequency_offset_hz=median_offset,
                p95_abs_frequency_offset_hz=_percentile(np.abs(offsets), 95.0),
                p95_abs_residual_hz=_percentile(np.abs(residual), 95.0),
                reliable=reliable,
            )
        )
    return metrics


def _reference_offset_bins_for_policy(
    *,
    policy: str,
    candidate_k: int,
    reference_k: int,
    skipped_guard_bins: int,
) -> tuple[int, int, str]:
    reference_offset_bins = int(skipped_guard_bins) + 1
    normalized = str(policy)
    if normalized == "fixed_skipped_guard":
        computed = int(reference_offset_bins)
        effective = max(MIN_REFERENCE_OFFSET_BINS, computed)
        note = (
            "none"
            if computed == effective
            else "clamped_to_min_reference_offset_bins"
        )
        return computed, effective, note
    if normalized == "fixed_hz_reference_spacing":
        computed = math.ceil(
            float(reference_offset_bins) * float(candidate_k) / float(reference_k)
        )
        effective = max(MIN_REFERENCE_OFFSET_BINS, int(computed))
        note = (
            "none"
            if int(computed) == effective
            else "clamped_to_min_reference_offset_bins"
        )
        return int(computed), int(effective), note
    raise ValueError(f"unsupported reference-spacing policy {policy!r}")


def _format_channel_list(channels: Sequence[int]) -> str:
    return ";".join(str(int(channel)) for channel in channels)


def _format_unique_text(values: Sequence[str]) -> str:
    return ";".join(sorted({str(value) for value in values if str(value)}))


def _candidate_reference_placement_summary(
    *,
    physical_channel: np.ndarray,
    receiver_profile_path: Path,
    candidate_k: int,
    reference_offset_bins: int,
) -> dict[str, object]:
    profile = load_receiver_profile(receiver_profile_path)
    core = DetectorCoreLayout(
        detector_window_samples=int(candidate_k),
        skipped_guard_bins=int(reference_offset_bins) - 1,
        reference_offset_bins=int(reference_offset_bins),
    )
    layouts = [
        target_layout(physical_channel=int(channel), profile=profile, core=core)
        for channel in physical_channel
    ]
    status_values = [str(layout["reference_placement_status"]) for layout in layouts]
    adaptive_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if str(layout["reference_placement_status"]) != REFERENCE_PLACEMENT_STATUS_NOMINAL
    ]
    dc_shifted_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout["dc_reference_shifted"])
    ]
    edge_wrapped_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout["edge_reference_wrapped"])
    ]
    skipped_guard_channels = [
        int(layout["physical_channel"])
        for layout in layouts
        if bool(layout["forbidden_tone_in_skipped_guard"])
    ]
    warnings = [
        f"DTV {int(layout['physical_channel'])}: {layout['placement_warnings']}"
        for layout in layouts
        if str(layout.get("placement_warnings", ""))
    ]
    unique_status = sorted(set(status_values))
    placement_status = (
        unique_status[0]
        if len(unique_status) == 1
        else f"mixed:{';'.join(unique_status)}"
    )
    num_dc_shifted_references = sum(
        int(bool(layout["lower_reference_dc_shifted"]))
        + int(bool(layout["upper_reference_dc_shifted"]))
        for layout in layouts
    )
    num_edge_wrapped_references = sum(
        int(bool(layout["lower_reference_edge_wrapped"]))
        + int(bool(layout["upper_reference_edge_wrapped"]))
        for layout in layouts
    )
    return {
        "reference_placement_status": placement_status,
        "placement_warnings": _format_unique_text(warnings),
        "num_channels_with_adaptive_reference": int(len(adaptive_channels)),
        "channels_with_adaptive_reference": _format_channel_list(adaptive_channels),
        "num_dc_shifted_references": int(num_dc_shifted_references),
        "channels_with_dc_shifted_reference": _format_channel_list(
            dc_shifted_channels
        ),
        "num_edge_wrapped_references": int(num_edge_wrapped_references),
        "channels_with_edge_wrapped_reference": _format_channel_list(
            edge_wrapped_channels
        ),
        "num_forbidden_tone_in_skipped_guard": int(len(skipped_guard_channels)),
        "channels_with_forbidden_tone_in_skipped_guard": _format_channel_list(
            skipped_guard_channels
        ),
    }


def choose_detector_k(
    *,
    frequency_offset: Path,
    output: Path,
    candidate_k: Sequence[int] = DEFAULT_CANDIDATE_K,
    candidate_reference_spacing_policy: Sequence[
        str
    ] = DEFAULT_REFERENCE_SPACING_POLICIES,
    min_peak_prominence_db: float = DEFAULT_MIN_PEAK_PROMINENCE_DB,
    max_capture_loss_db: float = DEFAULT_MAX_CAPTURE_LOSS_DB,
    reference_k: int = DEFAULT_REFERENCE_K,
    skipped_guard_bins: int = DEFAULT_SKIPPED_GUARD_BINS,
    receiver_profile: Path = DEFAULT_RECEIVER_PROFILE,
    recenter: bool = True,
) -> Path:
    """Write a K/reference-offset candidate summary table."""
    candidate_ks = [int(value) for value in candidate_k]
    if not candidate_ks:
        raise ValueError("at least one candidate K is required")
    for value in candidate_ks:
        if value <= 0:
            raise ValueError("candidate K values must be positive")
    if int(skipped_guard_bins) < MIN_SKIPPED_GUARD_BINS:
        raise ValueError(
            f"skipped_guard_bins must be at least {MIN_SKIPPED_GUARD_BINS}"
        )
    reference_offset_bins = int(skipped_guard_bins) + 1
    policies = [str(value) for value in candidate_reference_spacing_policy]
    if not policies:
        raise ValueError("at least one candidate reference-spacing policy is required")

    data = np.load(Path(frequency_offset))
    physical_channel = np.asarray(data["physical_channel"], dtype=np.int32)
    frequency_offset_hz = np.asarray(data["frequency_offset_hz"], dtype=np.float64)
    peak_prominence_db = np.asarray(data["peak_prominence_db"], dtype=np.float64)
    valid = np.asarray(data["valid"], dtype=np.uint8)
    sample_rate_hz = float(np.asarray(data["sample_rate_hz"]).reshape(()))
    if frequency_offset_hz.shape != peak_prominence_db.shape:
        raise ValueError("frequency_offset_hz and peak_prominence_db shapes differ")
    if frequency_offset_hz.shape != valid.shape:
        raise ValueError("frequency_offset_hz and valid shapes differ")
    if frequency_offset_hz.shape[1] != physical_channel.size:
        raise ValueError("pilot axis does not match physical_channel length")

    metrics = _channel_metrics(
        physical_channel=physical_channel,
        frequency_offset_hz=frequency_offset_hz,
        peak_prominence_db=peak_prominence_db,
        valid=valid,
        min_peak_prominence_db=float(min_peak_prominence_db),
        recenter=bool(recenter),
    )
    reliable = [metric for metric in metrics if metric.reliable]
    reliable_channels = [metric.physical_channel for metric in reliable]
    allowed_fraction = capture_loss_allowed_bin_fraction(float(max_capture_loss_db))
    data.close()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "K",
        "reference_spacing_policy",
        "reference_offset_bins",
        "reference_offset_bins_computed",
        "reference_offset_bins_effective",
        "skipped_guard_bins",
        "skipped_guard_bins_computed",
        "skipped_guard_bins_effective",
        "reference_spacing_policy_note",
        "fine_bin_width_hz",
        "allowed_p95_residual_hz",
        "num_reliable_channels",
        "num_passing_channels",
        "passing_physical_channels",
        "failing_physical_channels",
        "recommended",
        "reason",
        "max_p95_abs_residual_hz",
        "reliable_physical_channels",
        "min_peak_prominence_db",
        "max_capture_loss_db",
        "allowed_bin_fraction",
        "recentered_by_channel_median",
        "reference_placement_status",
        "placement_warnings",
        "num_channels_with_adaptive_reference",
        "channels_with_adaptive_reference",
        "num_dc_shifted_references",
        "channels_with_dc_shifted_reference",
        "num_edge_wrapped_references",
        "channels_with_edge_wrapped_reference",
        "num_forbidden_tone_in_skipped_guard",
        "channels_with_forbidden_tone_in_skipped_guard",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k_value in candidate_ks:
            fine_bin_width_hz = sample_rate_hz / float(k_value)
            allowed_hz = allowed_fraction * fine_bin_width_hz
            passing = [
                metric
                for metric in reliable
                if metric.p95_abs_residual_hz <= allowed_hz
            ]
            failing = [
                metric for metric in reliable if metric.p95_abs_residual_hz > allowed_hz
            ]
            max_residual = (
                max(metric.p95_abs_residual_hz for metric in reliable)
                if reliable
                else float("nan")
            )
            for policy in policies:
                (
                    offset_bins_computed,
                    offset_bins_effective,
                    offset_policy_note,
                ) = (
                    _reference_offset_bins_for_policy(
                        policy=policy,
                        candidate_k=k_value,
                        reference_k=int(reference_k),
                        skipped_guard_bins=int(skipped_guard_bins),
                    )
                )
                is_baseline = (
                    int(k_value) == int(reference_k)
                    and str(policy) == "fixed_hz_reference_spacing"
                    and int(offset_bins_effective) == int(reference_offset_bins)
                )
                placement_summary = _candidate_reference_placement_summary(
                    physical_channel=physical_channel,
                    receiver_profile_path=Path(receiver_profile),
                    candidate_k=int(k_value),
                    reference_offset_bins=int(offset_bins_effective),
                )
                if not reliable:
                    reason = "no_reliable_channels_above_prominence_threshold"
                elif is_baseline:
                    reason = "validated_baseline"
                elif len(failing) == 0 and int(k_value) < int(reference_k):
                    reason = "passes_reliable_channels_lower_resolution"
                elif len(failing) == 0 and int(k_value) == int(reference_k):
                    reason = "passes_reliable_channels_baseline_equivalent"
                elif len(failing) == 0 and int(k_value) <= 256:
                    reason = "passes_reliable_channels_candidate_for_canfar"
                elif len(failing) == 0:
                    reason = "passes_reliable_channels_exploratory"
                else:
                    reason = "fails_reliable_channel_residual_criterion"
                writer.writerow(
                    {
                        "K": int(k_value),
                        "reference_spacing_policy": str(policy),
                        "reference_offset_bins": int(offset_bins_effective),
                        "reference_offset_bins_computed": int(offset_bins_computed),
                        "reference_offset_bins_effective": int(offset_bins_effective),
                        "skipped_guard_bins": int(
                            max(0, int(offset_bins_effective) - 1)
                        ),
                        "skipped_guard_bins_computed": int(
                            max(0, int(offset_bins_computed) - 1)
                        ),
                        "skipped_guard_bins_effective": int(
                            max(0, int(offset_bins_effective) - 1)
                        ),
                        "reference_spacing_policy_note": str(offset_policy_note),
                        "fine_bin_width_hz": float(fine_bin_width_hz),
                        "allowed_p95_residual_hz": float(allowed_hz),
                        "num_reliable_channels": int(len(reliable)),
                        "num_passing_channels": int(len(passing)),
                        "passing_physical_channels": _format_channel_list(
                            [metric.physical_channel for metric in passing]
                        ),
                        "failing_physical_channels": _format_channel_list(
                            [metric.physical_channel for metric in failing]
                        ),
                        "recommended": bool(is_baseline),
                        "reason": reason,
                        "max_p95_abs_residual_hz": float(max_residual),
                        "reliable_physical_channels": _format_channel_list(
                            reliable_channels
                        ),
                        "min_peak_prominence_db": float(min_peak_prominence_db),
                        "max_capture_loss_db": float(max_capture_loss_db),
                        "allowed_bin_fraction": float(allowed_fraction),
                        "recentered_by_channel_median": bool(recenter),
                        **placement_summary,
                    }
                )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CHIME detector K candidates from frequency offsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frequency-offset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_K),
    )
    parser.add_argument(
        "--candidate-reference-spacing-policy",
        choices=list(DEFAULT_REFERENCE_SPACING_POLICIES),
        nargs="+",
        default=list(DEFAULT_REFERENCE_SPACING_POLICIES),
    )
    parser.add_argument(
        "--min-peak-prominence-db",
        type=float,
        default=DEFAULT_MIN_PEAK_PROMINENCE_DB,
    )
    parser.add_argument(
        "--max-capture-loss-db",
        type=float,
        default=DEFAULT_MAX_CAPTURE_LOSS_DB,
    )
    parser.add_argument("--reference-k", type=int, default=DEFAULT_REFERENCE_K)
    parser.add_argument(
        "--skipped-guard-bins", type=int, default=DEFAULT_SKIPPED_GUARD_BINS
    )
    parser.add_argument(
        "--receiver-profile",
        type=Path,
        default=DEFAULT_RECEIVER_PROFILE,
        help="Detector-coordinate receiver profile used to audit reference placement.",
    )
    parser.add_argument("--no-recenter", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = choose_detector_k(
        frequency_offset=args.frequency_offset,
        output=args.output,
        candidate_k=args.candidate_k,
        candidate_reference_spacing_policy=args.candidate_reference_spacing_policy,
        min_peak_prominence_db=float(args.min_peak_prominence_db),
        max_capture_loss_db=float(args.max_capture_loss_db),
        reference_k=int(args.reference_k),
        skipped_guard_bins=int(args.skipped_guard_bins),
        receiver_profile=args.receiver_profile,
        recenter=not bool(args.no_recenter),
    )
    print(f"k_candidate_summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
