# coding=utf-8
"""Combine per-pilot datatrawl analyzer products into PilotProxy's canonical products.

The detector analyzer fans out one ``<channel>.npz`` per coarse channel.
This step stacks those per-pilot products along the pilot axis, aligning
frames by (event, frame-in-file) identity, so pilots that processed different
event sets stack over exactly their common identities with drops reported --
and feeds the SAME writer functions ``run_chime_analysis`` uses, so the combined
``chime_detector_outputs`` / ``chime_spectrogram_cache`` / ``chime_reductions_10s``
/ ``mask_summary`` are byte-identical to a single-process run, which is what
keeps the existing plots and ``validate-products`` working unchanged on datatrawl
output.

A "per-pilot product" is exactly what ``PilotProxyDetectorAnalyzer.save`` writes:
the relevant fstat schema for one pilot, with
the per-frame 2-D arrays shaped ``(frames, 1)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pilot_proxy.chime.products import (
    ensure_run_dirs,
    write_detector_outputs,
    write_integrated_spectra,
    write_mask_summary,
    write_spectrogram_cache,
)
from pilot_proxy.chime.reductions import write_reductions_npz
from pilot_proxy.detector_contract import (
    CHIME_RUN_CONFIG_SCHEMA_VERSION,
    CHIME_STATS_SCHEMA_VERSION,
    normalized_positive_excess_policy,
)
from pilot_proxy.provenance import (
    detector_version_build_id,
    detector_version_geometry,
)
from pilot_proxy.product_contract import (
    CurrentProductContractError,
    validate_current_product_identity,
)
import json


def _write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


class CombineEmptyIntersectionError(ValueError):
    """No (event, frame) identity is shared by every product handed to combine."""


def _label(z: Mapping[str, Any]) -> str:
    ch = int(np.asarray(z["physical_channel"]).reshape(-1)[0])
    fid = int(np.asarray(z["freq_id"]).reshape(-1)[0])
    return f"ch{ch}/freq_id {fid}"


# Every per-frame array the analyzer writes (length n_frames along axis 0).
# Event-keyed alignment gathers exactly these; everything else in a product is
# per-pilot (scalars), per-unit (time/provenance axes), or per-bin (spectra).
_PER_FRAME_KEYS = (
    "frame_index", "p_target_u64", "p_ref_sum_u64", "coarse_power_ratio",
    "normalized_coarse_power_ratio_db", "pilot_excess_db", "estimated_data_shelf_snr_db", "normalized_pilot_excess",
    "reject_mask", "valid", "baseband_power_linear",
    "frame_unit_index", "frame_in_unit",
)


def _align_frames(
    products: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Event-keyed frame alignment: subset and reorder every product onto the
    per-frame identities they all share.

    A frame's identity is ``(source event, frame position within its file)``,
    which the analyzer records for every frame. The canonical order is the
    reference (lowest-channel) product's own order restricted to the common
    identities, so a fully aligned set passes through untouched (byte-parity
    with ``run_chime_analysis``) and a ragged set stacks exactly its overlap,
    reporting what each pilot dropped. Event-keyed identity is mandatory.
    """
    identities = [_frame_identity(z) for z in products]
    for z, ids in zip(products, identities):
        n = int(np.asarray(z["frame_index"]).reshape(-1).size)
        if ids.size != n:
            raise ValueError(
                "combine: frame identity length does not match frame_index length")
        if len(set(ids.tolist())) != ids.size:
            raise ValueError(
                f"combine: {_label(z)} contains duplicate (event, frame) "
                f"identities; one acquisition appears twice in that product")
    sets = [set(ids.tolist()) for ids in identities]
    common = set.intersection(*sets)
    if not common:
        counts = ", ".join(
            f"{_label(z)}: {len(s)} frames/"
            f"{len({i.split(chr(0))[0] for i in s})} events"
            for z, s in zip(products, sets))
        raise CombineEmptyIntersectionError(
            f"combine: the {len(products)} per-pilot products share no common "
            f"(event, frame) identity; there is nothing every pilot saw, so an "
            f"event-keyed stack over all of them is empty. Per-pilot inventory: "
            f"{counts}. Stack a channel subset instead (`pilot-proxy "
            f"chime-combine --report` shows the presence histogram and the "
            f"drop-curve; `--drop <freq_ids>` excludes channels).")
    ref_ids = identities[0].tolist()
    canonical = [i for i in ref_ids if i in common]
    aligned: list[dict[str, Any]] = []
    by_pilot: list[dict[str, Any]] = []
    kept_events = {i.split("\0")[0] for i in canonical}
    for z, ids in zip(products, identities):
        pos = {i: r for r, i in enumerate(ids.tolist())}
        rows = np.asarray([pos[i] for i in canonical], dtype=np.int64)
        out = dict(z)
        for key in _PER_FRAME_KEYS:
            if key in out:
                out[key] = np.asarray(out[key])[rows]
        aligned.append(out)
        pilot_events = {i.split("\0")[0] for i in ids.tolist()}
        by_pilot.append({
            "physical_channel": int(np.asarray(z["physical_channel"]).reshape(-1)[0]),
            "freq_id": (int(np.asarray(z["freq_id"]).reshape(-1)[0])
                        if "freq_id" in z else None),
            "n_frames_total": int(ids.size),
            "n_frames_dropped": int(ids.size - len(canonical)),
            "n_events_total": len(pilot_events),
            "n_events_dropped": len(pilot_events - kept_events),
        })
    info = {
        "mode": "event_keyed",
        "n_frames_common": len(canonical),
        "n_events_common": len(kept_events),
        "by_pilot": by_pilot,
        "frame_event_key": [i.split("\0")[0] for i in canonical],
        "frame_in_unit": [int(i.split("\0")[1]) for i in canonical],
    }
    dropped = [p for p in by_pilot if p["n_frames_dropped"]]
    if dropped:
        detail = ", ".join(
            f"ch{p['physical_channel']}"
            + (f"/freq_id {p['freq_id']}" if p["freq_id"] is not None else "")
            + f": -{p['n_frames_dropped']} frames/-{p['n_events_dropped']} events"
            for p in dropped)
        print(
            f"[combine] event-keyed alignment: kept {len(canonical)} frame(s) / "
            f"{len(kept_events)} event(s) common to {len(products)} pilot(s); "
            f"dropped {detail}", flush=True)
    frame_index = np.arange(len(canonical), dtype=np.int64)
    return aligned, frame_index, info


def report_products(product_paths: Sequence[str | Path]) -> str:
    """Event-presence report for a set of per-pilot products: per-pilot counts,
    the presence histogram, the all-pilot intersection, and the greedy
    drop-curve (intersection after removing the most-constraining pilot,
    repeatedly). This is the decision input for choosing a combine subset."""
    import collections
    ev: dict[str, set[str]] = {}
    for p in product_paths:
        with np.load(str(p), allow_pickle=False) as z:
            product = {name: z[name] for name in z.files}
        validate_current_product_identity(product)
        label = (
            f"ch{int(np.asarray(product['physical_channel']).reshape(-1)[0])}"
            + f"/freq_id {int(np.asarray(product['freq_id']).reshape(-1)[0])}"
        )
        events = set(
            np.asarray(product["source_event_keys"]).reshape(-1).astype(str).tolist()
        )
        ev[label] = events
    lines = [f"per-pilot products: {len(ev)}"]
    for label in sorted(ev):
        lines.append(f"  {label}: {len(ev[label])} events")
    if not ev or not any(ev.values()):
        lines.append("no event metadata present; report unavailable")
        return "\n".join(lines)
    union = set().union(*ev.values())
    presence = collections.Counter()
    for s in ev.values():
        for e in s:
            presence[e] += 1
    hist = collections.Counter(presence.values())
    lines.append(f"union: {len(union)} distinct events")
    lines.append("events by how many pilots hold them: "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())))
    lines.append(f"intersection of all {len(ev)} pilots: "
                 f"{len(set.intersection(*ev.values()))}")
    work = dict(ev)
    lines.append("drop-curve (removing the most-constraining pilot each step):")
    while len(work) > max(2, len(ev) // 2):
        best = None
        for c in work:
            n = len(set.intersection(*(work[x] for x in work if x != c)))
            if best is None or n > best[1]:
                best = (c, n)
        c, n = best
        del work[c]
        lines.append(f"  drop {c}: intersection of remaining {len(work)} = {n}")
    return "\n".join(lines)


def _detector_contract_from(
    products: Sequence[Mapping[str, Any]], nfft: int
) -> dict[str, Any]:
    """Return the required analyzer-stored detector contract."""
    del nfft  # retained in the call signature until the combine refactor lands
    try:
        contract = json.loads(
            str(np.asarray(products[0]["detector_contract_json"]).reshape(()).item())
        )
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "combine: current per-pilot product lacks a valid detector_contract_json"
        ) from exc
    if not isinstance(contract, dict) or not contract:
        raise ValueError(
            "combine: detector_contract_json must encode a non-empty object"
        )
    return contract


def _load_sorted(product_paths: Sequence[str | Path]) -> list[Mapping[str, Any]]:
    if not product_paths:
        raise ValueError("combine: no per-pilot product files given")
    products: list[dict[str, Any]] = []
    for path in product_paths:
        with np.load(str(path), allow_pickle=False) as product:
            loaded = {name: product[name] for name in product.files}
        try:
            validate_current_product_identity(loaded)
        except CurrentProductContractError as exc:
            raise ValueError(f"combine: {path}: {exc}") from exc
        products.append(loaded)
    products.sort(
        key=lambda z: int(np.asarray(z["physical_channel"]).reshape(-1)[0])
    )
    chans = [
        int(np.asarray(z["physical_channel"]).reshape(-1)[0]) for z in products
    ]
    dupes = sorted({channel for channel in chans if chans.count(channel) > 1})
    if dupes:
        raise ValueError(
            f"combine: ATSC physical channel(s) {dupes} appear in more than one "
            "per-pilot product. The combined schema is one pilot per ATSC "
            "channel; drop the duplicate receiver channel."
        )
    return products


def _frame_identity(z: Mapping[str, Any]) -> np.ndarray:
    required = {"source_event_keys", "frame_unit_index", "frame_in_unit"}
    missing = sorted(required.difference(z))
    if missing:
        raise ValueError(
            "combine: current per-pilot product is missing frame identity arrays: "
            + ", ".join(missing)
        )
    events = np.asarray(z["source_event_keys"]).reshape(-1).astype(str)
    unit_index = np.asarray(z["frame_unit_index"], dtype=np.int64).reshape(-1)
    frame_in_unit = np.asarray(z["frame_in_unit"], dtype=np.int64).reshape(-1)
    if unit_index.shape != frame_in_unit.shape:
        raise ValueError("combine: frame_unit_index and frame_in_unit shapes differ")
    if np.any(unit_index < 0) or np.any(unit_index >= events.size):
        raise ValueError("combine: frame_unit_index contains an out-of-range unit")
    return np.asarray(
        [f"{events[u]}\0{int(f)}" for u, f in zip(unit_index, frame_in_unit)],
        dtype=str,
    )


def _version_geometry(version: str) -> tuple:
    """The geometry-bearing tokens of a detector_version string: everything
    except the `pilot-proxy/<version>` and `source=<tree hash>` tokens, which
    are build provenance. A release version bump, or patches applied
    mid-survey, change those without touching detector math; the kernel hash,
    K, and schema tag are what stacking correctness needs. Defined once in
    pilot_proxy.provenance and shared with the detector's resume check."""
    return detector_version_geometry(version)


def _check_invariants(products: Sequence[Mapping[str, Any]],
                      keys, what: str) -> dict[str, Any]:
    """Assert all per-pilot products agree on geometry/config scalars before stacking.

    The combiner takes per-pilot frame arrays and the first product's metadata; it
    must verify the rest of the products were produced with the same geometry/config
    (nfft, K, spectral sense, schema, sample rate), or stacking would silently fuse
    inconsistent products into one canonical output.

    `detector_version` gets token-aware treatment: its `pilot-proxy/<version>`
    and `source=` components are build provenance rather than geometry, so products
    from different mid-survey builds (including builds on either side of a
    release version bump) stack freely as long as every other token (kernel
    hash, K, schema) matches. Returns provenance notes:
    {"detector_versions": [...]} when more than one build contributed, so the
    full stamps survive into the combined product.
    """
    notes: dict[str, Any] = {}
    ref = products[0]
    for key in keys:
        if key not in ref:
            raise ValueError(
                f"combine: current per-pilot product is missing {key!r}, needed "
                f"to verify {what}"
            )
        if key == "detector_version":
            versions = []
            for z in products:
                if key not in z:
                    raise ValueError(
                        f"combine: a product is missing '{key}', needed to "
                        f"verify {what}.")
                versions.append(str(np.asarray(z[key]).reshape(-1)[0]))
            geoms = {_version_geometry(v) for v in versions}
            if len(geoms) > 1:
                raise ValueError(
                    f"combine: per-pilot products disagree on detector_version "
                    f"geometry tokens ({sorted(' '.join(g) for g in geoms)!r}); "
                    f"refusing to stack mismatched {what}.")
            distinct = sorted(set(versions))
            if len(distinct) > 1:
                notes["detector_versions"] = distinct
                short = ", ".join(detector_version_build_id(v) for v in distinct)
                print(f"[combine] provenance: {len(distinct)} source builds "
                      f"with identical detector geometry contributed "
                      f"(builds={short}); stacking.", flush=True)
            continue
        base = np.asarray(ref[key]).reshape(-1)
        for z in products[1:]:
            if key not in z:
                raise ValueError(
                    f"combine: a product is missing '{key}', needed to verify {what}."
                )
            other = np.asarray(z[key]).reshape(-1)
            if base.shape != other.shape or not np.array_equal(base, other):
                raise ValueError(
                    f"combine: per-pilot products disagree on '{key}' "
                    f"({base.tolist()!r} vs {other.tolist()!r}); refusing to stack "
                    f"mismatched {what}."
                )
    return notes


def _common_sample_rate_hz(products: Sequence[Mapping[str, Any]]) -> float:
    """Return a shared sample rate, refusing mixed or partially missing timing."""
    per_product: list[np.ndarray] = []
    for z in products:
        values = np.asarray(z.get("unit_delta_time", []), dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values) & (values > 0.0)]
        per_product.append(finite)
    if not any(values.size for values in per_product):
        return float("nan")
    if not all(values.size for values in per_product):
        raise ValueError(
            "combine: timing metadata is present for only some per-pilot products"
        )
    reference = float(per_product[0][0])
    for values in per_product:
        if not np.allclose(values, reference, rtol=1e-12, atol=0.0):
            raise ValueError(
                "combine: per-pilot products disagree on unit_delta_time; refusing "
                "to construct a shared spectral frequency axis"
            )
    return float(1.0 / reference)


def _json_scalar(z: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = z.get(key)
    if raw is None:
        return {}
    try:
        value = json.loads(str(np.asarray(raw).reshape(()).item()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _combined_reference_placement_summary(
    products: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    parsed = [_json_scalar(z, "reference_placement_json") for z in products]
    present = [bool(summary) for summary in parsed]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "combine: reference-placement metadata is present for only some pilots"
        )
    summaries = parsed
    offsets = {int(summary.get("reference_offset_bins", 0)) for summary in summaries}
    guards = {int(summary.get("skipped_guard_bins", 0)) for summary in summaries}
    if len(offsets) != 1 or len(guards) != 1:
        raise ValueError("combine: reference-placement geometry differs between pilots")
    by_channel = [
        row
        for summary in summaries
        for row in summary.get("by_channel", [])
        if isinstance(row, dict)
    ]
    statuses = sorted({str(row.get("reference_placement_status", "unknown")) for row in by_channel})
    status = statuses[0] if len(statuses) == 1 else "mixed:" + ";".join(statuses)
    adaptive = [
        int(row["physical_channel"])
        for row in by_channel
        if str(row.get("reference_placement_status", "unknown")) != "nominal"
    ]
    dc_shifted = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("dc_reference_shifted", False))
    ]
    edge_wrapped = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("edge_reference_wrapped", False))
    ]
    skipped_guard = [
        int(row["physical_channel"])
        for row in by_channel
        if bool(row.get("forbidden_tone_in_skipped_guard", False))
    ]
    return {
        "reference_offset_bins": offsets.pop(),
        "skipped_guard_bins": guards.pop(),
        "reference_placement_status": status,
        "num_channels_with_adaptive_reference": len(adaptive),
        "channels_with_adaptive_reference": adaptive,
        "num_dc_shifted_references": sum(
            int(summary.get("num_dc_shifted_references", 0)) for summary in summaries
        ),
        "channels_with_dc_shifted_reference": dc_shifted,
        "num_edge_wrapped_references": sum(
            int(summary.get("num_edge_wrapped_references", 0)) for summary in summaries
        ),
        "channels_with_edge_wrapped_reference": edge_wrapped,
        "num_forbidden_tone_in_skipped_guard": len(skipped_guard),
        "channels_with_forbidden_tone_in_skipped_guard": skipped_guard,
        "forbidden_tone_policy": summaries[0].get("forbidden_tone_policy"),
        "by_channel": by_channel,
    }


def _stack_cols(products: Sequence[Mapping[str, Any]], key: str, dtype) -> np.ndarray:
    """Stack per-pilot (frames, 1) arrays into (frames, pilots)."""
    cols = [np.asarray(z[key], dtype=dtype).reshape(int(np.asarray(z[key]).shape[0]), 1)
            for z in products]
    return np.concatenate(cols, axis=1)


def _scalars(products: Sequence[Mapping[str, Any]], key: str, dtype) -> np.ndarray:
    return np.asarray(
        [np.asarray(z[key]).reshape(-1)[0] for z in products], dtype=dtype
    )


def combine_detector_products(
    product_paths: Sequence[str | Path],
    run_dir: str | Path,
    *,
    chunk_seconds: float = 10.0,
    drop_freq_ids: Sequence[int] | None = None,
) -> dict[str, Path]:
    """Stack per-pilot detector products and write the canonical detector products.

    Frames are aligned by (event, frame-in-file) identity: pilots that
    processed different event sets stack over their common identities, with
    per-pilot drops reported and recorded. ``drop_freq_ids`` excludes whole
    pilots up front (the subset-selection knob the drop-curve report feeds).
    """
    products = _load_sorted(product_paths)
    if drop_freq_ids:
        drop = {int(f) for f in drop_freq_ids}
        kept = [z for z in products
                if int(np.asarray(z.get("freq_id", -1)).reshape(-1)[0]) not in drop]
        excluded = len(products) - len(kept)
        if not kept:
            raise ValueError("combine: --drop excluded every per-pilot product")
        if excluded:
            print(f"[combine] --drop excluded {excluded} pilot(s): "
                  f"{sorted(drop)}", flush=True)
        products = kept
    invariant_notes = _check_invariants(
        products,
        ("schema_version", "nfft", "detector_window_samples", "sense",
         "detector_contract_json", "max_chunks_per_file", "num_input_streams",
         "weight_bank_sha256", "weight_manifest_sha256", "mask_rule",
         "detector_version", "pilot_below_data_db", "bin_enbw_hz",
         "dtv_bandwidth_hz", "pilot_capture_efficiency"),
        "detector geometry",
    )
    products_full = products
    products, frame_index, align_info = _align_frames(products_full)
    nfft = int(np.asarray(products[0]["nfft"]))

    # per-channel diagnostic paired with the integrated spectra, which are
    # accumulated at analyzer time over each pilot's FULL processed frame set
    # and cannot be re-subset here, so it is computed over the full set to match.
    def _masked_fraction(z: Mapping[str, Any]) -> float:
        rej = np.asarray(z["reject_mask"]).reshape(-1).astype(np.float64)
        n_valid = float(np.asarray(z["valid"]).reshape(-1).sum())
        return float(rej.sum() / n_valid) if n_valid > 0 else float("nan")

    masked_fraction = np.asarray(
        [_masked_fraction(z) for z in products_full], np.float64)

    physical_channel = _scalars(products, "physical_channel", np.int32)
    pilot_frequency_hz = _scalars(products, "pilot_frequency_hz", np.float64)
    chime_frequency_hz = _scalars(products, "chime_frequency_hz", np.float64)
    # freq_id is required by the current schema and identifies the receiver
    # coarse channel used by the later 6 MHz mask-expansion step.
    freq_id = _scalars(products, "freq_id", np.int64)

    p_target_u64 = _stack_cols(products, "p_target_u64", np.uint64)
    p_ref_sum_u64 = _stack_cols(products, "p_ref_sum_u64", np.uint64)
    coarse_power_ratio = _stack_cols(products, "coarse_power_ratio", np.float64)
    normalized_coarse_power_ratio_db = _stack_cols(products, "normalized_coarse_power_ratio_db", np.float64)
    pilot_excess_db = _stack_cols(products, "pilot_excess_db", np.float64)
    estimated_data_shelf_snr_db = _stack_cols(products, "estimated_data_shelf_snr_db", np.float64)
    # per-channel products renamed `mask` -> `reject_mask` at schema v2 (1 = discard,
    # positive excess); the canonical combined outputs keep the `mask` field name, so
    # only this read changes; write_* below stays byte-identical.
    mask = _stack_cols(products, "reject_mask", np.uint8)
    valid = _stack_cols(products, "valid", np.uint8)
    baseband_power_linear = _stack_cols(products, "baseband_power_linear", np.float64)
    # The current schema always carries the exact quantized-weight null point.
    target_norm_sq = _scalars(products, "target_norm_sq", np.int64)
    reference_norm_sum_sq = _scalars(products, "reference_norm_sum_sq", np.int64)
    null_power_ratio = _scalars(products, "null_power_ratio", np.float64)
    normalized_pilot_excess = _stack_cols(
        products, "normalized_pilot_excess", np.float64
    )

    # integrated spectra are per-channel 1-D [nfft] (not per-frame): stack along the
    # pilot axis -> [n_pilots, nfft]. masked fraction = valid-and-rejected / valid
    # per channel (NaN if a channel has no valid frames, e.g. out-of-band).
    spec_before = np.stack([
        np.asarray(z["integrated_spectrum_before_mask"], np.float64).reshape(-1)
        for z in products])
    spec_after = np.stack([
        np.asarray(z["integrated_spectrum_after_mask"], np.float64).reshape(-1)
        for z in products])

    # Sample rate for the spectra frequency axis is shared only when every
    # per-pilot product carries consistent timing metadata.
    sample_rate_hz = _common_sample_rate_hz(products)

    run_dir = Path(run_dir)
    ensure_run_dirs(run_dir)
    outputs: dict[str, Path] = {}
    outputs["detector_outputs"] = write_detector_outputs(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        p_target_u64=p_target_u64,
        p_ref_sum_u64=p_ref_sum_u64,
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        pilot_excess_db=pilot_excess_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        mask=mask,
        valid=valid,
        target_norm_sq=target_norm_sq,
        reference_norm_sum_sq=reference_norm_sum_sq,
        null_power_ratio=null_power_ratio,
        normalized_pilot_excess=normalized_pilot_excess,
    )
    outputs["spectrogram_cache"] = write_spectrogram_cache(
        run_dir,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        frame_index=frame_index,
        frame_size_samples=nfft,
        valid=valid,
    )
    outputs["integrated_spectra"] = write_integrated_spectra(
        run_dir,
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        integrated_spectrum_before_mask=spec_before,
        integrated_spectrum_after_mask=spec_after,
        masked_fraction_by_channel=masked_fraction,
        sample_rate_hz=sample_rate_hz,
        nfft=nfft,
        freq_id=freq_id,
    )
    outputs["reductions_10s"] = write_reductions_npz(
        run_dir,
        frame_index=frame_index,
        frame_size_samples=nfft,
        chunk_seconds=float(chunk_seconds),
        coarse_power_ratio=coarse_power_ratio,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    outputs["mask_summary"] = write_mask_summary(
        run_dir,
        physical_channel=[int(v) for v in physical_channel],
        pilot_frequency_hz=[float(v) for v in pilot_frequency_hz],
        chime_frequency_hz=[float(v) for v in chime_frequency_hz],
        mask=mask,
        valid=valid,
    )

    # run_config / stats / input_manifest, so validate-products accepts scan output.
    # These carry the schema-gated fields (detector_contract, mask_policy, geometry)
    # honestly labeled as chime-scan provenance rather than a byte-faithful imitation of
    # a single run_chime_analysis run.
    contract = _detector_contract_from(products, nfft)
    reference_placement = _combined_reference_placement_summary(products)
    if reference_placement is not None:
        contract = dict(contract)
        contract["reference_placement_summary"] = reference_placement
    mask_policy = normalized_positive_excess_policy()
    k = int(contract["detector_window_samples"])
    provenance_by_pilot = []
    for z in products:
        provenance_by_pilot.append({
            "physical_channel": int(np.asarray(z["physical_channel"]).reshape(-1)[0]),
            "freq_id": int(np.asarray(z["freq_id"]).reshape(-1)[0]),
            "weights_hash": str(np.asarray(z.get("weights_hash", "")).reshape(()).item()),
            "weight_bank_sha256": str(
                np.asarray(z.get("weight_bank_sha256", "")).reshape(()).item()
            ),
            "weight_manifest_sha256": str(
                np.asarray(z.get("weight_manifest_sha256", "")).reshape(()).item()
            ),
            "detector_version": str(np.asarray(z.get("detector_version", "")).reshape(()).item()),
            "mask_rule": str(np.asarray(z.get("mask_rule", "")).reshape(()).item()),
        })
    common = {
        "source": "chime-scan",
        "physical_channels": [int(v) for v in physical_channel],
        "pilot_frequency_hz": [float(v) for v in pilot_frequency_hz],
        "chime_frequency_hz": [float(v) for v in chime_frequency_hz],
        "frame_size_samples": int(nfft),
        "detector_window_samples": k,
        "num_input_streams": int(np.asarray(products[0].get("num_input_streams", 0))),
        "mask_policy": mask_policy,
        "detector_contract": contract,
        "detector_provenance_by_pilot": provenance_by_pilot,
    }
    if reference_placement is not None:
        common["reference_placement_summary"] = reference_placement
    common["freq_id_by_pilot"] = [int(v) for v in freq_id]
    _write_json(run_dir / "run_config.json",
                {"schema_version": CHIME_RUN_CONFIG_SCHEMA_VERSION, **common})
    if align_info.get("mode") == "event_keyed":
        identity_path = run_dir / "chime_frame_identity.npz"
        np.savez_compressed(
            str(identity_path),
            frame_event_key=np.asarray(align_info["frame_event_key"], dtype=str),
            frame_in_unit=np.asarray(align_info["frame_in_unit"], dtype=np.int64),
        )
        outputs["frame_identity"] = identity_path
    stats_alignment = {
        k: v for k, v in align_info.items()
        if k not in ("frame_event_key", "frame_in_unit")
    }
    _write_json(run_dir / "stats.json", {
        "schema_version": CHIME_STATS_SCHEMA_VERSION,
        "num_frames": int(frame_index.size),
        "num_pilots": len(products),
        "combine_alignment": stats_alignment,
        **invariant_notes,
        "windows_per_stream": int(nfft) // k,
        "rational_overflow_count_by_pilot": [
            int(np.asarray(z.get("rational_overflow_count", 0))) for z in products
        ],
        **common,
    })
    _write_json(run_dir / "input_manifest.json", {
        "schema_version": "fstat_chime_scan_input_manifest_v1",
        "source": "chime-scan",
        "physical_channels": [int(v) for v in physical_channel],
        "input_files": sorted({
            str(x) for z in products
            for x in np.asarray(z.get("unit_keys", np.asarray([], dtype=object)))
            .reshape(-1).tolist()
        }),
    })
    outputs["run_config"] = run_dir / "run_config.json"
    outputs["stats"] = run_dir / "stats.json"
    outputs["input_manifest"] = run_dir / "input_manifest.json"
    return outputs


__all__ = [
    "CombineEmptyIntersectionError",
    "combine_detector_products",
    "report_products",
]
