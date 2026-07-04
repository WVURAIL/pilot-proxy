# coding=utf-8
"""Combine per-pilot datatrawl analyzer products into PilotProxy's canonical products.

The detector analyzer fans out one ``<channel>.npz`` per coarse channel.
This step stacks those per-pilot products along the pilot axis -- aligning
frames by (event, frame-in-file) identity, so pilots that processed different
event sets stack over exactly their common identities with drops reported --
and feeds the SAME writer functions ``run_chime_analysis`` uses, so the combined
``chime_detector_outputs`` / ``chime_spectrogram_cache`` / ``chime_reductions_10s``
/ ``mask_summary`` are byte-identical to a single-process run -- which is what
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
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    build_chime_detector_contract,
    positive_excess_mask_policy,
)
import json


def _write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


class CombineEmptyIntersectionError(ValueError):
    """No (event, frame) identity is shared by every product handed to combine."""


def _label(z: Mapping[str, Any]) -> str:
    ch = int(np.asarray(z["physical_channel"]).reshape(-1)[0])
    if "freq_id" in z:
        return f"ch{ch}/freq_id {int(np.asarray(z['freq_id']).reshape(-1)[0])}"
    return f"ch{ch}"


# Every per-frame array the analyzer writes (length n_frames along axis 0).
# Event-keyed alignment gathers exactly these; everything else in a product is
# per-pilot (scalars), per-unit (time/provenance axes), or per-bin (spectra).
_PER_FRAME_KEYS = (
    "frame_index", "p_target_u64", "p_ref_sum_u64", "fstat_raw",
    "fstat_level_db", "pnr_bin_db", "snr_shelf_db", "pilot_excess_corrected",
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
    reporting what each pilot dropped. Products predating the identity tags
    fall back to the strict positional check unchanged.
    """
    identities = [_frame_identity(z) for z in products]
    if not all(identity is not None for identity in identities):
        # legacy products: strict positional semantics, exactly as before
        return [dict(z) for z in products], _check_frames(products), {
            "mode": "strict_positional"}
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
            f"(event, frame) identity -- there is nothing every pilot saw, so an "
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
        with np.load(str(p)) as z:
            label = (f"ch{int(np.asarray(z['physical_channel']).reshape(-1)[0])}"
                     + (f"/freq_id {int(np.asarray(z['freq_id']).reshape(-1)[0])}"
                        if "freq_id" in z.files else ""))
            events = set(np.asarray(z["source_event_keys"]).reshape(-1).astype(str)
                         .tolist()) if "source_event_keys" in z.files else set()
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


def _detector_contract_from(products: Sequence[Mapping[str, Any]], nfft: int) -> dict:
    """Prefer the analyzer-stored contract; else rebuild one from product geometry."""
    raw = products[0].get("detector_contract_json")
    if raw is not None:
        try:
            c = json.loads(str(np.asarray(raw)))
            if isinstance(c, dict) and c:
                return c
        except Exception:  # pragma: no cover - fall back to a rebuilt contract
            pass
    sense = int(np.asarray(products[0].get("sense", 1)))
    k = int(np.asarray(products[0]["detector_window_samples"]))
    time_reverse = sense == -1
    return build_chime_detector_contract(
        detector_window_samples=k,
        skipped_guard_bins=1,
        reference_offset_bins=2,
        num_weight_terms=3,
        weight_coordinate_system=(
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE if time_reverse
            else WEIGHT_COORDINATE_RAW_INPUT
        ),
        time_reverse_detector_windows_before_kernel=time_reverse,
    )


def _load_sorted(product_paths: Sequence[str | Path]) -> list[Mapping[str, Any]]:
    if not product_paths:
        raise ValueError("combine: no per-pilot product files given")
    products = [dict(np.load(str(p))) for p in product_paths]
    products.sort(key=lambda z: int(np.asarray(z["physical_channel"]).reshape(-1)[0]))
    chans = [int(np.asarray(z["physical_channel"]).reshape(-1)[0]) for z in products]
    dupes = sorted({c for c in chans if chans.count(c) > 1})
    if dupes:
        # Two coarse channels (freq_ids) that map to the same ATSC physical channel.
        raise ValueError(
            f"combine: ATSC physical channel(s) {dupes} appear in more than one "
            f"per-pilot product. The combined schema is one pilot per ATSC channel, "
            f"so stacking two freq_ids that resolve to the same channel would put "
            f"two pilots under one label. Drop one, or extend the schema to "
            f"represent multiple coarse channels per ATSC channel."
        )
    return products


def _frame_identity(z: Mapping[str, Any]) -> np.ndarray | None:
    required = {"source_event_keys", "frame_unit_index", "frame_in_unit"}
    if not required.issubset(z):
        return None
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


def _check_frames(products: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return the shared frame grid, or fail with a diagnostic if pilots disagree.

    Each product's ``frame_index`` is a 0-based *positional* counter (the analyzer
    writes ``arange(n_frames)``), so it is only comparable across pilots when they
    processed the *same files in the same order*. A length/grid mismatch therefore
    means the pilots saw different files -- typically a quarantined or missing file
    for one channel -- which shifts every subsequent frame in time. Stacking then
    would silently fuse time-misaligned frames, so we refuse and explain instead of
    intersecting (a positional intersection across differing file sets is a
    footgun, not a fix).
    """
    ref_fi = np.asarray(products[0]["frame_index"], dtype=np.int64)
    grids = [np.asarray(z["frame_index"], dtype=np.int64) for z in products]
    grids_match = all(
        g.shape == ref_fi.shape and np.array_equal(g, ref_fi) for g in grids[1:]
    )

    # Frame counts matching is necessary but NOT sufficient: frame_index is a
    # 0-based positional counter, so two pilots with the same count but different
    # source events (e.g. freq_id 829 over events A,B and freq_id 844 over A,C)
    # would also pass. Compare the per-pilot ordered source-event keys (the file
    # identities with each product's own freq_id token removed) so that only
    # pilots that saw the same events in the same order are accepted.
    have_events = all("source_event_keys" in z for z in products)
    events = (
        [np.asarray(z["source_event_keys"]).reshape(-1).tolist() for z in products]
        if have_events else None
    )
    events_match = events is None or all(e == events[0] for e in events[1:])
    identities = [_frame_identity(z) for z in products]
    present_identities = [identity is not None for identity in identities]
    if any(present_identities) and not all(present_identities):
        raise ValueError(
            "combine: only some per-pilot products contain per-frame identity tags"
        )
    have_identities = all(present_identities)
    identity_match = True
    if have_identities:
        ref_identity = identities[0]
        assert ref_identity is not None
        if ref_identity.shape != ref_fi.shape:
            raise ValueError(
                "combine: frame identity length does not match frame_index length"
            )
        identity_match = all(
            identity is not None
            and identity.shape == ref_identity.shape
            and np.array_equal(identity, ref_identity)
            for identity in identities[1:]
        )

    if grids_match and events_match and identity_match:
        return ref_fi

    lines = []
    for i, z in enumerate(products):
        g = grids[i]
        rng = f"[{int(g[0])}..{int(g[-1])}]" if g.size else "[empty]"
        ev = (f", events={events[i]}" if events is not None else "")
        lines.append(f"  {_label(z)}: {g.size} frames {rng}{ev}")
    if not grids_match:
        why = ("different frame counts -- the pilots processed different numbers "
               "of files")
    elif not events_match:
        why = ("equal frame counts but different source events -- the pilots "
               "processed different acquisitions, so frame N is a different time "
               "in each")
    else:
        why = ("equal frame counts and event order but different per-frame unit "
               "positions -- a file has a different number of frames for at least "
               "one pilot")
    raise ValueError(
        "combine: per-pilot products are not time-aligned and cannot be stacked "
        f"({why}). Because frame_index is a 0-based positional counter, frames only "
        "align when every pilot saw the same events in the same order; a "
        "quarantined/missing file for one channel shifts every later frame in "
        "time. Fix the inputs -- re-pull the missing file(s) for the short "
        "channel(s), or drop the affected channel(s) from this combine.\n"
        "Per-pilot frame inventory:\n" + "\n".join(lines)
    )


def _check_invariants(products: Sequence[Mapping[str, Any]], keys, what: str) -> None:
    """Assert all per-pilot products agree on geometry/config scalars before stacking.

    The combiner takes per-pilot frame arrays and the first product's metadata; it
    must verify the rest of the products were produced with the same geometry/config
    (nfft, K, spectral sense, schema, sample rate), or stacking would silently fuse
    inconsistent products into one canonical output.
    """
    ref = products[0]
    for key in keys:
        if key not in ref:
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
    _check_invariants(
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
    # and cannot be re-subset here -- computed over the full set to match.
    def _masked_fraction(z: Mapping[str, Any]) -> float:
        rej = np.asarray(z["reject_mask"]).reshape(-1).astype(np.float64)
        n_valid = float(np.asarray(z["valid"]).reshape(-1).sum())
        return float(rej.sum() / n_valid) if n_valid > 0 else float("nan")

    masked_fraction = np.asarray(
        [_masked_fraction(z) for z in products_full], np.float64)

    physical_channel = _scalars(products, "physical_channel", np.int32)
    pilot_frequency_hz = _scalars(products, "pilot_frequency_hz", np.float64)
    chime_frequency_hz = _scalars(products, "chime_frequency_hz", np.float64)
    # freq_id is recorded by the analyzer (derived from the centre frequency); it
    # is the coarse-channel handle the deferred 6 MHz mask-expansion needs to find
    # a detected pilot's sibling channels. Guard for older products without it.
    freq_id = (
        _scalars(products, "freq_id", np.int64)
        if all("freq_id" in z for z in products)
        else None
    )

    p_target_u64 = _stack_cols(products, "p_target_u64", np.uint64)
    p_ref_sum_u64 = _stack_cols(products, "p_ref_sum_u64", np.uint64)
    fstat_raw = _stack_cols(products, "fstat_raw", np.float64)
    fstat_level_db = _stack_cols(products, "fstat_level_db", np.float64)
    pnr_bin_db = _stack_cols(products, "pnr_bin_db", np.float64)
    snr_shelf_db = _stack_cols(products, "snr_shelf_db", np.float64)
    # per-channel products renamed `mask` -> `reject_mask` at schema v2 (1 = discard,
    # positive excess); the canonical combined outputs keep the `mask` field name, so
    # only this read changes -- write_* below stays byte-identical.
    mask = _stack_cols(products, "reject_mask", np.uint8)
    valid = _stack_cols(products, "valid", np.uint8)
    baseband_power_linear = _stack_cols(products, "baseband_power_linear", np.float64)
    # Weight-norm zero-point fields (norm-corrected mask). Present in every
    # product at or after the corrected rule; guarded like freq_id so a combine
    # of legacy products still writes a legacy-shaped file.
    has_norms = all(
        "target_norm_sq" in z and "ref_norm_sum_sq" in z and "mu0" in z
        and "pilot_excess_corrected" in z
        for z in products
    )
    target_norm_sq = _scalars(products, "target_norm_sq", np.int64) if has_norms else None
    ref_norm_sum_sq = (
        _scalars(products, "ref_norm_sum_sq", np.int64) if has_norms else None
    )
    mu0 = _scalars(products, "mu0", np.float64) if has_norms else None
    pilot_excess_corrected = (
        _stack_cols(products, "pilot_excess_corrected", np.float64)
        if has_norms
        else None
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
        fstat_raw=fstat_raw,
        fstat_level_db=fstat_level_db,
        pnr_bin_db=pnr_bin_db,
        snr_shelf_db=snr_shelf_db,
        mask=mask,
        valid=valid,
        target_norm_sq=target_norm_sq,
        ref_norm_sum_sq=ref_norm_sum_sq,
        mu0=mu0,
        pilot_excess_corrected=pilot_excess_corrected,
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
        fstat_raw=fstat_raw,
        fstat_level_db=fstat_level_db,
        snr_shelf_db=snr_shelf_db,
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
    # honestly labelled as chime-scan provenance -- not a byte-faithful imitation of
    # a single run_chime_analysis run.
    contract = _detector_contract_from(products, nfft)
    reference_placement = _combined_reference_placement_summary(products)
    if reference_placement is not None:
        contract = dict(contract)
        contract["reference_placement_summary"] = reference_placement
    mask_policy = positive_excess_mask_policy()
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
    if freq_id is not None:
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
