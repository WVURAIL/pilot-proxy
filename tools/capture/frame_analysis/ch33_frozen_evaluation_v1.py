#!/usr/bin/env python3
"""Evaluate the frozen CH33 ladder and describe its existing archive support."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

CALIBRATION_SHA = "036a134b0132bb84e95e4626afa39e0d144502a78be291a514953e62bf17bc44"
PLAN_SHA = "94a598941f3bb6f41ee6878e239ddba39953fd13b070612c5f64e5e21da5abd7"
MEMBERSHIP_SHA = "3ddafefa99e62280b8e1e905116dd79049185dba186cd4b1fc776f5f41bc7753"


def require(condition, message):
    """Keep release checks active under optimized Python."""
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc(seconds):
    return datetime.fromtimestamp(float(seconds), timezone.utc).isoformat()


def exact_keep(target, reference, nt, nr, ratio):
    """Use Python integers so the threshold comparison cannot overflow uint64."""
    a, b = ratio
    if a <= 0 or b <= 0 or nt <= 0 or nr <= 0:
        raise ValueError("Threshold and weight norms must be positive.")
    return np.array([
        int(r) > 0 and int(t) * int(nr) * int(b) <= int(r) * int(nt) * int(a)
        for t, r in zip(target, reference, strict=True)
    ], dtype=bool)


def unique_join(old_ids, new_ids):
    if len(set(old_ids)) != len(old_ids) or len(set(new_ids)) != len(new_ids):
        raise ValueError("Frame identities must be unique.")
    lookup = {key: i for i, key in enumerate(new_ids)}
    return np.array([lookup.get(key, -1) for key in old_ids], dtype=np.int64)


def retention_bounds(kept, matched, requested):
    if not 0 <= kept <= matched <= requested or requested == 0:
        raise ValueError("Invalid missingness denominator.")
    return [kept / requested, (kept + requested - matched) / requested]


def read_psd_window(path, columns, rows, nfft):
    """Stream the compressed array; hold only the pilot neighborhood in memory."""
    out = np.empty((rows, len(columns)), dtype=np.int16)
    with zipfile.ZipFile(path) as archive, archive.open("psd_frame_db_i16.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported array format: {version}")
        if shape != (rows, nfft) or fortran or dtype != np.dtype("int16"):
            raise ValueError("Unexpected spectral array layout.")
        for start in range(0, rows, 128):
            stop = min(start + 128, rows)
            count = (stop - start) * nfft * dtype.itemsize
            data = handle.read(count)
            if len(data) != count:
                raise ValueError("Truncated spectral array.")
            out[start:stop] = np.frombuffer(data, dtype=dtype).reshape(-1, nfft)[:, columns]
        if handle.read(1):
            raise ValueError("Unexpected trailing spectral data.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/djg/rail"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("Output already exists; preserve completed evaluations.")
    release = root / "output/dissertation-implementation-2026-09-19/frame-policies/corrected-ch33-calibration-v1"
    calibration_path, plan_path = release / "summary.json", release / "plan.json"
    membership_path = release / "calibration_membership.csv"
    require(sha(calibration_path) == CALIBRATION_SHA, 'Calibration summary hash mismatch.')
    require(sha(plan_path) == PLAN_SHA, 'Frozen calibration plan hash mismatch.')
    require(sha(membership_path) == MEMBERSHIP_SHA, 'Calibration membership hash mismatch.')
    frozen, prior_plan = json.loads(calibration_path.read_text()), json.loads(plan_path.read_text())
    for name, expected in prior_plan["source_hashes"].items():
        require(sha(name) == expected, f"Changed frozen input: {name}")
    sys.path.insert(0, str(root / "RFIsher/src"))
    from rfisher_results.archive.blocks import split_blocks

    product_path = root / "datasets/ch33_rescan/work/552.npz"
    old_path = root / "results/canfar_histogram_reference_comparison_2026-09-09/frames/ch33.npz"
    rescan = root / "output/channel-ruling-execution-2026-09-14/rebuild/author_actions/ch33_rescan/products"
    exposed_summary = rescan / "ch33_measured_bank_summary.json"
    nominal_era_path = root / "results/canfar_reanalysis_2026-09-09/archive/channels/ch33/eras.json"
    input_manifest_path = rescan / "input_manifest.json"
    source_paths = [Path(__file__).resolve(), calibration_path, plan_path, membership_path, input_manifest_path,
                    exposed_summary, nominal_era_path, root / "RFIsher/src/rfisher_results/archive/blocks.py"]
    source_hashes = dict(prior_plan["source_hashes"])
    source_hashes.update({str(path): sha(path) for path in source_paths})
    scope = json.loads((rescan / "scan_scope.json").read_text())
    require(scope["complete"] and scope["totals"]["failed"] == 0, 'Corrected scan must be complete with zero failed inputs.')

    fields = ["physical_channel", "freq_id", "frame_index", "frame_unit_index", "frame_in_unit",
              "unit_keys", "source_event_keys", "unit_time0_ctime", "unit_time0_fpga", "unit_delta_time",
              "unit_input_map_sha256", "unit_git_version_tag", "nfft", "sample_rate_hz", "sense",
              "pilot_frequency_hz", "chime_frequency_hz", "p_target_u64", "p_ref_sum_u64", "valid",
              "target_norm_sq", "reference_norm_sum_sq", "coarse_power_ratio", "weight_bank_sha256",
              "weight_manifest_sha256", "rational_overflow_count", "psd_db_step_per_code", "psd_db_invalid_code"]
    with np.load(product_path, allow_pickle=False) as handle:
        new = {key: handle[key] for key in fields}
    with np.load(old_path, allow_pickle=False) as handle:
        old = {key: handle[key] for key in handle.files}
    require(int(new["nfft"]) == 16384 and int(new["rational_overflow_count"]) == 0, 'Corrected geometry must contain 16384 samples per frame and no rational overflow.')
    require(str(new["weight_bank_sha256"]) == frozen["bank_sha256"], 'Corrected detector bank does not match the frozen calibration bank.')
    require(str(new["weight_manifest_sha256"]) == frozen["manifest_sha256"], 'Corrected weight manifest does not match the frozen calibration manifest.')
    nt, nr = frozen["target_norm_sq"], frozen["reference_norm_sum_sq"]
    require(int(new["target_norm_sq"][0]) == nt and int(new["reference_norm_sum_sq"][0]) == nr, 'Corrected integer weight norms differ from the frozen calibration norms.')
    old_ids = [(str(old["unit_keys"][u]), int(i)) for u, i in zip(old["frame_unit_index"], old["frame_in_unit"])]
    new_ids = [(str(new["unit_keys"][u]), int(i)) for u, i in zip(new["frame_unit_index"], new["frame_in_unit"])]
    joined = unique_join(old_ids, new_ids)
    requested_units = set(json.loads(input_manifest_path.read_text())["input_files"])
    require(requested_units == set(new["unit_keys"]), 'Corrected acquisition identities differ from the requested inventory.')
    require(all(old_ids[i][0] not in requested_units for i in np.flatnonzero(joined < 0)), 'Unmatched frames unexpectedly belong to requested acquisitions.')
    selected = old["selected"] & old["is_current"] & np.isfinite(old["frame_time"])
    split = split_blocks(old["frame_unit_index"], old["unit_time"], selected, frame_time=old["frame_time"])
    require(split.calibration_frames == frozen["original_calibration_frames"] == 11853, 'Original calibration membership count changed.')
    require(split.evaluation_frames == frozen["original_later_block_frames"] == 11842, 'Original later-block membership count changed.')
    require(split.boundary_time == prior_plan["original_split"]["boundary_time"], 'Original acquisition split boundary changed.')
    require(not set(old["unit_keys"][old["frame_unit_index"][split.calibration]]) & set(old["unit_keys"][old["frame_unit_index"][split.evaluation]]), 'Calibration and evaluation acquisitions overlap.')
    matched = np.flatnonzero(joined >= 0)
    for i in matched:
        j = joined[i]
        ou, nu = old["frame_unit_index"][i], new["frame_unit_index"][j]
        require(old["source_event_keys"][ou] == new["source_event_keys"][nu], 'Matched frames have different source-event identities.')
        require(old["unit_time0_ctime"][ou] == new["unit_time0_ctime"][nu], 'Matched frames have different acquisition UTC origins.')
        require(old["unit_delta_time"][ou] == new["unit_delta_time"][nu] or (np.isnan(old["unit_delta_time"][ou]) and np.isnan(new["unit_delta_time"][nu])), 'Matched frames have different sample intervals.')
    require(set(new_ids) <= set(old_ids), 'Corrected frames include identities absent from the original archive.')
    require(all(new["valid"][:, 0]) and all(new["p_ref_sum_u64"][:, 0] > 0), 'Corrected frames require valid scores and positive reference power.')
    inherited_rows = np.flatnonzero(split.calibration & (joined >= 0))
    with membership_path.open(newline="") as handle:
        membership = list(csv.DictReader(handle))
    require([int(row["nominal_frame_row"]) for row in membership] == inherited_rows.tolist(), 'Calibration membership differs from the frozen original frame rows.')
    require([int(row["corrected_frame_row"]) for row in membership] == joined[inherited_rows].tolist(), 'Calibration membership differs from the frozen corrected frame rows.')

    out.mkdir(parents=True)
    write_json(out / "plan.json", {
        "schema": "ch33-frozen-evaluation-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_frozen": frozen["policies"], "calibration_summary_sha256": CALIBRATION_SHA,
        "split": prior_plan["original_split"], "source_hashes": source_hashes,
        "selection": "Preserve every original later-block identity; match corrected scores by unit key and frame index.",
        "time_summary": "UTC calendar months and half-years; report missingness and acquisition weighting without significance claims.",
        "spectral_summary": "Monthly median PSD in nominal pilot +/-250 recorded FFT bins; 23.84185791015625 Hz per bin.",
        "independence": "Retrospective: the bank and inherited era used previously examined data. No threshold refitting or new era selection for evaluation.",
        "scientific_ruling": "undetermined", "held_out_validation": False,
    })
    (out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())

    target, reference = new["p_target_u64"][:, 0], new["p_ref_sum_u64"][:, 0]
    score = np.array([float(Fraction(int(t) * nr, int(r) * nt)) for t, r in zip(target, reference)])
    require(np.allclose(score, new["coarse_power_ratio"][:, 0] / (2 * nt / nr), rtol=2e-15, atol=0),
            "Stored normalized scores disagree with exact integer ratios.")
    policy_keep = {p["policy"]: exact_keep(target, reference, nt, nr, p["eta_integer_ratio"]) for p in frozen["policies"]}
    for policy in frozen["policies"]:
        require(policy["eta"].as_integer_ratio() == tuple(policy["eta_integer_ratio"]), 'Frozen threshold float and integer ratio disagree.')
        require(int(policy_keep[policy["policy"]][joined[inherited_rows]].sum()) == policy["calibration_kept"], 'Frozen calibration retained-frame count changed.')
    new_to_old = {int(joined[i]): int(i) for i in matched}
    old_months = np.array([utc(t)[:7] if np.isfinite(t) else "missing" for t in old["frame_time"]])
    new_time = new["unit_time0_ctime"][new["frame_unit_index"]] + new["frame_in_unit"] * 16384 * new["unit_delta_time"][new["frame_unit_index"]]
    new_months = np.array([utc(t)[:7] if np.isfinite(t) else "missing" for t in new_time])
    new_selected = np.array([bool(old["selected"][new_to_old[j]]) for j in range(len(score))])
    blocks = {}
    monthly = []
    policies = []
    for block, mask in [("calibration", split.calibration), ("later", split.evaluation)]:
        original_rows = np.flatnonzero(mask)
        available = original_rows[joined[original_rows] >= 0]
        new_rows = joined[available]
        blocks[block] = {
            "requested_frames": len(original_rows), "matched_frames": len(new_rows),
            "missing_frames": len(original_rows) - len(new_rows),
            "requested_acquisitions": len(np.unique(old["frame_unit_index"][original_rows])),
            "matched_acquisitions": len(np.unique(old["frame_unit_index"][available])),
            "first_utc": utc(old["frame_time"][available].min()), "last_utc": utc(old["frame_time"][available].max()),
            "score_q10_q50_q90": np.quantile(score[new_rows], [.1, .5, .9]).tolist(),
        }
        for name, keep in policy_keep.items():
            unit_values = [float(np.mean(keep[joined[available[old["frame_unit_index"][available] == unit]]])) for unit in np.unique(old["frame_unit_index"][available])]
            kept = int(keep[new_rows].sum())
            policies.append({"block": block, "policy": name, "kept_frames": kept, "matched_frames": len(new_rows),
                             "frame_retention": kept / len(new_rows), "equal_acquisition_retention": float(np.mean(unit_values)),
                             "all_original_frame_retention_bounds": retention_bounds(kept, len(new_rows), len(original_rows))})
        for month in np.unique(old_months[original_rows]):
            oi = original_rows[old_months[original_rows] == month]
            mi = oi[joined[oi] >= 0]
            ni = joined[mi]
            row = {"block": block, "month": month, "requested_frames": len(oi), "matched_frames": len(ni),
                   "missing_frames": len(oi) - len(ni), "matched_units": len(np.unique(old["frame_unit_index"][mi])),
                   "score_median": float(np.median(score[ni])) if len(ni) else None}
            for name, keep in policy_keep.items():
                row[f"{name}_kept"] = int(keep[ni].sum())
                row[f"{name}_retention"] = float(np.mean(keep[ni])) if len(ni) else None
            monthly.append(row)
    write_csv(out / "block_months.csv", monthly)
    write_json(out / "block_policies.json", policies)

    eval_rows = []
    for i in np.flatnonzero(split.evaluation):
        j = int(joined[i])
        row = {"nominal_frame_row": int(i), "corrected_frame_row": j, "unit_key": old_ids[i][0],
               "frame_in_unit": old_ids[i][1], "utc": utc(old["frame_time"][i]), "matched": j >= 0,
               "p_target_u64": int(target[j]) if j >= 0 else None,
               "p_ref_sum_u64": int(reference[j]) if j >= 0 else None}
        row.update({name: bool(keep[j]) if j >= 0 else None for name, keep in policy_keep.items()})
        eval_rows.append(row)
    write_csv(out / "later_membership.csv", eval_rows)
    missing_requests = []
    for block, mask in [("calibration", split.calibration), ("later", split.evaluation)]:
        missing = np.flatnonzero(mask & (joined < 0))
        for unit in np.unique(old["frame_unit_index"][missing]):
            rows = missing[old["frame_unit_index"][missing] == unit]
            missing_requests.append({"block": block, "physical_channel": 33, "freq_id": 552,
                                     "unit_key": str(old["unit_keys"][unit]),
                                     "source_event_key": str(old["source_event_keys"][unit]),
                                     "requested_frames": len(rows), "first_utc": utc(old["frame_time"][rows].min()),
                                     "last_utc": utc(old["frame_time"][rows].max())})
    write_csv(out / "missing_acquisitions.csv", missing_requests)

    center = float(new["chime_frequency_hz"][0])
    nominal = float(new["pilot_frequency_hz"][0])
    sense = int(new["sense"])
    bin_hz = float(new["sample_rate_hz"]) / 16384
    nominal_bin = int(round(sense * (nominal - center) / bin_hz)) % 16384
    columns = (nominal_bin + np.arange(-250, 251)) % 16384
    rf = center + sense * (((columns + 8192) % 16384) - 8192) * bin_hz
    codes = read_psd_window(product_path, columns, len(score), 16384)
    values = codes.astype(np.float32) * float(new["psd_db_step_per_code"])
    values[codes == int(new["psd_db_invalid_code"])] = np.nan
    full_months, spectra = [], []
    for month in sorted(set(new_months) - {"missing"}):
        rows = np.flatnonzero(new_selected & (new_months == month))
        if not len(rows):
            continue
        units = np.unique(new["frame_unit_index"][rows])
        days = np.unique(np.floor(new_time[rows] / 86400))
        median_score = float(np.median(score[rows]))
        level_db = float(np.median(10 * np.log10(score[rows])))
        spectrum = np.nanmedian(values[rows], axis=0)
        spectra.append(spectrum)
        peak = int(np.nanargmax(spectrum))
        full_months.append({
            "month": month, "frames": len(rows), "acquisitions": len(units), "days": len(days),
            "populated": len(rows) >= 30 and len(units) >= 5 and len(days) >= 3,
            "score_median": median_score, "score_level_db": level_db,
            "state": "proxy-high" if level_db >= 1 else "proxy-low" if level_db <= .5 else "ambiguous",
            "score_q10": float(np.quantile(score[rows], .1)), "score_q90": float(np.quantile(score[rows], .9)),
            "peak_rf_hz": float(rf[peak]), "peak_offset_from_corrected_bank_hz": float(rf[peak] - frozen["effective_pilot_frequency_hz"]),
            "peak_relative_frame_reference_db": float(spectrum[peak]), "peak_at_window_edge": peak in (0, 500),
            "input_maps": ";".join(sorted(set(new["unit_input_map_sha256"][units]))),
            "software_tags": len(set(new["unit_git_version_tag"][units])),
        })
    write_csv(out / "full_archive_months.csv", full_months)
    np.savez_compressed(out / "monthly_spectra.npz", months=np.array([row["month"] for row in full_months]),
                        relative_frame_reference_db=np.stack(spectra), rf_hz=rf)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    month_dates = [datetime.fromisoformat(row["month"] + "-15") for row in full_months]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(month_dates, [row["score_median"] for row in full_months], marker=".", linestyle="none")
    axes[0].axvspan(datetime(2020, 11, 1), datetime(2022, 10, 1), color="gray", alpha=.12)
    axes[0].text(datetime(2021, 10, 15), 4, "No recorded months", ha="center", fontsize=9)
    for color, policy in zip(["C0", "C1", "C2"], frozen["policies"]):
        axes[0].axhline(policy["eta"], linestyle="--", color=color, alpha=.7, label=policy["policy"])
    axes[0].set_ylabel("Monthly median normalized score")
    axes[0].legend(ncol=3, fontsize=9)
    high = [row for row in full_months if row["populated"] and row["state"] == "proxy-high"]
    axes[1].plot([datetime.fromisoformat(row["month"] + "-15") for row in high],
                 [row["peak_offset_from_corrected_bank_hz"] for row in high], marker=".")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_ylabel("Peak offset from corrected bank (Hz)")
    ev_months = [row for row in monthly if row["block"] == "later"]
    for name in policy_keep:
        axes[2].plot([datetime.fromisoformat(row["month"] + "-15") for row in ev_months],
                     [row[f"{name}_retention"] for row in ev_months], marker=".", label=name)
    axes[2].set_ylabel("Later-block frame retention")
    axes[2].set_ylim(-.03, 1.03)
    axes[2].legend(ncol=3, fontsize=9)
    axes[0].axvline(datetime.fromtimestamp(split.boundary_time, timezone.utc), color="gray", ls=":")
    axes[0].set_title("Channel 33: frozen thresholds on previously examined archive data")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlabel("UTC date (gaps are not transmitter states)")
    fig.savefig(out / "ch33_stability.png", dpi=150)
    plt.close(fig)

    later_new = joined[np.flatnonzero(split.evaluation & (joined >= 0))]
    cohort_rows = []
    for key in ["unit_input_map_sha256", "unit_git_version_tag"]:
        labels = new[key][new["frame_unit_index"][later_new]]
        for label in sorted(set(labels)):
            rows = later_new[labels == label]
            row = {"field": key, "value": str(label), "frames": len(rows), "score_median": float(np.median(score[rows]))}
            row.update({name: float(np.mean(keep[rows])) for name, keep in policy_keep.items()})
            cohort_rows.append(row)
    write_csv(out / "later_metadata_cohorts.csv", cohort_rows)

    # A separate Fraction implementation checks every recorded later decision.
    independent_checks = 0
    for policy in frozen["policies"]:
        bound = Fraction(*policy["eta_integer_ratio"])
        for j in later_new:
            expected = Fraction(int(target[j]) * nr, int(reference[j]) * nt) <= bound
            require(expected == policy_keep[policy["policy"]][j], 'Independent rational comparison disagrees with the saved decision.')
            independent_checks += 1
    current_months = [row for row in full_months if row["month"] >= "2023-12" and row["populated"]]
    summary = {
        "schema": "ch33-frozen-evaluation-v1", "channel": 33, "scientific_ruling": "undetermined",
        "effective_pilot_frequency_hz": frozen["effective_pilot_frequency_hz"],
        "calibration_summary_sha256": CALIBRATION_SHA, "bank_sha256": frozen["bank_sha256"],
        "blocks": blocks, "policies": policies, "independent_arithmetic_checks": independent_checks,
        "full_corrected_frames": len(score), "full_original_frames": len(old_ids),
        "full_original_unmatched_frames": int((joined < 0).sum()),
        "unmatched_frames_belong_to_unrequested_acquisitions": True,
        "corrected_scan_requested_acquisitions": len(requested_units),
        "original_health_selected_corrected_frames": int(new_selected.sum()),
        "corrected_frames_without_finite_time": int((~np.isfinite(new_time)).sum()),
        "inherited_current_months": len(current_months),
        "inherited_current_months_proxy_high": sum(row["state"] == "proxy-high" for row in current_months),
        "inherited_current_score_median_range": [min(row["score_median"] for row in current_months), max(row["score_median"] for row in current_months)],
        "inherited_current_peak_offset_range_hz": [min(row["peak_offset_from_corrected_bank_hz"] for row in current_months), max(row["peak_offset_from_corrected_bank_hz"] for row in current_months)],
        "held_out_validation": False, "thresholds_refitted": False, "physical_response_qualified": False,
        "era_requalified": False,
        "era_conclusion": "The inherited nominal-bank proxy-low state is contradicted by the corrected-bank monthly scores. The fixed calibration/evaluation membership remains unchanged for this retrospective diagnostic; a current stationary distribution has not been established.",
        "independence_limit": "Threshold fitting used only the inherited calibration identities, but the bank selection, nominal-era construction, and preexisting corrected whole-archive summary exposed the broader record. Neither later records nor September captures are claimed as unseen confirmation.",
        "missingness_limit": "Retentions describe available frames. Deterministic all-original bounds allow each unmatched frame either decision; missingness is not assumed random.",
        "source_hashes": source_hashes, "plan_sha256": sha(out / "plan.json"),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
