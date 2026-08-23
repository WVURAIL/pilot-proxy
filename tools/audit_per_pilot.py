#!/usr/bin/env python3
"""Deep integrity audit for current PilotProxy per-pilot products.

Re-derives every internally checkable quantity from first principles and the
repository's own contract functions, and verifies provenance hashes against
the shipped weight bank. Safe on live checkpoints (reads only).

Usage: python tools/audit_per_pilot.py <dir-with-<freq_id>.npz> [--repo <pilot-proxy checkout>]

Lives in the repository so every machine audits with the current version
(git pull is the update mechanism; do not keep local copies).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot_proxy.detector_contract import null_power_ratio_from_weight_norms, NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.fine_reduction import fine_bin_count
from pilot_proxy.product_contract import (
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    current_decision_contract,
)

CHIME_HZ_PER_CHANNEL = 400e6 / 1024.0          # 390625 Hz
PILOT_BASE_MHZ = 470.309441                    # ATSC 14 pilot
FREQ_TABLE = {14 + i: fid for i, fid in enumerate(
    [844, 829, 813, 798, 783, 767, 752, 736, 721, 706, 690, 675, 660,
     644, 629, 614, 598, 583, 568, 552, 537, 521, 506])}


class Audit:
    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, cond: bool, label: str):
        if not bool(cond):
            self.failures.append(label)

    def note(self, text: str):
        self.notes.append(text)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_file(path: Path, bank_sha: str | None, manifest_sha: str | None) -> tuple[Audit, dict]:
    a = Audit(path.name)
    z = np.load(str(path), allow_pickle=False)
    g = lambda k: np.asarray(z[k])
    r = lambda k: g(k).reshape(-1)
    scalar = lambda k: g(k).reshape(()).item() if g(k).size == 1 else g(k).reshape(-1)[0]

    # ---- provenance -------------------------------------------------------
    sv = str(scalar("schema_version"))
    dv = str(scalar("detector_version"))
    a.check(sv == PER_PILOT_PRODUCT_SCHEMA_TOKEN, f"schema_version={sv}")
    a.check(json.loads(str(scalar("decision_contract_json"))) == current_decision_contract(),
            "decision_contract_json")
    # kernel core 2.x family: additive minor bumps (2.0.0 survey cohort,
    # 2.1.0 fine-power stage) share the detector contract; a major bump
    # must still fail this check loudly. Cohort binary hashes are pinned
    # in the run ledger, not here.
    a.check(re.search(r"kernel=2\.\d+\.\d+", dv) is not None and "K=128" in dv
            and PER_PILOT_PRODUCT_SCHEMA_TOKEN in dv,
            "detector_version tokens")
    src = next((t[len("source="):][:12] for t in dv.split() if t.startswith("source=")), "?")
    a.check(str(scalar("mask_rule")) == NORMALIZED_POSITIVE_EXCESS_MASK_RULE, "mask_rule string")
    if bank_sha is not None:
        a.check(str(scalar("weight_bank_sha256")) == bank_sha, "weight_bank_sha256 vs shipped bank")
    if manifest_sha is not None:
        a.check(str(scalar("weight_manifest_sha256")) == manifest_sha,
                "weight_manifest_sha256 vs shipped manifest")
    contract = json.loads(str(scalar("detector_contract_json")))
    a.check(contract.get("schema_version") == "pilotproxy_detector_contract_v1",
            "detector_contract schema")
    a.check(contract.get("fine_reduction", {}).get("pad_factor") == 2, "contract fine pad_factor")

    # ---- geometry / identity ---------------------------------------------
    ch = int(scalar("physical_channel")); fid = int(scalar("freq_id"))
    nfft = int(scalar("nfft")); K = int(scalar("detector_window_samples"))
    a.check(FREQ_TABLE.get(ch) == fid, f"channel {ch} <-> freq_id {fid} mapping")
    a.check((nfft, K) == (16384, 128), f"nfft={nfft} K={K}")
    a.check(int(scalar("num_input_streams")) == 2048, "num_input_streams=2048")
    pilot_hz = float(scalar("pilot_frequency_hz"))
    chime_hz = float(scalar("chime_frequency_hz"))
    a.check(abs(pilot_hz - (PILOT_BASE_MHZ + 6.0 * (ch - 14)) * 1e6) < 1.0,
            "pilot_frequency vs ATSC formula")
    a.check(abs(chime_hz - (800e6 - fid * CHIME_HZ_PER_CHANNEL)) < 1.0,
            "chime_frequency vs freq_id formula")
    a.check(abs(pilot_hz - chime_hz) <= CHIME_HZ_PER_CHANNEL / 2, "pilot inside coarse channel")
    a.check(int(scalar("pilot_in_band")) == 1, "pilot_in_band flag")

    # ---- per-frame arrays: lengths ---------------------------------------
    F = r("coarse_power_ratio").astype(np.float64)
    n = F.size
    # Schema v3 removed every derived fine column (the float ratio, the CFAR
    # location/scale/threshold/mode, the null-bulk exceedance fraction, and the
    # ragged exceedance list). The scan stores only the exact terms and decides
    # nothing, so those are recomputed in post-processing and are not audited
    # here as stored fields.
    per_frame = ["p_target_u64", "p_ref_sum_u64", "valid", "reject_mask",
                 "normalized_coarse_power_ratio_db", "pilot_excess_db", "estimated_data_shelf_snr_db",
                 "baseband_power_linear", "frame_index", "frame_unit_index",
                 "frame_in_unit", "normalized_pilot_excess"]
    for k in per_frame:
        a.check(r(k).size == n, f"len({k})=={n}")
    a.check(np.array_equal(r("frame_index"), np.arange(n)), "frame_index == arange")

    # ---- exact integer contract ------------------------------------------
    pt = r("p_target_u64"); pr = r("p_ref_sum_u64")
    a.check(pt.dtype == np.uint64 and pr.dtype == np.uint64, "u64 dtypes")
    v = r("valid").astype(bool); m = r("reject_mask").astype(bool)
    a.check(np.array_equal(v, pr != 0), "valid == (p_ref_sum != 0)")
    tn = int(scalar("target_norm_sq")); rn = int(scalar("reference_norm_sum_sq"))
    expected_mask = np.array([bool(vv) and (int(p) * rn > tn * int(q))
                              for vv, p, q in zip(v, pt.tolist(), pr.tolist())])
    a.check(np.array_equal(m, expected_mask), "reject_mask exact integer rule")
    # v3 stopped storing null_power_ratio because it is exactly recoverable from
    # the two norms beside it; derive it rather than reading a field that would
    # only be able to disagree with its own inputs.
    null_power_ratio = null_power_ratio_from_weight_norms(tn, rn)
    a.check(np.isfinite(null_power_ratio) and null_power_ratio > 0.0,
            "null_power_ratio derived from weight norms is finite and positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        f_re = 2.0 * pt.astype(np.float64) / pr.astype(np.float64)
    a.check(np.allclose(F[v], f_re[v], rtol=1e-9, atol=0), "coarse_power_ratio == 2*pt/pr")
    exc = r("normalized_pilot_excess").astype(np.float64)
    a.check(np.allclose(exc[v], F[v] / null_power_ratio - 1.0, rtol=1e-9, atol=1e-12),
            "normalized_pilot_excess == F/null_power_ratio - 1")
    a.check(np.all(np.isnan(exc[~v])) if (~v).any() else True, "excess NaN on invalid")
    a.check(int(r("rational_overflow_count").sum()) == 0, "rational_overflow_count == 0")

    # ---- fine products ----------------------------------------------------
    a.check(str(scalar("fine_status")) == "enabled", "fine_status enabled")
    bins = int(scalar("fine_num_bins"))
    a.check(bins == fine_bin_count(nfft // K) == 256, f"fine_num_bins={bins}")
    # The exact terms are the whole stored fine surface under v3:
    # (frames, [target, lower reference, upper reference], fine bins), uint64.
    ff = g("fine_power_u64")
    a.check(ff.shape == (n, 3, bins), f"fine_power_u64 shape {ff.shape}")
    a.check(ff.dtype == np.uint64, f"fine_power_u64 dtype {ff.dtype}")
    a.check(bool(np.all(ff[v] > 0)) if v.any() else True,
            "fine reference terms are non-zero on valid frames")
    # The deployed ratio is a post-processing quantity; check it is recoverable
    # from what is stored rather than expecting a stored column.
    with np.errstate(divide="ignore", invalid="ignore"):
        fine_ratio = (2.0 * ff[:, 0, :].astype(np.float64)
                      / (ff[:, 1, :].astype(np.float64) + ff[:, 2, :].astype(np.float64)))
    a.check(bool(np.all(np.isfinite(fine_ratio[v]))) if v.any() else True,
            "fine ratio recomputable and finite on valid frames")
    a.check(str(scalar("decision_contract_json")).strip().startswith("{"),
            "decision_contract_json present for exact replay")

    # ---- integrated spectra ----------------------------------------------
    sb = r("integrated_spectrum_before_mask").astype(np.float64)
    sa = r("integrated_spectrum_after_mask").astype(np.float64)
    a.check(sb.size == nfft and sa.size == nfft, "spectra length == nfft")
    a.check(np.all(sb >= 0) and np.all(sa >= 0), "spectra nonnegative")
    a.check(np.all(sa <= sb * (1 + 1e-12) + 1e-6), "after <= before elementwise")

    # ---- units ------------------------------------------------------------
    uo = r("unit_order"); U = uo.size
    for k in ("unit_time0_ctime", "unit_time0_fpga", "unit_event_id",
              "unit_delta_time", "archive_version"):
        a.check(r(k).size == U, f"len({k})=={U}")
    a.check(len(set(uo.tolist())) == U, "unit keys unique")
    fui = r("frame_unit_index")
    a.check(fui.min() >= 0 and fui.max() < U, "frame_unit_index in range")
    a.check(np.all(np.diff(fui) >= 0), "frame_unit_index nondecreasing")
    fiu = r("frame_in_unit")
    ok_fiu = all(np.array_equal(fiu[fui == u], np.arange((fui == u).sum())) for u in range(U))
    a.check(ok_fiu, "frame_in_unit == arange per unit")
    t0 = r("unit_time0_ctime").astype(np.float64)
    a.check(np.all((t0 > 1.4e9) & (t0 < 1.9e9)), "unit times plausible (2014-2030)")

    stats = {
        "ch": ch, "fid": fid, "units": U, "frames": n,
        "valid%": 100.0 * v.mean(), "mask%": 100.0 * m.mean(),
        "medF/null_power_ratio": float(np.median(F[v] / null_power_ratio)) if v.any() else float("nan"),
        # Recomputed from the stored exact terms; v3 stores no fine decision.
        "med_fine_ratio": float(np.nanmedian(fine_ratio[v])) if v.any() else float("nan"),
        "span": (datetime.datetime.fromtimestamp(t0.min(), datetime.timezone.utc).strftime("%Y-%m")
                 + ".." + datetime.datetime.fromtimestamp(t0.max(), datetime.timezone.utc).strftime("%Y-%m")),
        "source": src,
    }
    return a, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("per_pilot_dir", type=Path)
    ap.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = ap.parse_args()

    bank = args.repo / "weights/chime_dtv_weights_k128.bin"
    man = args.repo / "weights/chime_dtv_weights_k128.bin.manifest.json"
    bank_sha = sha256_of(bank) if bank.exists() else None
    manifest_sha = sha256_of(man) if man.exists() else None

    files = sorted(args.per_pilot_dir.glob("*.npz"))
    if not files:
        print("no products found", file=sys.stderr)
        return 1
    all_ok = True
    rows = []
    for p in files:
        a, s = audit_file(p, bank_sha, manifest_sha)
        rows.append(s)
        status = "PASS" if not a.failures else "FAIL"
        all_ok &= not a.failures
        print(f"{p.name:10s} {status}  ({s['units']} units, {s['frames']} frames, "
              f"src={s['source']})")
        for f in a.failures:
            print(f"    FAIL: {f}")
    print()
    hdr = f"{'ch':>3} {'fid':>4} {'units':>5} {'frames':>6} {'valid%':>6} " \
          f"{'mask%':>6} {'medF/npr':>10} {'med_fine':>10} {'span':>16}"
    print(hdr)
    for s in rows:
        print(f"{s['ch']:>3} {s['fid']:>4} {s['units']:>5} {s['frames']:>6} "
              f"{s['valid%']:>6.1f} {s['mask%']:>6.1f} {s['medF/null_power_ratio']:>10.4f} "
              f"{s['med_fine_ratio']:>10.4f} {s['span']:>16}")
    print("\nOVERALL:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
