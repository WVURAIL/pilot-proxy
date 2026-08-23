#!/usr/bin/env python3
"""Per-CHIME-channel keep/discard list over the ATSC 1.0 range (470-608 MHz).

CHIME: 1024 channels over 400-800 MHz, width 25/64 MHz;
center(fid) = 800 - 400*fid/1024. ATSC allocation N (14..36):
[470+6(N-14), 476+6(N-14)] MHz; pilot at lower edge + 309.441 kHz.

Rules (Dylan, 2026-08-18):
- kept allocations (clean/keep-and-mask/pending): keep every CHIME channel
  they touch;
- excised allocations: discard interior channels EXCEPT the boundary
  channels shared with a neighboring allocation (kept regardless) and the
  pilot-bin channel (kept for monitoring).

Which allocations are excised comes from the masking policy's data file
(exports/artifacts/policy_data.json), so a policy revision propagates here
by regenerating, not by editing constants. Channels 14-15 sit below the
surveyed span and stay pending-kept. Writes the disposition CSV and the
team-wiki bad-channel tables.

    python3 analysis/channel_dispositions.py --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

W = 25.0 / 64.0                     # CHIME channel width, MHz
FID_RANGE = range(492, 846)         # CHIME channels touching 470-608 MHz
ALLOCATIONS = range(14, 37)         # ATSC 1.0 UHF allocations
PILOT_OFFSET_MHZ = 0.309441
# below the surveyed span (16-36); kept until the survey reaches them
PENDING = {14, 15}
NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def alloc_lo(n):
    return 470.0 + 6.0 * (n - 14)


def pilot_fid(n):
    """CHIME channel containing allocation N's pilot."""
    return round(1024.0 * (800.0 - alloc_lo(n) - PILOT_OFFSET_MHZ) / 400.0)


def policy_sets(policy_path: Path):
    """(kept, excised, keep_and_mask) allocation sets from the policy data."""
    with open(policy_path, encoding="utf-8") as fh:
        pd = json.load(fh)
    channels = {int(c): rec for c, rec in pd["channels"].items()}
    excised = {ch for ch, rec in channels.items()
               if rec["chosen"][0] == "excised"}
    keep_and_mask = {ch for ch, rec in channels.items()
                     if rec["recommendation"]["action"].startswith("keep and mask")}
    kept = PENDING | (set(channels) - excised)
    for ch, rec in channels.items():   # geometry cross-check against policy
        if pilot_fid(ch) != rec["fid"]:
            raise SystemExit(f"pilot-bin geometry disagrees with policy for "
                             f"ch{ch}: computed {pilot_fid(ch)}, policy {rec['fid']}")
    return kept, excised, keep_and_mask


def dispositions(kept, excised, keep_and_mask):
    fid_pilot = {pilot_fid(n): n for n in ALLOCATIONS}
    rows = []
    for fid in FID_RANGE:
        c = 800.0 - 400.0 * fid / 1024.0
        lo, hi = c - W / 2, c + W / 2
        allocs = [n for n in ALLOCATIONS
                  if lo < alloc_lo(n) + 6.0 and hi > alloc_lo(n)]
        shared = len(allocs) == 2
        touches_kept = any(n in kept for n in allocs)
        if touches_kept:
            disp = "keep"
            if shared:
                reason = f"edge shared: ch{allocs[0]}/ch{allocs[1]}"
            else:
                n = allocs[0]
                tag = (" (pending 14-15)" if n in PENDING
                       else " (keep-and-mask, eta=1.4)" if n in keep_and_mask
                       else "")
                reason = f"kept allocation ch{n}{tag}"
        elif fid in fid_pilot and fid_pilot[fid] in excised:
            disp = "keep"
            reason = f"pilot monitor, excised ch{fid_pilot[fid]}"
        elif shared:
            # edges are kept only when shared with a kept allocation
            disp = "discard"
            reason = f"edge shared: ch{allocs[0]}/ch{allocs[1]}, both excised"
        elif hi > 608.0 or lo < 470.0:
            disp = "keep"
            reason = "band edge (shares non-DTV spectrum)"
        else:
            disp = "discard"
            reason = f"interior of excised ch{allocs[0]}"
        rows.append(dict(freq_id=fid, f_lo_mhz=f"{lo:.6f}", f_hi_mhz=f"{hi:.6f}",
                         dtv=("/".join(str(a) for a in allocs) or "-"),
                         disposition=disp, reason=reason))
    return rows


def ranges(fids):
    """Contiguous (first, last) runs of a sorted freq_id list."""
    if not fids:
        return []
    out, a, b = [], fids[0], fids[0]
    for f in fids[1:]:
        if f == b + 1:
            b = f
        else:
            out.append((a, b)); a = b = f
    out.append((a, b))
    return out


def wikitable(rows, excised, policy_date):
    """The team-wiki bad-channel page: discard ranges + kept exceptions."""
    bad = [r for r in rows if r["disposition"] == "discard"]
    kept_exc = [r for r in rows if r["disposition"] == "keep"
                and ("pilot monitor" in r["reason"]
                     or "both excised" in r["reason"])]
    exc_list = ", ".join(str(c) for c in sorted(excised))
    n_exc = NUM_WORD.get(len(excised), str(len(excised)))
    both_pairs = sorted({r["dtv"] for r in bad if "both excised" in r["reason"]})
    if both_pairs:
        n_pairs = NUM_WORD.get(len(both_pairs), str(len(both_pairs)))
        plural = len(both_pairs) > 1
        both_clause = (f"the {n_pairs} edge{'s' if plural else ''} shared by "
                       f"two excised allocations ({', '.join(both_pairs)}) "
                       f"'''{'are' if plural else 'is'}''' discarded")
    else:
        both_clause = "no edge is shared by two excised allocations"

    L = []
    L.append("== DTV bad-channel list (470–608 MHz) ==")
    L.append("")
    L.append(f"Discard {len(bad)} of the {len(rows)} CHIME frequency channels "
             "in the ATSC 1.0 range, per the pilot-proxy survey's masking "
             f"policy ({policy_date}: {n_exc} excised allocations — DTV "
             f"{exc_list}). Edge channels shared with a kept allocation and "
             "the excised allocations' pilot channels are '''not''' in this "
             "list (edges carry kept spectrum; pilots are monitoring taps); "
             f"{both_clause}. freq_id convention: centre = 800 − "
             "400·freq_id/1024 MHz, width 25/64 MHz.")
    L.append("")
    fids = sorted(int(r["freq_id"]) for r in bad)
    dtv_of = {int(r["freq_id"]): r["dtv"] for r in bad}

    def label(a, b):
        chans = set()
        for f in range(a, b + 1):
            chans.update(int(x) for x in dtv_of[f].split("/"))
        return "/".join(str(c) for c in sorted(chans))

    L.append('{| class="wikitable"')
    L.append("|-")
    L.append("! DTV Chan. !! CHIME Chan.")
    for a, b in sorted(ranges(fids), key=lambda r: label(*r)):
        L.append("|-")
        L.append(f"| {label(a, b)} || {a}–{b}")
    L.append("|}")
    L.append("")
    L.append("=== Kept exceptions inside excised allocations ===")
    L.append("")
    L.append('{| class="wikitable"')
    L.append("|-")
    L.append("! freq_id !! lower edge (MHz) !! upper edge (MHz) !! why kept")
    for r in kept_exc:
        L.append("|-")
        L.append(f"| {r['freq_id']} || {float(r['f_lo_mhz']):.4f} || "
                 f"{float(r['f_hi_mhz']):.4f} || {r['reason']}")
    L.append("|}")
    L.append("")
    return "\n".join(L), len(bad), len(kept_exc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path,
                    default=(Path(__file__).resolve().parents[1]
                             / "exports" / "artifacts" / "policy_data.json"),
                    help="policy_data.json to derive the excised set from")
    ap.add_argument("--policy-date", default="2026-08-18",
                    help="policy snapshot date quoted on the wiki page")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    kept, excised, keep_and_mask = policy_sets(args.policy)
    rows = dispositions(kept, excised, keep_and_mask)

    csv_path = args.out / "chime_dtv_channel_dispositions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_keep = sum(r["disposition"] == "keep" for r in rows)
    print(f"{len(rows)} CHIME channels in 470-608 MHz: keep {n_keep}, "
          f"discard {len(rows) - n_keep}")

    print("\nDISCARD ranges (freq_id):")
    for a, b in ranges([r["freq_id"] for r in rows
                        if r["disposition"] == "discard"]):
        ch = next(r["reason"].split("ch")[-1] for r in rows
                  if r["freq_id"] == a)
        print(f"  {a}-{b}  ({b-a+1} ch, excised DTV {ch})")
    print("\nKEEP exceptions inside excised allocations:")
    for r in rows:
        if r["disposition"] == "keep" and ("pilot monitor" in r["reason"]
                                           or "both excised" in r["reason"]):
            print(f"  fid {r['freq_id']} ({r['f_lo_mhz'][:7]}-"
                  f"{r['f_hi_mhz'][:7]} MHz): {r['reason']}")

    wiki_path = args.out / "chime_dtv_bad_channels.wiki"
    text, n_bad, n_exc = wikitable(rows, excised, args.policy_date)
    with open(wiki_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nwrote {csv_path} and {wiki_path}: "
          f"{n_bad} bad channels, {n_exc} kept exceptions")


if __name__ == "__main__":
    main()
