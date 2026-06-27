#!/usr/bin/env python3
"""Pre-flight coverage report for a datatrawl CHIME-baseband survey inventory.

Groups the inventory's file records by freq_id; reports per-channel file count and
data volume; flags channels missing or short against an expected set; and estimates
the sequential, download-bound run time two ways (by file count and by data volume).
Read-only. Exit 1 if any expected freq_id has no coverage.

Usage:
  check_inventory.py INVENTORY.jsonl [--expect 506,521,...] [--rate-mbps 19] [--secs-per-file 17.4]
"""
import argparse, json, sys
from collections import defaultdict

PILOT_FREQ_IDS = [506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675,
                  690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844]
GiB = 1024 ** 3
TiB = 1024 ** 4


def _fmt(secs: float) -> str:
    d = secs / 86400.0
    return f"{d:.1f} days ({secs/3600:.0f} h)" if d >= 1.0 else f"{secs/3600:.1f} h"


def _vol(b: int) -> str:
    return f"{b/TiB:.2f} TiB" if b >= TiB else f"{b/GiB:.1f} GB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inventory")
    ap.add_argument("--expect", default=None,
                    help="comma-separated freq_ids to require (default: the 23 ATSC pilots)")
    ap.add_argument("--rate-mbps", type=float, default=19.0,
                    help="effective MB/s for the volume-based estimate (default 19)")
    ap.add_argument("--secs-per-file", type=float, default=17.4,
                    help="measured wall-seconds/file for the count-based estimate (default 17.4)")
    args = ap.parse_args(argv)

    expect = ([int(x) for x in args.expect.split(",") if x.strip()]
              if args.expect else list(PILOT_FREQ_IDS))
    expect_set = set(expect)

    files: dict = defaultdict(int)
    nbytes: dict = defaultdict(int)
    sizes: dict = defaultdict(list)
    events: set = set()
    total_files = total_bytes = bad = 0

    try:
        fh = open(args.inventory)
    except OSError as e:
        print(f"cannot open {args.inventory}: {e}", file=sys.stderr)
        return 2
    with fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
                fid = int(d["freq_id"]); sz = int(d["size_bytes"])
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                bad += 1
                continue
            files[fid] += 1; nbytes[fid] += sz; sizes[fid].append(sz)
            events.add(d.get("event")); total_files += 1; total_bytes += sz

    present = set(files)
    missing = sorted(expect_set - present)
    unexpected = sorted(present - expect_set)

    print(f"inventory: {args.inventory}")
    print(f"  {total_files} file record(s)  |  {len(events)} distinct event(s)  |  "
          f"{len(present & expect_set)}/{len(expect_set)} expected freq_id(s) present"
          + (f"  |  {bad} unparseable line(s)" if bad else ""))
    print()
    hdr = f"  {'freq_id':>7}  {'files':>6}  {'volume':>11}  {'median':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for fid in sorted(expect_set | present):
        c = files.get(fid, 0); b = nbytes.get(fid, 0)
        ss = sorted(sizes.get(fid, []))
        med = ss[len(ss) // 2] / 1e6 if ss else 0.0
        flag = ""
        if fid not in present:
            flag = "  <== MISSING"
        elif c < 3:
            flag = "  <== short"
        if fid not in expect_set:
            flag += "  (unexpected)"
        print(f"  {fid:>7}  {c:>6}  {b/GiB:>8.2f} GB  {med:>6.0f} MB{flag}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':>7}  {total_files:>6}  {_vol(total_bytes):>11}")
    print()

    secs_count = total_files * args.secs_per_file
    secs_vol = total_bytes / (args.rate_mbps * 1e6) if total_bytes else 0.0
    print("  est. sequential run time (download-bound):")
    print(f"    by file count ({total_files} x {args.secs_per_file:.1f}s): {_fmt(secs_count)}")
    print(f"    by volume ({_vol(total_bytes)} / {args.rate_mbps:.0f} MB/s): {_fmt(secs_vol)}")
    if secs_count and secs_vol and max(secs_count, secs_vol) > 1.5 * min(secs_count, secs_vol):
        print("    (estimates diverge -> file sizes vary widely; trust the volume figure)")
    print()

    if missing:
        print(f"  !! MISSING {len(missing)} expected freq_id(s): {','.join(map(str, missing))}")
        return 1
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected freq_id(s) present: {','.join(map(str, unexpected))}")
    print("  OK: every expected freq_id has coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
