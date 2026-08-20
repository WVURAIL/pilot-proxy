#!/usr/bin/env python3
"""Print the dissertation's unified 20-channel LaTeX table rows.

Reads report_data.json and policy_data.json (the data behind the published
artifact pages; defaults to the committed exports/artifacts/ snapshots) and
prints an audit line per channel followed by the LaTeX rows for the
dissertation's unified channel table --- a drafting aid for manual paste,
not a file generator. The era descriptions and status macros are editorial
and live here.

    python3 analysis/latex_channel_table.py [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STATUS = {
    14: r"\stBound", 15: r"\stBound",
    16: r"\stBound",
    17: r"\stExc", 18: r"\stBound", 19: r"\stBound/\stExc",
    20: r"\stBound/\stExc", 21: r"\stBound", 22: r"\stExc",
    23: r"\stBound", 24: r"\stExc", 25: r"\stBound",
    26: r"\stBound/\stExc", 27: r"\stBound/\stExc", 28: r"\stExc",
    29: r"\stBound", 30: r"\stExc", 31: r"\stExc", 32: r"\stCond",
    33: r"\stCond", 34: r"\stBound", 35: r"\stCond/\stExc", 36: r"\stExc",
}

ERA = {
    14: "trace; episodic faint carrier",
    15: "steady faint excess, no transition",
    16: "trace; warm-season swell",
    17: "persistent; two on-epochs",
    18: "trace, rising since 2024",
    19: "sign-off Dec 2024",
    20: "step-down Sep 2022",
    21: "trace",
    22: "persistent, steady",
    23: "trace",
    24: "persistent; curtailed archive",
    25: "trace",
    26: "sign-off Apr 2023",
    27: "sign-off in 2021--22 gap",
    28: "always-on",
    29: "always-on, faintest shelf",
    30: "always-on; collection cut Sep 2023",
    31: "always-on, occupancy wall",
    32: "station change 2021",
    33: "always-on, off-nominal carrier",
    34: "always-on",
    35: "sign-on Nov 2021",
    36: "always-on (propagation-fed)",
}


def wf(p, eta):
    v = p["window_frac"].get(eta)
    return "---" if v is None else f"{100*v:.1f}\\%"


def tau_cell(p):
    t = p["tau"]
    if t["quality"] == "bounded_above":
        return rf"$\le${round(t['seconds']/60)} min"
    if t["measured"]:
        return f"{round(t['seconds']/60)} min"
    return "cap"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=(Path(__file__).resolve().parents[1]
                             / "exports" / "artifacts"),
                    help="directory holding report_data.json / policy_data.json")
    args = ap.parse_args(argv)
    with open(args.data_dir / "report_data.json", encoding="utf-8") as fh:
        rd = json.load(fh)
    with open(args.data_dir / "policy_data.json", encoding="utf-8") as fh:
        pd = json.load(fh)
    report = {r["phys"]: r for r in rd["channels"]}
    policy = {int(k): v for k, v in pd["channels"].items()}

    # audit pass: show raw fields
    for ch in sorted(STATUS):
        r = report[ch]
        p = policy[ch]
        tau = p["tau"]
        print(f"ch{ch}: events={r['n_events']} arch={r['frac_excess_pos']:.4f} "
              f"w1.0={p['window_frac']['1.0']} w1.4={p['window_frac']['1.4']} "
              f"tau_s={tau['seconds']} meas={tau['measured']} q={tau['quality']} "
              f"cls={p['cls']} trans={p['transition']}")

    print("\n% ---- LaTeX rows ----")
    for ch in sorted(STATUS):
        r = report[ch]
        p = policy[ch]
        print(f"ch\\,{ch} & {r['n_events']:,} & "
              f"{100*r['frac_excess_pos']:.1f}\\% & "
              f"{wf(p, '1.0')} & {wf(p, '1.4')} & "
              f"{tau_cell(p)} & {ERA[ch]} & {STATUS[ch]} \\\\")


if __name__ == "__main__":
    main()
