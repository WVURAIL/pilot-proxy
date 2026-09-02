#!/usr/bin/env bash
# One-off repair for shard 2's quarantine ledger (2026-09-01 event).
# Run on notebook2:  bash /arc/home/dgormley/pp_switch/canfar_shard2_repair.sh
#
# What happened: shard 2 was interrupted on notebook1 and its staging dir was
# removed while the scan was still draining; 7 staged files vanished mid-read
# (06:04:07-06:04:15Z) and were misclassified as permanently unreadable units.
# Quarantine rows are persistent, so those 7 readable units would be skipped
# on every resume. This script stops the scan cleanly, removes exactly those
# 7 staging-ENOENT rows (keeping the 1 predeclared sub-frame row), and leaves
# a backup of the untouched ledger beside it.
set -euo pipefail
OUT=/arc/home/dgormley/pp_runs/chime_pilots_rebuild_20260829_canfar_shard2
Q=$OUT/_per_pilot/quarantine.jsonl

echo "== stopping shard 2 cleanly (no staging removal) =="
pid=$(pgrep -f "pilot-proxy chime-scan.*canfar_shard2" | head -1 || true)
if [ -n "${pid:-}" ]; then
  kill -INT "$pid"
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "SIGINT ignored (pre-fix launcher); escalating to SIGTERM (crash-safe by design)"
    kill -TERM "$pid"
    for i in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "still running -- NOT safe to edit the ledger; aborting. Last resort: kill -9 $pid"; exit 1
  fi
  echo "stopped pid $pid"
else
  echo "no running scan found on this node (already stopped)"
fi

echo "== repairing quarantine ledger =="
test -f "$Q" || { echo "no quarantine ledger at $Q"; exit 1; }
cp "$Q" "$Q.pre_repair_20260901"
python3 - "$Q" <<'PYREP'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
keep = [r for r in rows if "shorter than one transform" in r["reason"]]
drop = [r for r in rows if "shorter than one transform" not in r["reason"]]
for r in drop:
    ok = ("No such file or directory" in r["reason"]
          and "/tmp/pp_stage_shard2/" in r["reason"])
    assert ok, "row is NOT the staging-ENOENT class -- aborting, inspect by hand: " + r["reason"][:120]
with open(p, "w") as fh:
    for r in keep:
        fh.write(json.dumps(r) + "\n")
print("kept %d predeclared sub-frame row(s); removed %d staging-ENOENT row(s):" % (len(keep), len(drop)))
for r in drop:
    print("  -", r["unit_name"], r["time"])
PYREP
echo "backup: $Q.pre_repair_20260901"
echo
echo "now resume shard 2:"
echo "  PP_RESUME_CONFIRM=YES bash /arc/home/dgormley/pp_switch/canfar_shard.sh 2 resume"
echo "then, after ~10 min:"
echo "  bash /arc/home/dgormley/pp_switch/canfar_shard.sh 2 tripwire"
