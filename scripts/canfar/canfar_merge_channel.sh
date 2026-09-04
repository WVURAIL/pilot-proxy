#!/usr/bin/env bash
# Copy-ahead merge: hand a channel finished by shard 3 to the shard that owns
# it in its --select list, so the owner skips the channel when it arrives.
#
#   bash /arc/home/dgormley/pp_switch/canfar_merge_channel.sh <channel>
#   (run on any session -- it works on /arc, and takes the same lock the scans do)
#
# Why this is sound: the per-channel product carries the full detector
# provenance (source hash, kernel hash, weights, contract), and the owner's
# resume() refuses anything that does not match its own. All three shards run
# the same frozen source and the same qualified kernel, so a product built by
# shard 3 is indistinguishable from one the owner would have built. Once the
# product AND that channel's quarantine rows are in the owner's directory, the
# owner finds every unit already disposed and prints
# "nothing to do -- selection already complete in this product".
#
# Refuses unless: shard 3 has fully disposed the channel (no failed, no
# unprocessed), the owner has no product for it yet, the owner is not
# currently working it, and the provenance matches an existing owner product.
set -euo pipefail
CH="${1:-}"
case "$CH" in 767|813|829) OWNER=1 ;; 783|798|844) OWNER=2 ;;
  *) echo "usage: canfar_merge_channel.sh <767|813|829|783|798|844>"; exit 2 ;; esac
die(){ echo "MERGE-BLOCK: $*" >&2; exit 1; }

R=/arc/home/dgormley/pp_runs
SRC=$R/chime_pilots_rebuild_20260829_canfar_shard3_b59b5c0
DST=$R/chime_pilots_rebuild_20260829_canfar_shard${OWNER}_b59b5c0
SRCP=$SRC/_per_pilot; DSTP=$DST/_per_pilot
LOGDIR=$R/logs
VENV="$HOME/pp-venv-$(hostname)"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "== merge channel $CH: shard3 -> shard$OWNER =="

# 1. shard 3 has fully disposed the channel
test -f "$SRCP/$CH.npz" || die "no product $SRCP/$CH.npz"
python - "$SRC/scan_scope.json" "$CH" <<'PY' || die "shard 3 has not fully disposed channel (see above)"
import json, os, sys
scope = json.load(open(sys.argv[1])); ch = int(sys.argv[2])


def entry_channel(p):
    """Scope entries name their channel by product path, not a freq_id key."""
    prod = p.get("product") or ""
    base = os.path.basename(prod)
    if base.endswith(".npz") and base[:-4].isdigit():
        return int(base[:-4])
    sel = p.get("selection")
    if isinstance(sel, (list, tuple)) and len(sel) == 1:
        return int(sel[0])
    if isinstance(sel, int):
        return int(sel)
    return None


ent = next((p for p in scope["pilots"] if entry_channel(p) == ch), None)
assert ent, "channel not in shard-3 scope"
# A channel with quarantines reports status "partial" by design (the run's
# predeclared disposition), so status is not the completeness test --
# failed and unprocessed both being zero is.
assert int(ent.get("enumerated", -1)) == (
    int(ent.get("completed", 0)) + int(ent.get("quarantined", 0))
), "completed + quarantined does not account for every enumerated unit"
print(f"  shard3 scope: status={ent.get('status')} completed={ent.get('completed')} "
      f"quarantined={ent.get('quarantined')} failed={ent.get('failed')} unprocessed={ent.get('unprocessed')}")
assert int(ent.get("failed", 1)) == 0, "failed units remain (retryable) -- let shard 3 finish"
assert int(ent.get("unprocessed", 1)) == 0, "unprocessed units remain -- let shard 3 finish"
PY

# 2. owner has no product yet and is not on this channel
test ! -e "$DSTP/$CH.npz" || die "owner already has $DSTP/$CH.npz"
last=$(grep 'select=\[' "$LOGDIR/canfar_shard${OWNER}_b59b5c0.log" | tail -1 | grep -oE 'select=\[[0-9]+\]' || true)
[ "$last" = "select=[$CH]" ] && die "owner shard$OWNER is currently working channel $CH -- too late to merge"
if [ -e "$DSTP/.$CH.npz.datatrawl.lock" ]; then
  flock -n "$DSTP/.$CH.npz.datatrawl.lock" true || die "owner holds the lock on $CH -- it is working it"
fi
echo "  owner shard$OWNER: no product for $CH, currently on ${last:-?}"

# 3. provenance must match an existing owner product
ref=$(ls "$DSTP"/*.npz | head -1)
python - "$SRCP/$CH.npz" "$ref" <<'PY' || die "provenance mismatch -- do not merge"
import sys, numpy as np
def prov(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: str(z[k]) for k in ("detector_version", "weights_hash", "weight_bank_sha256",
                                       "weight_manifest_sha256", "mask_rule", "schema_version",
                                       "target_norm_sq", "reference_norm_sum_sq")}
a, b = prov(sys.argv[1]), prov(sys.argv[2])
bad = [k for k in a if a[k] != b[k]]
assert not bad, "differs in: " + ", ".join(bad)
print("  provenance: identical to owner's existing product on all 8 identity fields")
PY

# 4. atomic copy of the product, verified
tmp="$DSTP/.$CH.npz.merge.tmp"
cp "$SRCP/$CH.npz" "$tmp"; chmod 600 "$tmp"
s1=$(sha256sum "$SRCP/$CH.npz" | cut -d' ' -f1); s2=$(sha256sum "$tmp" | cut -d' ' -f1)
[ "$s1" = "$s2" ] || { rm -f "$tmp"; die "copy verification failed"; }
mv "$tmp" "$DSTP/$CH.npz"
echo "  product: copied, sha256 $s1"

# 5. append this channel's quarantine rows under the scan's own lock, deduplicated
n=$(python - "$SRCP/quarantine.jsonl" "$DSTP/quarantine.jsonl" "$CH" "$DSTP/.quarantine.jsonl.datatrawl.lock" <<'PY'
import fcntl, json, os, sys
src, dst, ch, lock = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
suffix = f"_{ch}.h5"
rows = [json.loads(l) for l in open(src) if l.strip()] if os.path.exists(src) else []
rows = [r for r in rows if str(r.get("unit_name", "")).endswith(suffix)]
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
try:
    have = set()
    if os.path.exists(dst):
        for l in open(dst):
            if l.strip():
                have.add(json.loads(l).get("quarantine_key"))
    new = [r for r in rows if r.get("quarantine_key") not in have]
    with open(dst, "a") as fh:
        for r in new:
            fh.write(json.dumps(r) + "\n")
        fh.flush(); os.fsync(fh.fileno())
finally:
    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
print(len(new))
PY
)
echo "  quarantine: appended $n row(s) for channel $CH to owner ledger (locked, deduplicated)"

# 6. provenance trail in the owner's directory
echo "{\"channel\": $CH, \"from\": \"shard3\", \"sha256\": \"$s1\", \"quarantine_rows\": $n, \"merged_at_utc\": \"$(date -u +%FT%TZ)\", \"by\": \"$(hostname)\"}" \
  >> "$DST/merged_from_shard3.jsonl"
echo "MERGE OK: channel $CH now owned by shard$OWNER; it will skip it on arrival"
