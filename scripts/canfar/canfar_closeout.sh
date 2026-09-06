#!/usr/bin/env bash
# Closeout for the completed 2026-09 CHIME archive run.
#
#   bash /arc/home/dgormley/pp_switch/canfar_closeout.sh report
#   bash /arc/home/dgormley/pp_switch/canfar_closeout.sh combine [drop-list]
#
# The combine is event-keyed: it stacks the frames every selected channel
# saw. No event in this archive was captured on all 23 channels, so a
# 23-channel stack is empty and the drop list chooses the subset. `report`
# prints the presence histogram and the drop-curve behind that choice.
#
# The scans stopped before the terminal combine by design ("incomplete
# requested scope", because the predeclared sub-frame units can never be
# processed), so the canonical products are stacked here instead.
#
# The 23 channels live in TWO shard directories and do not overlap:
#   shard1 (11): 506 521 537 552 568 583 675 752 767 813 829
#   shard2 (12): 598 614 629 644 660 690 706 721 736 783 798 844
# They are passed explicitly rather than copied into one directory, so no
# product is duplicated on disk and each is read from where it was written.
# shard3's directory is a deliberate duplicate of six of these and is NOT
# part of the canonical stack; it is kept as reproduction evidence.
set -uo pipefail
say(){ printf '\n===== %s =====\n' "$*"; }

MODE="${1:-report}"
# Channels to exclude from the stack, comma-separated, e.g. 598,690,568,660.
# The combine is event-keyed: it stacks only frames every selected channel
# saw, so sparse channels shrink the intersection sharply. `report` prints the
# drop-curve that quantifies the trade.
DROP="${2:-}"
REV=b59b5c05fed2a9509a31e206f0911e76ca2d2885
PKG=3722012957975f7d5698c24ab3bf36b59ff26dd94fd84ae75b2eb0820d8ea34a
R=/arc/home/dgormley/pp_runs
S1=$R/chime_pilots_rebuild_20260829_canfar_shard1_b59b5c0/_per_pilot
S2=$R/chime_pilots_rebuild_20260829_canfar_shard2_b59b5c0/_per_pilot
OUT=$R/chime_pilots_rebuild_20260829_COMBINED
PP="$HOME/pilot-proxy"
VENV="$HOME/pp-venv-$(hostname)"
die(){ echo "CLOSEOUT-BLOCK: $*" >&2; exit 1; }

CH1="506 521 537 552 568 583 675 752 767 813 829"
CH2="598 614 629 644 660 690 706 721 736 783 798 844"

say "0. environment"
cd "$PP" || die "no checkout at $PP"
test "$(git rev-parse HEAD)" = "$REV" || die "REV mismatch: closeout must run at the frozen source"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
got=$(python -c "from pilot_proxy.provenance import package_source_sha256 as p; print(p())")
test "$got" = "$PKG" || die "package sha mismatch: $got"
echo "source    : $REV"
echo "package   : $PKG"

say "1. assemble the canonical 23 (no copying -- explicit paths)"
ARGS=(); n=0
for c in $CH1; do
  f="$S1/$c.npz"; test -f "$f" || die "missing $f"
  ARGS+=(--product "$f"); n=$((n+1))
done
for c in $CH2; do
  f="$S2/$c.npz"; test -f "$f" || die "missing $f"
  ARGS+=(--product "$f"); n=$((n+1))
done
test "$n" -eq 23 || die "expected 23 products, found $n"
echo "products  : $n, all present"
python - "$S1" "$S2" "$CH1" "$CH2" <<'PYID'
import sys
import numpy as np
s1, s2, c1, c2 = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
seen = {}
for base, chans in ((s1, c1), (s2, c2)):
    for c in chans:
        with np.load(f"{base}/{c}.npz", allow_pickle=False) as z:
            seen.setdefault(str(z["detector_version"]), []).append(c)
            assert int(np.asarray(z["freq_id"]).reshape(-1)[0]) == int(c), f"{c}: freq_id mismatch"
print(f"  distinct detector_version across all 23: {len(seen)}")
for v, chans in seen.items():
    print(f"    {v[:78]}...")
    print(f"      on {len(chans)} channel(s)")
assert len(seen) == 1, "products were not all built by the same source+kernel"
PYID

if [ "$MODE" = "report" ]; then
  say "2. event-presence report (no output written)"
  pilot-proxy chime-combine "${ARGS[@]}" --report
  say "DONE (report only). Run with 'combine' to write the combined products."
  exit 0
fi

say "2. combine"
if [ -n "$DROP" ]; then
  OUT="${OUT}_drop$(echo "$DROP" | tr ',' '-')"
  echo "dropping  : $DROP"
  echo "output    : $OUT"
  DROPARG=(--drop "$DROP")
else
  echo "dropping  : nothing (all 23; expect an empty intersection unless the"
  echo "            archive covers every channel for at least one event)"
  DROPARG=()
fi
test ! -e "$OUT" || die "output dir exists: $OUT (move it aside for a fresh combine)"
umask 077
pilot-proxy chime-combine "${ARGS[@]}" "${DROPARG[@]}" --output-dir "$OUT"   || die "combine failed (if the intersection was empty, pass a drop list: bash closeout.sh combine 598,690,568,660 -- see the report's drop-curve)"

say "3. validate the combined run directory"
pilot-proxy validate-products --run-dir "$OUT" || die "validate failed"

say "4. what was written"
ls -la "$OUT" | sed 's/^/  /'
say "DONE -- combined products at $OUT"
