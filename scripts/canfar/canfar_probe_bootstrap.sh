#!/usr/bin/env bash
# CANFAR probe bootstrap v2 -- pilot-proxy v5 archive run, frozen tag.
# Run inside ONE CANFAR session terminal (notebook1):
#   bash /arc/home/dgormley/pp_switch/canfar_probe_bootstrap.sh
# v2: phase 3 now restores the runtime OFFLINE from the freeze bundle's own
# wheelhouse (cut at the frozen rev), per the lock's "install only from the
# bundled wheelhouse" doctrine.  v1 tried PyPI and was correctly refused by
# hash checking (different manylinux wheel variants for the same versions).
# Read-only against the archive except ~25 timed test downloads to /tmp.
set -uo pipefail
say(){ printf '\n===== %s =====\n' "$*"; }

TAG=archive-run-source-20260829
REV=65b49971ffa673f0c52987b3fe16ef7ecc8aec63
PKG=e30ec73fd037ff68407194b323766bb3317e4168526a6456e1f0520a074b0c4c
TARSHA=f37fe0410bff5233b4a5afd99262f47d9451d446a7be9274b943f5373e079ae6
SW=/arc/home/dgormley/pp_switch
PP="$HOME/pilot-proxy"
FRZ="$HOME/pp_freeze"
BUNDLE="$FRZ/archive-local-65b49971ffa6"
VENV="$HOME/pp-venv-$(hostname)"

say "0. node identity"
hostname; nproc; free -g | head -2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
python3 --version

say "1. certificate"
CERT="$HOME/.ssl/cadcproxy.pem"
if [ ! -f "$CERT" ]; then
  echo "NO CERT at $CERT -- run:  cadc-get-cert -u dgormley --days-valid 30"
  echo "then re-run this script."; exit 1
fi
chmod 600 "$CERT"
openssl x509 -in "$CERT" -noout -enddate

say "2. frozen source"
if [ ! -d "$PP/.git" ]; then
  git clone --quiet https://github.com/WVURAIL/pilot-proxy "$PP"
fi
cd "$PP"
git fetch --quiet origin --tags
git checkout --quiet "$TAG"
test "$(git rev-parse HEAD)" = "$REV" || { echo "REV MISMATCH"; exit 1; }
test -z "$(git status --porcelain)" || { echo "TREE NOT CLEAN"; exit 1; }
echo "at $TAG = $REV (clean)"

rm -rf "$HOME/pp-venv"  # pp-venv-old-cleanup: v1 partial install
say "3. runtime restore (offline, from the freeze bundle wheelhouse)"
if [ ! -e "$BUNDLE/.restored_ok" ]; then
  echo "verifying bundle tar..."
  echo "$TARSHA  $SW/pp_runtime_freeze.tar" | sha256sum --check --strict || exit 1
  mkdir -p "$FRZ"
  tar -C "$FRZ" -xf "$SW/pp_runtime_freeze.tar"
  ( cd "$BUNDLE" && sha256sum --check --quiet SHA256SUMS ) || { echo "BUNDLE SHA256SUMS FAILED"; exit 1; }
  touch "$BUNDLE/.restored_ok"
fi
echo "bundle verified: $BUNDLE"
[ -e "$VENV/bin/activate" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
LOG=/tmp/pp_pip_install.log
pip install --no-index --find-links "$BUNDLE/wheelhouse" -r "$BUNDLE/requirements.lock" >"$LOG" 2>&1 \
  || { echo "OFFLINE LOCK INSTALL FAILED -- tail of $LOG:"; tail -25 "$LOG"; exit 1; }
grep -E '^Successfully installed' "$LOG" | tail -1
pip install --no-index --no-build-isolation -e . >>"$LOG" 2>&1 \
  || { echo "EDITABLE INSTALL FAILED -- tail of $LOG:"; tail -25 "$LOG"; exit 1; }
python -c "from pilot_proxy.provenance import package_source_sha256 as p; s=p(); assert s=='"$PKG"', s; print('package sha OK: '+s[:16]+'...')"

say "4. kernel build + digest for THIS node arch"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
make -C cuda clean >/dev/null; make -C cuda SM="$SM" >/dev/null || { echo "KERNEL BUILD FAILED"; exit 1; }
SHA=$(sha256sum cuda/libfstatistic.so | cut -d' ' -f1)
KDIR="/arc/home/dgormley/pp_kernels"
mkdir -p "$KDIR"
KLIB="$KDIR/pilotproxy-detector-core-2.3.0-sm${SM}-${SHA:0:12}.so"
cp --no-clobber cuda/libfstatistic.so "$KLIB"; chmod 555 "$KLIB" 2>/dev/null || true
echo "preserved: $KLIB"
echo "sha256   : $SHA"

say "5. kernel gates (CPU reference + load check)"
make -C cuda test_ref 2>&1 | tail -2
PYTHONPATH=src python - "$KLIB" <<'PY'
import sys
from pilot_proxy.kernel import FStatKernel
k = FStatKernel(sys.argv[1])
print("kernel", k._get_version().as_string(), "fine:", k.supports_fine_powers(), "fused:", k.supports_fused_fine())
PY

if [ "${PP_SKIP_PROBE:-}" = "1" ]; then
  say "6. probe skipped (PP_SKIP_PROBE=1)"
  say "DONE"
  exit 0
fi
say "6. FETCH THROUGHPUT PROBE (the decision number)"
mkdir -p /tmp/pp_probe && cd /tmp/pp_probe
run_probe () {
  local workers=$1; shift
  local t0 t1 bytes=0
  t0=$(date +%s.%N)
  xargs -P "$workers" -I{} -a "$SW/probe_uris.txt" \
    sh -c 'cadcget --cert "$HOME/.ssl/cadcproxy.pem" "{}" -o "/tmp/pp_probe/$(basename {})" 2>/dev/null'
  t1=$(date +%s.%N)
  bytes=$(du -cb /tmp/pp_probe/*.h5 2>/dev/null | tail -1 | cut -f1)
  python3 -c "print(f'  workers=$workers  bytes={$bytes:,}  secs={$t1-$t0:.1f}  MiB/s={($bytes/1048576)/($t1-$t0):.1f}')"
  rm -f /tmp/pp_probe/*.h5
}
echo "single-stream:"; run_probe 1
echo "four-stream  :"; run_probe 4
echo "eight-stream :"; run_probe 8

say "DONE -- paste everything from '0. node identity' down"
