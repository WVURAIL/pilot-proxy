# Validation gates for the local archive run

These are the mandatory gates for the approved local historical estimation and
sufficient-statistic reprocessing run. The coarse positive-excess flag is
retained as a bootstrap diagnostic; the fine rank/eta decision is inactive and
no calibrated detection policy is applied.

Gate evidence must match one clean source revision, its package-source digest,
and the preserved SM89 library. A source change after a gate invalidates that
gate. The scientific and product pins are in
[`RERUN_PARAMETER_REGISTER.md`](RERUN_PARAMETER_REGISTER.md), the sole
production command is in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md), and the
actual values belong in the run ledger.

## Ground truth already closed

The 2026-07-29 frame correction and 23-channel parity census established the
active CHIME frame convention and DC-centered weight bank. The active receiver
profile and weight manifest resolve that convention. Historical half-band
weights remain under `weights/legacy_halfband/` and are not operational inputs.

The original supporting gates remain:

1. `tools/framing_audit.py` reports `VERDICT: ALIGNED` for event 68399317,
   `freq_id` 844 when run with four chunks.
2. The production-geometry NumPy replica recovers the known signal with a mean
   coarse excess above 1200 percent, while the historical bank remains near the
   null.
3. The parity census covers all 23 selected channels. An additional odd-channel
   framing audit is optional.
4. The shipped CHIME bank selects `K = 128`, one skipped guard cell, and a
   two-cell reference offset.

Changing the detector window, weights, references, repack, frame convention, or
fine transform creates a new run contract and requires new approval.

## Final-source CPU gates

Run these from a clean checkout of the exact revision recorded in the ledger:

```bash
test -z "$(git status --porcelain)"
git rev-parse HEAD
PYTHONPATH=src python - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
PYTHONPATH=src python -m pytest tests -q
make -C cuda test_ref
```

Pass criteria:

- the source tree is clean;
- the complete test tree has zero failures;
- optional tests skip only when their documented external dependency is absent;
- the CPU kernel reference exits zero; and
- the commit and package-source digest are recorded in the ledger.

## Final-source RTX/SM89 gates

The approved host is the local RTX 5000 Ada workstation. Do not replace the
already preserved library during this run. The explicit load below and the
real-file gates exercise that exact artifact. The later build and kernel-test
commands exercise a disposable SM89 build from the same frozen CUDA source.

```bash
KERNEL_LIB=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.so
KERNEL_MANIFEST=/home/djg/rail/kernels/pilotproxy-detector-core-2.3.0-sm89-f6cd8529ca4b.manifest.json
printf '%s  %s\n' \
  f6cd8529ca4b4581aaa37a6007a372d5afb4afa8c730d8a4372a8eaf25e807f2 \
  "$KERNEL_LIB" \
  d781d3d4dfbe15dd336b1c89e412a91522376f69a30cfc9543fd52ff6a954cf0 \
  "$KERNEL_MANIFEST" | sha256sum --check --strict

PYTHONPATH=src python - "$KERNEL_LIB" <<'PY'
import sys
from pilot_proxy.kernel import FStatKernel

kernel = FStatKernel(sys.argv[1])
assert kernel.version.as_string() == "2.3.0"
assert kernel.supports_fine_powers()
assert kernel.supports_fused_fine()
print("kernel", kernel.version.as_string(), kernel.get_fine_specs())
PY

make -C cuda test_cuda SM=89
PYTHONPATH=src python -m pytest tests/kernel -q
```

Pass criteria:

- the preserved byte digest matches;
- the library reports core 2.3.0 and the required exact fine capabilities;
- the disposable SM89 CUDA reference and fixed-point parity tests exit zero;
- the kernel test directory has zero failures and zero skips against that
  disposable build; and
- the preserved bytes pass the explicit capability check and real-file gates.

## Real-file and resume gates

Repeat both gates after the source revision is frozen:

1. Process one full detector chunk from the 2048-stream `freq_id` 844 file.
   Require product validation, nonempty receiver-state identity, and peak VRAM
   below 13,900 MiB.
2. Process eight archive units with four download workers, eight staged-file
   slots, and a two-unit rehearsal checkpoint. Interrupt after a durable
   checkpoint, rerun the identical command, and require eight unique unit keys,
   no duplicate frames, valid v5 products, and empty staging.

The exact commands and evidence paths are in `LOCAL_PROCESSING.md`. These gates
must embed the final package-source digest and preserved kernel digest. A capped
rehearsal output is never reused for production.

For every per-pilot v5 product inspected during these gates, require:

- all valid frames remain in the per-frame arrays on both sides of zero excess;
- exact target, lower-reference, upper-reference, and summed-reference powers;
- exact fine powers with shape `[N, 3, 256]` and unsigned 64-bit dtype;
- one nonempty `unit_scope`, raw `unit_git_version_tag`, and
  `unit_input_map_sha256` per unit;
- every input-map identity is a 64-character lowercase SHA-256 digest;
- `fine_status == enabled` and an inactive fine candidate decision;
- the positive-excess bit affects only the recorded bootstrap flag and the
  after-mask diagnostic spectrum; and
- the source, kernel, profile, bank, and inventory identities match the ledger.

The authoritative products are `_per_pilot/<freq_id>.npz`. The combined product
is a derived common-frame projection and is not a substitute for them.

## Production settings

The approved local command uses:

- the frozen 165,682-unit inventory and all 23 selected `freq_id` values;
- four download workers and eight staged-file slots;
- a 250-unit production checkpoint interval;
- the preserved SM89 library named above;
- `fine_products=on`; this retains exact fine powers and does not enable fine
  detection;
- no file cap, no chunk cap, and no partial-run acknowledgement; and
- a fresh output directory outside the checkout.

The full synthetic detection frontier, residual-contamination calibration,
rho/eta selection, and final detection policy are later work. They are not
prelaunch gates because this run preserves the sufficient statistics needed to
perform them without reacquiring the archive.
