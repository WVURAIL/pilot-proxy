# CANFAR Runbook

This runbook describes bounded pre-production CHIME DTV pilot runs with
`pilot-proxy` and `datatrawl`. We begin with one file and one chunk because the
detector needs a GPU, a matching weight bank, and archive metadata that agrees
with the requested CHIME coarse channel. After that run validates, we expand the
same workflow to the selected archive.

For archive-scale work, use:

```bash
pilot-proxy chime-scan ...
```

The older `pilot-proxy chime-run` command reads an already-staged HDF5
directory in one process. We retain it for local calibration and regression
comparisons, but new CADC/CANFAR work should start with `chime-scan`.

---

## Baseline detector contract

The current executable detector contract is:

```text
detector_window_samples = 128
skipped_guard_bins = 1
reference_offset_bins = 2
mask mode = positive_excess
```

The detector core supports compile-time `K = 64` and `K = 128` builds. The
CHIME receiver profile and shipped CHIME weight bank select `K = 128`; it is
not a runtime tuning parameter. Another value would require a separately
supported kernel geometry, weight bank, profile, and validation campaign.

No GPU session is required for the CPU-only synthetic publication sweeps in
item 2 of `docs/PUBLICATION_VALIDATION.md`. Run those sweeps with:

```bash
pilot-proxy evaluate-snr \
  --input-iq <capture> \
  --detector-backend cpu-reference \
  --noise-source python
```

(The input capture is generated once with GNU Radio; item 2, step 1 of
`docs/PUBLICATION_VALIDATION.md` shows how.)

## Launch a GPU session

On your workstation first: clone this repository and install the `canfar`
client (`pip install canfar`); the launch script below runs locally, and the
clone inside the session happens later.

The `pilot-proxy-detector` path requires a CUDA GPU. The helper script launches
a CANFAR GPU notebook session through the `canfar` client, or reuses a running
or pending session with the same name. The client also needs a Harbor CLI secret
to pull the session image. Obtain that secret from
`https://images.canfar.net` under your profile, then either export it or let
`setup_env.sh` prompt for and store it.

```bash
export CANFAR_REGISTRY_USER=<your-cadc-username>
export CANFAR_REGISTRY_SECRET=<your-cli-secret>

python scripts/launch_gpu_session.py            # launch or reuse; print the connect URL
python scripts/launch_gpu_session.py --status   # status + URL only
python scripts/launch_gpu_session.py --destroy  # tear it down when done
```

The default session name is `cupy-gpu`. Open the printed URL and complete the
environment setup in that session's terminal.

---

## Environment setup

Clone both repositories before running the setup script. The script requires
both checkouts and recreates the target virtual environment from scratch.

```bash
git clone https://github.com/WVURAIL/pilot-proxy.git ~/pilot-proxy
git clone https://github.com/WVURAIL/datatrawl.git ~/datatrawl
cd ~/pilot-proxy

VENV_DIR=~/pilot-proxy-datatrawl DATATRAWL_DIR=~/datatrawl PILOT_PROXY_DIR=~/pilot-proxy bash scripts/setup_env.sh

source ~/pilot-proxy-datatrawl/bin/activate
```

Do not point `VENV_DIR` at an environment that must be preserved. The script
uses `python -m venv --clear`, installs both repositories in editable mode,
checks plugin discovery, and builds the CUDA library when a GPU and `nvcc` are
available. It refuses protected, checkout-overlapping, and unowned non-empty
targets and keeps its ownership record beside the environment so an interrupted
rebuild can be retried. If this is the first guarded rerun of a genuine virtual
environment created by an older checkout, add
`PILOT_PROXY_ADOPT_LEGACY_VENV=1` to the command once; do not use that opt-in for
an arbitrary directory.

In the expected CANFAR notebook environment, the home directory is on
persistent `/arc` storage. Activate the virtual environment in every new
session before running `pip` or `pilot-proxy`. A bare install against the
session image's Python may fail because that Python environment is read-only.
Rerun `setup_env.sh` only when the environment should be rebuilt.

If the environment should not be cleared, use the manual installation path:

```bash
python3.12 -m venv --system-site-packages ~/pilot-proxy-datatrawl   # keeps the image's CuPy importable
source ~/pilot-proxy-datatrawl/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e "$HOME/datatrawl[cadc,survey]"
python -m pip install -e "$HOME/pilot-proxy[datatrawl,chime,test]"   # CuPy comes from the image via datatrawl's accel, so no cuda extra
```

Then confirm that datatrawl can discover both PilotProxy plugins:

```bash
datatrawl list | grep -E 'pilot-proxy-detector|chime-baseband-packed'
```

The production analyzer also needs a working CUDA/CuPy runtime and a built or
staged `libfstatistic.so`.

---

## Pre-launch validation gates (mandatory)

`docs/VALIDATION_GATES.md` is authoritative; this is the on-node sequence.
Run in order after environment setup, stop at the first failure. Do not
launch any scan until every gate passes.

```bash
# S2.1 - library built from the current sources and reporting kernel core 2.3.0
make -C cuda clean && make -C cuda
python - <<'PY'
from pilot_proxy.kernel import FStatKernel
v = FStatKernel().version.as_string()
assert v == "2.3.0", f"stale kernel library: {v}"
print("kernel core", v)
PY

# S2.2 - CUDA regression + exact row-sum parity (integers, no tolerance)
make -C cuda test_cuda

# S2.3 - GPU pytest gates: numpy-reference equality, bit-exact coarse marginal
#        identity, and the pre-registered ULP gate on the cupy fine FFT.
#        These FAIL (not skip) on a stale library.
python -m pytest tests/kernel -q

# Full suite (GPU + datatrawl tests now run instead of skipping)
python -m pytest tests -q
```

Frame parity is already CLOSED by the 23-channel census over the
legacy-epoch integrated spectra (see gate S2.4 in
`docs/VALIDATION_GATES.md`); no odd-channel profile override is needed.
Optional belt-and-braces with any archived odd-`freq_id` baseband file:

```bash
python tools/framing_audit.py /path/to/baseband_<event>_<odd_freq_id>.h5
```

The bounded smoke test below is gate S2.5. Its current per-pilot-schema pass criteria and the
interrupt/resume check are listed at the end of that section.

---

## Required inputs

For a local scan, provide CHIME HDF5 baseband files whose names end in the
selected `freq_id`, such as `baseband_<event>_844.h5`. For other layouts, pass
`--source-freq-id-regex '<regex-with-one-capturing-group>'`; the value is
stored as `source_freq_id_regex` for the datatrawl source, and an explicit
`--set 'source_freq_id_regex=...'` takes precedence over the flag.

For a CADC/CANFAR scan, provide:

- a valid CADC proxy certificate;
- an `inventory.jsonl` produced by `datatrawl survey`;
- the CHIME `freq_id` values to scan, or an inventory from which they can be
  inferred.

The detector resolves these runtime artifacts unless explicit alternatives are
passed:

- `configs/receiver_profiles/chime_dtv_fengine.json` - must be the
  frame-verified `chime_dtv_fengine` profile
  (`baseband_frame.channel_center_normalized = 0.0`);
- `weights/chime_dtv_weights_k128.bin` - manifest
  `receiver_profile_hash` starts `da047cdf453764e4`; the legacy half-band
  bank lives only under `weights/legacy_halfband/` and must never be
  deployed;
- `cuda/libfstatistic.so`, or its staged cache copy;
- a working CuPy/CUDA runtime.

Check the receiver profile and layout before staging archive data:

```bash
pilot-proxy check-profile \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json

pilot-proxy check-layout \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --stream-map configs/stream_maps/chime_feed_pol_example.json
```

---

## Selection convention

`pilot-proxy chime-scan --select` uses the CHIME `freq_id` coarse-channel
identifier. It does not use the ATSC physical-channel number.

For ATSC physical channels 14 through 36, the corresponding pilot `freq_id`
set is:

```text
506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
```

For a one-channel smoke test, choose a `freq_id` present in the inventory or
local directory. `844` is the expected coarse channel for the ATSC 14 pilot.
Do not substitute `396-412`; those are not the `freq_id` values for this DTV
14-36 pilot selection.

---

## CADC inventory

Renew the CADC proxy certificate:

```bash
cadc-get-cert -u <your-cadc-username>
```

Begin with a bounded inventory:

```bash
datatrawl survey \
  --telescope chime \
  --source cadc-datatrail \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --name chime-pilots \
  --max-events 5
```

The survey resolves its output location through datatrawl's canonical
inventory root and prints the path it wrote, for example:

```text
~/datatrawl-inventories/chime-pilots/inventory.jsonl
```

Inspect the inventory without downloading baseband data:

```bash
datatrawl explore \
  --source cadc-datatrail \
  --telescope chime \
  --inventory ~/datatrawl-inventories/chime-pilots/inventory.jsonl
```

With `--inventory-name`, `chime-scan` resolves the inventory through the same
datatrawl resolver, so it works from any directory. For an inventory stored
elsewhere, pass `--source-root <survey-root>` or `--inventory <path>`
explicitly. Increase `--max-events` only after the bounded scan succeeds.

---

## Bounded detector smoke test

First verify the GPU runtime:

```bash
nvidia-smi
python - <<'PY'
import cupy as cp
print("CuPy", cp.__version__)
print("GPU count", cp.cuda.runtime.getDeviceCount())
PY
```

Then process one file and one full analysis chunk for `freq_id 844`:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

The file cap intentionally leaves part of the inventory unprocessed, so this
bounded smoke test explicitly acknowledges that with `--allow-partial`.
`scan_scope.json` remains the authoritative record of the omitted units.

Validate the combined products and generate the diagnostic plots:

```bash
pilot-proxy validate-products \
  --run-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --output-json "$HOME/pilot_proxy_runs/detector_smoke_844/product_validation.json"

pilot-proxy chime-plot \
  --run-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --clean-figures
```

After these commands, the run directory should contain:

```text
run_config.json
stats.json
input_manifest.json
product_validation.json
chime_detector_outputs.npz
chime_spectrogram_cache.npz
chime_integrated_spectra.npz
chime_reductions_10s.npz
tables/mask_summary_by_pilot.csv
figures/*.png
```

An event-keyed combine also writes `chime_frame_identity.npz`. Current products
must contain event-keyed frame identity; combine refuses inputs without it.

Gate S2.5 pass criteria on the smoke product (current schema):

```bash
python - <<'PY'
import numpy as np, glob
import os
p = sorted(glob.glob(os.path.expanduser(
        "~/pilot_proxy_runs/detector_smoke_844/**/_per_pilot/844.npz"),
        recursive=True))
z = np.load(p[-1], allow_pickle=False)
sv = str(np.asarray(z["schema_version"]).reshape(()).item())
ev = str(np.asarray(z["source_event_key_schema_version"]).reshape(()).item())
dv = str(np.asarray(z["detector_version"]).reshape(()).item())
fs = str(np.asarray(z["fine_status"]).reshape(()).item())
assert sv == "pilotproxy_per_pilot_product_v3", sv
assert ev == "pilotproxy_namespaced_source_event_key_v1", ev
assert "pilot-proxy/" in dv and "kernel=2.3.0" in dv, dv
assert fs == "enabled", fs
assert z["fine_power_ratio"].shape[1] == int(z["fine_num_bins"]), z["fine_power_ratio"].shape
r = np.asarray(z["fine_null_bulk_exceedance_fraction"]).reshape(-1)
print("schema", sv)
print("event identity", ev)
print("fine bins", int(z["fine_num_bins"]),
      "| null-bulk exceedance median",
      float(np.nanmedian(r)))
PY
```

The null-bulk exceedance fraction is an in-sample threshold diagnostic, not an
independent measured false-alarm rate. Interpret it together with the stored
fine spectrum. For the resume half of the
gate, run the same `chime-scan` command in a **new** output directory with
`--max-files 2` and no chunk cap; the smoke product above was built with
`max_chunks_per_file=1`, and a capped product refuses completion under a
different cap by design. Interrupt the run after the first per-pilot
checkpoint lands (watch for `_per_pilot/<freq_id>.npz` appearing in the
output directory), run the identical command again, and confirm the frame
count continues without duplication and `validate-products` passes on the
result. If the bounded run finishes before it can be interrupted, relaunch
the identical command with a higher `--max-files` on the same directory;
the relaunch restores the checkpoint and continues through the same resume
path. Keep `--allow-partial` on these deliberately bounded resume commands;
remove both the file cap and that acknowledgement for the production run.

---

## H0 zero-point check

Before spending GPU time on the full archive, test the detector on a channel
that should approximate the no-pilot hypothesis, H0. Choose a DTV pilot
frequency that lies inside the selected CHIME coarse channel but whose physical
channel has no station listed in the 500-mile census. This is a census-based
control selection rather than a propagation prediction. Do not choose an arbitrary
coarse channel with no nominal ATSC pilot in band: the analyzer marks that case
invalid and does not form an local-reference power ratio.

Two constraints bound that selection for this archive. First, match the
census epoch to the data epoch: the shipped census reflects its 2026
retrieval date, while archive events span earlier years, and a channel that
is quiet in the census can have been occupied at observation time (the
channel 27 / `freq_id` 644 interior analysis records 38 per cent detections
in 2020 Q3 against quiet 2025-26 endpoints). Second, the shipped census
lists at least one station on every physical channel 14-36, so a strictly
station-free control does not exist in this pilot set; use the most
isolated channels as approximate controls and take the operating zero point
from the empirical calibration (`analysis/build_empirical_thresholds.py`)
rather than from a census assumption.

Run the bounded scan for the census-control channel and for one quiet channel
with a known pilot. Then:

1. Read `null_power_ratio` from `chime_detector_outputs.npz` or from the authoritative
   `_per_pilot/<freq_id>.npz` product. The `chime-run` batch path also records
   `null_power_ratio_by_channel` in `stats.json`, but the combined `chime-scan` statistics do
   not currently duplicate that array.
2. Over frames with `valid = 1`, compare the mean `coarse_power_ratio` with `null_power_ratio`. Also
   inspect the valid-frame mask fraction. Under the tested white-noise model,
   the corrected threshold gives a fraction near one half; on real data this is
   a diagnostic expectation rather than a pass condition by itself.
3. If the control result is strongly displaced, check the weight bank,
   `mask_rule`, channel selection, and structured interference before expanding
   the scan.

`tests/core/test_mask_zero_point.py` performs the corresponding white-noise
regression with the shipped weights. The on-sky check tests the additional
instrument and archive path that the synthetic regression cannot cover.

## Definitive-run pre-flight

For a run intended to process the archive once and never again, hold the
launch to this checklist. Every item is a real failure mode rather than
a hypothetical one.

1. **All gates, in full.** S2.1--S2.5 with no shortcuts. The S2.5
   interrupt/resume rehearsal is mandatory, since a multi-day run will
   cross session restarts, and resume is the mechanism that makes a
   session death cost minutes instead of days.
2. **Binary insurance at launch.** The survey pins
   `kernel_sha256`, and kernel builds are not byte-reproducible: identical
   sources, flags, and toolkit produce different bytes on each invocation
   (nvcc embeds per-invocation artifacts). The copy is the insurance,
   never the rebuild:

   ```bash
   cp cuda/libfstatistic.so cuda/libfstatistic.so.$(git rev-parse --short HEAD)
   sha256sum cuda/libfstatistic.so*    # record both in the run ledger
   ```

   Any resume after any rebuild passes the detector option
   `lib_path=<preserved copy>`.
3. **Inventory completeness before the run.** Confirm the survey
   inventory covers every selected `freq_id` (all 23 for the DTV band)
   and explain any channel whose unit selection looks anomalous;
   sparse channels and late-starting spans are only acceptable when the
   archive genuinely holds no more data, and that should be established
   before the run rather than discovered after it.
4. **Session-lifetime plan.** Know why the previous scan session ended
   (expiry vs crash) and budget the relaunch cadence around the session
   lifetime. Checkpointing (`--checkpoint-every 50`) plus unit-level
   resume bounds the loss --- but only if items 1 and 2 hold.
5. **First-product tripwire.** After the first ~50 units of the first
   channel, stop and check before the remaining channels burn days: the
   measured pilot line sits at its known offset (`freq_id` 506: KSKN at
   +739 Hz), the coarse excess is in family with the prior cohort, and
   the product passes validation. Kill criteria belong at hour one, not
   week two.
6. **Per-channel acceptance while running.** Validate and audit each
   product as it completes (`tools/audit_per_pilot.py` re-derives every
   internally checkable quantity from first principles and is safe on live
   checkpoints); check the run heartbeat
   (`find <run> -mmin -60`) at least daily.
7. **Cohort record.** Each run's products pin one binary and one source
   commit; record hash, commit, image/CUDA version, and channel list in
   the run ledger at launch, and never mix cohorts inside one output
   directory.
8. **`cupy` on the node, not just the `.so`.** The detector pre-flight
   fails closed with "cupy/CUDA is not importable" before any unit is
   staged. A node can carry a freshly built `cuda/libfstatistic.so` and
   still be unusable. Check it at hour zero:

   ```bash
   python -c "import cupy; print(cupy.__version__)"
   ```

9. **Read the quarantine ledger before accepting the run.** A unit whose
   bytes cannot yield one complete transform is rejected at probe time and
   quarantined, so the run survives it rather than aborting --- but
   quarantine is *persistent*, and a later run will skip that unit without
   re-examining it. The scan prints each one as `QUARANTINE <name>: ...`
   and the scope gate reports `quarantined=<n>`. Confirm every entry is a
   genuinely short acquisition (the file opens cleanly and holds its full
   declared extent) rather than a staging problem. Observed 2026-08-23 in
   the CANFAR pilots: two acquisitions of 1953 and 3906 samples against a
   typical 200000, each appearing on two channels.
10. **Expect the terminal combine over all channels to be empty.** The
   stack keeps only `(event, frame)` identities common to *every* selected
   pilot, and a triggered event lights a few surrounding channels rather
   than all 23. Measured over 115 events across the 23 DTV channels, no
   event was present in all of them. That is not a failure --- the
   per-pilot products are the deliverable, and the scan exits zero --- but
   decide before launch whether a full-band stack is wanted at all, or
   whether the run should target channel subsets from the start.

---

## Full pilot detector run

After the smoke test and H0 check are acceptable, run the selected inventory.
For the archive source, omitting `--select` selects every `freq_id` present in
the inventory. The command prints the resolved set before staging data, and
`--inventory-name` implies `--source cadc-datatrail`.

The production run uses a fresh name: products reduced with a different
weight bank or kernel must never be resumed or combined - the
`detector_version` invariant refuses this by design. `fine_products=on` hard-fails if the node's kernel library
is stale, which is the correct failure mode for a production launch.

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots-current" \
  --inventory-name chime-pilots \
  --set fine_products=on \
  --checkpoint-every 50
```

The terminal combine aligns frames by `(event, frame-in-file)` identity. It
keeps only identities common to every completed channel, records the per-channel
drops under `combine_alignment` in `stats.json`, and writes the retained
identities to `chime_frame_identity.npz`.

Some archives are ragged: different channels may contain different event sets.
If no event is common to all selected channels, the per-pilot products remain
complete and the terminal stack is skipped. Inspect channel presence and choose
a subset:

```bash
pilot-proxy chime-combine --report --work-dir "$HOME/pilot_proxy_runs/chime-pilots-current/_per_pilot"
pilot-proxy chime-combine \
  --work-dir "$HOME/pilot_proxy_runs/chime-pilots-current/_per_pilot" \
  --drop 598,690 \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots-current-subset"
```

The report gives the event count per channel, the presence histogram, and a
greedy drop curve. Use those quantities to state which channels are retained;
the drop curve is a decision aid rather than an automatic scientific
selection.

Validate and plot whichever directory contains the final combined products:

```bash
pilot-proxy validate-products \
  --run-dir "$HOME/pilot_proxy_runs/chime-pilots-current" \
  --output-json "$HOME/pilot_proxy_runs/chime-pilots-current/product_validation.json"

pilot-proxy chime-plot \
  --run-dir "$HOME/pilot_proxy_runs/chime-pilots-current" \
  --clean-figures
```

If a subset combine was required, replace `chime-pilots-current` with
`chime-pilots-current-subset` in both commands.

---

## Local staged-data equivalent

For HDF5 data already on disk, run the same detector analyzer through the local
source:

```bash
export LOCAL_H5=/path/to/chime_hdf5

pilot-proxy chime-scan \
  --input-dir "$LOCAL_H5" \
  --output-dir "$HOME/pilot_proxy_runs/local_detector_smoke_844" \
  --source local \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

If the filenames do not end in `_<freq_id>.h5`, pass the parser override with
one capturing group:

```bash
--source-freq-id-regex '<regex-with-one-capturing-group>'
```

The flag is stored as `source_freq_id_regex` for the datatrawl
`LocalDirectorySource`; an explicit `--set 'source_freq_id_regex=...'` takes
precedence over the flag.

---

## Control freq_ids (no pilot)

`pilot-proxy-detector` is undefined on a coarse channel without a pilot: it
marks every frame invalid or refuses at `begin()`. Control bins (the protected
608–614 MHz null band, mid-allocation transfer bins, canary bins) go through
the `pilot-proxy-control` analyzer instead, via raw `datatrawl scan` against a
surveyed control inventory:

```bash
datatrawl scan \
  --inventory ~/datatrawl-inventories/chime-controls/inventory.jsonl \
  --analyzer pilot-proxy-control \
  --select 484,491,515,545,591,745 \
  --max-files 2 --max-frames-per-file 4     # drop the caps for the full pass
```

No GPU is needed (these scans are network-bound; a CPU-only session is fine),
and one resumable `<freq_id>.npz` is written per selected bin — see
`docs/DATA_PRODUCTS.md`, "Control-Band NPZ", for the schema, the
deployed-geometry `F(b)` recipe, and the Parseval self-check. A capped
smoke-test product is stamped and cannot be resumed by an uncapped pass;
point the full run at a fresh `--out`.

Before trusting control-bin `F` values, run the same analyzer once on a pilot
`freq_id` and compare against the detector product (powers should match
per-frame in native units; `F` at the pilot bin agrees at the
weight-quantization level, not bit-exactly).

---

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | CPU-only host or unavailable driver utility | Move the detector run to a GPU node |
| `nvcc: command not found` | CUDA compiler toolkit is not on `PATH` | Load the CUDA toolkit/module or set `NVCC`/`PATH` |
| `pilot-proxy-detector needs cupy/CUDA` | Production detector started in a CPU-only environment | Use a GPU node |
| `no files matched` | `--select` does not match the inventory or local filename `freq_id` | Run `datatrawl explore` or inspect the filenames |
| first file's center implies a different `freq_id` | Inventory or filename label disagrees with HDF5 metadata | For local data, pass the current parser with `--set source_freq_id_regex=...`; for archive data, rebuild the inventory |
| combine finds no common events | The selected channels contain different event sets | Run `chime-combine --report`, choose a stated subset, and recombine with `--drop` |
| all frames are invalid | The selected coarse channel does not contain the nominal pilot, or the reference denominator is zero | Check `freq_id`, HDF5 frequency metadata, and the detector weights |

---

## Restart policy

Use a new output directory when the existing products are suspect. Resume only
when the partial run is known to have the same channel selection, frame cap,
weights, detector geometry, and provenance. The analyzer rejects several
incompatible resume cases, but that validation does not classify a scientifically
bad run.

**Rebuilding the kernel mid-survey.** Resumed products pin the kernel by
hash (`kernel_sha256=` in `detector_version`), and that token is strict:
resume forgiveness covers only the Python `source=` stamp. Any
`make build-kernel` therefore makes existing partial products refuse to
resume under the new `.so`, even for an additive ABI change. Before
rebuilding on a machine with an in-progress survey, preserve the current
library (for example
`cp cuda/libfstatistic.so cuda/libfstatistic.so.<git-short-hash>`) and
resume with the detector option `lib_path=<preserved copy>`; move the
survey to the new build only at a survey epoch boundary.

Keep a failed run until its failure has been classified. Do not commit generated
products while diagnosing the run.

---

## Calibration export (baonoise interface)

After a scan's products validate, export the forecasting-side calibration
bundle (the "PilotProxy -> baonoise export specification" files) directly
from the per-pilot products --- no baseband access needed:

```bash
python tools/export_baonoise_calibration.py \
  --per-pilot-dir <run>/_per_pilot \
  --out <run>/baonoise_export
```

The tool derives the per-frame decision statistic (max of the fine
statistic over the per-channel designated window, measured-line anchored),
histograms it per (channel, quarter), measures the empirical null from
off-pilot windows and transmitter-off epochs, substitutes per-event
maxima for the spec's 10 s windows (baseband snapshots contain no 10 s
integrations --- recorded in `provenance.json`; an exploratory
integrated-event pair adds the detect-on-integrated-spectra policy), and
runs the spec's
ingest-side validations before exiting. A nonzero exit means the bundle
must not be shipped. Note the leakage caveat recorded in provenance: on
occupied channels during on-epochs, off-pilot windows carry transmitter
leakage; the `off_epoch_anchor_window` rows are the calibration-grade
null.

---

## Archive policy

For each accepted run, archive:

- the validated products and `product_validation.json`;
- the receiver profile and any stream map used;
- the weight bank manifest;
- the source commit hash or source archive;
- the exact CADC inventory used.

Do not commit CANFAR products, local HDF5 files, generated figures, or CUDA build
artifacts to the source repository.

## Compatibility note: datatrawl inventory metadata

`pilot-proxy chime-scan` selects the `chime-baseband-packed` reader for the
`pilot-proxy-detector` analyzer. A raw `datatrawl scan` may instead infer the
canonical unpacked CHIME reader from inventory metadata. In that case, pass the
packed reader explicitly. See
[INTEGRATION.md](../INTEGRATION.md#compatibility-note-datatrawl-inventory-metadata)
for the direct datatrawl commands and the reason for the override.
