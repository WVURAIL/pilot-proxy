# CANFAR Runbook

This runbook describes bounded pre-production CHIME DTV pilot runs using
`pilot-proxy` and `datatrawl`.

The recommended archive-scale entry point is:

```bash
pilot-proxy chime-scan ...
```

The older `pilot-proxy chime-run` path remains useful for local, already-staged
HDF5 data and regression comparisons, but new CANFAR/CADC work should start with
`chime-scan`.

---

## Baseline detector contract

Use the current validated detector contract:

```text
detector_window_samples = 128
skipped_guard_bins = 1
reference_offset_bins = 2
mask mode = positive_excess
candidate K values = 128, 256
```

The detector window is not a runtime tuning parameter. It is determined by the
kernel contract and the weight bank.

---

## Launch a GPU session

The detector path (`pilot-proxy-detector`) needs a CUDA GPU node.
`scripts/launch_gpu_session.py` launches (or reuses) a CANFAR GPU notebook session
via the `canfar` client. It needs your Harbor CLI secret to pull the session image
(https://images.canfar.net -> your profile -> CLI secret); `setup_env.sh` prompts for
and stores it, or export it yourself:

```bash
export CANFAR_REGISTRY_USER=<your-cadc-username>
export CANFAR_REGISTRY_SECRET=<your-cli-secret>

python scripts/launch_gpu_session.py            # launch or reuse; print the connect URL
python scripts/launch_gpu_session.py --status   # status + URL only
python scripts/launch_gpu_session.py --destroy  # tear it down when done
```

It reuses a session of the same name (`cupy-gpu`) rather than duplicating it. Open the
printed URL, then run the environment setup below inside that session's terminal.

---

## Environment setup

Use the setup script from the fstat checkout:

```bash
cd ~/pilot-proxy

VENV_DIR=~/pilot-proxy-datatrawl DATATRAWL_DIR=~/datatrawl PILOT_PROXY_DIR=~/pilot-proxy bash scripts/setup_env.sh

source ~/pilot-proxy-datatrawl/bin/activate
```

The script recreates the target venv. Do not point VENV_DIR at a venv you need to
preserve.

Confirm plugin discovery:

```bash
datatrawl list | grep -E 'pilot-proxy-detector|pilot-proxy-offset|chime-baseband-packed'
```

On a CPU-only node, only the offset analyzer should be run. The detector analyzer
requires a GPU node with CUDA/CuPy and a built/staged `libfstatistic.so`.

---

## Required inputs

For local scans:

- CHIME HDF5 baseband files named with the terminal `freq_id`, for example
  `baseband_<event>_844.h5`, or an explicit `--source-channel-regex`.

For CADC/CANFAR scans:

- a valid CADC proxy certificate;
- an `inventory.jsonl` built by `datatrawl survey`;
- the desired CHIME `freq_id` selection.

For detector scans:

- `configs/receiver_profiles/chime_dtv_fengine.json`;
- `weights/chime_dtv_weights_k128.bin`;
- `cuda/libfstatistic.so` or the staged cache copy;
- a working CuPy/CUDA runtime.

Before a detector run, confirm the profile and layout:

```bash
pilot-proxy check-profile   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json

pilot-proxy check-layout   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --stream-map configs/stream_maps/chime_feed_pol_example.json
```

---

## Selection convention

`pilot-proxy chime-scan --select` uses CHIME `freq_id` coarse-channel indices, not
ATSC physical-channel numbers.

For the default ATSC physical-channel range 14-36, the corresponding CHIME pilot
`freq_id` set is:

```text
506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
```

For a one-channel smoke test, use a `freq_id` that is present in the inventory or
local directory. `844` is the expected coarse channel for the ATSC 14 pilot.

Do not use `396-412` for the default DTV 14-36 run.

---

## CADC inventory

Renew the CADC certificate:

```bash
cadc-get-cert -u <your-cadc-username>
```

Build a bounded first inventory:

```bash
datatrawl survey \
  --telescope chime \
  --source cadc-datatrail \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --name chime-pilots \
  --max-events 5
```

This writes:

```text
data/chime-pilots/inventory.jsonl
```

Inspect without downloading data:

```bash
datatrawl explore \
  --source cadc-datatrail \
  --telescope chime \
  --inventory data/chime-pilots/inventory.jsonl
```

Run the fstat scan commands below from the same directory where `datatrawl survey`
wrote the `data/` tree. If you run from another directory, add
`--source-root <survey-root>` to each `pilot-proxy chime-scan` command.

Increase `--max-events` only after a bounded scan succeeds.

---

## Bounded offset smoke test

The offset analyzer is CPU-only and should be the first end-to-end test:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/offset_smoke_844" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-offset \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1
```

Expected products:

```text
frequency_offset_outputs.npz
tables/frequency_offset_summary_by_pilot.csv
_per_pilot/844.npz
```

Use `choose-detector-k` on offset products:

```bash
pilot-proxy choose-detector-k   --frequency-offset "$HOME/pilot_proxy_runs/offset_smoke_844/frequency_offset_outputs.npz"   --output "$HOME/pilot_proxy_runs/offset_smoke_844/tables/k_candidate_summary.csv"   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json
```

Offset-only directories are not full detector run directories. Do not expect them
to pass detector product validation unless detector products are also present.

---

## Bounded detector smoke test

Run this only on a GPU node:

```bash
nvidia-smi
python - <<'PY'
import cupy as cp
print("CuPy", cp.__version__)
print("GPU count", cp.cuda.runtime.getDeviceCount())
PY
```

Then:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1
```

Validate and plot:

```bash
pilot-proxy validate-products   --run-dir "$HOME/pilot_proxy_runs/detector_smoke_844"   --output-json "$HOME/pilot_proxy_runs/detector_smoke_844/product_validation.json"

pilot-proxy chime-plot   --run-dir "$HOME/pilot_proxy_runs/detector_smoke_844"   --clean-figures
```

Expected detector outputs:

```text
run_config.json
stats.json
input_manifest.json
chime_detector_outputs.npz
chime_spectrogram_cache.npz
chime_reductions_10s.npz
tables/mask_summary_by_pilot.csv
figures/*.png
```

---

## Full pilot detector run

After the bounded detector smoke test passes:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/detector_pilots" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-detector \
  --select 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
```

Validate:

```bash
pilot-proxy validate-products   --run-dir "$HOME/pilot_proxy_runs/detector_pilots"   --output-json "$HOME/pilot_proxy_runs/detector_pilots/product_validation.json"
```

Plot:

```bash
pilot-proxy chime-plot   --run-dir "$HOME/pilot_proxy_runs/detector_pilots"   --clean-figures
```

---

## Local staged-data equivalent

For data already on disk:

```bash
export LOCAL_H5=/path/to/chime_hdf5

pilot-proxy chime-scan   --input-dir "$LOCAL_H5"   --output-dir "$HOME/pilot_proxy_runs/local_detector_smoke_844"   --source local   --analyzer pilot-proxy-detector   --select 844   --max-files 1   --max-chunks-per-file 1
```

If filenames do not end in `_<freq_id>.h5`, pass:

```bash
--source-channel-regex '<regex-with-one-capturing-group>'
```

---

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | CPU-only host or driver utility unavailable | Run only `pilot-proxy-offset`, or move detector run to GPU node |
| `nvcc: command not found` | CUDA compiler module/toolkit not loaded | Load CUDA module or set `NVCC`/`PATH` |
| `pilot-proxy-detector needs cupy/CUDA` | Detector run on CPU-only env | Use GPU node or run `pilot-proxy-offset` only |
| `no files matched` | `--select` does not match local filename/inventory `freq_id` | Run `datatrawl explore` or inspect filenames |
| first file's center implies a different `freq_id` | filename/inventory mismatch | Fix filename regex or rebuild inventory |
| combine rejects time alignment | selected pilots did not process the same events in same order | use matched files/inventory, or drop affected channel |
| validation requires detector products | offset-only run passed to detector validator | validate detector runs only, or use offset-specific checks |
| all frames invalid for a pilot | selected coarse channel does not contain pilot or no valid refs | verify `freq_id` selection and HDF5 metadata |

---

## Restart policy

Do not append into a suspect output directory. Start a new output directory for
retries unless you are deliberately resuming a known-good partial run.

Keep failed run directories until the failure is classified. Remove generated
products before committing or packaging the source tree.

---

## Archive policy

Archive:

- validated run products;
- `product_validation.json`;
- the exact receiver profile;
- the exact stream map if used;
- the exact weight manifest;
- the exact commit hash or source archive;
- the exact inventory used for CADC scans.

Do not commit CANFAR products, local HDF5 files, generated figures, or CUDA build
artifacts into the source tree.

## Compatibility note: datatrawl inventory metadata

`pilot-proxy chime-scan` selects the correct reader per analyzer automatically
(`chime-baseband` for `pilot-proxy-offset`, `chime-baseband-packed` for
`pilot-proxy-detector`). When driving raw `datatrawl scan` directly, the detector run
must override the inferred reader. See
[INTEGRATION.md](../INTEGRATION.md#compatibility-note-datatrawl-inventory-metadata)
for the full explanation and the override commands.
