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

The CUDA kernel and the shipped weight bank fix the detector window at
`K = 128`; it is not a runtime tuning parameter. `K = 256` remains a future
candidate discussed in the design documents, but it is not part of this run
contract.

No GPU session is required for the CPU-only synthetic publication sweeps in
item 2 of `docs/PUBLICATION_VALIDATION.md`. Run those sweeps with:

```bash
pilot-proxy evaluate-snr \
  --detector-backend cpu-reference \
  --noise-source python
```

## Launch a GPU session

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
available.

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

## Required inputs

For a local scan, provide CHIME HDF5 baseband files whose names end in the
selected `freq_id`, such as `baseband_<event>_844.h5`. The current datatrawl
local source reads the option key `source_freq_id_regex`. PilotProxy's advertised
`--source-channel-regex` flag still writes the older key
`source_channel_regex`, so it does not override the current source parser. Until
that interface is repaired, pass the current key through `--set` as shown in the
local workflow below.

For a CADC/CANFAR scan, provide:

- a valid CADC proxy certificate;
- an `inventory.jsonl` produced by `datatrawl survey`;
- the CHIME `freq_id` values to scan, or an inventory from which they can be
  inferred.

The detector resolves these runtime artifacts unless explicit alternatives are
passed:

- `configs/receiver_profiles/chime_dtv_fengine.json`;
- `weights/chime_dtv_weights_k128.bin`;
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

This command writes:

```text
data/chime-pilots/inventory.jsonl
```

Inspect the inventory without downloading baseband data:

```bash
datatrawl explore \
  --source cadc-datatrail \
  --telescope chime \
  --inventory data/chime-pilots/inventory.jsonl
```

Run `chime-scan` from the directory in which `datatrawl survey` wrote the
`data/` tree. If the scan starts elsewhere, add `--source-root <survey-root>`
to each command. Increase `--max-events` only after the bounded scan succeeds.

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
  --max-chunks-per-file 1
```

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

An event-keyed combine also writes `chime_frame_identity.npz`. A combine of
legacy products without identity tags uses strict positional alignment and does
not write that sidecar.

---

## H0 zero-point check

Before spending GPU time on the full archive, test the detector on a channel
that should approximate the no-pilot hypothesis, H0. Choose a DTV pilot
frequency that lies inside the selected CHIME coarse channel but whose physical
channel has no station listed in the 500-mile census. This is a census-based
control selection, not a propagation prediction. Do not choose an arbitrary
coarse channel with no nominal ATSC pilot in band: the analyzer marks that case
invalid and does not form an F-statistic.

Run the bounded scan for the census-control channel and for one quiet channel
with a known pilot. Then:

1. Read `mu0` from `chime_detector_outputs.npz` or from the authoritative
   `_per_pilot/<freq_id>.npz` product. The `chime-run` batch path also records
   `mu0_by_pilot` in `stats.json`, but the combined `chime-scan` statistics do
   not currently duplicate that array.
2. Over frames with `valid = 1`, compare the mean `fstat_raw` with `mu0`. Also
   inspect the valid-frame mask fraction. Under the tested white-noise model,
   the corrected threshold gives a fraction near one half; on real data this is
   a diagnostic expectation, not a pass condition by itself.
3. If the control result is strongly displaced, check the weight bank,
   `mask_rule`, channel selection, and structured interference before expanding
   the scan.

`tests/core/test_mask_zero_point.py` performs the corresponding white-noise
regression with the shipped weights. The on-sky check tests the additional
instrument and archive path that the synthetic regression cannot cover.

## Full pilot detector run

After the smoke test and H0 check are acceptable, run the selected inventory.
For the archive source, omitting `--select` selects every `freq_id` present in
the inventory. The command prints the resolved set before staging data, and
`--inventory-name` implies `--source cadc-datatrail`.

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --inventory-name chime-pilots
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
pilot-proxy chime-combine --report --work-dir "$HOME/pilot_proxy_runs/chime-pilots/_per_pilot"
pilot-proxy chime-combine \
  --work-dir "$HOME/pilot_proxy_runs/chime-pilots/_per_pilot" \
  --drop 598,690 \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots-subset"
```

The report gives the event count per channel, the presence histogram, and a
greedy drop curve. Use those quantities to state which channels are retained;
the drop curve is a decision aid rather than an automatic scientific
selection.

Validate and plot whichever directory contains the final combined products:

```bash
pilot-proxy validate-products \
  --run-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --output-json "$HOME/pilot_proxy_runs/chime-pilots/product_validation.json"

pilot-proxy chime-plot \
  --run-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --clean-figures
```

If a subset combine was required, replace `chime-pilots` with
`chime-pilots-subset` in both commands.

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
  --max-chunks-per-file 1
```

If the filenames do not end in `_<freq_id>.h5`, override the current datatrawl
parser with one capturing group:

```bash
--set 'source_freq_id_regex=<regex-with-one-capturing-group>'
```

Do not rely on `--source-channel-regex` for the current repository pair. That
flag populates `source_channel_regex`, while `LocalDirectorySource` reads
`source_freq_id_regex`.

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

Keep a failed run until its failure has been classified. Do not commit generated
products while diagnosing the run.

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
