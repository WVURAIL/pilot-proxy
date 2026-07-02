# pilot-proxy <-> datatrawl integration

This integration lets `pilot-proxy` run CHIME DTV analyses through the `datatrawl`
streaming engine while keeping the PilotProxy science code in `pilot-proxy`.

The design goal is:

```text
shared streaming engine, private science analyzers
```

`datatrawl` supplies storage-safe enumeration, staging, streaming, checkpointing,
and per-channel fan-out. `pilot-proxy` supplies the CHIME/DTV readers, analyzers,
CUDA detector call, and canonical PilotProxy product writers.

---

## How it fits together

| datatrawl axis | pilot-proxy integration |
|---|---|
| instrument | datatrawl's bundled `chime` geometry: 800 MHz band top, 1024 coarse channels, `nfft=16384`, inverted spectral sense |
| source | datatrawl's `local` and `cadc-datatrail` sources |
| reader | `chime-baseband-packed` for detector runs |
| analyzer | `pilot-proxy-detector` |

The analyzer reuses PilotProxy's DSP rather than reimplementing it:

- `pilot-proxy-detector` wraps `pack_chime_block_for_detector` and the detector call.
  The production detector path uses the CUDA kernel by default.

The analyzer produces one per-pilot `<freq_id>.npz` product. The combine step then
stacks per-pilot products and writes PilotProxy's canonical outputs:

- `chime_detector_outputs.npz`
- `chime_spectrogram_cache.npz`
- `chime_reductions_10s.npz`
- CSV summaries under `tables/`
- JSON provenance under `run_config.json`, `stats.json`, and `input_manifest.json`

---

## Setup

The integrated workflow uses `scripts/setup_env.sh`, which recreates a clean
virtual environment, installs both repos editable, builds the CUDA kernel when
possible, verifies plugin discovery, and runs the datatrawl integration tests.
The full procedure --- the exact invocation, the manual fallback, and the
plugin-discovery check --- is in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#environment-setup).

For detector runs, use a GPU node with a CuPy build matching the CUDA runtime.

---

## Selection: CHIME `freq_id`, not ATSC channel

`pilot-proxy chime-scan --select` is in the CHIME coarse-channel namespace. It is
not the ATSC physical-channel namespace.

A CHIME baseband file is one coarse channel. The local source and CADC inventory
key files by `freq_id`, and archive-style filenames look like:

```text
baseband_<event>_<freq_id>.h5
```

For the default DTV physical-channel range 14-36, use the 23-value pilot `freq_id`
set in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#selection-convention).

Do not use `396-412` for the default DTV 14-36 workflow. That range is not the
ATSC 14-36 pilot set for the CHIME 400-800 MHz band geometry.

For a single-channel smoke test, use one selected `freq_id` that exists in the
inventory or local filenames. For ATSC physical channel 14, the expected CHIME
coarse-channel center is about 470.3125 MHz, corresponding to `freq_id=844`.

---

## Local source workflow

For local source mode, the default filename regex is:

```text
_(\d+)\.h5$
```

so names such as these work:

```text
baseband_<event>_844.h5
baseband_<event>_829.h5
```

If your files use another naming convention, pass `--source-channel-regex` with
one capture group containing the integer `freq_id`.

Inspect local files:

```bash
find "$LOCAL_H5" -maxdepth 1 -name "*.h5"   | sed -E 's/.*_([0-9]+)\.h5$/\1/'   | sort -n | uniq
```

Run a GPU detector smoke test:

```bash
pilot-proxy chime-scan   --input-dir "$LOCAL_H5"   --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844"   --source local   --analyzer pilot-proxy-detector   --select 844   --max-files 1   --max-chunks-per-file 1
```

---

## CADC / CANFAR workflow

Build a bounded inventory first:

```bash
cadc-get-cert -u <your-cadc-username>

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

Inspect the inventory without downloading data:

```bash
datatrawl explore \
  --source cadc-datatrail \
  --telescope chime \
  --inventory data/chime-pilots/inventory.jsonl
```

When using `--inventory-name`, run `pilot-proxy chime-scan` from the same
directory where `datatrawl survey` wrote `data/chime-pilots/`, or pass
`--source-root <survey-root>`.

Run the detector analyzer over a bounded selection:

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

Run the detector analyzer over all default pilot `freq_id`s:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/detector_pilots" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-detector \
  --select 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
```

---

## Post-processing

Detector products:

```bash
pilot-proxy validate-products --run-dir <detector_run>
pilot-proxy chime-plot --run-dir <detector_run>
```

---

## Order-safety constraint

The PilotProxy analyzers append frames in delivery order. `pilot-proxy chime-scan` forces
`download_workers=1` and `max_staged_files=1` so products remain time-aligned.
Driving `pilot-proxy-detector` directly through raw `datatrawl scan`
with non-default concurrency can deliver files in completion order and should be
avoided unless the analyzer is made order-insensitive.

---

## Current verification status

The offline `tests/datatrawl/` suite exercises the integration without requiring
GPU or CADC access. The parity tests compare synthetic products from
`chime-scan` against products from the classic PilotProxy runner.

The following remain operational CANFAR/GPU checks:

- real CUDA-kernel detector path end-to-end;
- real CADC `cadc-datatrail` streaming;
- real-data parity against a known validated CHIME run;

---

## File manifest

New or integration-specific files:

```text
INTEGRATION.md
src/pilot_proxy/datatrawl_plugins/__init__.py
src/pilot_proxy/datatrawl_plugins/_chime_coarse.py
src/pilot_proxy/datatrawl_plugins/detector.py
src/pilot_proxy/datatrawl_plugins/packed_reader.py
src/pilot_proxy/datatrawl_plugins/combine.py
src/pilot_proxy/datatrawl_plugins/scan.py
tests/datatrawl/test_packed_reader.py
tests/datatrawl/test_detector_analyzer_parity.py
tests/datatrawl/test_combine_parity.py
tests/datatrawl/test_scan_parity.py
tests/datatrawl/test_selection_validation.py
tests/datatrawl/test_scan_layer.py
```

Changed integration surface:

```text
pyproject.toml          # datatrawl optional extra and plugin entry points
src/pilot_proxy/cli.py    # chime-scan CLI wrapper
README.md               # canonical setup and corrected selection examples
docs/CANFAR_RUNBOOK.md  # CANFAR operating procedure for chime-scan
scripts/setup_env.sh    # one-shot setup and preflight script
```

## Compatibility note: datatrawl inventory metadata

Current `datatrawl survey` writes an inventory sidecar named
`inventory.meta.json`. That sidecar records the telescope, source, and the
telescope's canonical reader. For CHIME, the canonical reader is
`chime-baseband`.

That default is **not** correct for `pilot-proxy-detector`.
The detector analyzer needs PilotProxy's `chime-baseband-packed` reader so native
CHIME offset-binary 4+4-bit samples can be repacked losslessly for the CUDA
kernel.

The recommended entry point remains:

```bash
pilot-proxy chime-scan ...
```

`chime-scan` chooses the correct reader internally:

| analyzer | reader used by `pilot-proxy chime-scan` |
|---|---|
| `pilot-proxy-detector` | `chime-baseband-packed` |

If driving raw datatrawl directly, do not rely on inventory-metadata reader
inference for the detector. Override the reader explicitly:

```bash
# detector: override the inferred canonical reader
datatrawl scan \
  --name chime-pilots \
  --reader chime-baseband-packed \
  --analyzer pilot-proxy-detector \
  --select 844
```

Use `--inventory data/chime-pilots/inventory.jsonl` instead of
`--name chime-pilots` when scanning from a different working directory or when
the inventory is not under the current `data/` tree.

The PilotProxy analyzers now also include dtype guards so a wrong reader pairing fails
with an actionable error rather than producing invalid products.
