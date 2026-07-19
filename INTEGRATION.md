# pilot-proxy <-> datatrawl integration

CHIME archive runs require bounded staging, restartable products, and one
detector product per selected pilot. We use `datatrawl` for that data movement
while keeping the DTV analysis in `pilot-proxy`.

The design rule is:

```text
shared streaming engine, private science analyzers
```

Under this rule, `datatrawl` enumerates files, bounds the staging area, streams
arrays, requests checkpoints, and fans the run out by channel. `pilot-proxy`
provides the packed CHIME reader, DTV analyzer, CUDA detector call, and canonical
product writers. A conforming analyzer remains responsible for implementing its
checkpoint and resume contract; the supplied PilotProxy analyzer does so in its
per-pilot product.

---

## How it fits together

| datatrawl axis | pilot-proxy integration |
|---|---|
| instrument | datatrawl's bundled `chime` target geometry: 800 MHz band top, 1024 coarse channels, `nfft=16384`, inverted spectral sense |
| source | datatrawl's `local` and `cadc-datatrail` sources |
| reader | `chime-baseband-packed` for detector runs |
| analyzer | `pilot-proxy-detector` |

We keep one implementation of the detector path:

- `pilot-proxy-detector` calls `pack_chime_block_for_detector` and then the
  detector. The operational path uses the CUDA kernel by default.

The analyzer writes one `<freq_id>.npz` product for each selected pilot with
usable input. When the products share `(event, frame-in-file)` identities, the
combine step aligns the common identities, stacks the per-pilot arrays, and
writes the canonical PilotProxy outputs:

- `chime_detector_outputs.npz`
- `chime_spectrogram_cache.npz`
- `chime_reductions_10s.npz`
- CSV summaries under `tables/`
- JSON provenance under `run_config.json`, `stats.json`, and `input_manifest.json`

If the intersection across all completed pilots is empty, `chime-scan` does not
write a misleading empty stack. It leaves the completed per-pilot products under
`_per_pilot/` and reports how to choose a compatible subset with
`pilot-proxy chime-combine --report` and `--drop`.

The `nfft=16384` value is the target frame size for the CHIME Engine upgrade.
It is not a statement that the currently deployed CHIME frame has this size.
Both the datatrawl geometry and `chime_dtv_fengine.json` currently encode this
target, and the receiver profile is marked
`example_requires_data_product_verification`. Verify the frame size and the
remaining receiver metadata against the data product used for each operational
run.

---

## Setup

The integrated workflow uses `scripts/setup_env.sh`. The script recreates a
virtual environment, installs both repositories in editable mode, checks plugin
discovery, and runs the offline datatrawl integration tests. On a GPU node, it
requires `nvcc`, resolves CuPy, builds the CUDA kernel, and verifies that the
kernel loads. The exact invocation and the manual fallback are in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#environment-setup).

The operational detector has no CPU fallback in `chime-scan`. Run it on a GPU
node with a CuPy build that matches the CUDA runtime. The CPU detector injections
used in the test suite are fixtures for parity and orchestration tests.

---

## Selection: CHIME `freq_id`, not ATSC channel

`pilot-proxy chime-scan --select` accepts CHIME coarse-channel identifiers,
called `freq_id`s. It does not accept ATSC physical-channel numbers. For
`--source cadc-datatrail`, `--select` is optional. When it is omitted, we read
the distinct `freq_id`s from the inventory and print the resolved set before
staging any file. Either form plans one product per selected `freq_id`; a
selection with no matching file is skipped. A local source has no inventory from
which to infer this scope, so `--select` remains required.

A CHIME baseband file contains one coarse channel. Both the local source and the
CADC inventory identify that channel by `freq_id`. Archive-style filenames use:

```text
baseband_<event>_<freq_id>.h5
```

For the default DTV physical-channel range 14-36, use the 23 pilot `freq_id`s in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md#selection-convention).

The range `396-412` does not represent the ATSC 14-36 pilots in the CHIME
400-800 MHz geometry and should not be used for this workflow.

For a single-channel smoke test, select a `freq_id` present in the inventory or
local filenames. Under the bundled CHIME geometry, the coarse-channel center
used for the ATSC physical-channel 14 pilot is 470.3125 MHz, which maps to
`freq_id=844`. This mapping is conditional on that configured geometry.

---

## Local source workflow

The local source must recover a `freq_id` from each filename. Its default regular
expression is:

```text
_(\d+)\.h5$
```

Therefore, filenames ending in an underscore, an integer, and `.h5` match:

```text
baseband_<event>_844.h5
baseband_<event>_829.h5
```

The `chime-scan` CLI exposes `--source-channel-regex`, but the current adapter
stores that value as `source_channel_regex` while the paired datatrawl local
source reads `source_freq_id_regex`. Therefore, this flag does not currently
override the datatrawl regular expression. Until the adapter alias is corrected,
either use the default filename suffix or pass the datatrawl option through the
context explicitly:

```bash
pilot-proxy chime-scan ... --set 'source_freq_id_regex=<regex-with-one-capturing-group>'
```

The regular expression must contain one capture group for the integer
`freq_id`.

Inspect the identifiers present in a local directory before selecting one:

```bash
find "$LOCAL_H5" -maxdepth 1 -name "*.h5"   | sed -E 's/.*_([0-9]+)\.h5$/\1/'   | sort -n | uniq
```

Then run one file and one chunk as a bounded GPU smoke test:

```bash
pilot-proxy chime-scan   --input-dir "$LOCAL_H5"   --output-dir "$HOME/pilot_proxy_runs/detector_smoke_844"   --source local   --analyzer pilot-proxy-detector   --select 844   --max-files 1   --max-chunks-per-file 1
```

---

## CADC / CANFAR workflow

The archive path begins with a bounded inventory. The following request limits
the inventory to five events:

```bash
cadc-get-cert -u <your-cadc-username>

datatrawl survey \
  --telescope chime \
  --source cadc-datatrail \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --name chime-pilots \
  --max-events 5
```

The survey writes:

```text
data/chime-pilots/inventory.jsonl
```

Inspect the inventory before downloading any baseband file:

```bash
datatrawl explore \
  --source cadc-datatrail \
  --telescope chime \
  --inventory data/chime-pilots/inventory.jsonl
```

`--inventory-name` resolves relative to a survey root. Run
`pilot-proxy chime-scan` from the directory in which `datatrawl survey` created
`data/chime-pilots/`, or pass `--source-root <survey-root>` explicitly.

First run one selected `freq_id`, one file, and one chunk:

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

After the bounded run succeeds, the following command scans every `freq_id`
present in the inventory. `--select` defaults to that set, `--source` is inferred
from `--inventory-name`, and `pilot-proxy-detector` is the default analyzer:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --inventory-name chime-pilots
```

To scan only part of the inventory, pass the subset explicitly:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilot_proxy_runs/chime-pilots" \
  --inventory-name chime-pilots \
  --select 660,675,690,752
```

---

## Post-processing

Validate the combined detector products before plotting them:

```bash
pilot-proxy validate-products --run-dir <detector_run>
pilot-proxy chime-plot --run-dir <detector_run>
```

---

## Order-safety constraint

The PilotProxy analyzer appends frames in the order files are delivered. When
more than one file can download or remain staged, datatrawl may deliver files in
completion order rather than source order. That would change `frame_index` and
`relative_time_s`. Therefore, `pilot-proxy chime-scan` overrides caller values
and forces `download_workers=1` and `max_staged_files=1`.

A raw `datatrawl scan` does not apply this PilotProxy-specific override. Use the
same single-worker, single-staged-file constraint, or make the analyzer
order-insensitive before enabling concurrency.

---

## Current verification status

The offline `tests/datatrawl/` suite checks the integration without GPU or CADC
access. Its parity tests inject a CPU detector fixture and compare synthetic
`chime-scan` products against products from the earlier PilotProxy runner. These
tests cover array layout, orchestration, resume behavior, product combination,
selection, and metadata inference for the tested fixtures.

They do not replace the following operational CANFAR/GPU checks:

- end-to-end execution of the CUDA detector on a GPU node;
- streaming through the CADC `cadc-datatrail` source;
- real-data parity against a separately validated CHIME run.

---

## File manifest

The integration is implemented in these files:

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

It also changes these existing interfaces:

```text
pyproject.toml          # datatrawl optional extra and plugin entry points
src/pilot_proxy/cli.py    # chime-scan CLI wrapper
README.md               # canonical setup and corrected selection examples
docs/CANFAR_RUNBOOK.md  # CANFAR operating procedure for chime-scan
scripts/setup_env.sh    # one-shot setup and preflight script
```

## Compatibility note: datatrawl inventory metadata

`datatrawl survey` writes `inventory.meta.json` beside the inventory. The
sidecar records the telescope, source, and canonical reader. For CHIME, that
canonical reader is `chime-baseband`.

However, `chime-baseband` unpacks samples to complex64, while the detector needs
the native packed bytes. `pilot-proxy-detector` therefore uses
`chime-baseband-packed`, which converts the native CHIME offset-binary 4+4-bit
samples to the kernel's two's-complement int4 layout without a float
requantization step.

Use the supported wrapper:

```bash
pilot-proxy chime-scan ...
```

`chime-scan` selects the reader from the analyzer:

| analyzer | reader used by `pilot-proxy chime-scan` |
|---|---|
| `pilot-proxy-detector` | `chime-baseband-packed` |

When invoking datatrawl directly, the inventory metadata would select the wrong
reader for this analyzer. Override it explicitly:

```bash
# detector: override the inferred canonical reader
datatrawl scan \
  --name chime-pilots \
  --reader chime-baseband-packed \
  --analyzer pilot-proxy-detector \
  --select 844
```

`--name chime-pilots` resolves under the current `data/` tree. From another
working directory, or for an inventory stored elsewhere, use
`--inventory data/chime-pilots/inventory.jsonl` instead.

The analyzer also checks the input dtype. A reader that produces complex64
therefore stops with an error before the detector writes a product.
