# PilotProxy

`pilot-proxy` is a standalone CUDA F-statistic detector and GNU Radio ATSC 1.0
validation testbench for estimating and detecting sub-noise-floor ATSC 1.0 DTV
signals by using the ATSC pilot tone as a proxy for the data shelf.

The package has two supported operating modes:

1. **Standalone synthetic/testbench mode** for clean ATSC generation, quantization,
   CUDA-kernel evaluation, and controlled SNR sweeps.
2. **CHIME real-data mode** for baseband HDF5 data. The recommended archive-scale
   entry point is `pilot-proxy chime-scan`, which runs the PilotProxy analyzers through
   the `datatrawl` streaming engine and then combines the per-pilot products into
   PilotProxy's canonical CHIME outputs.

The standalone detector core remains telescope-independent. Receiver or telescope
integrations provide metadata and arrays; they do not change the CUDA kernel
contract.

---

## Choose your workflow

`pilot-proxy` supports several independent paths. Pick the one that matches your
goal before diving into the specialized sections below.

| Goal | Use this path | Needs GPU? | Needs datatrawl? |
| --- | --- | :---: | :---: |
| Check the package installs and the CLI loads | Minimal CPU-only smoke test (below) | No | No |
| Run CHIME offset diagnostics | `pilot-proxy chime-scan --analyzer pilot-proxy-offset` | No | Yes |
| Run the CHIME detector | `pilot-proxy chime-scan --analyzer pilot-proxy-detector` | Yes | Yes |
| Generate / audit synthetic ATSC | Standalone testbench | Partly | No |
| Run the CUDA detector / SNR evaluation | Standalone CUDA path | Yes | No |

---

## Fresh clone and minimal CPU-only smoke test

This verifies the Python package, the CLI entry point, and the reference detector
metadata **without** GNU Radio, datatrawl, CADC credentials, CUDA, or CHIME HDF5
data --- a good first step before any of the specialized workflows.

```bash
git clone https://github.com/WVURAIL/pilot-proxy.git ~/pilot-proxy
cd ~/pilot-proxy
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[test]"

pilot-proxy --help
python -m pytest tests/test_cli.py -q
pilot-proxy check-profile \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
pilot-proxy check-layout \
    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
```

> **Test commands.** `make test` is *not* a CPU-only check --- it runs
> `test-kernel` (which needs `nvcc`/CUDA) alongside the Python tests. On a CPU
> host run the Python tests directly (`make test-python`, or
> `pytest tests/test_cli.py -q` for just the CLI). `make release-check` adds the
> CPU C/C++ reference checks plus profile/layout and runtime-bundle validation
> (needs a C++ compiler, not a GPU). Use `make test-kernel SM=<arch>` only on a
> CUDA build host (`SM` is your GPU's compute capability, e.g. `89`).

---

## Contents

- `cuda/` - CUDA kernel, public C header, CPU C++ reference, and C++ tests.
- `src/pilot_proxy/` - Python package for kernel loading, detector geometry,
  reference channelization, DTV unit conversion, CHIME adapters, and testbench
  workflows.
- `src/pilot_proxy/testbench/` - GNU Radio ATSC generation, waveform audit,
  AWGN generation, quantization, and SNR evaluation.
- `src/pilot_proxy/datatrawl_plugins/` - PilotProxy readers/analyzers used by
  `datatrawl` and by `pilot-proxy chime-scan`.
- `weights/` - prebuilt ATSC reference detector weights for
  `detector_window_samples=128`, `num_weight_terms=3`,
  `skipped_guard_bins=1`, `reference_offset_bins=2`, physical channels 14-36,
  and 4+4 bit samples.
- `configs/` - detector-core, receiver-profile, and stream-map JSON examples for
  standalone and integration workflows.
- `scripts/setup_env.sh` - one-shot CANFAR / datatrawl setup script for the
  integrated CHIME workflow.
- `scripts/launch_gpu_session.py` - launch, reuse, or tear down a CANFAR CUDA GPU
  notebook session (skaha / `canfar` client) for the detector path.
- `INTEGRATION.md` - detailed `pilot-proxy` <-> `datatrawl` integration notes.
- `docs/METHOD_SPEC.md` - equation-first method contract for CHIME products.
- `docs/product_schema_v2.md` - per-pilot detector product schema (v2).
- `docs/CHIME_REAL_DATA_V0_2.md` - CHIME real-data adapter notes (v0.2).
- `docs/DATA_PRODUCTS.md` - emitted file, array, and table definitions.
- `docs/CANFAR_RUNBOOK.md` - bounded CANFAR operating procedure.
- `docs/KOTEKAN_INTERFACE_PREP.md` - runtime-bundle and Kotekan handoff notes.
- `docs/DESIGN_DECISIONS.md` - recorded detector and integration decisions.
- `docs/PilotProxy_DS001_v1_3_Data_Sheet.tex` - formal data sheet (build to PDF).
- `docs/PilotProxy_UG001_v1_3_User_Guide.tex` - formal user guide (build to PDF).
- `examples/quickstart.sh` - standalone release sanity-check workflow (CUDA +
  GNU Radio; environment-specific defaults --- override `SM`, `CUDA_PYTHON`,
  `GNURADIO_PYTHON`).

Documentation is committed as source. Generated PDFs, figures, products, and CUDA
shared libraries are build artifacts and should not be committed.

Built wheels include the shipped receiver profiles, stream map, weight bank, and weight
manifest. The CUDA shared library remains architecture-specific: build it from a source
checkout and stage it under ``~/.cache/pilot_proxy/libfstatistic.so`` before running the
GPU detector.

---

## Setup for the CHIME / CANFAR workflow

The integrated workflow needs both repositories checked out. The setup script
creates a clean virtual environment, installs both repos editable with the
CADC/survey and CHIME extras, resolves CuPy through datatrawl's `accel` module,
builds the CUDA kernel when `nvcc` is available, and verifies that the fstat
datatrawl plugins are discoverable:

```bash
git clone https://github.com/WVURAIL/datatrawl.git ~/datatrawl
git clone https://github.com/WVURAIL/pilot-proxy.git ~/pilot-proxy
cd ~/pilot-proxy

VENV_DIR=~/pilot-proxy-datatrawl DATATRAWL_DIR=~/datatrawl PILOT_PROXY_DIR=~/pilot-proxy bash scripts/setup_env.sh

source ~/pilot-proxy-datatrawl/bin/activate
```

The script **removes and recreates** the target venv (`python -m venv --clear`),
so do not point `VENV_DIR` at a venv you need to keep. On hosts without
`nvidia-smi` it skips the kernel build; those hosts can run the CPU-only
`pilot-proxy-offset` analyzer but not the GPU `pilot-proxy-detector`.

Manual fallback, when you do not want the script to recreate the venv:

```bash
python3.12 -m venv ~/pilot-proxy-datatrawl
source ~/pilot-proxy-datatrawl/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e "$HOME/datatrawl[cadc,survey]"
python -m pip install -e "$HOME/pilot-proxy[cuda,datatrawl,chime,test]"   # drop the cuda extra on CPU-only hosts
```

The complete CANFAR procedure --- GPU-session launch, Harbor registry
credentials, required inputs, and the bounded run sequences --- is in
[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md). Integration-specific setup
notes are in [INTEGRATION.md](INTEGRATION.md#setup).

---

## Receiver integration contract

Receiver integrations provide:

- `receiver_profile.json` describing RF band, channelizer geometry, spectral
  sense, frame size, input streams, quantization policy, bin ENBW, and pilot
  capture efficiency;
- optional `stream_map.json` describing the input-stream ordering;
- channelized complex input arrays or packed detector matrices;
- a generated weight bank built from the receiver profile.

The CUDA kernel sees only:

- packed int4 detector rows;
- packed int4 weights;
- uint64 target/reference powers.

CHIME real-data run products include a stable detector contract in
`run_config.json` and `stats.json` with:

- `schema_version = pilotproxy_chime_detector_contract_v1`;
- K, weight-term, skipped-guard, and reference-offset geometry;
- packed input and uint64 accumulator metadata;
- the all-row summation rule;
- the positive-excess mask policy;
- reference-placement summary metadata.

Validate integration metadata with:

```bash
pilot-proxy check-profile   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json

pilot-proxy check-layout   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --stream-map configs/stream_maps/chime_feed_pol_example.json
```

The shipped receiver profiles have different roles:

- `reference_800mhz_pfb.json` is the single-stream detector-coordinate reference
  profile used for shipped reference weights and tests.
- `chime_dtv_fengine.json` is the CHIME DTV real-data adapter profile: 2048
  feed-polarization streams, inverted spectral sense, and descending RF channel
  order.

---

## Weight bank

Generate the default CHIME DTV weight bank with:

```bash
pilot-proxy make-weights   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --detector-core-profile configs/detector_core/pilotproxy_cuda_fstat_v1.json   --physical-channel-range 14:36   --weight-coordinate-system post_spectral_sense_normalization   --output weights/chime_dtv_weights_k128.bin
```

The default detector path expects:

```text
weights/chime_dtv_weights_k128.bin
```

Export and validate a compact runtime bundle with:

```bash
pilot-proxy export-runtime-weight-bundle   --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json   --detector-core-profile configs/detector_core/pilotproxy_cuda_fstat_v1.json   --weight-coordinate-system post_spectral_sense_normalization   --physical-channel-range 14:36   --output-dir generated/deploy/chime_dtv_k128

pilot-proxy validate-runtime-weight-bundle   --bundle-dir generated/deploy/chime_dtv_k128
```

---

## CUDA kernel

On a GPU host:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Convert compute capability to `SM` by removing the decimal point. Examples:

| Compute capability | `SM` |
|---:|---:|
| 8.0 | 80 |
| 8.6 | 86 |
| 8.9 | 89 |
| 9.0 | 90 |

Build and stage the kernel:

```bash
make build-kernel SM=89
```

This builds `cuda/libfstatistic.so` and stages a copy to:

```text
~/.cache/pilot_proxy/libfstatistic.so
```

Validate by loading through Python:

```bash
PYTHONPATH=src python - <<'PY'
from pilot_proxy.kernel import FStatKernel
kernel = FStatKernel()
print(kernel.specs.as_descriptive_dict())
print(kernel.features.as_dict())
print(kernel.version.as_string())
PY
```

Do not execute the shared library directly.

Run compiled CUDA/C++ regression tests with:

```bash
make test-kernel SM=89
```

---

## CHIME / datatrawl archive workflow

The archive-scale entry point is `pilot-proxy chime-scan`, which runs the fstat
analyzers through `datatrawl` and combines the per-pilot products into fstat's
canonical CHIME outputs. Two things to know before running:

- **Selection is in the CHIME `freq_id` coarse-channel namespace**, not ATSC
  physical-channel numbers. The default ATSC 14-36 pilot set is
  `506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844`;
  `844` is the single-channel smoke-test default (the ATSC 14 pilot).
- **Use `chime-scan`, not raw `datatrawl scan`.** The PilotProxy analyzers are
  order-sensitive, and `chime-scan` forces the single-staged-file path that keeps
  frames time-aligned.

A bounded local offset smoke test, for orientation:

```bash
pilot-proxy chime-scan   --input-dir "$LOCAL_H5"   --output-dir "$HOME/pilot_proxy_runs/offset_smoke_844"   --source local   --analyzer pilot-proxy-offset   --select 844   --max-files 1   --max-chunks-per-file 1
```

For the full reference --- selection details, the local and CADC/CANFAR run
sequences, order-safety internals, and post-processing (`validate-products`,
`chime-plot`, `choose-detector-k`) --- see:

- **[INTEGRATION.md](INTEGRATION.md)** --- the datatrawl integration contract.
- **[docs/CANFAR_RUNBOOK.md](docs/CANFAR_RUNBOOK.md)** --- the step-by-step CANFAR
  operating procedure.

---

## Standalone synthetic/testbench workflow

The standalone path is unchanged. It still uses two Python interpreters when GNU
Radio is installed in the system Python:

- GNU Radio Python for clean ATSC generation and GNU Radio AWGN;
- CUDA Python for CuPy and the CUDA F-statistic library.

Generate a clean ATSC signal:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3   -m pilot_proxy.testbench.generate_atsc_signal   --output-iq generated/atsc/atsc_8vsb_complex64.cfile   --num-iq-samples 600000
```

Audit the clean waveform:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.audit_atsc_signal   --input-iq generated/atsc/atsc_8vsb_complex64.cfile
```

Pack detector input:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.quantize   --input-iq generated/atsc/atsc_8vsb_complex64.cfile   --physical-channel 14   --frame-size-samples 16384   --num-input-streams 1
```

Run a small SNR evaluation:

```bash
PYTHONPATH=src python   -m pilot_proxy.testbench.evaluate_snr   --input-iq generated/atsc/atsc_8vsb_complex64.cfile   --physical-channel 14   --frame-size-samples 16384   --num-input-streams 1   --requested-snr-shelf-db -26   --noise-trials 10
```

---

## Build documentation

Generated PDFs are ignored by git. Build locally when needed:

```bash
mkdir -p docs/out
(cd docs && latexmk -g -pdf -interaction=nonstopmode -halt-on-error   -outdir=out PilotProxy_DS001_v1_3_Data_Sheet.tex)
(cd docs && latexmk -g -pdf -interaction=nonstopmode -halt-on-error   -outdir=out PilotProxy_UG001_v1_3_User_Guide.tex)
```

---

## Commit hygiene

Before committing, remove generated products and local build artifacts:

```bash
make release-clean
make commit-check
```

## Compatibility note: datatrawl inventory metadata

`pilot-proxy chime-scan` selects the correct reader per analyzer automatically
(`chime-baseband` for `pilot-proxy-offset`, `chime-baseband-packed` for
`pilot-proxy-detector`), so the inventory-metadata default needs no attention on the
recommended path. The full explanation --- including how to override the reader
when driving raw `datatrawl scan` directly --- is in
[INTEGRATION.md](INTEGRATION.md#compatibility-note-datatrawl-inventory-metadata).
