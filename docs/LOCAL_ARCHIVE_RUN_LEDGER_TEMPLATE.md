# Local archive run ledger

Copy this file beside the run output. Complete every before-launch field before
processing. Fill the tripwire, renewal, and completion sections as the run
progresses. Do not record credentials or other secrets.

## Before launch: status

- Status: planned
- Run purpose: Historical estimation and sufficient-statistic reprocessing. The coarse positive-excess flag is retained as a bootstrap diagnostic; the fine rank/eta decision is inactive and no calibrated detection policy is applied.
- UTC start:
- Operator:
- Output directory:
- Staging directory:
- Complete command:

## Before launch: frozen source

- Source revision:
- Source branch:
- Source clean: yes / no
- Local revision equals pushed revision: yes / no
- Package-source SHA-256:
- Python environment: `/home/djg/rail/venvs/archive-local`

## Before launch: host and runtime

- Execution host: local WSL workstation
- Host CPU:
- WSL virtual CPUs:
- WSL memory and swap:
- GPU: RTX 5000 Ada, 16 GiB
- GPU compute capability: 8.9
- Free WSL ext4 space:
- Python version:
- NumPy version:
- CuPy version:
- Driver version:
- Toolkit version:

## Before launch: final-source gates

- Full test result:
- CPU kernel-reference result:
- SM89 CUDA test result:
- Kernel test result and skip count:
- One-file smoke evidence path:
- One-file embedded source SHA-256:
- One-file embedded kernel SHA-256:
- One-file input shape and dtype:
- One-file peak VRAM:
- One-file product validation:
- Four-worker/eight-slot resume evidence path:
- Resume interruption point:
- Resume embedded source SHA-256:
- Resume embedded kernel SHA-256:
- Resume unique units and frames:
- Resume product validation:
- Resume staging empty: yes / no

## Before launch: archive scope

- Instrument: `chime`
- Source: `cadc-datatrail`
- Reader: `chime-baseband-packed`
- Analyzer: `pilot-proxy-detector`
- Inventory path:
- Inventory SHA-256: `b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8`
- Source inventory path:
- Source inventory SHA-256: `0ecadee63791328ab3c71dd88c668d3df595bd778229e0e5f3980431943175e5`
- Exclusion ledger path:
- Exclusion ledger SHA-256: `d0ab78db9dec847fb5270fb94e3fc45efd6bbbc71adc376830fb71467693079a`
- Pending-event resolution path:
- Pending-event resolution SHA-256: `077915779d234ad9cca1e9b52171e28025a60474f20fad7aff5684f55ce0c4c7`
- Pending-event resolution: 3 events / 69 selected objects absent / 0 errors
- Inventory manifest path:
- Inventory manifest SHA-256: `5695e1cc9c007cb2c79ad39535cf9ed2fb20848ad00d3cf0215933670d0707e5`
- Source / excluded / frozen units: 170377 / 4695 / 165682
- Source / frozen events: 9214 / 8983
- Sparse coverage: `598: 1543 units through 2023-09-13; 690: 1767 units through 2026-04-16`
- Selected `freq_id` values: `506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844`
- File cap: none
- Chunk cap: none
- Partial-run acknowledgement: off
- Download workers: 4
- Maximum staged files: 8
- Checkpoint interval: 250 units
- Certificate expiry:
- Certificate renewal and restart plan:

## Before launch: detector artifacts

- Kernel core: 2.3.0
- Kernel architecture: SM89
- Preserved kernel path: `/home/djg/rail/pilot-proxy/cuda/libfstatistic-2.3.0-sm89-e48ffa59bb592be8.so`
- Preserved kernel SHA-256: `e48ffa59bb592be839218dfb6f920c8f9e9653b10abab97e856372cdcfa3bc8b`
- Receiver profile path: `configs/receiver_profiles/chime_dtv_fengine.json`
- Receiver profile file SHA-256: `bc59e77442a4c15f74c716d14eaeea4f10a69517d3bfb8c88ce10a7a42ea1e15`
- Receiver profile canonical SHA-256: `da047cdf453764e4b8c01514602034456e32a39256e8c5980f2c052554500e45`
- Weight bank path: `weights/chime_dtv_weights_k128.bin`
- Weight bank SHA-256: `1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259`
- Weight manifest path: `weights/chime_dtv_weights_k128.bin.manifest.json`
- Weight manifest SHA-256: `d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7`

## Before launch: expected and resolved geometry

- Expected `nfft`: 16384
- Resolved `nfft`:
- Expected frame time: 0.04194304 seconds
- Resolved frame time:
- Expected input streams: 2048
- Resolved input streams:
- Expected detector window: 128 samples
- Resolved detector window:
- Expected exact coarse terms: target, lower reference, upper reference, reference sum
- Resolved exact coarse terms:
- Expected fine terms: `[frames, 3, 256]` unsigned 64-bit integers
- Resolved fine terms:
- Expected fine status: enabled
- Resolved fine status:
- Expected fine decision: inactive
- Resolved fine decision:

## Before launch: decisions

- Terminal product: all 23 per-pilot v5 products
- Combined products: derived common-frame projections only
- Coarse positive-excess flag: recorded bootstrap diagnostic
- Fine decision: inactive
- Fine rank: unset
- Eta/Q16 multiplier: unset
- `fine_products=on`: retains sufficient statistics only
- Evaluation status: historical reprocessing, unblinded
- Holdout claim: none
- Future validation: future frozen epoch

## During run: first-checkpoint tripwire

- First `freq_id`:
- Checkpoint unit count: 250
- Product schema: `pilotproxy_per_pilot_product_v5`
- Embedded package-source SHA-256:
- Embedded kernel SHA-256:
- Fine shape and dtype:
- Fine status:
- Fine decision inactive: yes / no
- Download workers / staged slots / checkpoint: 4 / 8 / 250
- Per-pilot contract result:
- Per-pilot audit result:
- Stored designated bins:
- Stored census exclusions:

## During run: certificate renewals

For every renewal, record the checkpoint, clean stop, new expiry, and exact
unchanged resume command.

- Renewal 1:
- Renewal 2:
- Renewal 3:

## After run: completion

- Quarantine entries and dispositions:
- Final per-pilot audits:
- Canonical validation: pass / not applicable because combine was skipped
- Final inventory accounting:
- Final product manifest and SHA-256:
- UTC accepted:
