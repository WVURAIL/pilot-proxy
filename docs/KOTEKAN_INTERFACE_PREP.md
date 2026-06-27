# Kotekan Interface Preparation

This note records the expected runtime interface shape. It does not define or
implement a Kotekan stage.

## 1. Assumptions

- The same detector software can run on every node.
- The first frame provides an integer CHIME channel identifier.
- Non-pilot channels disable the detector.
- Pilot channels select exactly one runtime profile from a packaged bundle.
- The detector uses packed complex int4 input and packed int4 weights.
- The current Python analysis path computes the positive-excess mask from exact
  uint64 powers.

## 2. Runtime State Machine

```text
INIT
  -> WAIT_FOR_FIRST_FRAME
  -> DISABLED
  -> RUNNING
```

`WAIT_FOR_FIRST_FRAME` reads the first-frame channel identifier. If the channel
identifier is not present in `pilot_profiles.json`, the node enters
`DISABLED`. If it matches a pilot profile, the node enters `RUNNING` with the
selected weight-bank pointer and detector contract.

## 3. Runtime Bundle Format

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

`detector_contract.json` is the methods-level contract. `pilot_profiles.json`
maps physical DTV pilots and future CHIME channel IDs to byte offsets in
`weights.bin`. `weights.bin` is a compact concatenation of selected weight
profiles. `weights.manifest.json` records layout and provenance.

The runtime bundle is a host-side memory bank. A Kotekan stage may either upload
the entire bank to GPU memory at startup and select a device pointer by byte
offset, or copy only the selected profile to device memory after first-frame
channel selection. The CUDA kernel must receive a device pointer, not a host
pointer.

## 4. Weight Coordinate Convention

The CUDA kernel receives detector-coordinate windows. For an input channel with
inverted spectral sense, a runtime stage must choose exactly one convention:

- Time-reverse each detector window before the kernel and use
  `post_spectral_sense_normalization` weights.
- Skip time reversal and use `raw_input_frequency_coordinate` weights.

The bundle must declare `weight_coordinate_system` in `detector_contract.json`,
`pilot_profiles.json`, and `weights.manifest.json`. These files must agree. The
contract also records:

```json
{
  "input_coordinate_system": "post_spectral_sense_normalized",
  "input_preprocessing": {
    "time_reverse_detector_windows_before_kernel": true
  }
}
```

The current Python CHIME/CANFAR path uses
`post_spectral_sense_normalization` weights. Raw input-coordinate weights are a
future deployment option and must not be mixed with detector-window time
reversal.

## 5. Kernel ABI

- Selected weight-bank pointer.
- Packed complex int4 detector rows.
- Packed complex int4 target/lower-reference/upper-reference weights.
- Production valid/mask output.
- Optional uint64 target and reference power outputs for debug builds, audits,
  and product validation.

The current Python analysis path computes the positive-excess mask on the host
from exact uint64 powers. A production Kotekan stage may instead compute and
emit the valid/mask decision on the device, while optionally exposing powers in
debug builds.

## 6. Open Questions

- Exact Kotekan metadata field for the integer CHIME channel ID.
- Whether channel ID can change during a run.
- Whether non-pilot nodes should emit empty mask frames.
- Which frame/alignment key should identify emitted mask frames.
