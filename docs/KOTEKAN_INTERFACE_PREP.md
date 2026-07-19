# Kotekan Interface Preparation

This note separates the tested PilotProxy runtime bundle from the proposed
Kotekan integration. The repository exports and validates the bundle described
below, and the CUDA library exposes a device mask API. It does not contain a
Kotekan stage. The state machine, metadata binding, and frame alignment therefore
remain interface requirements rather than implemented behavior.

## 1. Assumptions

The proposed stage uses these assumptions:

- We deploy the same detector code on each participating node.
- The Kotekan frame metadata provides an integer CHIME channel identifier.
- We bind the node to a channel profile after reading its first frame.
- A channel absent from the runtime bundle disables pilot detection on that
  node.
- A matching pilot channel selects exactly one weight profile.
- Detector samples and weights use packed complex int4 values.
- The detector produces one binary rejection decision for each aligned detector
  frame.

Only the packed detector, runtime bundle, and bundle validator are tested in
this repository. In particular, the exact Kotekan channel field and its
stability during a run have not been verified.

The Python CHIME path currently reads exact `uint64` target and reference powers
back to the host and forms the norm-corrected positive-excess decision there.
The CUDA library also exposes the rational-half-threshold mask API needed to
form the same comparison on the device.

## 2. Runtime State Machine

The first accepted frame determines whether this detector instance runs:

```text
INIT
  -> WAIT_FOR_FIRST_FRAME
       -> DISABLED   (channel identifier has no bundle profile)
       -> RUNNING    (channel identifier selects one bundle profile)
```

In `WAIT_FOR_FIRST_FRAME`, the stage reads the integer channel identifier and
looks it up in `pilot_profiles.json`. A miss enters `DISABLED`. A match enters
`RUNNING` with the selected byte offset, weight-profile pointer, and detector
contract.

This state machine assumes that the identifier does not change. If Kotekan can
change it within one stage lifetime, the integration must define whether to
reject that transition, drain and reinitialize, or select a new profile at a
frame boundary.

## 3. Runtime Bundle Format

PilotProxy exports these five files:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

`detector_contract.json` records the detector geometry, coordinate convention,
mask rule, and input preprocessing. `weights.bin` concatenates one int8-packed
`(3, K)` weight profile per selected physical channel.
`weights.manifest.json` records the profile shape, source geometry, selected
channels, coordinate convention, reference placement, and hashes.

`pilot_profiles.json` maps each physical DTV channel to a profile index, byte
offset, byte count, calibration fields, and positive-excess rational threshold.
The current exporter writes `chime_channel_id = null`; a CHIME/Kotekan metadata
mapping must populate and validate that field before first-frame selection can
be implemented. Thus, the current bundle supports physical-channel identity but
does not yet constitute a live Kotekan channel map.

Export and validate a candidate CHIME bundle with:

```bash
pilot-proxy export-runtime-weight-bundle \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --detector-core-profile configs/detector_core/pilotproxy_cuda_fstat_v1.json \
  --weight-coordinate-system post_spectral_sense_normalization \
  --physical-channel-range 14:36 \
  --output-dir generated/runtime_bundle

pilot-proxy validate-runtime-weight-bundle \
  --bundle-dir generated/runtime_bundle
```

The validator checks file hashes, profile bounds and alignment, channel
uniqueness, detector-contract hashes, weight hashes, and coordinate consistency.
These checks validate the exported files; they do not validate the future
Kotekan metadata binding.

At startup, a stage can upload the complete `weights.bin` bank and select a
device offset, or copy only the selected profile after channel identification.
In either design, the CUDA kernel receives a device pointer. A host pointer is
not a valid substitute.

## 4. Weight Coordinate Convention

The CUDA kernel consumes detector-coordinate windows. For inverted-spectrum
CHIME input, the stage must choose one complete convention:

- Reverse each detector window and use
  `post_spectral_sense_normalization` weights.
- Preserve the raw time order and use `raw_input_frequency_coordinate`
  weights.

The three bundle metadata files must agree on `weight_coordinate_system`,
`input_coordinate_system`, and the preprocessing flag. For the current Python
CHIME path, the contract is:

```json
{
  "input_coordinate_system": "post_spectral_sense_normalized",
  "input_preprocessing": {
    "time_reverse_detector_windows_before_kernel": true
  }
}
```

Therefore the current path uses `post_spectral_sense_normalization` weights.
Raw input-coordinate weights remain a supported bundle-generation option for a
future stage, but they cannot be combined with detector-window reversal.

## 5. Kernel ABI

The proposed stage must bind the following CUDA inputs:

- packed complex int4 detector rows;
- a device pointer to the selected packed int4 target, lower-reference, and
  upper-reference weights;
- the rational half-threshold numerator and denominator stored in the selected
  profile;
- device storage for the target power, summed reference power, and rejection
  mask.

For the norm-corrected positive-excess rule, the bundle stores:

```text
positive_excess_half_threshold_num = target_norm_sq
positive_excess_half_threshold_den = ref_norm_sum_sq
```

The deployed CUDA comparison is:

```text
mask = (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

When `p_ref_sum == 0`, the CUDA API forces `mask = 0`. A consumer can therefore
derive `valid` as `p_ref_sum != 0`. Production may retain only the aligned mask
after this decision, while debug and validation builds should expose the exact
`uint64` powers and overflow counter long enough to compare the Kotekan path
with the Python reference.

The 16,384-sample framing used in the software profile is associated with the
planned CHIME engine upgrade. It should remain a configurable and recorded
interface value until the upgrade's Kotekan frame contract is published. A
12,288-sample current-frame value has been recalled but is still provisional;
the integration must obtain the active value from authoritative metadata rather
than infer it from this repository.

## 6. Open Questions

- Which Kotekan metadata field carries the integer CHIME channel identifier?
- How does that identifier map to `chime_channel_id` in the runtime bundle?
- Can the channel identifier change during one stage lifetime?
- Should a non-pilot node emit an all-zero mask frame or emit no mask product?
- Which frame or alignment key identifies each emitted mask?
- What frame length will the active CHIME engine expose, and where is that value
  carried in the Kotekan configuration or metadata?
- Which debug deployment will compare device mask decisions and `uint64` powers
  against the Python path before the powers are removed from production output?
