# Design Decisions

This document records project decisions that affect interpretation of CHIME DTV
products and should not be rediscovered during review or article writing.

## Positive-Excess Masking Is The CHIME Default

The CHIME real-data path uses thresholdless positive-excess masking:

```text
valid = p_ref_sum != 0
mask = valid && (p_target > (p_ref_sum >> 1))
```

This is the exact integer-power form of `F > 1`. It avoids empirical threshold
fitting before the bounded CANFAR pilot and keeps the detector policy auditable.

## K=128 Is The CANFAR Baseline

K=128 remains the validated baseline because it matches the current CUDA
contract, shipped detector core profile, generated manifests, and test coverage.

K=256 remains a plausible future candidate. It is not blocked by unavoidable
precision loss if implemented with int32 dot products, uint32 row powers, and
uint64 frame sums, but it is a separate detector configuration and is not
promoted until CANFAR offset and cleaning evidence supports it.

K=512 is not part of the current production candidate set.

## Guard/Reference Terminology Was Reset

The public terminology is:

```text
skipped_guard_bins
reference_offset_bins
```

with:

```text
reference_offset_bins = skipped_guard_bins + 1
```

User-authored detector-core configs specify `skipped_guard_bins` only. Generated
metadata may record both fields for auditability.

## Reference Placement Is Adaptive And Auditable

Reference placement keeps the requested offset when possible, but records any
adaptive handling explicitly.

Important CHIME cases:

- DTV 14 has the coarse-channel DC tone in the skipped guard region; this is
  valid and does not require moving the reference.
- DTV 21 has a lower reference that wraps across the coarse-channel edge; the
  reference wraps rather than being silently moved closer to the target.
- Target/DC collisions are invalid because the target signal cannot be moved.

Run products must preserve the reference-placement summary so these cases are
visible to validators and downstream reviewers.

## Runtime Bundle Is Preparation, Not Kotekan Integration

The runtime bundle exporter prepares:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

The intended future deployment model is:

```text
same detector software on every node
same weight bundle on every node
first-frame integer CHIME channel ID selects active profile
non-pilot channel disables detector
pilot channel selects one weight-bank pointer
```

No actual Kotekan stage is implemented in this project state.

## Deferred Work

The following work is intentionally deferred until after the bounded CANFAR
pilot:

- tone catalog and intermodulation classification,
- LimeSDR loopback,
- actual Kotekan stage,
- K=256 implementation,
- new threshold modes,
- additional empirical threshold fitting.

The current bottleneck is validation on real data, not new detector logic.
