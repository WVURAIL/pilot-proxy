# Design decisions

This file records choices that affect the interpretation of CHIME DTV
products. We keep the choice, its reason, and its boundary together so the
same question does not have to be reconstructed during review.

## The CHIME default is norm-corrected positive excess

The CHIME real-data path uses the following comparison:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer form of `F > mu0`, where

```text
mu0 = 2 * target_norm_sq / ref_norm_sum_sq
```

is the flat-floor `H0` zero point implied by the packed weights. Int4
quantization leaves the target and reference norms unequal. Independent
recomputation from the shipped ATSC 14--36 manifest gives a `mu0` range of
0.9853298815 to 1.0111111111 rather than exactly 1.

We compare each channel with its own `mu0` so that the rule is defined by the
shipped weights and can be reproduced from integer powers. This avoids
fitting a separate operational threshold before the bounded CANFAR test.
Products written under the earlier `F > 1` rule retain that rule in
`mask_rule` and must be interpreted under their recorded convention.

## `K = 128` is the CANFAR baseline

We retain `K = 128` because it matches the current CUDA contract, detector
core profile, shipped manifests, and regression tests.

`K = 256` remains a possible later configuration. The proposed implementation
uses int32 dot products, uint32 row powers, and uint64 frame sums to avoid the
current precision constraint. It is not implemented or tested here and would
require a separate weight bank, contract, and validation set. We do not
promote it without CANFAR cleaning evidence.

`K = 512` is outside the current candidate set.

## Guard and reference terms have separate names

The public terms are:

```text
skipped_guard_bins
reference_offset_bins
```

They obey:

```text
reference_offset_bins = skipped_guard_bins + 1
```

A user-authored detector-core configuration specifies
`skipped_guard_bins`. Generated metadata may also record
`reference_offset_bins` so that the selected geometry can be audited.

## Reference placement is adaptive and recorded

The resolver retains the requested reference offset when the receiver
geometry permits it. If a reference reaches a circular FFT edge, it wraps. If
a reference would collide with the forbidden coarse-channel DC tone, it moves
away. A target/DC collision is invalid because moving the target would change
the signal being tested.

The shipped CHIME bank includes two useful boundary cases:

- DTV 14 places the forbidden DC tone in the skipped guard. Neither the
  target nor a reference collides with it.
- DTV 21 wraps its lower reference across the coarse-channel edge instead of
  moving that reference toward the target.

Run products retain the reference-placement summary. Validators and later
analyses should use the recorded placement rather than reconstructing it from
the nominal offset.

## The runtime bundle prepares, but does not implement, Kotekan

The runtime exporter writes:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

The planned deployment model is:

```text
same detector software on every node
same weight bundle on every node
first-frame integer CHIME channel ID selects active profile
non-pilot channel disables detector
pilot channel selects one weight-bank pointer
```

This repository does not contain a Kotekan stage. The bundle defines the
inputs that a later stage would consume; it is not evidence of deployment.

## Deferred work

The following items are outside the minimum detector result and remain
separate tasks:

- tone catalog and intermodulation classification;
- LimeSDR loopback;
- Kotekan integration;
- a `K = 256` implementation;
- production integration of additional threshold modes;
- a common-mode power veto; and
- threshold fitting from CANFAR-measured means.

The analysis directory can test alternative thresholds after the scan. That
does not change the shipped real-data mask or add a new production mode. The
current paper should therefore distinguish the recorded detector output from
post-hoc sensitivity studies.
