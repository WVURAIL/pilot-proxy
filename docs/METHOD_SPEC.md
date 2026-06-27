# PilotProxy Method Specification

## Detector Geometry

The baseline detector uses:

```text
detector_window_samples = 128
num_weight_terms = 3
skipped_guard_bins = 1
reference_offset_bins = 2
sample_bits_per_component = 4
power_accumulator = uint64
```

`reference_offset_bins = skipped_guard_bins + 1`. The target bin is centered on
the ATSC pilot. Lower and upper reference bins estimate the local non-DTV floor.

## Target And Reference Powers

For each detector row `r`:

```text
P_target,r
P_ref_lower,r
P_ref_upper,r
```

Rows are combined by summing powers first:

```text
P_target = sum_r(P_target,r)
P_ref_sum = sum_r(P_ref_lower,r) + sum_r(P_ref_upper,r)
```

## F-Statistic

```text
F = 2 * P_target / P_ref_sum
```

Equivalently:

```text
F = P_target / mean(P_ref_lower, P_ref_upper)
```

The one-bin pilot excess is:

```text
rho = F - 1
```

## Positive-Excess Mask

The CHIME real-data workflow is thresholdless:

```text
valid = p_ref_sum != 0
mask = valid && (p_target > (p_ref_sum >> 1))
```

This is the exact integer-power form of:

```text
F > 1
```

Masked frames are excluded from before/after averages; they are not zero-filled.

## Reference-Placement Resolver

Reference placement is adaptive and auditable:

- requested references remain at `reference_offset_bins = 2` when valid;
- edge references wrap around the circular coarse-channel FFT;
- references that collide with the forbidden coarse-channel DC tone shift away;
- target/DC collisions fail because the target signal cannot be moved.

Known baseline cases:

- DTV 14 has the forbidden DC tone in the skipped guard region but no target or
  reference collision.
- DTV 21 uses an edge-wrapped lower reference.

## K Baseline And Candidate

`K=128` is the validated production baseline.

`K=256` is not blocked by precision loss if implemented with int32 dot
products, uint32 row powers, and uint64 frame sums, but it is a separate
detector configuration. It is not promoted until CANFAR offset and cleaning
evidence supports it.
