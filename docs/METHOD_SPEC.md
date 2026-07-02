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

The CHIME real-data workflow uses a fixed, parameter-free mask threshold at the
detector's own H0 zero-point. int4 quantization of the steering vectors leaves
the three weight-term squared norms unequal, so under a locally flat noise
floor

```text
E[P_term] = sigma^2 * ||w_term||^2
E[F]      = mu0 = 2 * target_norm_sq / ref_norm_sum_sq
```

with `target_norm_sq = ||w_target||^2` and `ref_norm_sum_sq = ||w_ref_lower||^2
+ ||w_ref_upper||^2` (exact integers, computed from the packed weights). Across
the shipped ATSC 14-36 bank, `mu0` spans about 0.985 to 1.011, so a mask at
`F > 1` would pin the H0 mask fraction toward 0 or 1 per channel. The mask
therefore compares against `mu0` exactly, in integers:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer-power form of:

```text
F > mu0
```

With `target_norm_sq : ref_norm_sum_sq = 1 : 2` it reduces to the legacy
`F > 1` rule (`p_target > p_ref_sum >> 1`), which products written before the
correction declare via their recorded `mask_rule`. The corrected pilot excess
is `rho_corrected = F/mu0 - 1` (recorded per frame as
`pilot_excess_corrected`). For the deployed kernel path, the same correction
is the existing rational half-threshold with
`half_num : half_den = target_norm_sq : ref_norm_sum_sq`.

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
detector configuration. It is not promoted until CANFAR cleaning
evidence supports it.
