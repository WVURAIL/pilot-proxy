# PilotProxy method specification

This specification fixes the detector geometry and the statistic used by the
CHIME real-data workflow. We state the operations in the order they are
applied so that a result can be reproduced from the recorded powers and
weight norms.

## Detector geometry

The baseline detector uses:

```text
detector_window_samples = 128
num_weight_terms = 3
skipped_guard_bins = 1
reference_offset_bins = 2
sample_bits_per_component = 4
power_accumulator = uint64
```

The two offset parameters obey
`reference_offset_bins = skipped_guard_bins + 1`. The target term is centered
on the nominal ATSC pilot. The lower and upper reference terms sample the
local non-DTV power on either side of that target.

## Target and reference powers

For each detector row `r`, the kernel returns three powers:

```text
P_target,r
P_ref_lower,r
P_ref_upper,r
```

We sum the powers over rows before forming a ratio:

```text
P_target = sum_r(P_target,r)
P_ref_sum = sum_r(P_ref_lower,r) + sum_r(P_ref_upper,r)
```

This order matters. Averaging per-row F-statistics is not equivalent to the
specified detector.

## F-statistic

The detector compares the target power with the mean of the two reference
powers:

```text
F = 2 * P_target / P_ref_sum
```

Equivalently,

```text
F = P_target / mean(P_ref_lower, P_ref_upper)
```

where the lower and upper powers are the row-summed quantities defined above.
The uncorrected one-bin pilot excess is

```text
rho = F - 1
```

## Norm-corrected positive-excess mask

The CHIME real-data path places its fixed mask threshold at the detector's own
null-hypothesis (`H0`) zero point. Under `H0`, the local spectrum contains no
pilot excess and is treated as flat across the three detector terms. Because
the steering vectors are quantized to int4, their squared norms are not
exactly equal. Therefore,

```text
E[P_term] = sigma^2 * ||w_term||^2
E[F]      = mu0 = 2 * target_norm_sq / ref_norm_sum_sq
```

Here `target_norm_sq = ||w_target||^2` and
`ref_norm_sum_sq = ||w_ref_lower||^2 + ||w_ref_upper||^2`. These are exact
integer norms computed from the packed weights. Independent recomputation
from the shipped ATSC 14--36 manifest gives a `mu0` range of 0.9853298815 to
1.0111111111. A common threshold at `F = 1` would therefore place different
channels on different sides of their own `H0` zero points.

We compare with `mu0` using integer powers:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer form of

```text
F > mu0
```

When `target_norm_sq : ref_norm_sum_sq = 1 : 2`, the comparison reduces to
the earlier `F > 1` rule, `p_target > (p_ref_sum >> 1)`. Products written
under that earlier rule identify it in `mask_rule`.

The corrected pilot excess is

```text
rho_corrected = F / mu0 - 1
```

and is stored per frame as `pilot_excess_corrected`. The CUDA ABI can express
the same comparison through the rational half-threshold
`half_num : half_den = target_norm_sq : ref_norm_sum_sq`. The current
`chime-scan` path reads the uint64 powers and applies the decision on the
host.

Masked frames are omitted from before/after averages. They are not replaced
with zeros.

## Dynamic range and frequency capture

The mask remains useful after the reported dB quantities have reached their
effective ceilings. We independently recomputed the following bounds from
the shipped packed weights and a pure-tone response:

- **Int4 weight crosstalk.** Pilot power entering a reference term through
  the quantized weights limits the reported F-statistic to `+39.055` to
  `+62.136` dB across ATSC channels 14--36. The corresponding individual
  reference-term crosstalk is `-68.244` to `-36.730` dB relative to the
  target term and depends on channel.
- **Pilot-frequency offset.** A transmitter offset `df` from the nominal
  pilot places power in the references at exactly `+/-2` fine bins. At
  `|df| = 300 Hz`, this effect limits the reported F-statistic to about
  `+25` to `+28` dB for the shipped weights. The exact value depends on the
  channel and the sign of the offset and is independent of additional pilot
  power.

Both ceilings remain well above `mu0`. Therefore, they do not change the
positive-excess decision for a strong pilot. They do mean that
`pnr_bin_db` and `snr_shelf_db` should be read as lower bounds for sufficiently
strong, offset transmitters.

For a rectangular `K`-sample window, the target-term capture factor is
`sinc^2(df / 3051.7578125 Hz)`, where
`sinc(x) = sin(pi*x)/(pi*x)`. Independent evaluation gives losses of `0.1385`
dB at `|df| = 300 Hz` and `0.3870` dB at `|df| = 500 Hz`. Apply the factor at
each measured offset. Whether that correction is negligible depends on the
retained calibration-uncertainty budget and should not be assumed here.

## Reference-placement resolver

The resolver starts with `reference_offset_bins = 2` and changes placement
only when the receiver geometry requires it:

- a valid requested reference remains at its requested offset;
- an edge reference wraps around the circular coarse-channel FFT;
- a reference that would land on the forbidden coarse-channel DC tone moves
  away from that tone; and
- a target/DC collision fails because the target signal cannot be moved.

The manifest records the selected placement. Two shipped cases exercise this
logic:

- DTV 14 has the forbidden DC tone in the skipped guard, but neither the
  target nor a reference collides with it.
- DTV 21 uses a lower reference that wraps across the coarse-channel edge.

## `K` baseline and candidate

`K = 128` is the implemented and tested software baseline. It matches the
current CUDA contract, shipped weight bank, result schema, and regression
tests.

`K = 256` is a separate detector configuration. The proposed implementation
uses int32 dot products, uint32 row powers, and uint64 frame sums to avoid the
current precision limit. That path is not implemented or tested here and
remains conditional on CANFAR cleaning evidence.
