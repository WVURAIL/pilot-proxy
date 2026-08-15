# Per-pilot product field reference

This document defines the fields in the only supported public per-pilot
product. The product identity and decision semantics are defined in
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md); fine-reduction-specific arrays are
listed in [`FINE_REDUCTION_PRODUCTS.md`](FINE_REDUCTION_PRODUCTS.md).

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 1
schema_version = "pilotproxy_per_pilot_product_v1"
```

The `pilot-proxy-detector` analyzer writes one product for one selected receiver
coarse channel. `reject_mask = 1` means reject. Event-keyed frame identity,
exact weight norms, detector provenance, and the decision contract are
mandatory. Products missing them are invalid and must be regenerated.

In the tables below, `N` is the number of detector frames and `U` is the number
of consumed input files.
## Identity and geometry

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `freq_id` | `(1,)` | `int64` | CHIME coarse-channel identifier |
| `physical_channel` | `(1,)` | `int32` | Nearest ATSC physical channel |
| `pilot_in_band` | `(1,)` | `uint8` | `1` when the selected coarse channel contains the nominal ATSC pilot |
| `pilot_frequency_hz` | `(1,)` | `float64` | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(1,)` | `float64` | CHIME coarse-channel center |
| `nfft` | scalar | `int64` | Analysis frame and FFT length used for this product |
| `detector_window_samples` | scalar | `int64` | Receiver-selected CUDA detector window `K` (128 for CHIME; 64 for the current CHORD profiles) |
| `num_input_streams` | scalar | `int64` | Input feed/polarization streams summed |
| `sense` | scalar | `int64` | Spectral sense, `+1` or `-1` |

The schema does not fix `nfft` to one value. The current software profile and
tests use 16,384 samples, which is associated with the planned CHIME engine
upgrade. The active acquisition value must be recorded from the run; a
12,288-sample current-frame value remains provisional until independently
verified. Any accepted value must be divisible by `K`.

If the nominal ATSC pilot does not fall within half a coarse-channel bandwidth,
the analyzer sets `pilot_in_band = 0`. It still emits one row per input frame,
but sets `valid = 0`, `reject_mask = 0`, and the integer detector powers to
zero. Derived detector values and `baseband_power_linear` are NaN, and neither
integrated spectrum receives the frame.

## Per-frame detector output (length `N`)

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `frame_index` | `(N,)` | `int64` | Zero-based positional frame counter |
| `p_target_u64` | `(N, 1)` | `uint64` | Target-bin power from the fixed-point detector |
| `p_ref_sum_u64` | `(N, 1)` | `uint64` | Lower plus upper reference power |
| `coarse_power_ratio` | `(N, 1)` | `float64` | `2*p_target/p_ref_sum` |
| `normalized_coarse_power_ratio_db` | `(N, 1)` | `float64` | `10*log10(F)` |
| `pilot_excess_db` | `(N, 1)` | `float64` | One-bin pilot-excess PNR |
| `estimated_data_shelf_snr_db` | `(N, 1)` | `float64` | Estimated ATSC data-shelf SNR |
| `valid` | `(N, 1)` | `uint8` | `p_ref_sum != 0` |
| `reject_mask` | `(N, 1)` | `uint8` | `1 = discard` under the recorded positive-excess rule |
| `normalized_pilot_excess` | `(N, 1)` | `float64` | `F/null_power_ratio - 1`, or NaN when invalid |
| `target_norm_sq` | `(1,)` | `int64` | Exact `||w_target||^2` of the int4 weights |
| `reference_norm_sum_sq` | `(1,)` | `int64` | Exact `||w_ref_lo||^2 + ||w_ref_up||^2` |
| `null_power_ratio` | `(1,)` | `float64` | `2*target_norm_sq/reference_norm_sum_sq` |
| `baseband_power_linear` | `(N, 1)` | `float64` | Mean non-coherent baseband power for the frame |

The product stores `p_target_u64` and `p_ref_sum_u64` without converting them to
a thresholded statistic. We can therefore recompute an alternative F threshold
or dB calibration from the same detector pass, provided the required calibration
constants are also used. This does not require rerunning the CUDA detector.

### Mask convention

The current `reject_mask` compares the target and reference powers after
correcting for unequal quantized weight norms. A valid frame is rejected when:

```text
reject_mask = valid && (p_target_u64 * reference_norm_sum_sq
                        > target_norm_sq * p_ref_sum_u64)
            = valid && (F > null_power_ratio)
```

The integer cross multiplication is the recorded `mask_rule`. It avoids a
floating-point threshold decision and uses the weight-norm flat-floor reference
`null_power_ratio = 2*target_norm_sq/reference_norm_sum_sq` rather than assuming `null_power_ratio = 1`.

Reporting can derive `keep_mask = 1 - reject_mask`; the per-pilot product does
not store a second copy.

## Integrated power spectra

For each full analysis frame, the analyzer computes a rectangular-window FFT of
the raw samples, sums `|FFT|^2` over input streams, and accumulates the result.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `integrated_spectrum_before_mask` | `(nfft,)` | `float64` | Sum over valid frames |
| `integrated_spectrum_after_mask` | `(nfft,)` | `float64` | Sum over valid frames with `reject_mask = 0` |

Therefore `before - after` is the accumulated spectrum of frames rejected by
the stored mask. The arrays are raw accumulated power; normalization by frame
count or input-stream count is a reporting choice.

Bin `k` maps to baseband frequency by:

```text
((k + nfft//2) % nfft - nfft//2) * fs / nfft
```

where `fs = 1 / unit_delta_time` when every contributing unit has the same
finite sample period. If the periods differ, one shared frequency axis is not
defined; the canonical combine rejects that mixture.

The production path uses CuPy when a GPU runtime is available, while tests can
use NumPy. Both implement the same FFT and float64 feed-sum accumulation, but
their results can differ by normal floating-point roundoff. The stored spectra
contain accumulated power rather than per-frame complex FFT values. As a
result, a different window or a different mask threshold requires a new spectral
pass; those choices cannot be reconstructed from these two accumulated arrays.

## Absolute-time axis

The packed HDF5 reader copies timing and event attributes from each file into a
per-unit table aligned with `unit_order`. Two per-frame arrays identify the
corresponding unit and the frame position within that unit. This avoids storing
one absolute timestamp per frame.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `unit_time0_ctime` | `(U,)` | `float64` | File start UNIX time, or NaN when absent |
| `unit_time0_fpga` | `(U,)` | `uint64` | FPGA count at file start, or 0 when absent |
| `unit_event_id` | `(U,)` | `int64` | CHIME event identifier, or `-1` when absent |
| `unit_delta_time` | `(U,)` | `float64` | Sample period in seconds, or NaN when absent |
| `archive_version` | `(U,)` | `str` | CHIME archive version, or an empty string when absent |
| `frame_unit_index` | `(N,)` | `int32` | Unit index `u` for each frame |
| `frame_in_unit` | `(N,)` | `int32` | Zero-based frame position within unit `u` |

For frame `f`, compute wall time with:

```python
u = frame_unit_index[f]
t = unit_time0_ctime[u] + frame_in_unit[f] * nfft * unit_delta_time[u]
```

If a synthetic or incomplete file lacks the root attributes, the analyzer stores
`NaN / 0 / -1 / ""` and continues. Any LST or wall-time analysis must first
exclude those missing values.

## Provenance and calibration

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `weights_hash` | scalar | `str` | SHA-256 of the selected packed weight profile |
| `weight_bank_sha256` | scalar | `str` | SHA-256 of the complete weight bank, or empty for injected weights |
| `weight_manifest_sha256` | scalar | `str` | SHA-256 of the adjacent manifest, or empty when unavailable |
| `detector_version` | scalar | `str` | Package, source-tree, kernel, schema, and `K` identity string |
| `mask_rule` | scalar | `str` | Integer rejection rule used for this product |
| `reference_placement_json` | scalar | `str` | Selected reference-placement metadata encoded as JSON |
| `rational_overflow_count` | scalar | `uint64` | Accumulated fixed-point overflow telemetry |
| `max_chunks_per_file` | scalar | `int64` | Per-file cap, or `-1` when uncapped |
| `detector_contract_json` | scalar | `str` | Full detector contract encoded as JSON |
| `pilot_below_data_db` | scalar | `float64` | Pilot-to-data-shelf calibration constant |
| `bin_enbw_hz` | scalar | `float64` | Detector-bin equivalent noise bandwidth |
| `dtv_bandwidth_hz` | scalar | `float64` | Assumed DTV bandwidth |
| `pilot_capture_efficiency` | scalar | `float64` | Pilot capture-efficiency factor |

The file also stores keys needed for resume and channel alignment:

| Array | Shape | Meaning |
|---|---:|---|
| `unit_keys` | `(U,)` | Sorted set of committed datatrawl unit keys |
| `unit_order` | `(U,)` | Unit keys in analyzer consumption order |
| `source_event_keys` | `(U,)` | Event identity used to align the same acquisition across `freq_id` products |

Resume checks the schema, `freq_id`, frame cap, detector geometry, calibration,
weights, mask rule, detector contract, and reference placement. A source-tree
hash change is allowed only when the remaining `detector_version` geometry and
kernel tokens match. Other mismatches stop the run and require a clean output
directory.

---

## Derived in reporting (not stored)

The frame-level fields support these downstream products without another CUDA
detector pass:

- `keep_mask = 1 - reject_mask`;
- local-reference power ratio histograms and survival curves from the raw integer powers;
- alternative frame-level thresholds from the raw powers and norms;
- masked fraction from `reject_mask` and `valid`;
- baseband before/after summaries from `baseband_power_linear` and a selected
  frame mask;
- per-frame wall time, and then LST where the timing attributes and telescope
  longitude are available.

The stored integrated spectra support the spectrum before/after comparison for
the mask used during the run. They do not preserve enough information to apply a
new frame threshold or FFT window after the fact.

Full 6 MHz DTV-channel mask expansion is not part of this product. It requires a
larger pass over the neighboring coarse channels after the pilot detector has
been validated.
