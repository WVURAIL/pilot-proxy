# PilotProxy Data Products

This document defines the canonical files produced by the CHIME real-data
workflows. `chime-run` writes the products directly from one staged directory.
`chime-scan` first writes one authoritative `<freq_id>.npz` product per coarse
channel and then combines those products on a shared event/frame identity. The
two paths use the same canonical detector, cache, reduction, table, and plotting
formats where those formats apply.

Unless stated otherwise, `mask = 1` means that the frame is rejected and
`valid = 1` means that the reference denominator is nonzero.

## JSON Files

### `run_config.json`

This file records the run-level detector contract and provenance. Both workflows
use:

- `schema_version = fstat_chime_run_config_v2`;
- `detector_contract`;
- `physical_channels`;
- `mask_policy`;
- `reference_placement_summary` when placement metadata are available.

The staged `chime-run` path also records fields such as `input_dir`,
`output_dir`, `weight_coordinate`, and file-level `provenance`. The combined
`chime-scan` path instead records `source = chime-scan`,
`freq_id_by_pilot`, and `detector_provenance_by_pilot`. Consumers should use
the schema and named fields rather than assume that the two producers have
byte-identical JSON.

### `stats.json`

This file records the detector geometry and run statistics. Shared fields
include:

- `schema_version = fstat_chime_stats_v2`;
- `detector_contract`;
- `num_frames` and `num_pilots`;
- `num_input_streams`;
- `windows_per_stream`;
- `mask_policy`;
- `reference_placement_summary` when available.

`chime-run` also records `detector_rows_per_frame`, `kernel_specs`,
`weight_coordinate`, and `mu0_by_pilot`. A `chime-scan` combine records
`combine_alignment`, `rational_overflow_count_by_pilot`, and any cross-build
provenance notes. The combined scan keeps `mu0` in
`chime_detector_outputs.npz` rather than copying it into `stats.json`.

The `detector_contract` states the coordinate convention through:

- `weight_coordinate_system`;
- `input_coordinate_system`;
- `input_preprocessing.time_reverse_detector_windows_before_kernel`.

### `input_manifest.json`

This file records the HDF5 inputs. The staged runner includes discovered dataset
metadata. The archive combine records its `chime-scan` source and the input unit
keys collected from the per-pilot products.

### `product_validation.json`

`pilot-proxy validate-products --output-json ...` writes this report; it is not
created merely by running the detector. Its principal fields are:

- `valid`;
- `num_errors`;
- `errors`.

## Detector NPZ

### `chime_detector_outputs.npz`

This is the canonical frame-by-pilot detector product.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `frame_index` | `(num_frames,)` | `int64` | frame | Contiguous positional frame index |
| `p_target_u64` | `(num_frames, num_pilots)` | `uint64` | power | Target-bin power |
| `p_ref_sum_u64` | `(num_frames, num_pilots)` | `uint64` | power | Lower plus upper reference power |
| `fstat_raw` | `(num_frames, num_pilots)` | `float64` | unitless | `2*p_target/p_ref_sum` |
| `fstat_level_db` | `(num_frames, num_pilots)` | `float64` | dB | `10*log10(F)` |
| `pnr_bin_db` | `(num_frames, num_pilots)` | `float64` | dB | One-bin pilot-excess PNR |
| `snr_shelf_db` | `(num_frames, num_pilots)` | `float64` | dB | Estimated ATSC data-shelf SNR; finite only where its transform is defined |
| `valid` | `(num_frames, num_pilots)` | `uint8` | 0/1 | `p_ref_sum != 0` |
| `mask` | `(num_frames, num_pilots)` | `uint8` | 0/1 | `1 = reject` under the recorded mask rule |
| `target_norm_sq` | `(num_pilots,)` | `int64` | unitless | Exact `||w_target||^2` of the int4 weights |
| `ref_norm_sum_sq` | `(num_pilots,)` | `int64` | unitless | Exact `||w_ref_lo||^2 + ||w_ref_up||^2` |
| `mu0` | `(num_pilots,)` | `float64` | unitless | `2*target_norm_sq/ref_norm_sum_sq`, the weight-norm H0 reference |
| `pilot_excess_corrected` | `(num_frames, num_pilots)` | `float64` | unitless | `F/mu0 - 1`, or NaN for invalid frames |

The current mask is the norm-corrected positive-excess comparison:

```text
valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the integer form of `F > mu0`. Products written before the correction
declare the legacy `F > 1` rule in `mask_rule` and may omit the four norm-related
arrays. Therefore readers should check the recorded contract before assuming
that those arrays exist.

For `chime-scan`, `num_frames` is the event/frame intersection retained by the
combine. The source per-pilot products can contain additional frames that were
not common to every channel.

## Spectrogram Cache NPZ

### `chime_spectrogram_cache.npz`

This cache carries the frame-level baseband power and the matching detector
mask used by the plotting functions.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `baseband_power_linear` | `(num_frames, num_pilots)` | `float64` | power | Mean non-coherent baseband power for the frame |
| `baseband_power_db` | `(num_frames, num_pilots)` | `float64` | dB | `10*log10(baseband_power_linear)` where power is positive |
| `mask` | `(num_frames, num_pilots)` | `uint8` | 0/1 | Detector rejection mask copied from the detector product |
| `valid` | `(num_frames, num_pilots)` | `uint8` | 0/1 | Detector validity copied from the detector product |
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `frame_index` | `(num_frames,)` | `int64` | frame | Contiguous positional frame index |
| `relative_time_s` | `(num_frames,)` | `float64` | s | `frame_index*nfft/390625` |

`relative_time_s` is accumulated data time. It does not restore gaps between
separate archive events and should not be interpreted as wall-clock time.

## Integrated Spectra NPZ

### `chime_integrated_spectra.npz`

The `chime-scan` combine writes a reporting stack of the integrated spectra in
the per-pilot products. The analyzer uses a rectangular-window FFT, sums
`|FFT|^2` over input streams, and accumulates over frames. `before` includes
valid frames; `after` includes valid frames with `reject_mask = 0`. Therefore
`before - after` is the accumulated spectrum of the rejected frames.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `schema_version` | scalar | `str` | — | `fstat_chime_integrated_spectra_v1` |
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `freq_id` | `(num_pilots,)` | `int64` | id | CHIME coarse-channel identifier when recorded |
| `integrated_spectrum_before_mask` | `(num_pilots, nfft)` | `float64` | power | Sum over valid frames |
| `integrated_spectrum_after_mask` | `(num_pilots, nfft)` | `float64` | power | Sum over valid, kept frames |
| `masked_fraction_by_channel` | `(num_pilots,)` | `float64` | 0..1 | Rejected valid frames divided by valid frames; NaN when none are valid |
| `sample_rate_hz` | scalar | `float64` | Hz | Shared per-channel sample rate, or NaN when timing metadata are unavailable |
| `nfft` | scalar | `int64` | bins | FFT length recorded by the per-pilot products |

Bin `k` maps to baseband frequency as:

```text
((k + nfft//2) % nfft - nfft//2) * sample_rate_hz / nfft
```

The authoritative copy remains in each `_per_pilot/<freq_id>.npz`; see
`product_schema_v2.md`. Integrated spectra are accumulated before terminal
event intersection, so they represent each pilot's full processed frame set.
The canonical frame arrays can represent a smaller all-channel intersection.
This distinction is recorded by the per-pilot products and
`stats.json.combine_alignment`.

## Reductions NPZ

### `chime_reductions_10s.npz`

This file groups the canonical frame arrays into approximately 10 s of
contiguous data time. The grouping uses `frame_index`, the analysis frame length,
and a 390,625 Hz channel sample rate; it does not use the per-file absolute-time
axis.

All arrays below have a leading dimension of `num_chunks`. Arrays that also
vary by pilot have shape `(num_chunks, num_pilots)`.

- `chunk_index`, `chunk_start_frame`, `chunk_stop_frame`;
- `input_power_mean`, `cleaned_power_mean`;
- `mask_fraction`, `mask_fraction_valid`, `mask_fraction_total`;
- `unmasked_count`, `total_count`, `valid_count`, `invalid_count`;
- `masked_count_valid`, `unmasked_count_valid`;
- `fstat_level_db_median`, `fstat_level_db_p95`, `fstat_level_db_max`;
- `snr_shelf_db_median`, `snr_shelf_db_p95`, `snr_shelf_db_max`.

`cleaned_power_mean` is the mean over valid frames with `mask = 0`; it is NaN
when a chunk contains no such frame. The term “cleaned” is a product-field name,
not a claim that all interference has been removed.

## Frame Identity NPZ

### `chime_frame_identity.npz`

An event-keyed `chime-scan` combine writes the identities retained in the
canonical stack:

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `frame_event_key` | `(num_frames,)` | `str` | Source event identity with the per-channel `freq_id` token removed |
| `frame_in_unit` | `(num_frames,)` | `int64` | Frame position within the source file |

Legacy products without identity tags fall back to strict positional alignment
and do not produce this sidecar.

## Tables

Tables are written under `tables/`:

- `mask_summary_by_pilot.csv` is written by the detector/combine path;
- `spectrum_before_after.csv` is written by `chime-run` and regenerated by the
  baseband spectrum plot;
- `fstat_summary_by_pilot.csv` and
  `snr_shelf_histogram_summary.csv` are written by `chime-plot` or
  `chime-run --plot`.

The SNR-shelf summary retains legacy field names. Its
`num_positive_excess_frames` and `positive_excess_fraction` count finite
`snr_shelf_db` values, which corresponds to `F > 1`. When `mu0 != 1`, use the
recorded mask or `mask_summary_by_pilot.csv` for the norm-corrected `F > mu0`
decision.

## Figures

`pilot-proxy chime-plot` or `chime-run --plot` writes these figures under
`figures/`:

- `snr_shelf_histogram_by_pilot.png`;
- `fstat_survival_by_pilot.png`;
- `fstat_level_spectrogram.png`;
- `baseband_spectrogram.png`;
- `baseband_spectrum_before_after_mask.png`;
- `mask_spectrogram.png`.

PNG is the default. Setting `PILOT_PROXY_FIGURE_FORMATS=png,pdf` adds PDF copies
without changing the numerical products.
