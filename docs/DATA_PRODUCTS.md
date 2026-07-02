# PilotProxy Data Products

This document defines the files emitted by a CHIME real-data run.

## JSON Files

### `run_config.json`

Run configuration and provenance.

Key fields:

- `schema_version = fstat_chime_run_config_v2`
- `detector_contract`
- `input_dir`
- `output_dir`
- `physical_channels`
- `weight_coordinate`
- `mask_policy`
- `reference_placement_summary`
- `provenance`

### `stats.json`

Run statistics and detector-layout metadata.

Key fields:

- `schema_version = fstat_chime_stats_v2`
- `detector_contract`
- `num_frames`
- `num_pilots`
- `num_input_streams`
- `detector_rows_per_frame`
- `windows_per_stream`
- `kernel_specs`
- `weight_coordinate`
- `mask_policy`
- `reference_placement_summary`

The `detector_contract` object records the required coordinate convention:

- `weight_coordinate_system`
- `input_coordinate_system`
- `input_preprocessing.time_reverse_detector_windows_before_kernel`

### `input_manifest.json`

Discovered CHIME HDF5 input files and dataset metadata.

### `product_validation.json`

Output from `pilot-proxy validate-products`.

Key fields:

- `valid`
- `num_errors`
- `errors`

## Detector NPZ

### `chime_detector_outputs.npz`

| Array                  |                        Shape | Dtype       | Units      | Meaning                            |
| ---------------------- | ---------------------------: | ----------- | ---------- | ---------------------------------- |
| `physical_channel`     |              `(num_pilots,)` | `int32`     | channel    | ATSC physical channel              |
| `pilot_frequency_hz`   |              `(num_pilots,)` | `float64`   | Hz         | ATSC pilot RF frequency            |
| `chime_frequency_hz`   |              `(num_pilots,)` | `float64`   | Hz         | CHIME coarse-channel center        |
| `frame_index`          |              `(num_frames,)` | `int64`     | frame      | Contiguous frame index             |
| `p_target_u64`         |   `(num_frames, num_pilots)` | `uint64`    | power      | Target-bin power                   |
| `p_ref_sum_u64`        |   `(num_frames, num_pilots)` | `uint64`    | power      | Lower plus upper reference power   |
| `fstat_raw`            |   `(num_frames, num_pilots)` | `float64`   | unitless   | `2*p_target/p_ref_sum`             |
| `fstat_level_db`       |   `(num_frames, num_pilots)` | `float64`   | dB         | `10*log10(F)`                      |
| `pnr_bin_db`           |   `(num_frames, num_pilots)` | `float64`   | dB         | One-bin pilot-excess PNR           |
| `snr_shelf_db`         |   `(num_frames, num_pilots)` | `float64`   | dB         | Estimated ATSC data-shelf SNR      |
| `valid`                |   `(num_frames, num_pilots)` | `uint8`     | 0/1        | `p_ref_sum != 0`                   |
| `mask`                 |   `(num_frames, num_pilots)` | `uint8`     | 0/1        | Positive-excess mask               |

## Spectrogram Cache NPZ

### `chime_spectrogram_cache.npz`

| Array                     |                        Shape | Dtype       | Units     | Meaning                                    |
| ------------------------- | ---------------------------: | ----------- | --------- | ------------------------------------------ |
| `baseband_power_linear`   |   `(num_frames, num_pilots)` | `float64`   | power     | Raw pilot-channel baseband power           |
| `baseband_power_db`       |   `(num_frames, num_pilots)` | `float64`   | dB        | Baseband power in dB                       |
| `mask`                    |   `(num_frames, num_pilots)` | `uint8`     | 0/1       | Detector mask copied from detector output  |
| `valid`                   |   `(num_frames, num_pilots)` | `uint8`     | 0/1       | Valid mask copied from detector output     |
| `physical_channel`        |              `(num_pilots,)` | `int32`     | channel   | ATSC physical channel                      |
| `pilot_frequency_hz`      |              `(num_pilots,)` | `float64`   | Hz        | ATSC pilot RF frequency                    |
| `chime_frequency_hz`      |              `(num_pilots,)` | `float64`   | Hz        | CHIME coarse-channel center                |
| `frame_index`             |              `(num_frames,)` | `int64`     | frame     | Contiguous frame index                     |
| `relative_time_s`         |              `(num_frames,)` | `float64`   | s         | Relative time from frame index             |

## Integrated Spectra NPZ

### `chime_integrated_spectra.npz`

Per-pilot integrated power spectra (rectangular-window `|FFT|^2` summed over feeds,
accumulated over frames), stacked along the pilot axis. `before` integrates every
valid frame; `after` integrates kept (not-rejected) frames, so `before - after` is
the spectrum the positive-excess mask removed. A reporting-only convenience; the
authoritative per-channel copy lives in each `<freq_id>.npz` (see
`product_schema_v2.md`). Bin `k` maps to baseband frequency
`((k + nfft//2) % nfft - nfft//2) * sample_rate_hz / nfft`.

| Array                              |               Shape | Dtype     | Units | Meaning                                  |
| ---------------------------------- | ------------------: | --------- | ----- | ---------------------------------------- |
| `physical_channel`                 |     `(num_pilots,)` | `int32`   | chan  | ATSC physical channel                    |
| `pilot_frequency_hz`               |     `(num_pilots,)` | `float64` | Hz    | ATSC pilot RF frequency                  |
| `chime_frequency_hz`               |     `(num_pilots,)` | `float64` | Hz    | CHIME coarse-channel center              |
| `freq_id`                          |     `(num_pilots,)` | `int64`   | id    | CHIME coarse-channel id (if recorded)    |
| `integrated_spectrum_before_mask`  | `(num_pilots, nfft)`| `float64` | power | Sum over valid frames (power spectrum)   |
| `integrated_spectrum_after_mask`   | `(num_pilots, nfft)`| `float64` | power | Sum over kept frames (power spectrum)    |
| `masked_fraction_by_channel`       |     `(num_pilots,)` | `float64` | 0..1  | valid-and-rejected / valid (NaN if none) |
| `sample_rate_hz`                   |              scalar | `float64` | Hz    | Per-channel sample rate (freq axis)      |
| `nfft`                             |              scalar | `int64`   | bins  | FFT length                               |

## Reductions NPZ

### `chime_reductions_10s.npz`

Chunk-level reductions for approximately 10 s time bins.

Key arrays:

- `chunk_index`, `chunk_start_frame`, `chunk_stop_frame`: shape `(num_chunks,)`
- `input_power_mean`, `cleaned_power_mean`: shape `(num_chunks, num_pilots)`
- `valid_count`, `invalid_count`: shape `(num_chunks, num_pilots)`
- `masked_count_valid`, `unmasked_count_valid`: shape `(num_chunks, num_pilots)`
- `mask_fraction_valid`, `mask_fraction_total`: shape `(num_chunks, num_pilots)`

## Tables

Tables are written under `tables/`:

- `fstat_summary_by_pilot.csv`
- `mask_summary_by_pilot.csv`
- `snr_shelf_histogram_summary.csv`
- `spectrum_before_after.csv`

## Figures

Figures are written under `figures/`:

- `snr_shelf_histogram_by_pilot.png`
- `fstat_survival_by_pilot.png`
- `fstat_level_spectrogram.png`
- `baseband_spectrogram.png`
- `baseband_spectrum_before_after_mask.png`
- `mask_spectrogram.png`
