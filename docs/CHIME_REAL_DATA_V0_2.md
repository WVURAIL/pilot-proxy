# PilotProxy v0.2 CHIME Real-Data Adapter

This revision adds an optional CHIME adapter layer under `pilot_proxy.chime`.
The CUDA kernel contract is unchanged:

```text
K = 128 detector samples
N = 3 weight terms
reference_offset_bins = 2 nominal
skipped_guard_bins = 1 nominal
packed complex int4 input
uint64 power accumulation
```

The standalone synthetic/testbench path still supports explicit shelf-SNR
thresholds. The CHIME real-data workflow does not use a shelf-SNR threshold or
threshold table; it applies the norm-corrected positive-excess rule:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

the exact integer form of `F > mu0`, where `mu0 = 2*target_norm_sq/
ref_norm_sum_sq` is the flat-floor H0 zero-point set by the int4 weight norms
(see `docs/METHOD_SPEC.md`).

`reference_offset_bins` is the primary internal/kernel term. It is the
target-to-reference spacing in detector fine bins. `skipped_guard_bins` is the
more intuitive user-facing gap between target and reference, and is always:

```text
skipped_guard_bins = reference_offset_bins - 1
```

The adapter streams segmented CHIME HDF5 files into the published detector:

```text
CHIME HDF5 segments
  -> normalized block (num_input_streams, 1, samples)
  -> packed detector input (frames, detector_rows_per_frame, 128)
  -> CUDA NumDen detector
  -> one F-statistic per frame per physical DTV pilot channel
```

## Data Contract

The CHIME input directory is set by `PILOT_PROXY_CHIME_INPUT_DIR`, and defaults to
`$HOME/dataset/canfar_pilots_10s` when that variable is unset:

```text
$PILOT_PROXY_CHIME_INPUT_DIR    # e.g. $HOME/dataset/canfar_pilots_10s
```

Inspection found 23 physical DTV pilot channels mapped from CHIME coarse-channel
directories `ch0844` through `ch0506`, corresponding to physical channels
14 through 36. Each file contains:

```text
dataset path: /baseband
shape: (time, input)
dtype: uint8
axis attr: ["time", "input"]
num input streams: 2048
frequency/channel: one CHIME coarse channel per directory
encoding: CHIME native offset-binary complex int4 packed in uint8
```

CHIME `freq_id` and `chNNNN` labels are treated as data-product identifiers.
For example, CHIME `ch0844` maps to PilotProxy receiver-profile
`coarse_channel_index = 843`; the CHIME label is one-based while the PilotProxy
coarse-channel index is zero-based.

The adapter converts CHIME offset-binary int4 bytes to the detector kernel's
two's-complement packed complex int4 format. Absolute telescope timestamps are
ignored; files are concatenated in deterministic sorted segment order and
reported by `frame_index`.

## Weight Coordinate Convention

The CHIME profile has inverted spectral sense. The runner transforms CHIME data
into the normal detector coordinate by reversing each detector window before the
kernel sees it. Therefore the runner uses detector-coordinate weights, not raw
inverted-coordinate weights:

```text
weight_coordinate_system = post_spectral_sense_normalization
input_spectral_sense = inverted
input_requires_time_reversal = true
```

Use the shipped/default reference weight bank for this CHIME runner. Do not
generate a CHIME weight bank from `chime_dtv_fengine.json` unless the weight
manifest explicitly declares `post_spectral_sense_normalization`; otherwise the
pilot fine-bin offset can be flipped twice. The runner records the validated
weight convention in `run_config.json` and `stats.json`, and rejects raw
inverted-coordinate manifests when time reversal is active.

## Canonical Calibration Workflow

Use this order for local 10 s CHIME calibration and as the CANFAR template:

```text
chime-run
  -> validate-products
```

`K=128`, `reference_offset_bins=2`, `skipped_guard_bins=1`, and the shipped
detector-coordinate weights remain the baseline. The CHIME cleaning rule is
norm-corrected positive excess:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

Set the input directory explicitly for local or CANFAR mounts:

```bash
export PILOT_PROXY_CHIME_INPUT_DIR="$HOME/dataset/canfar_pilots_10s"
```

When the variable is unset, the local script defaults to
`$HOME/dataset/canfar_pilots_10s`. CANFAR jobs should set
`PILOT_PROXY_CHIME_INPUT_DIR` to the CANFAR input mount/path.

## Commands

Inspect:

```bash
PYTHONPATH=src python -m pilot_proxy.cli chime-inspect \
  --input-dir "$PILOT_PROXY_CHIME_INPUT_DIR" \
  --max-files 20 \
  --dataset-path baseband
```

Check the detector layout:

```bash
PYTHONPATH=src python -m pilot_proxy.cli check-layout \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --stream-map configs/stream_maps/chime_feed_pol_example.json \
  --frame-size-samples 16384 \
  --num-selected-channels 1
```

Positive-excess detector run. This is the current local/CANFAR pilot workflow;
it reads the real samples once and writes detector and mask products into one
run directory:

```bash
PYTHONPATH=src python -m pilot_proxy.cli chime-run \
  --input-dir "$PILOT_PROXY_CHIME_INPUT_DIR" \
  --physical-channel-range 14:36 \
  --frames-per-chunk 2 \
  --output-dir generated/chime_real/canfar_pilots_10s_positive_excess_full \
  --plot
```

Validate the combined products:

```bash
PYTHONPATH=src python -m pilot_proxy.cli validate-products \
  --run-dir generated/chime_real/canfar_pilots_10s_positive_excess_full \
  --output-json generated/chime_real/canfar_pilots_10s_positive_excess_full/product_validation.json
```

## Output Products

Each run directory contains:

```text
run_config.json
input_manifest.json
stats.json
chime_detector_outputs.npz
chime_spectrogram_cache.npz
chime_reductions_10s.npz
tables/
figures/
```

Generated metadata may contain absolute local paths for the receiver profile,
weights, kernel library, input manifest, and output directory. Those paths are
informational and may not be portable between local, CANFAR, and review
systems. Provenance SHA256 fields are the authoritative identity checks for
configuration, weights, kernel, and input artifacts.

`run_config.json` and `stats.json` use schema versions
`fstat_chime_run_config_v2` and `fstat_chime_stats_v2`. Both files include the
same `detector_contract` object with `schema_version =
pilotproxy_chime_detector_contract_v1`. That contract is the methods-level summary
of the K=128 detector geometry, positive-excess mask, power accumulator, and
all-row summation rule.

The detector output arrays are shaped `(num_frames, num_pilots)`. Baseband
before/after spectra use masked-frame exclusion; masked frames are not
zero-filled. Positive-excess runs write:

```text
tables/fstat_summary_by_pilot.csv
tables/snr_shelf_histogram_summary.csv
tables/mask_summary_by_pilot.csv
tables/spectrum_before_after.csv
figures/snr_shelf_histogram_by_pilot.png
figures/fstat_survival_by_pilot.png
figures/fstat_level_spectrogram.png
figures/baseband_spectrogram.png
figures/baseband_spectrum_before_after_mask.png
figures/mask_spectrogram.png
```

Reference placement is adaptive and auditable. The requested reference offset is
never silently weakened to a closer adjacent bin. If a requested reference leaves
the coarse-channel edge, it wraps around the circular coarse-channel FFT. If a
requested reference collides with the coarse-channel DC/forbidden tone, that one
reference shifts farther from the target. If the target bin itself collides with
the forbidden DC tone, weight generation fails hard because the target cannot be
moved without changing the signal under test.
The DC/forbidden-tone collision rule is:

```text
forbidden_tone = coarse_channel_dc
forbidden_tone_normalized = 0.5
forbidden_collision_rule = circular_normalized_distance <= 0.5 / detector_window_samples
```

Use `reference_offset_bins` for the internal/kernel value and
`skipped_guard_bins` for the human-readable gap:

```text
reference_offset_bins = skipped_guard_bins + 1
reference_offset_bins = 2  # shipped K=128 baseline
skipped_guard_bins = 1     # one skipped fine bin between target and reference
```

The weight manifest records placement status and warning fields such as:

```text
reference_placement_status
edge_reference_wrapped
dc_reference_collision
dc_reference_shifted
forbidden_tone_in_skipped_guard
placement_warnings
```

The manifest also records human-readable selected/requested offsets:

```text
target_offset_hz
detector_fine_bin_width_hz
lower_reference_offset_hz
upper_reference_offset_hz
lower_reference_relative_to_target_hz
upper_reference_relative_to_target_hz
lower_reference_requested_offset_hz
upper_reference_requested_offset_hz
lower_reference_requested_relative_to_target_hz
upper_reference_requested_relative_to_target_hz
```

`run_config.json` and `stats.json` copy a compact
`reference_placement_summary` from the weight manifest, including adaptive
channels, DC-shifted references, edge-wrapped references, skipped-guard DC
channels, and the forbidden-tone policy.

For the shipped K=128, reference-offset-2 baseline, DTV 21 wraps its lower
reference across the coarse-channel edge rather than falling back to adjacent
`-1,+1` references.

The SNR-shelf histogram summary distinguishes:

```text
num_detector_valid_frames     # reference denominator > 0
num_positive_excess_frames    # finite SNR shelf, equivalent to F > mu0
positive_excess_fraction
```

Figures use LaTeX styling (Computer Modern mathtext by default;
`PILOT_PROXY_USE_TEX=1` for full TeX rendering) and are written as 300 dpi
PNG, with `PILOT_PROXY_FIGURE_FORMATS=png,pdf` adding vector PDFs for the
manuscript. Core figures include a finite \(\mathrm{SNR}_{\mathrm{shelf}}\) histogram,
F-statistic survival curves, F-statistic level spectrograms, baseband
before/after spectra, baseband spectrograms, and mask spectrograms. The
histogram x-axis is only \(\mathrm{SNR}_{\mathrm{shelf}}\): the top panel spans
\([-90, 0]\) dB, and the lower panel zooms to \([-90, -25]\) dB to cut off the
strong channel-30 outlier. The histogram is
a probability density over finite \(\mathrm{SNR}_{\mathrm{shelf}}\) values for
each pilot channel: counts are divided by the number of finite shelf-SNR
samples and by the bin width. Frames with \(F\leq1\) have no finite shelf-SNR
value, so their contribution is recorded separately by the
`positive_excess_fraction` column in
`tables/snr_shelf_histogram_summary.csv`.
Spectrogram plots use
relative time on the bottom axis with frame index on the top axis, and CHIME
coarse-channel frequency on the left axis with DTV physical channel labels on
the right axis. The F-statistic survival and F-statistic level spectrogram
figures include a second lower panel excluding the strong DTV-30 outlier. The
10-second local spectrograms include an explicit 10 s tick, and mask
spectrogram colorbars are discrete with only \(M=0\) and \(M=1\).
The SNR-shelf histogram intentionally uses only
\(\mathrm{SNR}_{\mathrm{shelf}}\) on the x-axis because the y-axis is a
probability density per shelf-SNR dB. The F-statistic survival plots keep
\(R_F=10\log_{10}F\) on the bottom axis and show the corresponding finite
\(\mathrm{SNR}_{\mathrm{shelf}}\) axis values on the top axis.

Generated CHIME analysis plots are written as high-DPI PNG files. The run
products do not write PDF or JPEG copies.
