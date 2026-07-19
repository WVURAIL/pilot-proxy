# PilotProxy v0.2 CHIME Real-Data Adapter

> **Status:** Historical adapter notes. For new archive-scale CADC/CANFAR work,
> use `pilot-proxy chime-scan` and `docs/CANFAR_RUNBOOK.md`. We retain the
> `chime-run` workflow for already-staged HDF5 data and regression comparisons.

The v0.2 adapter connects segmented CHIME HDF5 baseband data to the existing
fixed-point detector. It does not change the CUDA kernel contract:

```text
K = 128 detector samples
N = 3 weight terms
reference_offset_bins = 2 nominal
skipped_guard_bins = 1 nominal
packed complex int4 input
uint64 power accumulation
```

The synthetic testbench can apply an explicit shelf-SNR threshold. The CHIME
real-data path does not use that table. It forms the validity flag and the
norm-corrected positive-excess mask from the exact integer powers:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * ref_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This comparison is the integer form of `F > mu0`, where
`mu0 = 2*target_norm_sq/ref_norm_sum_sq`. The value `mu0` is the flat-floor
reference implied by the quantized weight norms; it is not assumed to be one.
See `docs/METHOD_SPEC.md` for the statistic and its calibration.

`reference_offset_bins` measures the target-to-reference separation in detector
fine bins. `skipped_guard_bins` counts the fine bins between them. Therefore:

```text
skipped_guard_bins = reference_offset_bins - 1
```

For each selected pilot, the adapter follows this data path:

```text
CHIME HDF5 segments
  -> normalized block (num_input_streams, 1, samples)
  -> packed detector input (frames, detector_rows_per_frame, 128)
  -> CUDA numerator/denominator detector
  -> one F-statistic per frame per selected DTV pilot channel
```

## Data Contract

The helper script `scripts/run_chime_local_calibration.sh` reads
`PILOT_PROXY_CHIME_INPUT_DIR` and otherwise uses
`$HOME/dataset/canfar_pilots_10s`. The `pilot-proxy chime-run` command itself
does not read that variable; it always requires `--input-dir`.

```text
$PILOT_PROXY_CHIME_INPUT_DIR    # e.g. $HOME/dataset/canfar_pilots_10s
```

The original data inspection found 23 DTV pilot channels in directories
`ch0844` through `ch0506`, corresponding to physical channels 14 through 36.
That observation came from the staged dataset and is not reconstructed by the
unit tests. The inspected files had:

```text
dataset path: /baseband
shape: (time, input)
dtype: uint8
axis attr: ["time", "input"]
num input streams: 2048
frequency/channel: one CHIME coarse channel per directory
encoding: CHIME native offset-binary complex int4 packed in uint8
```

The adapter treats the CHIME `freq_id` and `chNNNN` label as data-product
identifiers. In the current receiver profile, `ch0844` corresponds to
`coarse_channel_index = 843`: the CHIME label is one-based and the PilotProxy
profile index is zero-based. The receiver profile is marked
`example_requires_data_product_verification`, so this coordinate relation must
still agree with the metadata of the dataset being processed.

The adapter converts each CHIME offset-binary int4 byte to the detector's
two's-complement packed complex int4 representation. `chime-run` concatenates
files in deterministic sorted segment order, ignores absolute telescope time,
and reports the resulting position as `frame_index`.

The software profile and the commands below use a 16,384-sample analysis frame.
This value is the profile for the planned CHIME engine-upgrade frame and should
not be presented as a measurement of the currently deployed correlator. A
12,288-sample current-frame value has been discussed but remains provisional
until it is checked against an authoritative CHIME data product or upgrade
document. In all cases, record the frame length used for the run and require it
to be divisible by `K = 128`.

## Weight Coordinate Convention

The CHIME receiver profile declares inverted spectral sense. The runner reverses
each detector window before the CUDA kernel, which places the samples in the
detector frequency coordinate. It must therefore use weights defined in that
same coordinate:

```text
weight_coordinate_system = post_spectral_sense_normalization
input_spectral_sense = inverted
input_requires_time_reversal = true
```

Use the shipped detector-coordinate weight bank with this runner. A newly
generated CHIME bank is acceptable only when its manifest declares
`post_spectral_sense_normalization` and the matching time-reversal preprocessing.
Otherwise the pilot offset can be reversed twice. The runner records the
effective convention in `run_config.json` and `stats.json` and rejects a raw
input-coordinate manifest when time reversal is active.

## Legacy staged-data workflow

For a local staged dataset, run the detector and then validate the products:

```text
chime-run
  -> validate-products
```

The baseline remains `K = 128`, `reference_offset_bins = 2`,
`skipped_guard_bins = 1`, and the shipped detector-coordinate weights. The mask
is the norm-corrected positive-excess comparison shown above.

Set the local input path explicitly:

```bash
export PILOT_PROXY_CHIME_INPUT_DIR="$HOME/dataset/canfar_pilots_10s"
```

The helper script uses this value. A CANFAR job should point it at the mounted or
staged CANFAR directory.

## Commands

Inspect the staged files before running the detector:

```bash
PYTHONPATH=src python -m pilot_proxy.cli chime-inspect \
  --input-dir "$PILOT_PROXY_CHIME_INPUT_DIR" \
  --max-files 20 \
  --dataset-path baseband
```

Check the proposed 16,384-sample upgrade layout:

```bash
PYTHONPATH=src python -m pilot_proxy.cli check-layout \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --stream-map configs/stream_maps/chime_feed_pol_example.json \
  --frame-size-samples 16384 \
  --num-selected-channels 1
```

Run the positive-excess detector over physical channels 14 through 36. This
command reads the staged samples once and writes the frame products and mask to
one run directory:

```bash
PYTHONPATH=src python -m pilot_proxy.cli chime-run \
  --input-dir "$PILOT_PROXY_CHIME_INPUT_DIR" \
  --physical-channel-range 14:36 \
  --frames-per-chunk 2 \
  --output-dir generated/chime_real/canfar_pilots_10s_positive_excess_full \
  --plot
```

Validate the combined files:

```bash
PYTHONPATH=src python -m pilot_proxy.cli validate-products \
  --run-dir generated/chime_real/canfar_pilots_10s_positive_excess_full \
  --output-json generated/chime_real/canfar_pilots_10s_positive_excess_full/product_validation.json
```

## Output Products

The staged-data runner writes:

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

The archive-scale `chime-scan` path additionally writes authoritative per-pilot
products and `chime_integrated_spectra.npz`; see `docs/product_schema_v2.md` and
`docs/DATA_PRODUCTS.md`.

Generated metadata can contain absolute paths for the receiver profile, stream
map, weight bank, kernel library, input manifest, and output directory. Those
paths describe the machine that produced the run and may not be portable. Use
the corresponding SHA-256 fields to compare artifact contents across local,
CANFAR, and review systems.

`run_config.json` and `stats.json` use schema versions
`fstat_chime_run_config_v2` and `fstat_chime_stats_v2`. Both carry a matching
`detector_contract` object with schema
`pilotproxy_chime_detector_contract_v1`. The contract records the `K = 128`
geometry, the positive-excess rule, the `uint64` accumulator, and the
all-row-sum statistic.

The frame arrays have shape `(num_frames, num_pilots)`. Before/after baseband
summaries exclude invalid frames, and the after-mask value averages only frames
with `mask = 0`; rejected frames are not replaced with zeros.

After `--plot`, the staged-data run contains:

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

Reference placement is adaptive and recorded in the weight manifest. A
reference that crosses a coarse-channel edge wraps on the circular FFT grid. A
reference that collides with the forbidden coarse-channel DC tone moves farther
from the target. The algorithm does not silently reduce the requested offset to
the adjacent fine bin. If the target itself collides with the forbidden tone,
weight generation stops because moving the target would change the signal being
tested.

The collision rule is:

```text
forbidden_tone = coarse_channel_dc
forbidden_tone_normalized = 0.5
forbidden_collision_rule = circular_normalized_distance <= 0.5 / detector_window_samples
```

The internal offset and the human-readable gap remain related by:

```text
reference_offset_bins = skipped_guard_bins + 1
reference_offset_bins = 2  # shipped K=128 baseline
skipped_guard_bins = 1     # one skipped fine bin between target and reference
```

The manifest records the placement status and warnings:

```text
reference_placement_status
edge_reference_wrapped
dc_reference_collision
dc_reference_shifted
forbidden_tone_in_skipped_guard
placement_warnings
```

It also records the requested and selected offsets:

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
`reference_placement_summary`, including the channels with adaptive placement,
DC shifts, edge wraps, or a forbidden tone in the skipped guard. For the shipped
`K = 128`, offset-2 bank, DTV 21 wraps its lower reference across the
coarse-channel edge rather than substituting `-1,+1` references.

The SNR-shelf table uses legacy column names that need careful interpretation:

```text
num_detector_valid_frames
num_positive_excess_frames
positive_excess_fraction
mask_fraction
```

`num_positive_excess_frames` is currently the number of finite
`snr_shelf_db` values. Because `10*log10(F - 1)` is finite only for `F > 1`,
this count and `positive_excess_fraction` describe `F > 1`, not the
norm-corrected mask when `mu0 != 1`. The stored mask follows `F > mu0`.
`mask_fraction` is the table's direct mean of the stored binary mask over all
frames. Use `mask_summary_by_pilot.csv` when the valid-frame denominator must be
explicit.

## Injection-recovery and cleaning tradeoff

Four publication-analysis commands operate on the staged files or the
archive-scale products. Their procedures are in
`docs/PUBLICATION_VALIDATION.md`.

```text
pilot-proxy inject-pilot-tone
pilot-proxy analyze-cleaning-tradeoff
pilot-proxy analyze-injection-recovery
pilot-proxy chime-combine
```

`inject-pilot-tone` works in the file's native offset-binary 4+4-bit integer
domain, with component range `[-8, 7]`. A zero-amplitude pass preserves the
baseband bytes, and a nonzero pass records saturation while preserving sibling
datasets, attributes, and filenames. Therefore the output can be read by
`chime-scan --source local`.

`analyze-cleaning-tradeoff` evaluates
`tau = mu0 * 10^(x/10)` from the stored integer powers and weight norms. The
`x = 0` point must reproduce the stored mask before the remaining thresholds are
interpreted. The command then reports the masked-fraction and residual-power
operating curve.

The plotting code uses Computer Modern mathtext by default. Set
`PILOT_PROXY_USE_TEX=1` for full TeX rendering. The default output is a 300 dpi
PNG; `PILOT_PROXY_FIGURE_FORMATS=png,pdf` adds a vector PDF copy. JPEG output is
not implemented.

The current SNR-shelf figure uses an `[-90, 0]` dB overview and an
`[-90, -25]` dB detail panel. Each channel histogram is normalized over its
finite shelf-SNR samples and by bin width. Frames with `F <= 1` therefore do not
enter the histogram and must be counted separately.

The spectrograms place relative data time and frame index on the horizontal
axes, with CHIME coarse-channel frequency and DTV physical channel on the
vertical axes. The F-statistic survival and level-spectrogram figures include a
second panel without DTV 30. The 10 s local plots include a 10 s tick, and the
mask colorbar is discrete at `M = 0` and `M = 1`. These are reporting choices;
the NPZ products remain the numerical record.
