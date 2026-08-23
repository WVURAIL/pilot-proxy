# Current-geometry synthetic sensitivity study

This document specifies the staged experiment implemented by
`tools/current_geometry_sensitivity.py`. It is the experiment to use when the
scientific question is not merely whether the detector runs, but how much
sensitivity is lost as the ideal calculation is successively replaced by the
ATSC waveform model, signed-int4 inputs and weights, the frozen integer
transform, the Q16 decision, and the selected CUDA artifact.

The existing `pilot-proxy evaluate-snr` command is retained for its original
single-configuration response and regression uses. The study described here
does not change that command's interface or historical interpretation. In
particular, a small `evaluate-snr` example must not be relabeled as a
current-geometry fixed-versus-floating sensitivity measurement.

## Scientific question and scope

For a fixed synthetic-null false-alarm probability, the study estimates the
probability of detecting a generated ATSC 8-VSB pilot as a function of the
requested input data-shelf SNR. It reports detection curves for each ablation
stage and brackets the SNR at which a requested detection probability is
crossed. The primary comparison is the SNR required by the exact fixed/Q16
candidate minus the SNR required by the ATSC ideal-floating detector. A
positive difference therefore means a sensitivity loss in the fixed-point
candidate.

This is a conditional synthetic result. The ATSC signal passes through the
reference real-ADC and four-tap, 2048-point PFB model, retaining the
positive-frequency 1024 channels. Noise at the selected PFB output is
independent circular complex Gaussian, normalized by a deterministic PFB
noise-gain calculation. That model isolates implementation loss under
controlled conditions; it does not reproduce all cross-stream, frequency, or
time covariance in CHIME baseband. It also does not establish an on-sky
threshold, an astrophysical completeness function, or the performance of a
deployed Kotekan path.

The current scientific geometry is 2048 simultaneous input streams, 128
detector samples per window, 128 windows per stream, and a two-times-padded
256-bin fine transform. Every output row records `num_input_streams` and a
`scientific_scope`. Any run with fewer than 2048 streams is labeled
`reduced_stream_code_validation_not_scientific_evidence`; the program requires
an explicit `--allow-reduced-geometry` acknowledgement. Smoke results validate
code and parity only and must not be quoted as sensitivity evidence.

Production does **not** materialize a 2048-by-16384 packed input for every
Monte Carlo point. A literal run at the default counts would require about
1.29 million full frames: 46 channel/offset profiles multiplied by 4000 null
trials plus 16 SNR points multiplied by 1500 H1 trials. Re-measured on an RTX
5000 Ada on 2026-08-23, a full frame costs **about 3.8 seconds** (20 literal
null trials in 75.8 s; the 480-trial audit in 1734 s agrees at 3.6 s), so a
serial literal sweep would take roughly 55--60 GPU-days before reporting
overhead. The earlier 6--8 second figure quoted here was about twice that, and
scoping decisions taken against it were correspondingly pessimistic. A full
literal sweep is still not a credible default dissertation workflow, but a
*targeted* one is far cheaper than this section used to imply.

The production default is therefore `--simulation-backend
sufficient-statistic`. It evaluates a deterministic common-noise pool of 256
independent streams through every stated signal, quantization, weight, and
transform ablation. It retains each stream's three 256-bin fine powers at the
exact additive boundary immediately before the stream sum. For each nominal
2048-stream trial it uses a multivariate Gaussian multiplier-CLT draw fitted
to those per-stream vectors. One multiplier vector is shared across every
fine bin and every ablation, which preserves the empirical cross-bin and
cross-stage covariance represented in the pool. The fixed-stage draw is
clipped at zero if necessary, rounded to uint64, and only then passed to the
exact rational and Q16 comparisons.

This acceleration is an approximation, not a disguised full-frame run. The
report exposes the pool size, backend, maximum negative-power clipping
fraction, and conditional uncertainty statement. Wilson and paired-bootstrap
intervals describe repeated aggregate draws conditional on the finite pool;
they do not include uncertainty from estimating the per-stream distribution.
Additional shards with distinct base seeds provide independent pools and are
preferable when the production result is sensitive to pool realization.

The approximation is guarded by a declared literal full-frame audit. The
default audit is stratified over channels 14, 25, and 36, both centered and
half-bin offsets, and SNRs -54, -50, -46, and -42 dB. Each audit point uses an
actual 2048-by-16384 packed frame and the selected CUDA artifact. The report
compares full and accelerated response distributions with a declared
two-sample KS rule, requires the float and fixed-Q16 Wilson 95% intervals to
overlap at every audit SNR, and requires zero CPU/GPU power and decision
mismatches. It also requires at least 16 full-frame null trials and 16 trials
at each declared H1 point. These are approximation gates, not a claim that 16
trials alone measure `P_d` precisely.

## Paired experimental design

The experiment uses common random numbers. A trial seed is the SHA-256 digest
of the declared base seed, purpose, physical channel, frequency offset, and
trial index. SNR and ablation stage are deliberately absent. Consequently, a
given trial identity has the same noise realization at every SNR and in every
stage. This reduces Monte Carlo noise in stage differences without making
trials statistically independent when they share an identity.

The paired bootstrap respects those identities. It intersects explicit trial
keys across every SNR, resamples one common set of H1 identities for the whole
curve, and applies the same resampled indices to the float and fixed paths. It
also resamples paired null trials because the threshold is estimated rather
than known. A bootstrap replicate contributes to a sensitivity-loss interval
only when both resampled curves bracket the requested detection probability.
The report includes the number and fraction of valid bracketed replicates.

Each curve point also includes a binomial Wilson 95% interval. A crossing is
formed only between adjacent sampled SNR values whose observed detection
probabilities straddle the target. The program does not extrapolate. If either
the float or fixed curve lacks a bracket, both the point sensitivity-loss
claim and its bootstrap interval are withheld. A wider or finer SNR grid must
then be run.

## Ablation stages

The stages differ in one or a small number of explicit operations. They share
the same noise and signal amplitude within each trial identity. In the
accelerated backend, each stage is evaluated literally through its per-stream
fine powers and only the subsequent 2048-stream sum is modeled. In the audit
backend, the entire 2048-stream packed frame is evaluated literally.

| Stage | Signal and arithmetic represented |
|---|---|
| `ideal_tone_float` | An analytic coherent pilot tone at the predicted fine bin, with amplitude measured from the channelized 8-VSB pilot; unquantized input, ideal complex weights, and a complex128 padded FFT. This is an upper-bound signal model, not the deployed detector. |
| `atsc_8vsb_float` | The complete generated 8-VSB waveform after the reference channelizer, with unquantized input, ideal complex weights, and the floating transform. |
| `atsc_8vsb_input_int4_float` | The preceding stage with signed 4+4-bit input quantization and dequantization, leaving weights and subsequent arithmetic ideal. |
| `atsc_8vsb_weight_int4_float` | The floating 8-VSB input with the shipped signed 4+4-bit weights, dequantized before floating matched filtering. |
| `atsc_8vsb_joint_int4_float_transform` | Both input and weight quantization, followed by floating matched filtering, transform, accumulation, and decision statistic. |
| `atsc_8vsb_fixed_transform_float_decision` | Packed inputs and weights, exact integer row projections, the frozen `fxfft256` transform, exact uint64 accumulation, and an unquantized threshold multiplier. |
| `atsc_8vsb_fixed_transform_q16_cpu` | The exact fixed transform with the null-calibrated multiplier rounded upward to Q16 and evaluated by integer cross-products on the CPU. |
| `atsc_8vsb_packed_gpu_fine_exact_q16` | The selected CUDA artifact's packed path in full-frame smoke/audit shards, with every fine-power result and decision checked against the exact CPU reference. Sufficient-statistic shards instead parity-check the GPU sum over their literal per-stream pool and do not emit a simulated GPU decision curve. |

The comparison between adjacent stages localizes a loss; it does not prove
that all effects are statistically additive. The primary fixed-minus-float
quantity compares `atsc_8vsb_fixed_transform_q16_cpu` with
`atsc_8vsb_float`. The GPU stage is an implementation-equivalence check on the
same exact candidate, not an additional numerical approximation that should
produce a separate sensitivity curve.

## Frequency profiles and coordinates

Production defaults cover all 23 shipped physical-channel profiles, channels
14 through 36. They include a centered pilot and a half-fine-bin offset. The
channel-14 profile deliberately exercises its circularly wrapped reference;
the report records `channel14_circular_wrap_profile`.

The reference rFFT channelizer already emits the
`post_spectral_sense_normalized` coordinate consumed by the current weight
bank. The synthetic study therefore does not reapply either the CHIME raw
input time reversal or the legacy archive lower-edge phase conversion. Cache
creation independently predicts the fine-bin anchor from geometry, measures
the strongest clean pilot line, and refuses to proceed unless the line lies
inside the designated circular bin set. This gate prevents a coordinate
mistake from silently becoming a sensitivity result.

## Threshold definition

Null trials are generated separately from H1 trials. For every floating stage,
the threshold is a conservative observed order statistic of the null response
ratio at the requested `P_fa`. The fixed threshold starts from the fixed
transform's null ratio and is rounded upward to Q16. Fixed decisions are then
evaluated with Python integer cross-products over the stored exact numerator
and denominator fields rather than a rounded floating reconstruction.

The default requested `P_fa` is 0.001. At least 1000 null trials are needed
for even one expected tail event, and materially more are preferable for a
stable operating point. The default production plan uses 4000 null trials and
1500 H1 trials per SNR/profile combination. Those counts are planning values,
not an automatic guarantee of adequate precision. The emitted null
exceedances, Wilson intervals, curve brackets, and bootstrap valid fraction
must be inspected before a dissertation number is accepted.

This calibration exists only for the synthetic study. It does not activate a
fine mask, replace the archive/live bundle calibration, or establish a
per-channel on-sky false-alarm rate.

## Kernel provenance and execution forms

Pass the exact artifact being evaluated with `--lib-path` on every command.
The immutable study configuration records its resolved path, byte count, and
SHA-256 digest; every GPU shard also records the loaded kernel version,
execution form, and digest. Results from a separately built current-source
library must use a different output directory and retain their own artifact
hash. They must not be pooled with results from a supplied binary merely
because the exported API is compatible.

Kernel 2.3 and later can execute fused fine powers and the Q16 epilogue on the
device. Kernel 2.1 exposes exact row projections and fine powers but not the
fused Q16 epilogue; for that artifact the study composes those two GPU calls
and applies the same exact Q16 comparison on the host. The report labels these
forms respectively as
`fused_gpu_fine_powers_and_device_q16_epilogue` and
`composed_gpu_row_projections_and_fine_powers_plus_host_exact_q16_decision`.
Both forms require bit-for-bit agreement with CPU fine powers and decisions on
every literal full-frame trial. Accelerated shards additionally check the GPU
fine-power sum over the literal input pool, and label that narrower validation
scope separately. A pool parity check is not counted as a 2048-stream
full-frame audit trial.

An unrelated or historical binary can be recorded without loading it by
using `--historical-kernel-artifact`. This is provenance only. A binary passed
to `--lib-path --gpu` is dynamically loaded and executed, so its source,
symbols, dependencies, architecture metadata, and digest must be inspected
before the run.

## Reproducible workflow

Generate and audit a deterministic clean waveform. The waveform and audit are
reproducible run inputs and are ignored build products; do not commit the
multi-megabyte IQ file.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 \
  -m pilot_proxy.testbench.generate_atsc_signal \
  --output-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --output-ts generated/atsc/atsc_payload.ts \
  --num-iq-samples 600000 --seed 12345

PYTHONPATH=src /usr/bin/python3 \
  -m pilot_proxy.testbench.audit_atsc_signal \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --output-json generated/atsc/atsc_waveform_audit.json \
  --fail-on-quality
```

Run a reduced smoke test first. The explicit reduced-geometry flag is
required, and the resulting report remains non-scientific even if all parity
checks pass.

```bash
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  --mode smoke --stage smoke --allow-reduced-geometry --gpu \
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
  --waveform-audit generated/atsc/atsc_waveform_audit.json \
  --lib-path /absolute/path/to/libfstatistic.so \
  --output-dir generated/current_geometry_sensitivity_smoke
```

For production, define one shell array of arguments and reuse it unchanged so
every phase binds to the same waveform, weight bank, profile grid, SNR grid,
and kernel hash. The program writes an immutable `study_config.json` and
refuses to mix incompatible shards.

```bash
study_args=(
  --mode production
  --input-iq generated/atsc/atsc_8vsb_complex64.cfile
  --waveform-audit generated/atsc/atsc_waveform_audit.json
  --lib-path /absolute/path/to/libfstatistic.so
  --output-dir results/current_geometry_sensitivity
)

PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage prepare
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage null --gpu
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage calibrate
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage sweep --gpu
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage audit --gpu
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage report
```

The production defaults declare all channels 14--36, offsets 0 and 0.5 fine
bins, 2048 streams, and the planned `-60` through `-30` dB grid in 2 dB
increments. It uses the 256-stream sufficient pool described above, while
still reporting `num_input_streams=2048` because the modeled additive sum has
the current geometry. The declared grid, acceleration method, pool size, and
audit design form part of the immutable study identity. Large runs can
execute a subset without changing that identity:

```bash
PYTHONPATH=src python3 tools/current_geometry_sensitivity.py \
  "${study_args[@]}" --stage sweep --gpu \
  --run-physical-channel 14 --run-offset-fine-bins 0.5 \
  --sweep-snr-db -50 --trials 250 --trial-start 0 --seed 20260820
```

The default `audit` stage runs the three declared channels, two offsets, one
null point, and four H1 points, for 480 literal full-frame trials. Restricting
`--run-physical-channel` does not weaken this audit declaration. To split the
audit over jobs, declare the same study and set explicit
`--run-audit-physical-channel`, `--run-audit-offset-fine-bins`, and
`--run-audit-snr-db` execution subsets while leaving the declared
`--audit-*` design unchanged, then use non-overlapping `--trial-start` ranges.
The report will not mark the audit complete until the entire immutable
declaration has the required count.

On the local RTX 5000 Ada validation host, one 256-stream sufficient shard
with 4000 null trials took 3.3 seconds, and one with 1500 H1 trials took 2.1
seconds after signal-cache preparation. These are engineering measurements,
not scientific results. They imply roughly 30--40 minutes for the primary 46
profile/16-SNR sweep on that host, before the paired bootstrap report. The
default 480-trial literal audit is approximately another **29 minutes** at the
re-measured 3.8 seconds per frame (observed 2026-08-23: 1734 s). The
2000-replicate paired report adds a further 29 minutes on the same host.
Record the emitted per-shard wall times on the final host rather than treating
these estimates as guaranteed throughput.

Two cautions learned on 2026-08-23. The accelerated backend's response is
**conditional on the finite pool in a way that matters**: re-drawing the
256-stream pool via `--seed` shifted median responses by −11% to +14%, with the
sign varying by channel, where the literal full-frame path moved by ~1%. And
`PFB_GAIN_SEED_BASE` is *not* the pool knob --- it seeds only the converged
`pfb_noise_gain` normalization, which is common to both paths and cancels out of
the full-vs-primary comparison. Use `--seed` for a second pool realization.

`--simulation-backend full-frame` remains available for an explicitly bounded
run. Do not combine it with the default million-frame trial plan. Use it for
smoke tests, the declared audit, and targeted follow-up when the audit exposes
a discrepancy.

Additional non-overlapping shards use a different `--trial-start`, base seed,
or both. Exact duplicate trial identities are rejected during aggregation.
`--resume` is enabled by default and preserves an existing shard instead of
overwriting it. The report may be regenerated as shards accumulate; missing
profiles, missing SNR points, and unbracketed crossings remain visibly
incomplete.

## Outputs and dissertation acceptance

The run directory contains:

- `study_config.json`, the immutable scientific configuration and input
  provenance;
- `signal_cache/*.npz`, the reference-channelized 8-VSB signal and analytic
  tone for each profile/offset;
- `shards/null` and `shards/h1`, resumable primary trial responses, exact
  integer fields, identities, clipping fractions, sufficient-pool diagnostics,
  and pool-level GPU parity evidence;
- `shards/audit_null` and `shards/audit_h1`, literal 2048-stream packed-frame
  responses and per-trial CPU/GPU parity evidence for the declared audit;
- `calibration.json`, stage-specific synthetic-null thresholds and Q16
  rounding;
- `sensitivity_curves.csv`, trial counts, detections, Wilson intervals,
  geometry, and parity counts for every observed point; and
- `sensitivity_report.json`, crossing brackets, paired bootstrap intervals,
  kernel execution forms, scope labels, conditional-pool diagnostics,
  full-versus-accelerated KS and Wilson comparisons, and claim status.

A dissertation table or figure should identify the waveform digest, weight
digest, kernel digest and execution form, physical channels, frequency
offsets, 2048-stream geometry, null and H1 counts, requested and empirical
false-alarm rates, and common-random-number/bootstrap method. It should plot
all staged curves needed to identify the source of loss, with Wilson
intervals, and state explicitly when a crossing is merely bracketed between
sampled SNRs.

It must also state that the production curve uses the multiplier-CLT
sufficient-statistic backend, give the per-stream pool size and seed coverage,
report the negative-power clipping bound, and tabulate the literal audit's
channels, offsets, counts, KS comparisons, Wilson overlaps, artifact digest,
and parity results. Omitting that distinction would incorrectly describe an
accelerated conditional model as 1.29 million direct CUDA evaluations.

Do not promote `claim_status=preliminary_or_code_validation` to a scientific
result. Even `eligible_for_precision_review` is a prompt for human review, not
an automatic publication assertion. Acceptance requires zero CPU/GPU parity
mismatches, adequate tail resolution, adequate Wilson precision through the
transition, both requested crossings bracketed for both primary curves, a
useful fraction of bracketed bootstrap replicates, stable conclusions under a
finer local SNR grid, negligible negative-power clipping, a complete literal
audit that passes the predeclared KS and Wilson-overlap rules, and a written
statement of the synthetic model's limits. The code deliberately withholds
`eligible_for_precision_review` when any one of those machine-checkable
preconditions is absent.
