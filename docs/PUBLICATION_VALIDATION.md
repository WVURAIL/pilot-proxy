# Publication validation runbook

This runbook defines the five checks required for a publication result. The
repository contains the analysis commands. The remaining work is to run the
GPU and on-sky cases, retain their products, and apply the acceptance tests
below. Until a check passes, its associated manuscript number is provisional.

| # | Item | Depends on | Effort | Tooling |
|---|------|-----------|--------|---------|
| 1 | On-sky norm-corrected mask validation | -- | ~1 GPU day (incl. control) | shipped (runbook + `validate-products`) |
| 2 | Detection-rate curves at publication trial counts | -- | hours (parallel with 1) | shipped (`evaluate-snr` Wilson fields) |
| 3 | Injection-recovery on real baseband | 1 | ~1-2 days of runs after preflight | injection and analysis commands shipped; realized-power coordinate must be added before the publication run |
| 4 | Radiometer baseline comparison | 3 | free after item 3 | same command and same realized-power precondition as item 3 |
| 5 | Cleaning tradeoff in science terms | 1 (products) | hours (post-hoc) | shipped (`analyze-cleaning-tradeoff`) |

### Running these checks during a survey

The checks do not require the production scan to finish. Each invocation uses
its own staging directory, and each per-pilot checkpoint is installed by an
atomic rename. Thus, a visible `work/<freq_id>.npz` is a complete checkpoint.
Use a separate output directory for every bounded run and coordinate CADC
access with the active survey rather than assuming its load is negligible.

While a survey is running:

- Run item 2 and the item 1 control scan on a second GPU session. Run the
  item 3--4 ladder only after its realized-power preflight is closed.
- The DTV 18 and DTV 20 zero-point checks in item 1 can use a snapshot of the
  survey checkpoints. Copy `work/<freq_id>.npz` into `snapshot/`, then run:

  ```bash
  pilot-proxy chime-combine --product snapshot/<freq_id>.npz \
    --output-dir snapshot_combined/<freq_id>
  pilot-proxy validate-products --run-dir snapshot_combined/<freq_id>
  ```

- Item 5 can be rehearsed on a completed channel snapshot with
  `analyze-cleaning-tradeoff --run-dir snapshot_combined/<freq_id>`. The
  publication result must still use the declared final survey set.
- `chime-combine` requires a common event set across channels. It rejects a
  complete channel combined with a partial channel and prints a frame-grid
  diagnostic. Combine compatible completed channels, or process one channel
  at a time.

Close item 1 before interpreting downstream on-sky results because it tests
the corrected mask rule. Item 2 is independent. Items 4 and 5 operate on
stored products and do not require a new detector run.

For items 1 and 3, use the A100, `setup_env.sh`, and CADC-certificate setup in
`docs/CANFAR_RUNBOOK.md`. For camera-ready figures, set
`PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf`.

---

## 1. On-sky validation of the norm-corrected mask

This test asks whether the real-data `H0` distribution follows each channel's
`mu0` rather than 1. It also measures whether the corrected comparison gives
an `H0` mask fraction consistent with approximately 0.5 for the tested
channels. This is the on-sky counterpart of
`tests/core/test_mask_zero_point.py` and the runbook's "H0 zero-point check"
section.

1. **Export and validate the runtime bundle.** This establishes that the
   runtime profiles carry the rational correction `nt:(nl+nu)`:

   ```bash
   pilot-proxy export-runtime-weight-bundle \
     --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
     --weight-coordinate-system post_spectral_sense_normalization \
     --physical-channel-range 14:36 \
     --output-dir bundles/norm_corrected
   pilot-proxy validate-runtime-weight-bundle --bundle-dir bundles/norm_corrected
   ```

   Accept the bundle only when the validator reports `valid` and every row in
   `pilot_profiles.json` contains `target_norm_sq`, `ref_norm_sum_sq`, `mu0`,
   and `positive_excess_half_threshold_num/den`.

2. **Choose three `freq_id` values** from the transmitter census in the calibration
   report:
   - **Control:** use a normal DTV pilot profile whose nominal ATSC pilot is
     inside the selected coarse-channel passband, but for which the 500-mile
     census lists no station on that physical channel. This is a census-based
     selection, not a propagation prediction. Do not select an arbitrary
     pilot-free coarse channel: the detector analyzer rejects a target outside
     the passband.
     Inspect several staged files with
     `pilot-proxy chime-inspect --input-dir <staged>` before running the
     detector. This is a pilot-free control for the tested sample, not a
     proof that the channel is always free of RFI.
   - **Extreme-`mu0` quiet channels:** DTV 18 (largest shipped `mu0`, 1.0111)
     and DTV 20 (smallest, 0.9853). These are where the legacy rule pinned
     most strongly, so they provide the clearest comparison. The values are
     rounded from an independent recomputation over the shipped manifest;
     whether the selected intervals are quiet is determined from the data.

3. **Run bounded scans** using the pre-production pattern:

   ```bash
   pilot-proxy chime-scan --inventory-name chime-pilots --select <freq_id> \
     --max-files 50 --checkpoint-every 10 --output-dir runs/h0check_<freq_id>
   pilot-proxy validate-products --run-dir runs/h0check_<freq_id>/<...>
   ```

   Aim for at least 1000 valid frames per channel. The `--max-files 50` bound
   is an initial allocation; use the measured valid-frame count rather than
   assuming it is sufficient.

4. **Apply the acceptance criteria** to each channel using `stats.json` and
   `chime_detector_outputs.npz`. The numerical bounds below are planned
   acceptance criteria, not existing measurements:
   - `mu0_by_pilot` matches the weight-manifest value exactly.
   - For the control and quiet channels, require
     `|mean(fstat_raw[valid]) - mu0| < |mu0 - 1| / 3`. Record the frame count
     and measured standard error. The approximation
     `sqrt(1.5/rows_per_frame)/sqrt(nframes)` is a planning estimate, not a
     substitute for the measured uncertainty.
   - Require a mask fraction of **0.45--0.55** over valid frames for the
     control and for quiet-channel intervals. If compatible pre-correction
     products exist, use the same channels in the before/after panel.
   - `validate-products` passes (it now checks the *declared* corrected
     rule with exact integer math).
   - For a DTV-loud interval, report the mask fraction and distribution of
     `pilot_excess_corrected`. Do not replace these measured values with
     qualitative terms such as "well above" or "strongly positive" in the
     paper.

5. **Exercise resume once.** Stop one channel mid-scan, resume it, and verify
   that provenance accepts the recorded norms and rule. Compare its combined
   product with an uninterrupted rerun using the same inputs.

The paper artifact is a two-panel figure. The first panel compares measured
`mean F` with `mu0` and includes `F = 1` for reference. The second reports the
`H0` mask fraction before and after the correction for matched data where
available.

---

## 2. Detection-rate curves at publication trial counts

This test measures `P_d` as a function of shelf SNR for each frequency offset.
We evaluate both the -32 dB science threshold and the positive-excess rule,
and we report Wilson 95% intervals. The summary already emits
`positive_excess_detection_rate`, `threshold_detection_rate`, and their
`*_wilson95_lo/hi` fields. The publication run increases the trial count.

1. On the GPU node, use the audited golden ATSC capture, or regenerate and
   audit it with
   `pilot-proxy generate-atsc ...` then `pilot-proxy audit-atsc`.

2. First sweep the threshold region with 300 trials per point as a shakedown:

   ```bash
   pilot-proxy evaluate-snr --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
     --physical-channel 14 --frame-size-samples 16384 --num-input-streams 4 \
     --snr-start-db -38 --snr-stop-db -24 --snr-step-db 1 \
     --standard-frequency-offset-sweep \
     --threshold-snr-shelf-db -32 \
     --noise-trials 300 --output-dir results/pd_curves
   ```

   The standard sweep is `-1000`, `0`, and `+1000` Hz. The
   `--requested-snr-shelf-db` and `--frequency-offset-hz` options are
   repeatable when an explicit grid is preferable. Each trial is one detector
   batch.

   The `16384`-sample frame is the project-supplied configuration for the
   planned CHIME engine upgrade, not a measurement of the current live
   correlator. Until a public upgrade paper or other citable record exists,
   describe it as the planned configuration and retain the internal source
   used to set it.

   A 300-trial point cannot support a 2--3% precision claim: at `P_d = 0.5`,
   its independently recomputed Wilson 95% half-width is approximately 5.6%.
   After the shakedown, rerun the final grid with `--noise-trials 1500`, which
   gives approximately 2.5% at `P_d = 0.5` (1000 trials gives approximately
   3.1%). Use the emitted interval fields, rather than these worst-case
   planning values, for the final acceptance decision.

**CPU path.** The full sweep can also run without a GPU:
`--detector-backend cpu-reference` computes the primary fields with the
exact-integer CPU reference, and every row records `detector_backend`. The
kernel parity suite checks the CPU and CUDA implementations. Before using a
CPU sweep in the paper, repeat one mid-transition, same-seed point on the GPU
and require identical detection counts.

3. Build the publication figure from the summary CSV/JSON:

   ```bash
   PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf \
     python analysis/fig3_publication.py \
       results/pd_curves/dtv_snr_summary.csv "1500 trials/point"
   ```

   Use this backend-aware publication script for CPU-reference sweeps. The
   general `pilot-proxy plot-results` response plot currently labels its packed
   fixed-point column `GPU fixed-point` even when that column was produced by
   `--detector-backend cpu-reference`; do not use that label as backend
   provenance.

   Plot `positive_excess_detection_rate` and the -32 dB threshold rate against
   requested shelf SNR. Show the Wilson bounds and one curve per offset.

4. Accept the result when the rates are statistically consistent with a
   monotone response and the Wilson half-width is at most 3% through the
   transition. The current `analysis/fig3_publication.py` check rejects a
   downward step only when the adjacent Wilson intervals do not overlap; keep
   that rule fixed before the final run.

   Report the `P_d = 0.5` and `P_d = 0.9` crossings. The current helper
   linearly interpolates point estimates but does not calculate crossing
   intervals. If the manuscript quotes those intervals, derive them from the
   Wilson envelopes or a retained bootstrap and record the method. For the
   standard sweep, compare the `+/-1 kHz` shifts with the `1.5923` dB
   rectangular-window capture-loss prediction. That value was independently
   recomputed from the `sinc^2` expression in `docs/METHOD_SPEC.md` and is
   already used by `analysis/fig3_publication.py`.

---

## 3. Injection–recovery on real CHIME baseband

This test requests a complex tone at the pilot frequency in real baseband.
We then measure corrected pilot excess against the realized injected tone
power. The expected signal-dominated log-log slope is 1, and the linear-fit
floor is the ambient pilot excess of the selected channel. These are model
predictions to test, not existing measurements.

The preprocessing harness is already implemented as
`pilot-proxy inject-pilot-tone`. It operates in the file's offset-binary
4+4-bit domain, whose components span `[-8, 7]`. An `a = 0` run is checked for
byte identity. For nonzero amplitudes, the harness adds the tone, rounds the
sum once to the integer domain, clips to the component range, and records the
clip count. It verifies that sibling datasets and attributes are unchanged
and preserves each filename. Therefore, the resulting directory can be read
by `chime-scan --source local` without a detector-specific input path.

The current analysis is not yet sufficient for a publication ladder. It uses
the requested `amplitude_lsb**2` as the injected-power coordinate. The
manifest records the source RMS and clip count, but not the realized coherent
tone power after rounding and clipping. In particular, any requested
amplitude below 0.5 LSB makes no sample change under the current rounding
rule, so the earlier `0.05`--`0.4` LSB points were not valid injections.

1. **Close the realized-power preflight.** Update the injection manifest to
   record the realized post-quantization delta and its coherent power at the
   pilot frequency. Compute that quantity from the decoded output-minus-source
   samples after rounding and clipping; retain enough information to audit the
   projection. Update `analyze-injection-recovery` to use this measured value
   rather than `amplitude_lsb**2`, and add a regression test covering a point
   altered by rounding. Do not make a linearity or sensitivity claim until
   that change is implemented and tested.

2. **Build the ladder** on the quiet channel from item 1. Choose 6--8
   amplitudes whose realized powers span the transition, plus the required
   `a = 0` control. Populate `VALIDATED_AMPLITUDES` from the preflight; do not
   include a nonzero point below 0.5 LSB with the current integer-rounding
   path:

   ```bash
   for a in "${VALIDATED_AMPLITUDES[@]}"; do
     pilot-proxy inject-pilot-tone --input staged/<freq_id>/ \
       --output-dir injected/a$a --amplitude-lsb $a \
       --physical-channel <ch> --phase-seed 20260701
   done
   ```

   Run a second ladder at the measured transmitter offset with
   `--pilot-frequency-hz <pilot + df>`. This tests the capture-loss model on
   the same baseband distribution.

   Before scanning, require every ladder point to list the same source files,
   event identities, frame grid, phase seed, and pilot frequency, except for
   the intentionally offset ladder. The current analysis command does not
   enforce all of these cross-point identity checks, so retain and audit the
   comparison explicitly.

3. **Run the same pipeline** over each injected directory and retain the
   injection manifest with the products:

   ```bash
   pilot-proxy chime-scan --source local --input-dir injected/a$a \
     --analyzer pilot-proxy-detector --select <freq_id> \
     --output-dir runs/inj_a$a
   pilot-proxy validate-products --run-dir runs/inj_a$a/<...>
   cp injected/a$a/injection_manifest.json runs/inj_a$a/<...>/
   ```

4. **Analyze the ladder.** This command also produces the comparison used in
   item 4:

   ```bash
   pilot-proxy analyze-injection-recovery \
     --point runs/inj_a0/<...> --point runs/inj_a<a1>/<...> ... \
     --false-alarm-rate 1e-2 --output-dir results/injection_recovery
   ```

5. Accept the ladder when the fitted floor is consistent with the `a = 0`
   point, the reported `signal_dominated_log_slope` is consistent with 1.00,
   and the offset ladder is consistent with the predicted `sinc^2` factor.
   Report the clip count at every point; do not call it negligible without a
   numerical bound. The linearity figure and item 2's synthetic curves form
   the primary sensitivity check.

---

## 4. Radiometer baseline comparison

This test compares the pilot-informed F-statistic with a total-power
radiometer on the same frames and integration time. Both statistics are
already present in the run products, so no additional detector run is needed.

The `analyze-injection-recovery` command from item 3 uses
`baseband_power_linear` for the radiometer. For each requested `P_fa`, it sets
both thresholds from empirical quantiles of the `a = 0` control and reports a
Wilson 95% interval for each detector's rate. It requires at least `10/P_fa`
valid control frames.

The realized-power and cross-point identity preconditions from item 3 also
apply here. The current command does not independently prove that each ladder
point contains the same source events or frame identities. Verify those
identities before comparing the two detection curves, and use only the
matched valid-frame intersection if a frame is missing from any point.

The deliverable is `detector_vs_radiometer_pd`: `P_d` versus injected tone
power at matched `P_fa`. Report the horizontal difference at `P_d = 0.9` in
dB only when both curves cross that level within the sampled ladder. The
analytic comparison should state its assumed effective number of independent
bins and integration convention. The approximate scaling
`10*log10(N_bins_effective)` is a prediction to test, not a replacement for
the measured difference.

Accept the comparison when it uses identical valid frames, thresholds derived
from the same control data, and intervals on both rates. Then report whether
the measured difference is consistent with the stated analytic model.

---

## 5. Cleaning tradeoff in science terms

This test reports how the retained frame fraction changes with the detector
threshold. It is a detector-space cleaning study, not a BAO integration-time
forecast. The product schema stores `p_target_u64`, `p_ref_sum_u64`, and the
per-pilot weight norms, so the threshold can be recomputed after the scan.

`pilot-proxy analyze-cleaning-tradeoff` implements the sweep. Before using any
alternative threshold, it requires the `x = 0` float recomputation to match
the stored exact-integer mask on every valid frame.

1. **Run the threshold sweep** (`pilot-proxy analyze-cleaning-tradeoff --run-dir
   <combined> --control-run-dir <control> --survey-hours <H>`) over the
   combined survey products:
   - Use `tau = mu0 * 10^(x/10)` for `x` from 0 through 12 dB. The `x = 0`
     point is the stored operating point.
   - The implemented sweep forms `F` from the stored powers in float and
     applies `F > tau`. At `x = 0`, it asserts equality with the stored mask,
     which was formed by exact integer cross-multiplication.
   - For each channel and `tau`, report the masked fraction over valid frames.
     When a control run is provided, also report the cleaned mean
     `baseband_power_linear` relative to the control mean in dB.

2. **Build the operating curve.** Plot residual relative to the tested
   control against masked fraction, with one curve per DTV channel and
   `tau = mu0` marked. The current command emits a descriptive curve and
   frame counts, but no residual uncertainty intervals. Add a retained
   per-event block bootstrap before making an inferential claim about a knee
   or floor agreement. Otherwise, describe the curve as descriptive and do
   not imply that plotted lines are confidence intervals.

3. **Compute retained bandwidth** at the operating point:

   `recovered = sum over affected freq_ids of (1 − mask_fraction) × 0.390625 MHz`

   Compare this value with discarding the affected coarse channels outright.
   If `survey_hours` is supplied, also report the corresponding MHz-hours.
   Any percentage must name its denominator; do not describe the denominator
   as the amount "currently lost" unless that operational flagging policy is
   documented separately.

4. **Bound the interpretation.** The control comparison is in stored
   baseband-power units. Convert it to a system-temperature contribution only
   when a calibrated transfer and its uncertainty are available. The
   radiofisher uncertainty-tolerance study is the appropriate place to
   propagate that result into a BAO integration-time forecast.

Accept the analysis when the `x = 0` mask reproduces the stored decision on
every valid frame and the retained-bandwidth curve is monotone with threshold.
Report any non-monotone residual behavior rather than filtering it out. The
headline value must include the valid-frame count, affected-channel set, and
survey hours used to compute it.

---

## Figures produced by these checks

`docs/PAPER_PLAN.md` assigns the final figure and table numbers. The five core
checks produce:

1. Measured `mean F` vs `mu0` per channel + H0 mask-fraction before/after
   (item 1).
2. `P_d`(shelf SNR) with Wilson bars, per offset, synthetic (item 2).
3. Recovered vs injected pilot excess on real baseband, with capture-loss
   ladder (item 3).
4. F-statistic vs radiometer `P_d` at matched `P_fa` (item 4).
5. Masked fraction vs residual + recovered-bandwidth statement (item 5).

---

## Release checklist

1. Freeze the result set. Change `0.2.0.dev0` to `0.2.0` in `pyproject.toml`
   and `CITATION.cff`, update the `CHANGELOG.md` heading, and create the tag.
2. Archive the tagged release. Add the resulting DOI to the `doi` field in
   `CITATION.cff` and to the manuscript software citation. Repeat this step
   for datatrawl.
3. Build both formal documents from the tag and attach their PDFs to the
   release. `PILOT_PROXY_USE_TEX` does not affect these LaTeX documents.
4. Run `make release-check` and the kernel suite on the GPU node at the tag.
   Record the tag, test result, and CI URL in the reproducibility note.
