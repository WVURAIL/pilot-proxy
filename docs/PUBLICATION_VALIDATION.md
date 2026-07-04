# Publication validation runbook

Five items stand between the current repos and a publication-ready result set.
All tooling now ships in this repository; the remaining work is GPU/on-sky
runs and the analysis over their products. Ordering, effort, and tooling
status:

| # | Item | Depends on | Effort | Tooling |
|---|------|-----------|--------|---------|
| 1 | On-sky norm-corrected mask validation | -- | ~1 GPU day (incl. control) | shipped (runbook + `validate-products`) |
| 2 | Detection-rate curves at publication trial counts | -- | hours (parallel with 1) | shipped (`evaluate-snr` Wilson fields) |
| 3 | Injection-recovery on real baseband | 1 | ~1-2 days of runs | shipped (`inject-pilot-tone`, `analyze-injection-recovery`) |
| 4 | Radiometer baseline comparison | 3 | free (same command) | shipped (`analyze-injection-recovery`) |
| 5 | Cleaning tradeoff in science terms | 1 (products) | hours (post-hoc) | shipped (`analyze-cleaning-tradeoff`) |

**Mid-survey execution.** Nothing below requires the production scan to
finish, and running alongside it is safe by design: staging directories are
per-invocation (concurrent scans cannot delete each other's files), per-pilot
checkpoints are atomic renames (any `work/<freq_id>.npz` you read is a
complete, consistent product), and the bounded runs below are a negligible
CADC load next to the survey. Concretely, while the survey runs:

- Items 2 (synthetic curves), 1 (control-channel scan), and 3-4 (the
  injection ladder) run on a second GPU session in that order.
- Item 1's DTV 18/20 zero-point checks need no new scan at all: snapshot the
  survey's own checkpoints (`cp work/<freq_id>.npz snapshot/`), combine, and
  read `mu0`/mask fractions from the snapshot --

  ```bash
  pilot-proxy chime-combine --product snapshot/<freq_id>.npz \
    --output-dir snapshot_combined/<freq_id>
  pilot-proxy validate-products --run-dir snapshot_combined/<freq_id>
  ```

- Item 5's machinery can be rehearsed the same way on any completed channel's
  snapshot (`analyze-cleaning-tradeoff --run-dir snapshot_combined/<freq_id>`);
  final numbers still come from the full combined survey.
- `chime-combine` across multiple channels requires them to have processed
  the same event set; combining a complete channel with a partial one is
  refused with a frame-grid diagnostic. That refusal is correct -- combine
  completed channels together, or one channel at a time.

Run 1 first — it gates everything downstream on the corrected rule. Item 2 is
independent and can run on the same GPU session. Items 4 and 5 are pure
analysis over stored products; no new detector code.

Environment for 1 and 3: the existing `docs/CANFAR_RUNBOOK.md` setup (A100
session, `setup_env.sh`, CADC cert). All figures: run with
`PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf` for camera-ready
output.

---

## 1. On-sky validation of the norm-corrected mask

Goal: demonstrate on real data that E[F] under H0 tracks the per-channel
`mu0`, not 1, and that the corrected mask restores a channel-independent ~0.5
H0 mask fraction. This is the on-sky counterpart of
`tests/core/test_mask_zero_point.py` and the runbook's "H0 zero-point check"
section.

1. **Export and validate the runtime bundle** so the kernel rational
   half-thresholds `nt:(nl+nu)` are the deployed correction:

   ```bash
   pilot-proxy export-runtime-weight-bundle \
     --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
     --weight-coordinate-system post_spectral_sense_normalization \
     --physical-channel-range 14:36 \
     --output-dir bundles/norm_corrected
   pilot-proxy validate-runtime-weight-bundle --bundle-dir bundles/norm_corrected
   ```

   Acceptance: validator reports valid; `pilot_profiles.json` rows carry
   `target_norm_sq`, `ref_norm_sum_sq`, `mu0`,
   `positive_excess_half_threshold_num/den`.

2. **Choose three freq_ids** from the transmitter census in the calibration
   report:
   - **Control**: a coarse channel in the DTV band containing *no* ATSC pilot
     within capture (use the census 500-mile table; confirm the band is quiet
     with `pilot-proxy chime-inspect --input-dir <staged>` on a few staged
     files before burning GPU time).
   - **Extreme-mu0 quiet channels**: DTV 18 (largest shipped `mu0`, 1.0111)
     and DTV 20 (smallest, 0.9853). These are where the legacy rule pinned
     hardest, so they are the discriminating measurement.

3. **Bounded scans** (same pattern as the pre-production gates):

   ```bash
   pilot-proxy chime-scan --name h0check --select <freq_id> \
     --max-files 50 --checkpoint-every 10 --output-dir runs/h0check_<freq_id>
   pilot-proxy validate-products --run-dir runs/h0check_<freq_id>/<...>
   ```

   Aim for ≥1000 valid frames per channel (50 files is ample at 10 s frames).

4. **Acceptance criteria**, per channel, from `stats.json` and
   `chime_detector_outputs.npz`:
   - `mu0_by_pilot` matches the weight-manifest value exactly.
   - Control + quiet channels: `mean(fstat_raw[valid])` agrees with `mu0`
     far better than with 1 — require `|mean − mu0| < |mu0 − 1| / 3` and note
     the frame count. With ≥1000 frames the standard error on the mean is
     ~`sqrt(1.5/rows_per_frame)/sqrt(nframes)`, orders below the 1↔mu0 gap.
   - Mask fraction over valid frames in **0.45–0.55** on the control and on
     quiet channels at quiet hours (compare the same channels' pinned 0/1
     fractions from any pre-correction products for the paper's
     before/after panel).
   - `validate-products` passes (it now checks the *declared* corrected
     rule with exact integer math).
   - DTV-loud channel/hours: mask fraction well above 0.5 and
     `pilot_excess_corrected` strongly positive — the signal side still
     detects.

5. **Kill/resume once** mid-scan on one channel and confirm the resume is
   accepted (provenance now gates on the norms and rule string) and the
   combined product is deterministic against an uninterrupted rerun — the
   same check as the pre-production gate, under the new rule.

Paper artifact: a two-panel figure — per-channel measured `mean F` vs `mu0`
(with the `F=1` line for contrast), and H0 mask-fraction before/after the
correction.

---

## 2. Detection-rate curves at publication trial counts

Goal: `P_d` vs shelf SNR with Wilson 95% intervals, per frequency offset, at
the −32 dB science threshold and for the parameter-free positive-excess rule.
The summary machinery already emits `positive_excess_detection_rate` and
`threshold_detection_rate` with `*_wilson95_lo/hi` — the only change from your
existing sweeps is the trial count.

1. On the GPU node, reuse the golden ATSC capture (or regenerate:
   `pilot-proxy generate-atsc ...` then `pilot-proxy audit-atsc`).

2. Sweep the threshold region densely with ≥300 trials/point:

   ```bash
   pilot-proxy evaluate-snr --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
     --physical-channel 14 --frame-size-samples 16384 --num-input-streams 4 \
     --snr-start-db -38 --snr-stop-db -24 --snr-step-db 1 \
     --standard-frequency-offset-sweep \
     --threshold-snr-shelf-db -32 \
     --noise-trials 300 --output-dir results/pd_curves
   ```

   (`--requested-snr-shelf-db` / `--frequency-offset-hz` are repeatable if you
   prefer explicit lists to the start/stop/step grid.) Budget: each trial is
   one kernel batch; 15 points × offsets × 300 trials is minutes on the A100,
   so 300 is a floor, not a ceiling — use 1000 for the final figure if the
   session allows.

**No GPU available?** The full sweep also runs GPU-free:
`--detector-backend cpu-reference` computes the primary fields with the
validated exact-integer CPU reference (Python-integer rational-half mask, so
decisions are exact); each row records its `detector_backend`. The kernel <->
reference equivalence is CI-gated by the kernel parity suite, so a
CPU-produced curve is the deployed detector's curve up to that gate; before
the camera-ready figure, tie them together with a same-seed GPU spot check
(one mid-transition SNR point, both backends, assert identical detection
counts). Sweep cost on CPU is minutes at these geometries.

3. Figures from the summary CSV/JSON:

   ```bash
   PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf \
     pilot-proxy plot-results \
       --input-csv results/pd_curves/dtv_snr_summary.csv \
       --output-png results/pd_curves/dtv_snr_sweep.png
   ```

   The paper figure is `positive_excess_detection_rate` (and the −32 dB
   threshold curve) vs requested shelf SNR with the Wilson bounds as error
   bars, one curve per offset.

4. Acceptance: monotone curves; Wilson half-widths ≲2–3% through the
   transition; quote the `P_d = 0.5` and `P_d = 0.9` crossings with their
   intervals, and confirm the ±300 Hz offset curve shift is consistent with
   the METHOD_SPEC capture-loss bound (≤0.14 dB).

---

## 3. Injection–recovery on real CHIME baseband

Goal: inject a complex tone of known amplitude at the pilot frequency into
*real* baseband and show recovered pilot excess tracks injected amplitude —
slope 1, intercept = the channel's ambient floor. No harness exists yet; build
it as a preprocessing script so the production pipeline runs untouched and
every product carries normal provenance.

**Tooling:** `pilot-proxy inject-pilot-tone` works in the file's own integer
domain (offset-binary 4+4-bit, components [-8, 7]): a zero-amplitude pass is
byte-identical to the source unconditionally, injected deltas are exact apart
from counted saturation, and sibling datasets/attributes/filenames are
preserved so the output tree runs through `chime-scan --source local`
unchanged.

1. **Build the ladder** on the quiet channel from item 1. Choose 6-8
   amplitudes spanning shelf-SNR-equivalent -40 to -20 dB (the manifest
   records each file's measured per-component RMS in LSB, so `--amplitude-lsb`
   converts to dB after the fact), plus the mandatory `a = 0` control:

   ```bash
   for a in 0 0.05 0.1 0.2 0.4 0.8 1.6; do
     pilot-proxy inject-pilot-tone --input staged/<freq_id>/ \
       --output-dir injected/a$a --amplitude-lsb $a \
       --physical-channel <ch> --phase-seed 20260701
   done
   ```

   A second ladder at the channel's *measured* transmitter offset
   (`--pilot-frequency-hz <pilot + df>`) tests the capture-loss bound on-sky.

2. **Run the untouched pipeline** over each tree, then keep the manifest with
   its products:

   ```bash
   pilot-proxy chime-scan --source local --input-dir injected/a$a \
     --analyzer pilot-proxy-detector --select <freq_id> \
     --output-dir runs/inj_a$a
   pilot-proxy validate-products --run-dir runs/inj_a$a/<...>
   cp injected/a$a/injection_manifest.json runs/inj_a$a/<...>/
   ```

3. **Analyze the ladder** (this also produces item 4's comparison):

   ```bash
   pilot-proxy analyze-injection-recovery \
     --point runs/inj_a0/<...> --point runs/inj_a0.05/<...> ... \
     --false-alarm-rate 1e-2 --output-dir results/injection_recovery
   ```

4. Acceptance: weighted-fit floor consistent with the `a = 0` ambient level;
   signal-dominated log-log slope `1.00` within error (reported as
   `signal_dominated_log_slope`); the offset ladder depressed by the
   predicted `sinc^2` capture loss; clip counts negligible at every point
   (they are in the CSV). This figure plus item 2's synthetic curves is the
   sensitivity validation a referee will look for first.

---

## 4. Radiometer baseline comparison

Goal: quantify, on identical data and timescales, the advantage of the
pilot-informed F-statistic over the classical total-power detector. Pure
analysis: both statistics are already in the products.

**Tooling:** produced by the same `analyze-injection-recovery` run as item 3
-- the radiometer statistic is `baseband_power_linear` on identical frames,
thresholds are empirical quantiles of the `a = 0` control at each requested
`P_fa` (the tool refuses a `P_fa` the control's frame count cannot support:
it needs >= 10/P_fa valid frames), and both detectors' rates carry Wilson
95% intervals.

Deliverable: the `detector_vs_radiometer_pd` figure -- `P_d` vs injected tone
power for both detectors at matched `P_fa`; the horizontal gap at
`P_d = 0.9` is the headline sensitivity advantage in dB. Add the analytic
overlay in the manuscript: the pilot occupies one fine bin while the
radiometer integrates the full band's noise, so the expected processing-gain
gap is `~10*log10(N_bins_effective)` scaled by the per-frame integration;
state the prediction and show the measurement matches within the ladder's
CIs.

Acceptance: measured gap consistent with the analytic expectation; the
comparison uses *identical frames*, thresholds set on *data*, and both curves
carry binomial intervals.

---

## 5. Cleaning tradeoff in science terms

Goal: an operating curve a BAO referee can act on — masked fraction versus
residual contamination, with the recovered-bandwidth headline. Entirely
post-hoc: the schema keeps raw `p_target_u64`/`p_ref_sum_u64` verbatim and the
norms per pilot precisely so thresholds are a recompute, never a re-run.

**Tooling:** `pilot-proxy analyze-cleaning-tradeoff` implements the sweep
below; the `x = 0` anchor against the stored mask is enforced as a hard gate.

1. **Threshold sweep** (`pilot-proxy analyze-cleaning-tradeoff --run-dir
   <combined> --control-run-dir <control> --survey-hours <H>`) over the
   combined survey products:
   - Grid `tau = mu0 * 10^(x/10)` for `x` in 0…12 dB (x = 0 is the shipped
     operating point).
   - Exact integer mask per frame:
     `p_target * ref_norm_sum_sq * den > target_norm_sq * p_ref_sum * num`
     with `num/den` the rational for `10^(x/10)` (or float is fine for the
     sweep; the x = 0 point must reproduce the stored mask exactly — assert
     it).
   - Per channel and per `tau`: masked fraction over valid frames, and the
     **residual metric** — mean cleaned baseband power (mask applied to
     `baseband_power_linear`) minus the control channel's mean, in dB.

2. **Operating curve figure**: residual (dB above control floor) vs masked
   fraction, one curve per DTV-affected channel, shipped point `tau = mu0`
   marked. Loud channels show the knee; quiet channels hug the floor.

3. **Recovered-bandwidth headline**: at the operating point,

   `recovered = sum over affected freq_ids of (1 − mask_fraction) × 0.390625 MHz`

   quoted alongside the alternative (discarding those coarse channels
   outright) and scaled by the survey hours — "X MHz·h recovered at ≤Y dB
   residual above the pilot-free floor". Express X also as a percentage of
   the CHIME band currently lost to DTV flagging.

4. **BAO tie-in, minimal defensible version**: convert the residual dB excess
   into a fractional system-temperature contribution per channel and state it
   against the survey's noise budget; full power-spectrum propagation can be
   future work if the residual is ≪ the thermal floor, which the control
   comparison establishes.

Acceptance: the x = 0 sweep point reproduces the stored products bit-exactly;
curves are monotone; the headline number carries the frame counts and hours
behind it.

---

## Suggested paper-figure inventory from these items

The full editorial mapping (venue, section outline, figure/table numbers,
statements-to-artifacts) lives in `docs/PAPER_PLAN.md`.

1. Measured `mean F` vs `mu0` per channel + H0 mask-fraction before/after
   (item 1).
2. `P_d`(shelf SNR) with Wilson bars, per offset, synthetic (item 2).
3. Recovered vs injected pilot excess on real baseband, with capture-loss
   ladder (item 3).
4. F-statistic vs radiometer `P_d` at matched `P_fa` (item 4).
5. Masked fraction vs residual + recovered-bandwidth statement (item 5).

---

## Release checklist (at submission)

1. Freeze results, then tag: bump `0.2.0.dev0` -> `0.2.0` in `pyproject.toml`
   and `CITATION.cff`, update `CHANGELOG.md`'s heading, tag and push.
2. Archive the tagged release (e.g. Zenodo's GitHub integration) and add the
   minted DOI to `CITATION.cff` (`doi:` field) and the manuscript's software
   citation. Do the same for `datatrawl`.
3. Rebuild the formal documents from the tag
   (`PILOT_PROXY_USE_TEX` irrelevant here; they are LaTeX sources) and attach
   the PDFs to the release.
4. Re-run `make release-check` and the kernel suite on the GPU node at the
   tag; record the CI run URL in the paper's reproducibility note.
