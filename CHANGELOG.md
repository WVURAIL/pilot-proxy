# Changelog

## 0.2.0.dev0 - Unreleased

- `pilot-proxy evaluate-snr` now inherits the testbench evaluator's parser
  directly (`parents=`) and calls it in-process instead of hand-mirroring
  arguments into a subprocess. This fixes `--detector-backend` (and 15 other
  testbench options that had silently never been exposed on the CLI:
  channelizer geometry/rate overrides, `--scale`, `--clip-sigma`,
  `--spectral-sense`, `--waveform-audit-json`, archive-phase toggles, and
  the experimental knobs) and adds a parity test so the CLI can never drift
  from the testbench surface again.

- `evaluate-snr --detector-backend cpu-reference`: publication detection
  sweeps without a GPU. The primary fields come from the validated
  exact-integer CPU reference (shared result-builder with the kernel path,
  exact Python-integer rational-half mask); rows record their backend, and
  the runbook documents the same-seed GPU spot check that ties CPU curves to
  the deployed kernel.

- `pilot-proxy chime-combine`: standalone access to the scan's combine step,
  so per-pilot checkpoint snapshots can be stacked into canonical products
  mid-survey (validate-products / chime-plot / analyze-cleaning-tradeoff on
  completed channels without waiting for the full scan). Frame-grid
  mismatches between complete and partial channels are refused with the
  existing diagnostic. `docs/PUBLICATION_VALIDATION.md` gains the mid-survey
  execution guide.

- `docs/PAPER_PLAN.md`: editorial plan for the publication (venue, section
  outline, figure/table inventory mapped to producing commands and status).

- `pilot-proxy analyze-injection-recovery`: post-hoc analysis of an injection
  ladder's run products -- weighted recovery-linearity fit (floor + gain, with
  a signal-dominated log-log slope check) and the F-statistic vs radiometer
  detection comparison at matched false-alarm rates (empirical control
  quantiles, Wilson 95% intervals). `docs/PUBLICATION_VALIDATION.md` now
  ships the full publication runbook in-repo.

- Publication-analysis commands: `pilot-proxy inject-pilot-tone` (integer-
  domain pilot-tone injection into real baseband copies; zero-amplitude pass
  is byte-identical, saturation counted, siblings preserved) and `pilot-proxy
  analyze-cleaning-tradeoff` (post-hoc mask-threshold sweep over stored
  num/den with an exact x=0 anchor against the shipped mask; operating-curve
  and recovered-bandwidth outputs).

- Norm-corrected positive-excess mask: the mask now compares against the
  detector's exact H0 zero-point `mu0 = 2*target_norm_sq/ref_norm_sum_sq`
  (integer cross-multiplication) instead of `F > 1`. int4 weight quantization
  leaves the three weight-term norms unequal (`mu0` spans ~0.985..1.011 across
  the shipped ATSC 14-36 bank), which pinned the H0 mask fraction toward 0 or
  1 per channel under the old rule. Detector products gain per-pilot
  `target_norm_sq`, `ref_norm_sum_sq`, `mu0`, and per-frame
  `pilot_excess_corrected` (`F/mu0 - 1`); runtime bundles declare the
  per-channel kernel rational half-threshold `nt : (nl+nu)`; `validate-products`
  checks whichever rule a product declares, and resume refuses to mix rules.
- `datatrawl` integration: `chime-scan` streams the detector analyzer over the
  CADC archive, one resumable per-pilot product per CHIME coarse channel.
- Resume compatibility now binds detector products to the exact selected weights,
  detector contract, Python implementation, CUDA library, feed count, and run cap.
- Combined products verify per-frame event positions and sample-rate consistency.
- Wheels include the shipped profiles, stream map, weight bank, and manifest.
- CHIME real-data adapter.
- Schema-v2 per-pilot detector products with per-unit time axis and provenance
  (`weights_hash`, `detector_version`, `mask_rule`).
- Positive-excess masking.
- Adaptive reference placement.
- Runtime weight-bundle exporter and validator.
- Guard/reference terminology cleanup: `skipped_guard_bins` and
  `reference_offset_bins`.

## 0.1.0 - 2026-05-29

- Standalone CUDA F-statistic kernel and C/C++ tests.
- GNU Radio ATSC 8VSB clean-waveform generator.
- GNU Radio and Python AWGN injection paths.
- Reference ADC/PFB conversion to packed 4+4 bit detector input.
- DTV SNR evaluator using exact uint64 GPU power readback.
- ATSC waveform audit for pilot frequency, pilot/data ratio, occupied
  bandwidth, shelf flatness, and edge rolloff.
