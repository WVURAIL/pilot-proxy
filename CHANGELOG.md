# Changelog

## 0.2.0.dev0 - Unreleased

- Mask boundary aligned to the contract: the CUDA kernel used inclusive
  `>=` where the schema, method spec, and Python reference declare strict
  positive excess (`>`). Behavior differs only on exact integer equality
  (measure-~zero on real u64 accumulations; existing products unaffected,
  `mask_rule` string unchanged). Python-level equality-boundary regression
  added; C-harness and CUDA-path boundary tests follow with the next GPU
  release-check. Credit: external review.

- pilotcal is retired. The one artifact the analysis still needs from it --
  the high-resolution time-averaged spectrum per coarse channel -- is
  carried by every per-pilot scan product
  (`integrated_spectrum_before/after_mask`: 23.8 Hz bins, accumulated over
  the full processed frame set), and `analyze-transmitter-census
  --lines-from-run <work_dir>` now extracts the carrier line list directly
  from those spectra (windowed peak detection about the nominal pilot with
  a median floor; window, SNR threshold, and separation are CLI knobs; the
  derived list is written alongside the outputs for provenance).

- `analyze-transmitter-census`: the census loader now accepts a precomputed
  `detectability_db` column (e.g. a propagation-model field strength) as the
  association ranking score, with blanks ranking last tie-broken by distance;
  the ERP/distance^2 path remains for schemas without one. Class
  normalization handles real FCC/ISED export strings ("Translator (LPTV)",
  "Low-power (LPTV)", "Relay", "Class A") via punctuation-insensitive keys.

- Documentation rolled to DS001/UG001 v1.6 (files renamed to match): kernel
  build examples updated for SM auto-detection, the `detect` example now
  relies on the quantize metadata.json sidecar for the pilot identity, and
  the environment prerequisites reference the repository venv (conda
  references removed).

- New `analyze-transmitter-census` command: the 500-mile case-study analysis
  over a detected-carrier line list and the FCC/ISED transmitter census. Associates
  detected carrier lines to census entries per RF channel (rank pairing by
  SNR vs an ERP/distance^2 detectability score, with a dominant/secondary
  fallback strategy), and produces the class-split offset-dispersion figure,
  the per-channel spread-vs-composition figure with Spearman rank
  correlation and bootstrap CI, the association table, and an SNR-threshold
  stability sweep. Input schemas are declared in the module docstring;
  nothing touches the archive.

- Documentation builds are now part of the tooling: `make docs` builds the
  data sheet and user guide with latexmk (`docs/auxil/` scratch, PDFs in
  `docs/out/`), and the README's Build-documentation section records the
  verified Debian/Ubuntu TeX package set, including the extras that
  `PILOT_PROXY_USE_TEX=1` figure rendering needs.

- Event-keyed combine: per-pilot frames now align by (event, frame-in-file)
  identity instead of positionally. Pilots that processed different event
  sets (the archive is ragged: not every channel holds every event) stack
  over exactly their common identities; per-pilot drops are echoed, recorded
  in `stats.json` (`combine_alignment`), and the kept identities are written
  to `chime_frame_identity.npz`. Fully aligned inputs pass through
  byte-identically (the `run_chime_analysis` parity guarantee is unchanged),
  and products predating the identity tags keep the strict positional check.

- `pilot-proxy chime-scan`: the terminal combine can no longer fail an
  archive-scale run. When no event is common to every completed channel, the
  scan finishes successfully with complete per-pilot products, explains, and
  defers stacking to `chime-combine`.

- `pilot-proxy chime-combine`: new `--report` (per-pilot event counts,
  presence histogram, all-pilot intersection, greedy drop-curve -- the
  decision input for subset selection) and `--drop <freq_ids>` (exclude
  channels from the stack). `--output-dir` is required only when combining.

- `pilot-proxy chime-scan`: `--select` is now optional for the archive source
  -- omitted, the scan covers every freq_id the inventory contains (sorted;
  companion rows without a freq_id are skipped), prints the resolved set
  before any staging, and notes when the survey sidecar's requested
  `freq_ids` disagree with the rows (patchy replication, partial surveys).
  One product per freq_id either way; the analyzer-level explicit-selection
  guard is unchanged. `--source local` still requires `--select` (no
  inventory to derive the scope from).

- `pilot-proxy chime-scan`: `--source` is inferred from the flags that name
  it (`--inventory`/`--inventory-name` select `cadc-datatrail`; bare
  `--source-root` keeps the historic local default). The previously silent
  wrong pairings are now hard errors (`--source local` with inventory flags;
  `--input-dir` with the archive source), and `--instrument` is
  cross-checked against the inventory sidecar's recorded `telescope`.

- `pilot-proxy detect`: the pilot identity comes from the `metadata.json`
  sidecar quantize writes next to the packed matrix (`dtv_pilot_hz` is
  authoritative); an explicit `--physical-channel`/`--dtv-pilot-mhz` must
  agree with it within the pilot-frequency tolerance. Behavior change: with
  neither a flag nor a sidecar, `detect` refuses instead of silently
  assuming channel 14.

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
