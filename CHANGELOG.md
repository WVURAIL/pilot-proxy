# Changelog

## 0.2.0.dev0 - Unreleased

- `datatrawl` integration: `chime-scan` streams the detector and offset
  analyzers over the CADC archive, one resumable per-pilot product per CHIME
  coarse channel.
- Resume compatibility now binds detector products to the exact selected weights,
  detector contract, Python implementation, CUDA library, feed count, and run cap.
- Offset products now support checkpoint/resume with configuration and source checks.
- Combined products verify per-frame event positions and sample-rate consistency.
- Wheels include the shipped profiles, stream map, weight bank, and manifest.
- CHIME real-data adapter.
- Schema-v2 per-pilot detector products with per-unit time axis and provenance
  (`weights_hash`, `detector_version`, `mask_rule`).
- Positive-excess masking.
- Adaptive reference placement.
- Frequency-offset diagnostic.
- Time-averaged FFT spectrum diagnostic.
- Runtime weight-bundle exporter and validator.
- Guard/reference terminology cleanup: `skipped_guard_bins` and
  `reference_offset_bins`.
- Removed the standalone `chime-frequency-offset` CLI; offset diagnostics
  remain via `chime-run --frequency-offset-diagnostic`.

## 0.1.0 - 2026-05-29

- Standalone CUDA F-statistic kernel and C/C++ tests.
- GNU Radio ATSC 8VSB clean-waveform generator.
- GNU Radio and Python AWGN injection paths.
- Reference ADC/PFB conversion to packed 4+4 bit detector input.
- DTV SNR evaluator using exact uint64 GPU power readback.
- ATSC waveform audit for pilot frequency, pilot/data ratio, occupied
  bandwidth, shelf flatness, and edge rolloff.
