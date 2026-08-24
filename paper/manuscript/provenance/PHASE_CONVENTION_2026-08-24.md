# Phase-convention reconciliation (2026-08-24)

Records the resolution of an apparent conflict between the 2026-07-16 sweep
verification and the 2026-08-24 evaluate-snr phase-convention fix, so neither
record is read as invalidating the other.

## The apparent conflict

The 2026-08-24 fix measured that applying the reference archive phase
convention (conjugation plus the (-1)^n half-rate factor) displaces the clean
channel-14 pilot from the weight layout's fine bin 1 to bin 63, so the
detector statistic reads noise at every SNR regardless of spectral sense
(clean normalized ratio about 1.1 against 121 for the matched combination).
Under the assumption that the 2026-07-16 1000-trial sweep ran with that
convention applied, the sweep could not have reproduced the ideal-benchmark
crossings it records.

## Why there is no conflict

- The July sweep demonstrably ran a matched chain: its crossings sit +0.27 to
  +0.32 dB from the mu0-corrected ncF benchmark, and the raw 45,000-row
  bundle re-aggregates to the archived summary exactly (T2 memo). A chain
  measuring noise cannot do either.
- The assumption that it ran with the archive phase applied is an inference
  from the current tool's later defaults, not a recorded fact. The producing
  commit (7d5ae68) predates the public history, and the sweep-era CLI in
  these memos (`--requested-snr-shelf-db`, `--threshold-snr-shelf-db`) is a
  vocabulary the current tool no longer accepts: the testbench was
  substantially rewritten between the sweep and the public initial commit,
  and the `--reference-archive-phase` step belongs to that later work. The
  exact sweep commands live in the raw bundle's shard.logs
  (run_pd_curves_cpu_1000.tar.gz, sha256 in provenance_blobs.sha256).
- The current matched configuration (no archive phase, normal sense)
  reproduces the July-verified behaviour: the 2026-08-24 transfer curve
  (~/rail/transfer_2026-08-24, channel 14, 4 streams, 31 points x 60 trials)
  tracks identity within 0.8 dB over -34..-12 dB with its knee at about
  -33 dB, consistent with the sweep's ~= -31.8 dB threshold-rule crossing and
  the -32.09 dB benchmark. A real per-pilot product (freq_id 767) confirms
  the archive stream is plain-conjugated relative to physical frequency
  (pilot line at +81,181.5 Hz for a physical -81,184 Hz offset) with the
  deployed bank matched (coarse F ~= 31 on every valid frame).

## Ruling

The 2026-07-16 sweep verification and T2 resolution stand as written for
their era; Fig. 3 and its provenance chain are unaffected. The current
defaults are the matched configuration of the current tool, enforced at
startup by the clean-pilot guard, with the 2026-08-24 transfer curve as
their verification. Any future change to the reference channelizer, the
phase conventions, or the weight-bank coordinates must re-run the guard and
record a fresh four-quadrant measurement alongside this note.
