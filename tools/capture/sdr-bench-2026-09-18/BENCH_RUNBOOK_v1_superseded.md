# Single-input SDR bench: terminated nulls and attenuator ladder

Predeclared 2026-09-17, before any capture; revised the same day after an
adversarial check, still before any capture. Spec: `protocol_spec.json` (run_id
SDR-BENCH-2026-09-17-single-input-reference-v1). Rig: LimeSDR Mini serial
1D423D9108F273, RF terminators, step attenuators, SMA cables. No antenna, no
radiated emission.

## Fixed settings (unchanged from the 2026-09-09 controls)

500 MHz LO, 2 MHz sample rate, native RX gain 30 on LNAW, RX LPF 1.5 MHz, TX LPF
5 MHz, TX2 path, native TX gain 50 with readback 50, tone at +100 kHz with
digital amplitude 0.005. The spec's `rx_gain_setting_db` 18 and
`tx_gain_setting_db` 38 are the same settings in the protocol schema's
convention (native gain = setting + 12), not a second gain. Every record is 2 s
(4,000,000 samples, within the helper's 4 s maximum) because the unchanged
analyzer accepts exactly this geometry (47 complete frames). Analysis: mixing by
-100 kHz, 25/128 resampler, K = L = 128, bins 0/-2/+2, sample scale 500 fixed.
Nothing is fitted to these data.

Base pad: a fixed 24 dB attenuator (or fixed pad combination summing to 24 dB,
identified in the log like every other part) stays in line on every tone slot,
including slot 1, in addition to the commanded ladder value. The commanded
ladder values are the ladder variable; the in-line attenuation is 24 dB plus the
commanded value. The reason is the level budget below. The base pad is fixed
before the first record and is never changed during the session.

## Schedule (26 records, one per spec slot, in slot order)

1. Slot 0, txzero: both ports terminated, TX active with zero IQ.
2. Slot 1, tone at commanded 0 dB: cabling check, TX cable C1 to base pad and
   ladder, ladder to cable C2 to RX.
3. Slot 2, noise: RX and TX terminated, TX disabled. Terminated null, before.
4. Slots 3 to 23, tone: ladder 0, 6, 12, 18, 24, 30, 36 dB commanded, pass A
   ascending, pass B descending, pass C ascending. Three repeats per rung,
   21 records.
5. Slot 24, noise: terminated null, after.
6. Slot 25, txzero: terminated TX-zero, after.

Commanded 0 dB means cables, adapters and the base pad only. The order is fixed;
no redraw, retry, extension, gain change, pad change or interval change after a
failure. A failed attempt is kept, the remaining slots stay unattempted, and the
session is reported as stopped, with one predeclared exception: a tone slot whose
receipt carries the worker error `RX overload guard` (any RX chunk with a
component at or above 0.98 or an rms at or above 0.35 of full scale) is recorded
as overloaded and the session continues with the next slot. An overloaded record
is kept, counts as attempted and not successful, and never enters analysis. No
other failure is exempt.

## Level budget (from the 2026-09-09 records, written before any bench data)

The 2026-09-09 over-the-air tone (30 cm, same radio settings) sits at raw rms
0.0122 to 0.0140 and resampled tone amplitude about 0.0025 (accepted interval).
Measured against that level:

- the worker overload guard (chunk rms 0.35) is reached about 28 dB above it;
  the peak guard (0.98) about 26 dB above it on the raw peak, so a record fails
  roughly 26 to 28 dB above the 30 cm level;
- the analyzer's 4-bit quantizer at sample scale 500 saturates a resampled
  component above 0.015; with the observed noise peak of 0.0052 the first
  saturated component appears when the tone is about 12 dB above the 30 cm
  level, so a rung with zero saturated components must stay at or below about
  +10 dB;
- the qualification floor (10 dB above the terminated null in the target bin,
  null target power about 0.014) is about 19 dB below the 30 cm level, because
  the 30 cm tone stands 29.5 dB above the null in the target bin.

A qualified rung therefore lies in a window about 30 dB wide, from 19 dB below to
10 dB above the 30 cm level. The conducted level at commanded 0 dB is higher than
the 30 cm level by the over-the-air link loss L, which was never measured:
free-space loss at 0.5 wavelength is 16 dB and the small antennas add an unknown
amount, so L is plausibly 20 to 45 dB. Without a base pad, commanded rungs below
L - 27 dB fail the record (for L above 27 dB the session would have stopped at
slot 1) and rungs below L - 10 dB saturate, leaving fewer than four qualified
rungs for any L above 28 dB. With the 24 dB base pad the in-line
ladder runs 24 to 60 dB, which holds at least four qualified rungs for L between
about 23 and 52 dB and never reaches the overload guard for L below 51 dB. If L
is below 23 dB the bottom rungs fall under the qualification floor and the
ladder is reported as not established, not as passed. This budget sets the pad;
it is not a calibration and no number in it enters the analysis.

RX range: the RX port never sees more than the declared TX bound of -18 dBm
minus the cabling, well inside the 0 dBm limit in `rf_limits` and far from any
damage level; the range question is the recorded digital range above, not the
hardware.

Pad inventory: the ladder needs pads summing to 60 dB (24 base plus 36). If the
inventory is short, regenerate the spec with a shorter ladder (`make_spec.py`)
before the first record; do not shorten it during the session.

## Steps

Interpreter for steps 5 and 7: the analysis venv of the 2026-09-09 run,
`/home/djg/rail/output/fisher-fixes-2026-09-08/analysis-venv/bin/python`,
named in its `analysis-command.json` and `prepare-command.json`; below it is
written `$PY`.

1. Create the session directory `session/` here and `session_log.csv` with columns
   slot, utc_start, helper_mode, commanded_attenuation_db, base_pad_parts,
   attenuator_parts, cable_tx, cable_rx, terminator_ids, room_temperature_c,
   notes. Photograph the bench before the first record.
2. Label the parts once: cables C1 and C2, the base pad, each attenuator pad by
   its marked value and serial or body mark, terminators T1 and T2. Write the
   same identities on every row that uses them.
3. Validate the spec and keep the output: `/home/djg/rail/venvs/archive-local/bin/python
   /home/djg/rail/pilot-proxy/tools/prepare_sdr_protocol.py validate --spec protocol_spec.json`.
   Recorded output on 2026-09-17: protocol_valid true, hardware_adapter_available
   false, power_check average_conducted_dbm, upper_power_dbm -18.0,
   maximum_power_dbm 0.0 (`validate_output.txt`).
4. Set `PYTHONPATH=/home/djg/rail/pilot-proxy/src` and all numerical thread
   variables to 1. Power the radio, wait ten minutes before slot 0.
5. For each slot, connect the ports as the slot says, fill the log row, then run
   `$PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py prepare
   --output session/slot-NN-<mode> --build-manifest
   /home/djg/rail/results/sdr_reference_transport_2026-09-09/build-v1/build.json
   --serial 0x1d423d9108f273 --mode <noise|txzero|tone> --record-seconds 2`, followed
   by `$PY /home/djg/rail/pilot-proxy/tools/lime_reference_capture_v1.py capture
   --output session/slot-NN-<mode> --hardware-authorized --rf-confined-authorized`,
   adding `--transmit` for txzero and tone. Change attenuators only between
   records, with the helper not running. On a failure, read `rx_error` in the
   slot's `receipt.json` worker report: `RX overload guard` on a tone slot is the
   predeclared exception above; anything else stops the session.
6. After the last slot, hash every `receipt.json` and the log with `sha256sum`
   into `session/manifest.sha256` before analysis.
7. Analyse every successful record with the unchanged per-record function:
   `$PY -c "import sys; sys.path.insert(0, '/home/djg/rail/pilot-proxy/tools');
   from analyze_sdr_antenna_controls_v1 import analyze_capture;
   analyze_capture('session/slot-NN-<mode>', 'analysis/slot-NN-<mode>')"`.
   The function refuses a record whose receipt is not successful, so overloaded
   records are tabulated from their receipts only. Tabulate
   `natural_mean_frame_term_powers`, `natural_ratio`, `packed_ratio` and
   `adapter_metadata.sample_quantization.saturated_component_count` from each
   `record.json`, plus the receipt's `rx_peak_component` and `rx_max_chunk_rms`,
   into `analysis/ladder.csv`, one row per slot, overloaded slots marked.

## What is recorded

Per record, by the unchanged worker: LO, rate, LPF and gain readbacks for RX and
TX, TX-disabled register readbacks before and after (noise mode), stream
counters, per-chunk peak and rms, timestamps, cleanup returns and the attempt
receipt. By hand: the base pad, the attenuator settings and part identities,
cable and terminator identities, room temperature if a thermometer is present.
The worker does not read the chip temperature and is not modified, so chip
temperature is recorded as not available.

## Acceptance criteria (written before the data)

Two clipping measures appear below and are not the same: the worker's overload
guard (raw chunk component 0.98 or rms 0.35, fails the record) and the
analyzer's `saturated_component_count` (resampled component beyond the 4-bit
rails at scale 500, counted on successful records).

1. Completeness: every terminated slot (0, 2, 24, 25) and every commanded rung at
   12 dB or more succeeds with confirmed cleanup and exact counts; 26 of 26 slots
   attempted. A commanded 0 or 6 dB record may be overloaded; any other failure
   fails this criterion.
2. Clipping: every terminated record and every commanded rung at 12 dB or more has
   zero saturated components and passed the worker's guard. Overloaded or
   saturated commanded 0 or 6 dB records stay in the table and are excluded from
   the fit; they are reported, not corrected.
3. Qualified rung: successful, no saturated components, and its natural target
   power at least 10 dB above the mean natural target power of the two terminated
   nulls.
4. Ladder linearity: least-squares line of natural target power (dB) on commanded
   attenuation over the qualified records; slope between -1.05 and -0.95 dB/dB;
   every qualified residual within 1.0 dB (step-attenuator accuracy class); at least
   four qualified rungs, and the three repeats of any rung within 0.5 dB of each other.
   Fewer than four qualified rungs means the ladder is not established, not failed.
5. Terminated null width: for each null, finite natural ratio median within 0.85 to
   1.15 and frame standard deviation (ddof 1) between 0.07 and 0.20 (ideal
   F(256,512) gives 0.109); the before and after medians differ by at most 0.05;
   both null target powers at or below 0.0142, the median natural target power of
   the five 2026-09-09 receiver-only ambient records (their minimum, 0.0081, is a
   single noisy draw and is not the bar).
6. Terminated TX-zero: same width window as the nulls; its target power compared
   with the null is reported as a number, without a claim.

## Scope

Meeting these criteria closes the evidence-matrix cell "single-input noise and
steady-signal references: physical reference pending" with a physical single-input
reference: a matched-load null and a conducted steady tone at commanded levels,
in the detector's own units. Against the acceptance list in
`pilot-proxy/docs/SDR_UPGRADE_ADAPTER.md`, it closes step 3 (terminated noise,
then stable signal plus that noise at predeclared levels) and the single-input
part of step 4; it does not address step 1 (frequency sign and offset with known
positive and negative offsets; the +100 kHz sign is fixed only by the 2026-09-09
tone landing in the target bin), step 2 (transfer, leakage and image response
across the supports) or step 5 (stateful streaming and the fine-detector
handoff). It does not establish sensitivity, a receiver operating
characteristic, a thermal noise temperature, a telescope or array deployment
result, or absolute dBm; the power file in the spec is a declared bound, not a
meter reading. The 2048-input endpoint is measured on CHIME itself.

Schema placeholders in `protocol_spec.json` that are not claims: the ladder
slots carry `data_shelf_snr_db` equal to minus the commanded attenuation because
the schema requires a number for `mixture` slots (it is not an SNR);
`antenna_separation_cm` 30 is a required positive field and names the cable run;
`noise_model` is the schema's null model behind the F(256,512) reference width,
not a statement that the terminated null is independent Gaussian.
