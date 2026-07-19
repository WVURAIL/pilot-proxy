# T2 estimator-convention and sample-span provenance (2026-07-18)

`T2` is the project label for the second technical review item: whether the
reported ATSC pilot-to-data ratio changes with the estimator convention or
with the sample span. We keep the two effects separate because either one can
move the calibration number without changing the detector.

Until the missing audits and stationary-span sweep are retained, T2 remains
open and every number that depends on this calibration is provisional.

## Artifacts committed in this directory

- `audit_orig_fullspan.json` is the extended audit of the original capture at
  `--max-samples 600000`. It uses 524288 samples, or 8 complete segments, and
  reports `measured_pilot_below_data_direct_db = 11.5950992961` dB.
- `audit_trim_fullspan.json` applies the same audit to the trimmed capture. It
  uses 524288 samples and reports
  `measured_pilot_below_data_direct_db = 11.0828912346` dB.
- `block_profile_20260718.txt` records the ten-block coherent projection that
  identifies the generator startup transient.

## Recorded 2-by-2 comparison

The default-span direct audits use 262144 samples and record 12.024 dB for the
original capture and 11.497 dB for the trimmed capture. Together with the
full-span records, the comparison gives:

- sample-span changes of `-0.43/-0.41` dB at fixed method;
- method-versus-projection differences of `-0.24/-0.22` dB at the long span;
- trim shifts of `-0.527/-0.512/-0.530` dB across the three recorded
  conventions; and
- data power equal to nine significant digits between the two full-span
  captures.

For the first two bullets, each pair is ordered original/trimmed. The three
trim-shift values are ordered default-span direct/full-span direct/projection.
These values are the recorded decomposition; it is not yet self-contained
because the default-span audit JSON files are not both present in this
directory.

## Status of the remaining artifacts

- **Still to retain:** `audit_v3.json`, the default-span audit of the trimmed
  capture with direct result 11.497 dB. The planning record places this file
  at `~/audit_v3.json` on `cupy-gpu`; it is not in this checkout.
- **Completed at the repository root:** `trim_report.json` has been
  regenerated after the `gain_db` formula correction. It now records
  amplitude factor `1.0629299002401351` and power gain
  `0.5300924787490547` dB. The earlier value 0.265 dB was a reporting-formula
  error; the trimmed capture itself was not the error.
- **Partially completed:** `trim_report_stationary.json` is committed at the
  repository root. The cropped capture and `audit_v4.json` are not committed,
  and the crop -> retrim -> audit -> sweep sequence required by the G3b
  protocol remains unfinished.

T2 closes only after the missing audit JSON files, stationary input identity,
and stationary-span sweep are retained together with the exact commands used
to produce them.
