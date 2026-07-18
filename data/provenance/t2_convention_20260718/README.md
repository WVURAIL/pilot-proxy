# T2 estimator-convention / sample-span provenance (2026-07-18)

Committed here:
- audit_orig_fullspan.json  extended audit of the ORIGINAL capture at
  --max-samples 600000 (num_samples_used 524288 = 8 whole segments);
  direct field 11.595 dB.
- audit_trim_fullspan.json  same for the TRIMMED capture; direct field
  11.083 dB.
- block_profile_20260718.txt  ten-block projection profile identifying
  the generator startup transient.

The 2x2 decomposition with the default-span audits (original 12.024 /
trimmed 11.497, span 262144) gives: span effect -0.43/-0.41 dB at fixed
method; method-vs-projection -0.24/-0.22 dB at the long span; trim shift
-0.527/-0.512/-0.530 dB across all conventions; data power identical to
nine significant digits between captures.

STILL TO ADD (Dylan):
- audit_v3.json (default-span audit of the trimmed capture; direct
  11.497) — file at ~/audit_v3.json on cupy-gpu.
- regenerated trim_report.json (the committed one predates the gain_db
  formula fix: it shows 0.265 where the amplitude factor 1.0629299
  implies 0.530092 dB power gain; the capture itself is correct).
- the stationary-span set (crop -> retrim -> audit -> sweep): cropped
  capture trim_report_stationary.json + audit_v4.json, per the G3b
  protocol in the manuscript provenance memos.
