# Where to pick up after 2026-09-18

State of record: dissertation commit 4b14826 on main (github.com/djgormley/dissertation), CI green, 285 pages.
The ruling: five DTV channels (17, 30, 31, 33, 35) are excised for the 21 cm BAO measurement on the forecast's
baselines (20 m and longer) with both credits against the least strict tolerance; every other channel is undetermined
with its reason; no channel keeps. Amendments 1 to 9 and both adversarial audits are in
tools/capture/frame_analysis/ (FRAME_ANALYSIS_PREDECLARATION.md, RULING_AUDIT_*.md) and in the dissertation's
chapter 9 record section. The repository notes HANDOFF.md, RESULTS_PLAN.md and archive_completion_checklist.md in the
dissertation carry a "State of record, 2026-09-18" section.

## Records and data
- Analysis tree (scripts, CSVs, notes): ~/rail/output/channel-ruling-execution-2026-09-14/rebuild/author_actions/
  (capture-runbook/reduce/frame_analysis/ is the working directory of every table and report). Mirrored on the WVU RAIL
  OneDrive under RFI Mitigation/Datasets/matched-capture-2026-09/. Scripts also here in tools/capture/.
- Reduced dump products: ~/rail/datasets/pilot_reduce_<event>/ (14 events) and on frb-analysis under
  /data/user-data/dgormley/pilot_reduce/. Raw dumps are on the CHIME archive (datatrail, baseband_<event>).
- Channel 33 rescan products: author_actions/ch33_rescan/products/ (and OneDrive ch33_rescan/).
- Table of record: frame_analysis/table_of_record.csv (sha256 8f1a510b...), vendored in the dissertation as
  figure_src/data/record/table_of_record.csv; fragments regenerate with
  `~/.venvs/dissertation/bin/python scripts/table_of_record_tex.py figure_src/data/record/table_of_record.csv`.

## To rebuild the ruling (in frame_analysis/, venv ~/rail/venvs/archive-local)
1. python ruling_baseline.py table_of_record_before_amendment8.csv table_of_record_v10.csv ../../../ch33_rescan/products
2. python robustness_v8.py; python cadence_report.py CADENCE_REPORT.md cadence_tau.csv lag_coherence_14.csv table_of_record.csv cadence_tau_a3.csv
3. In the dissertation: copy the CSVs to figure_src/data/record/, run the two generators, then
   `make manifest PYTHON=~/.venvs/dissertation/bin/python`, commit, push (main is Overleaf-linked: never rebase).
Inputs the ruling reads: class_excess_epochs.csv, class_floor_bins.csv (class_excess_epochs.py), cadence_tau*.csv
(cadence_tau.py --class ew,ns --level polmax --trim archive), lag_coherence_tenclasses_D.csv (cadence_lags_tenclasses.py),
the kernel run directories ../kernel_<event>_k230 and _k230_ch33measured, the archive per-pilot products.

## Open items, in order of value
1. Your read of the abstract, chapter 9's record section and chapter 11 as the author.
2. Post the 1681 reply (draft in the session notes; close it). 1673 and 1677 are mergeable and await reviewers.
3. Rik: the one-line follow-up on whether nodes will again be dropped for RFI after the X-engine upgrade
   (bins 767, 752, 690, 598, 583, 537). Upgrade hoped for October 2026; all 1024 channels planned.
4. Post-upgrade dump (one 1.4 s dump once the upgraded correlator records all channels): trigger_dump.py, then
   process_dump.sh <event>, then steps 1 to 3 above. Closes channel 24 and lets the six pilot bins' masks be read.
   Include a control block outside the DTV band if possible (audit S1).
5. SDR bench (optional; closes an evidence-matrix cell): ~/rail/output/sdr-bench-2026-09-17/BENCH_RUNBOOK.md and
   protocol_spec.json; pads must give a fixed 24 dB plus a 0 to 36 dB ladder in 6 dB steps.
6. Future work named in the text: rebuild the RFIsher bank with the as-built CHIME baseline density (RadioFisher CHIME
   layout has Dmin 20 m; audit round 2, R3) so the 10 to 20 m classes are priced; calibrated-gain stacking to lower the
   estimator floor; a per-range uncertainty on the bars.
7. Housekeeping: the chimestack recall on fir was cleaned off within hours (drop that route); the fir gain files were
   released; the buffer key on outrigger-buffer (preflight only) can stay.

## Memory
Claude Code memory for this project: ~/.claude/projects/-home-djg-rail/memory/ (frame-analysis-state-2026-09-17.md
holds the full chronology and the resume recipe; MEMORY.md is the index).
