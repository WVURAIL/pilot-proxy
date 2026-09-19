# Where to pick up after 2026-09-18

State of record: dissertation commit 4b14826 on main (github.com/djgormley/dissertation), CI green, 285 pages.
The ruling: six DTV channels are excised for the 21 cm BAO measurement with both credits against the least strict
tolerance. 17, 30, 31, 33 and 35 on the long BAO baselines (bars 24.0, 30.8, 26.4, 26.5, 17.0 dB) and 15 on the short
(19.8 dB). Channel 16 is marginal. Every other channel is undetermined with its reason; no channel keeps, and no
rescue exists anywhere (the one keep cell, channel 23 on the short range, fails the no-credit rescue test by 14.5 dB).

Amendment 11 (2026-09-19) took channel 22 out. Amendment 5 reads a class's coherence time on the polarisation that
sets its level; after amendment 8 made the ruling per class that became a per-class choice and was never implemented
as one, because the class files were produced without the predeclared --pol-table, so cadence_tau.py fell back to a
per-class argmax of the raw median level over all fourteen epochs. On four of the seven then-excised channels that
picked a different polarisation from the one the ruling prices. Channel 22's only measured class is 9.8 m on
polarisation 0, where the estimator REFUSES (trim spread 471); the 2068 s behind its 16.2 dB bar came from
polarisation 1, whose level there is 1.6 floor scatters, at the control floor. cadence_tau.py now takes --pol 0|1 and
is run per class on each polarisation (the fallback run reproduces the files of record exactly); ruling_baseline.py
reads the row matching the class's priced polarisation. Item 2 aggregates a range's gain only over classes whose
excess is measured above the floor, which removed the table's only keep cell (channel 18). The amendment removes an
excision and adds none. That direction test is the thing to quote if anyone asks whether the amendments were fitted
to the result.

Amendment 10 (2026-09-18) is why the count moved from five to seven, and it matters for anyone re-reading the record.
Audit finding R3, and amendment 9 which acted on it, claimed the forecast prices no baseline below 20 m and withdrew
the 9.8 and 19.5 m range from the ruling. That was wrong. The four response banks behind both tolerance worlds are
config chime2022, built on CHIME's as-built baseline density (RadioFisher/chime2021/array_config/nx_CHIME_800.dat,
support 0.18 to 101.8 m, 11 per cent of the integral below 20 m). The audit had read a hardcoded label in the bank
metadata instead of the experiment settings recorded beside it. No bank rebuild was ever needed. Evidence:
frame_analysis/forecast_baseline_domain.py and FORECAST_BASELINE_DOMAIN.txt. Amendment 10 also made the least trim
probe a condition of every excision (this demotes channel 16 to marginal) and prints a rescue coherence on every
excised row, the coherence a per-day ground filter would have to reach to save the channel: 0.982 to 0.999 against a
measured 0.34 to 0.84. Amendment 12 (2026-09-19) is the systematic version of the check that caught amendment 11. The whole
predeclaration was audited against the whole implementation: 68 text-versus-code differences claimed, 24 verified
adversarially, 15 confirmed, 9 refuted, none moving a verdict. Four corrections adopted (a bound class now enters
its range's coherence time at its lower end, per amendment 7's own words, raising 17, 30 and 33 to 24.0, 30.8 and
26.5 dB with no verdict change; least trim probe read at the record's precision; lowest-epoch bound on the
per-class polarisation; the at-floor decibel figure renamed the detection gate). Four declined with reasons
recorded, including per-polarisation rho, which would excise channel 16 by WITHDRAWING a credit and so is barred
by the asymmetric rule. If anyone asks whether the ruling is fitted, this audit is the answer.

Amendment 13 (2026-09-19) finally rules channel 24, the one channel that had no data at all. The 2020
CANFAR per-pilot baseband cohort (~/rail/datasets/baseband/canfar_pilots_10s) covers its band. Through
the same estimator on the same long BAO classes it is 48.6 scatters above channel 35, whose transmitter
was off until 2021-10 and so gives a transmitter-off null IN THE SAME FRAMES. Over tolerance by 30.4 dB
at its archive-measured 7777 s, and by 3.2 dB even at 15 s. CONDITIONAL on the transmitter still being
on: the archive has no declared off epoch through 2026-04 and its bin stopped on 2026-04-16 because the
node went away. It is NOT merged into the 2026 table (no ch37 bin in the cohort, one coarse bin per
file, 8 frames vs 33). Reproduce: ch24_2020_basis.py, also in ~/rail/output/channel-ruling-campaign-2026-09-19/.
The same cohort is a positive control and dates the silence of 26, 27 and 32. And every undetermined row
now publishes its detection gate in dB over tolerance as {range}_gate_db, so undetermined is a measured
sensitivity limit with a number, not a blank.

Read the disclosure at the end of the predeclaration before quoting the two short-range rows:
they clear the three-scatter detection gate by 0.1 to 1.1 scatters, where the five long-range rows clear it by 19 to 320.

Amendments 1 to 14 and both adversarial audits are in
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
- Table of record: frame_analysis/table_of_record.csv (sha256 9e090aa7...; the amendment-9 table is kept as table_of_record_before_amendment10.csv, sha256 8f1a510b, and the amendment-10 table as table_of_record_before_amendment11.csv, sha256 939546f4), vendored in the dissertation as
  figure_src/data/record/table_of_record.csv; fragments regenerate with
  `~/.venvs/dissertation/bin/python scripts/table_of_record_tex.py figure_src/data/record/table_of_record.csv`.

## To rebuild the ruling (in frame_analysis/, venv ~/rail/venvs/archive-local)
1. python ruling_baseline.py table_of_record_before_amendment8.csv table_of_record.csv ../../../ch33_rescan/products
   (reproduces sha256 9e090aa7 exactly; check that before trusting any re-run)
   If cadence_tau_*_pol{0,1}.csv are missing, regenerate them first:
     for C in 0,1 0,32 0,64 0,128 0,255 1,0 2,0 3,0; do for P in 0 1; do
       python cadence_tau.py --level polmax --trim archive --class $C --pol $P <out>.csv <label>=<dir> ... ; done; done
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
5. SDR bench: RUN AND CLOSED, 2026-09-19. Result: tools/capture/sdr-bench-2026-09-18/RESULT_2026-09-19.md.
   The commanded-amplitude ladder failed (that knob is a quantization floor below 0.005 on this device, measured);
   amendment 2 replaced it with a transmit-gain ladder, which passes its predeclared criteria: slope 0.941 dB per dB,
   residual scatter 0.19 dB over 24 dB, bracket -0.27 dB, two independent passes. Floor stable to 0.2 dB, cabled
   floor equal to the terminated floor within 0.1 dB (no ambient contamination), TX-on minus TX-off 0.09 dB, the two
   pads equal within 0.2 dB. THE REMAINING ACTION is to carry this into the dissertation: the evidence matrix cell
   "single-input noise and steady-signal references: physical reference pending" is now closed, and appA, the SDR
   paragraphs of ch02/ch06 and STUBS.md should say so, with the amplitude-ladder failure reported as its own finding
   and the slope's 6 percent shortfall stated as inseparable between the device gain table and the estimator.
   After editing: make manifest, commit, push, CI (the usual dissertation loop).
   PRIOR DESIGN, superseded: redesigned 2026-09-18 for the kit in hand (LimeSDR Mini,
   2 terminators, 2 x 30 dB pads, 1 cable): the fine ladder is the commanded tone amplitude in six 6.02 dB steps and
   the pads move it by a known 30 dB, with two coincidence pairs where the analog and digital routes must agree.
   Runbook and the six validated rung protocols: ~/rail/output/sdr-bench-2026-09-17/BENCH_RUNBOOK.md and rungs/a0..a5
   (copy without payloads in tools/capture/sdr-bench-2026-09-18/; regenerate payloads with make_spec_v2.py).
   36 records, about forty minutes. Nothing in it is tuned after a record.
6. Future work named in the text: rebuild the RFIsher bank with the as-built CHIME baseline density (RadioFisher CHIME
   layout has Dmin 20 m; audit round 2, R3) so the 10 to 20 m classes are priced; calibrated-gain stacking to lower the
   estimator floor; a per-range uncertainty on the bars.
7. Housekeeping: the chimestack recall on fir was cleaned off within hours (drop that route); the fir gain files were
   released; the buffer key on outrigger-buffer (preflight only) can stay.

## Memory
Claude Code memory for this project: ~/.claude/projects/-home-djg-rail/memory/ (frame-analysis-state-2026-09-17.md
holds the full chronology and the resume recipe; MEMORY.md is the index).
