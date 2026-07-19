# Paper plan

This file maps the paper's claims to figures, tables, source products, and
repository commands. We use it to keep the argument tied to finished
artifacts. Procedures and acceptance tests for unfinished work are in
`docs/PUBLICATION_VALIDATION.md`.

This is a planning record, not evidence. A statement marked pending remains
provisional until its listed artifact exists and passes the corresponding
acceptance test.

## Working title and target claim

*Recovering DTV-contaminated bandwidth for 21 cm cosmology: a pilot-informed
F-statistic detector for CHIME.*

The target headline is that a pilot-informed F-statistic can retain usable
parts of ATSC-contaminated CHIME coarse channels at the approximately 10 s
frame level. The detector uses quantized weights and places the mask at its
own `H0` zero point. Synthetic curves, on-sky null tests, real-baseband
injections, and a matched total-power comparison must establish the size and
limits of the improvement before we state this as a result.

## Venue

We retain two venue options. The artifact inventory is the same for both.

- **Primary: MNRAS or RASTI** (RAS Techniques & Instruments). The paper joins
  an instrumentation method to a cosmology use case. LaTeX class `mnras`.
- **Alternate: Journal of Astronomical Instrumentation** (prior RAIL venue,
  known editorial fit). Class `ws-jai`.

We choose the venue after validation items 3 and 4 exist. Their measured
sensitivity difference is the evidence needed to argue for the primary
venue.

## Section outline

1. **Introduction.** Define the DTV problem in the 400--800 MHz band, explain
   the cost of whole-channel flagging for BAO work, review total-power and
   CFAR methods and the Canary paper, and state the contributions.
2. **ATSC pilot structure and detection problem.** Define the pilot
   placement, summarize the 500-mile transmitter census, and explain why the
   CHIME workflow makes a frame-level mask useful.
3. **Method.** Derive the F-statistic from the three per-row matched filters.
   Then introduce the int4 weights, the exact `F > mu0` comparison, and the
   dynamic-range and capture-loss bounds in `docs/METHOD_SPEC.md`.
4. **Implementation.** Describe the PilotProxy CUDA kernel, rational
   thresholds, and product provenance. Describe datatrawl only to the degree
   needed to explain bounded staging and resume. Defer interface detail to
   the two repositories and their formal documents.
5. **Validation.** Present the five runbook items in causal order: the
   synthetic curves, on-sky zero point, injection recovery, radiometer
   comparison, and cleaning tradeoff.
6. **Survey results.** Report the 23-channel mask fractions, carrier peaks,
   and the bounded recovered-bandwidth result.
7. **Implications and future work.** Separate the measured detector result
   from later BAO propagation, Kotekan deployment, and extensions to other
   telescopes or broadcast standards.
8. **Data and software availability and reproducibility appendix.**

## Figure inventory

| # | Figure | Producing command | Source products | Status |
|---|--------|-------------------|-----------------|--------|
| 1 | Pilot/census context: transmitter map or pilot-frequency vs CHIME channel layout | manuscript-side from the census CSV | UHF station catalog | data done; figure at writing time |
| 2 | Detector geometry / weight response `|W(f)|^2` schematic | small script over `DetectorWeightBank` (writing time) | shipped weight bank | pending (trivial) |
| 3 | Synthetic detection curves: `P_d`(shelf SNR) per offset, Wilson bars, -32 dB threshold | `pilot-proxy evaluate-snr` (300-trial shakedown; 1500-trial publication grid) + `analysis/fig3_publication.py` | evaluate-snr summary CSV/JSON | runnable now on CPU (`--detector-backend cpu-reference`); same-seed GPU spot check ties it to the kernel (item 2) |
| 4 | On-sky H0 zero-point: mean F vs `mu0` per channel + mask-fraction before/after | `chime-scan` runs + small notebook over `stats.json` / detector NPZ | item-1 run products | **pending on-sky** (item 1) |
| 5 | Injection--recovery linearity (`injection_recovery_linearity`) | `pilot-proxy analyze-injection-recovery` | item-3 ladder products | injection tooling exists; **blocked on realized-power coordinate, then runs** |
| 6 | F-statistic vs radiometer `P_d` at matched `P_fa` (`detector_vs_radiometer_pd`) | same command as Fig. 5 | same | same realized-power and matched-frame preconditions as Fig. 5; **pending** |
| 7 | Cleaning operating curve (`cleaning_tradeoff_operating_curve`) | `pilot-proxy analyze-cleaning-tradeoff` | combined survey + control | tooling shipped; **pending survey** |
| 8 | Recovered bandwidth vs threshold (`recovered_bandwidth_vs_threshold`) | same command as Fig. 7 | same | same |
| 9 | Survey gallery: spectrogram + mask + before/after spectra for one loud and one quiet channel | `pilot-proxy chime-plot` | production-run figure set | tooling shipped; **pending survey** |
| 10 | Case study Fig. A: carrier-offset dispersion by service class (per detected line, pooled) | `pilot-proxy analyze-transmitter-census --lines-from-run` | per-pilot integrated spectra + census v2 | previewed at slab depth (27 lines / 14 ch); final at full survey |
| 11 | Case study Fig. B: per-channel spread vs non-primary composition, Spearman + bootstrap CI | same command | same | previewed; verdict deferred to full depth under the pre-registered n>=3 rule |
| 12 | Coherent before/after: bearing-steered beamformed spectrum, all-frames vs mask-kept accumulators | visibility analyzer (planned) over a bounded companion scan | companion-scan products + gains companion | design recorded: geometry from the CHIME Overview (arXiv:2201.07869; 305 mm pitch and published 0.071 deg rotation); feed axis in file order; serial-to-position map declared as input; gains supplied separately from the baseband event datasets. Analyzer pending build |

For camera-ready figures, run with
`PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf` and use the PDF
outputs in the manuscript.

## Deferred mask-expansion documentation

The full 6 MHz sibling-span mask-expansion result is not a PilotProxy-paper
figure. It belongs with the future CHIME overview update, where the deployed
Kotekan path and its operational context can be described together. A single
spectrogram may remain in project documentation to illustrate how a
pilot-frame mask expands across sibling channels, but it must be labeled as
an illustration rather than validation evidence or a survey result.

We retain the earlier planning record as a provisional TODO. The 2026-07-08
record reports a 1,028-event channel-32 span probe, a mean of 13.3 of 17
archived sibling channels (78%), gap counts of 973/168/354/114, and an
estimated size of approximately 4,000 files per span. Those numbers have not
been independently recomputed because the referenced
`data/sibling-probe-ch32` inventory is absent from this checkout. Restore that
inventory before using the record in the future CHIME work; do not cite these
numbers in the PilotProxy paper.

## Table inventory

| # | Table | Source | Status |
|---|-------|--------|--------|
| 1 | Detector parameters (K, offsets, guard bins, int4, thresholds) | `docs/METHOD_SPEC.md` / detector contract | done |
| 2 | Census summary (channels, nearest transmitters, measured offsets) | merged-emitter census v2 + `--lines-from-run` extraction | catalog done (v2, sharing partners merged); offsets extract from survey product spectra |
| 3 | Per-channel survey summary (mask fraction, `mu0`, mean excess) | combined `stats.json` + tradeoff CSV | **pending survey** |
| 4 | Validation acceptance summary (items 1--5, pass criteria, measured values) | `docs/PUBLICATION_VALIDATION.md` acceptance lines | pending runs |

## Pre-registered analysis decisions

We registered the following choices on 2026-07-08. At that time, the first
capped production pass was approximately two-thirds complete and no
full-depth statistic existed. Any later change is an amendment and must be
identified as such.

1. **Combine subset selection.** We start with all completed channels. At
   each step, we drop the channel that most limits the common event set only
   if the drop increases that set by at least 50%. We stop otherwise and
   retain at least 16 channels. We report the full drop curve in the appendix
   regardless of the selected stopping point.

2. **Case-study Figure B channel inclusion.** A channel enters the
   spread-versus-composition statistic only when at least 3 lines are
   extracted. Fewer points do not support the intended MAD dispersion
   estimate. We show excluded channels in gray, and `summary.json` records
   the Spearman statistic for both the qualifying subset and all channels.

## Claims and required evidence

- **Sensitivity relative to the radiometer:** use the horizontal difference
  at `P_d = 0.9` in Fig. 6. The current helper does not derive an interval on
  that crossing difference, so add and retain the interval method before
  reporting one in dB.
- **Recovered bandwidth:** use the `analyze-cleaning-tradeoff` result, "X of
  Y MHz recovered at `tau = mu0`; Z MHz-hours," with the affected-band
  denominator stated explicitly.
- **Zero-point behavior:** use the Fig. 4 ratio test and cite the Monte Carlo
  regression gate in `tests/core/test_mask_zero_point.py`.
- **Reproducibility:** retain the exact numerator, denominator, and weight
  norms in each product, and require the tradeoff tool's `x = 0` anchor.

## Data and software availability targets

- Software: `WVURAIL/pilot-proxy` and `WVURAIL/datatrawl`, tagged release
  with Zenodo DOIs (release checklist in `docs/PUBLICATION_VALIDATION.md`);
  cite via each `CITATION.cff`.
- Data: CHIME baseband via CADC/datatrawl under CHIME policy; derived products
  (detector NPZ, tradeoff CSVs, ladder manifests) archived with the paper.
- Documents: PilotProxy DS001/UG001 v1.6 and Datatrawl DS001/UG001 v1.0
  attached to the tagged releases.

## Assembly order

We first close Figs. 3 and 4 because they establish the detector response and
zero point. We then close the injection ladder in Figs. 5 and 6. The survey
products supply Figs. 7--9 and Tables 2--4 after those gates are satisfied.
Figures 10--12 remain case studies or extensions and do not replace the five
core validation items. The full mask-expansion result remains outside this
paper.
