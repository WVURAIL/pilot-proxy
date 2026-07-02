# Paper plan

Planning skeleton for the publication. This maps every figure, table, and
statement class to the repository command that produces it and its current
status, so the manuscript is assembled from finished artifacts rather than
written ahead of them. Procedures for the pending items are in
`docs/PUBLICATION_VALIDATION.md`; this file is the editorial side of the same
plan. The manuscript itself is deliberately not started yet.

## Working title and claim

*Recovering DTV-contaminated bandwidth for 21 cm cosmology: a pilot-informed
F-statistic detector for CHIME.*

The paper's single headline claim: ATSC DTV coarse channels currently
discarded from CHIME analyses can be recovered at the ~10 s frame level by
detecting the pilot tone with a quantized-weight F-statistic whose mask sits
at the detector's own exact H0 zero-point, validated end-to-end by synthetic
curves, on-sky null tests, and injection--recovery on real baseband --- at a
quantified sensitivity advantage over total-power flagging.

## Venue

Two-tier plan; the artifact inventory below is identical for both.

- **Primary: MNRAS or RASTI** (RAS Techniques & Instruments). Instrumentation
  + cosmology-adjacent methods fit; RASTI in particular is built for exactly
  this paper shape. LaTeX class `mnras`.
- **Alternate: Journal of Astronomical Instrumentation** (prior RAIL venue,
  known editorial fit). Class `ws-jai`.

Decision point: after item-3/4 results exist (the sensitivity-advantage
number is what argues for the higher-tier venue).

## Section outline

1. **Introduction** --- DTV contamination of the 400--800 MHz band; cost to
   BAO surveys of whole-channel flagging; prior art (total-power/CFAR
   flagging, the Canary paper); contribution list.
2. **ATSC pilot structure and the detection problem** --- pilot placement,
   the 500-mile transmitter census summary, why per-frame detection (not
   excision) is the right unit for CHIME's pipeline.
3. **Method** --- F-statistic from per-row matched filters; int4 quantized
   weights; the norm-corrected positive-excess mask (`F > mu0`, exact
   integer form) and why quantization makes `mu0 != 1`; dynamic-range and
   capture-loss bounds (from `docs/METHOD_SPEC.md`).
4. **Implementation** --- pilot-proxy (CUDA kernel + rational thresholds,
   product schema with verbatim num/den for post-hoc rethresholding) and
   datatrawl (storage-safe streaming, resume, provenance); one paragraph
   each, deferring to the DS/UG documents and repositories.
5. **Validation** --- the five items, one subsection each, in the runbook's
   order: synthetic detection curves; on-sky H0 zero-point; injection--
   recovery; radiometer baseline; cleaning tradeoff.
6. **Survey results** --- 23-channel production scan: per-channel mask
   fractions, carrier-peaks findings, recovered-bandwidth headline.
7. **Implications for BAO analyses and future work** --- residual budget vs
   thermal floor; kotekan deployment path; other telescopes/standards.
8. **Data and software availability; reproducibility appendix.**

## Figure inventory

| # | Figure | Producing command | Source products | Status |
|---|--------|-------------------|-----------------|--------|
| 1 | Pilot/census context: transmitter map or pilot-frequency vs CHIME channel layout | manuscript-side from the census CSV | UHF station catalog | data done; figure at writing time |
| 2 | Detector geometry / weight response `|W(f)|^2` schematic | small script over `DetectorWeightBank` (writing time) | shipped weight bank | pending (trivial) |
| 3 | Synthetic detection curves: `P_d`(shelf SNR) per offset, Wilson bars, -32 dB threshold | `pilot-proxy evaluate-snr` (>=300 trials) + `plot-results` | evaluate-snr summary CSV/JSON | **pending GPU run** (item 2) |
| 4 | On-sky H0 zero-point: mean F vs `mu0` per channel + mask-fraction before/after | `chime-scan` runs + small notebook over `stats.json` / detector NPZ | item-1 run products | **pending on-sky** (item 1) |
| 5 | Injection--recovery linearity (`injection_recovery_linearity`) | `pilot-proxy analyze-injection-recovery` | item-3 ladder products | tooling shipped; **pending runs** |
| 6 | F-statistic vs radiometer `P_d` at matched `P_fa` (`detector_vs_radiometer_pd`) | same command as Fig. 5 | same | tooling shipped; **pending runs** |
| 7 | Cleaning operating curve (`cleaning_tradeoff_operating_curve`) | `pilot-proxy analyze-cleaning-tradeoff` | combined survey + control | tooling shipped; **pending survey** |
| 8 | Recovered bandwidth vs threshold (`recovered_bandwidth_vs_threshold`) | same command as Fig. 7 | same | same |
| 9 | Survey gallery: spectrogram + mask + before/after spectra for one loud and one quiet channel | `pilot-proxy chime-plot` | production-run figure set | tooling shipped; **pending survey** |

All figures render with `PILOT_PROXY_USE_TEX=1 PILOT_PROXY_FIGURE_FORMATS=png,pdf`;
manuscripts take the PDFs.

## Table inventory

| # | Table | Source | Status |
|---|-------|--------|--------|
| 1 | Detector parameters (K, offsets, guard bins, int4, thresholds) | `docs/METHOD_SPEC.md` / detector contract | done |
| 2 | Census summary (channels, nearest transmitters, measured offsets) | station catalog + survey carrier peaks | catalog done; offsets from survey |
| 3 | Per-channel survey summary (mask fraction, `mu0`, mean excess) | combined `stats.json` + tradeoff CSV | **pending survey** |
| 4 | Validation acceptance summary (items 1--5, pass criteria, measured values) | `docs/PUBLICATION_VALIDATION.md` acceptance lines | pending runs |

## Statements to support with artifacts

- Sensitivity advantage over the radiometer: the horizontal gap at
  `P_d = 0.9` from Fig. 6, quoted in dB with CIs.
- Recovered bandwidth: the `analyze-cleaning-tradeoff` headline
  ("X of Y MHz recovered at `tau = mu0`; Z MHz-hours"), plus percentage of
  the CHIME band currently lost to DTV flagging.
- Zero-point correctness: the Fig. 4 ratio test and the CI-run MC gate
  (`tests/core/test_mask_zero_point.py`) cited in the reproducibility
  appendix.
- Exactness/reproducibility: verbatim num/den + norms in products; the
  tradeoff tool's hard `x = 0` anchor.

## Data and software availability (draft wording targets)

- Software: `WVURAIL/pilot-proxy` and `WVURAIL/datatrawl`, tagged release
  with Zenodo DOIs (release checklist in `docs/PUBLICATION_VALIDATION.md`);
  cite via each `CITATION.cff`.
- Data: CHIME baseband via CADC/Datatrail per CHIME policy; derived products
  (detector NPZ, tradeoff CSVs, ladder manifests) archived with the paper.
- Documents: PilotProxy DS001/UG001 v1.4 and Datatrawl DS001/UG001 v1.0
  attached to the tagged releases.

## Assembly order

Fig. 3 and Fig. 4 first (they gate trust in everything else), then the
ladder (Figs. 5--6), then the survey completes Figs. 7--9 and Tables 2--4.
Manuscript writing starts once Figs. 3--6 exist; survey figures drop in as
the production scan finishes.
