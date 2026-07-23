# paper/ -- RASTI article + dissertation supplement

LaTeX source for "A pilot-informed F-statistic detector for digital
television in CHIME baseband data" (RASTI, in prep) and the companion
dissertation supplement. Everything here builds from the analysis chain in
`../analysis/` against the frozen 2026-07 snapshot products.

## Build

Both documents build with `latexmk` (TeX Live 2023+):

    cd paper/manuscript  && latexmk -pdf draft_article    # working draft, [DYLAN:] notes visible
    cd paper/manuscript  && latexmk -pdf main             # journal wrapper, same body (needs mnras.cls)
    cd paper/supplement  && latexmk -pdf dissertation_supplement

Build products follow the project's generated-artifact convention: if
`PP_OUT` is set (see `analysis/_paths.py`), each document's `.latexmkrc`
routes the compiled PDF and all aux files to `$PP_OUT/tex/manuscript/` or
`$PP_OUT/tex/supplement/` and the source tree stays untouched. With
`PP_OUT` unset (fresh clone, CI) the build is in-tree and gitignored.
Either way, revisions travel as dated bundles, not through git. (WSL +
SumatraPDF users: the routed PDF is reachable from Windows at
`\\wsl.localhost\<distro>\home\<user>\paper\out\tex\manuscript\draft_article.pdf`.)

## Layout

    manuscript/            article source (abstract/body/appendices .tex, refs.bib)
      figs/                the 10 figures the article \includegraphics's (committed)
      provenance/          decision memos, referee-triage rounds, verification
                           notes, small result tables, and hashes.sha256
    supplement/            dissertation_supplement.tex + figure generators
      figs/                the 12 figures the supplement includes, plus
                           per-figure generator scripts (figS_*.py, hist_explainer.py)

## Figure regeneration map

Article figures (written to `$PP_OUT`, then copied into `manuscript/figs/`;
inputs resolve via `analysis/_paths.py`, i.e. `$PP_DUMPS`, `$PP_OUT`):

| figure                          | generator                              |
|---------------------------------|----------------------------------------|
| fig1_census_context             | analysis/build_fig1.py                 |
| fig2_detector_geometry          | analysis/build_fig2.py                 |
| fig3_detection_curves           | analysis/fig3_publication.py           |
| fig_empirical_zero_points       | analysis/zero_point_study.py           |
| fig_pfb_zero_point_prediction   | analysis/pfb_zero_point_prediction.py  |
| fig_aggressive_masking_tradeoff | analysis/aggressiveness_study.py       |
| fig_seasonal_propagation        | analysis/seasonal_propagation.py       |
| fig_secular_rates               | analysis/survey_composition.py         |
| fig_tail_decomposition          | analysis/tail_decomposition.py         |
| fig_threearm_veto               | analysis/threearm_fulldepth.py         |

Supplement figures: `fig_excess_threshold_all23` from
analysis/build_excess_histograms.py; `fig_spectra_all23_*` from
analysis/build_spectra_all23.py; `fig_diurnal_mask_fraction` from
analysis/zero_point_study.py; `figS_*` from supplement/make_supp_figs.py,
supplement/make_tone_gallery.py, and the standalone scripts in
supplement/figs/. The supplementary-plots zip (spectra + histograms for
every channel, secular FRB-stratum figure, etc.) is a built deliverable
regenerated from the same scripts and is not committed.

## What is deliberately NOT in git

Large binary provenance -- results-bundle tarballs, the archived
pre-import sweep set (`run_pd_curves_cpu_1000.tar.gz`), GPU parity zips,
`all_spectra.npz` and other npz dumps -- stays in the archive channel.
Two manifests keep their identity under version control even though their
bytes are not: `manuscript/provenance/hashes.sha256` fingerprints the
paper's physical inputs (the original capture and the shipped weight
bank), and `manuscript/provenance/provenance_blobs.sha256` fingerprints
every excluded blob. If a hash and an archived blob disagree, trust
neither and regenerate.

`manuscript/provenance/*.md` are working memos (decisions, status
snapshots, adversarial referee-triage rounds). They document how results
were reached and superseded conclusions are marked as such -- read them as
lab-notebook pages, not polished text.
