# analysis/ — paper analysis chain for pilot-proxy

Everything that turns scan products into the paper's figures and tables,
with no hardcoded user paths. All locations come from `_paths.py` and can be
overridden by environment variables (see `env.example.sh`); defaults assume:

    ~/paper/dumps/           perframe.npz, power.npz, all_spectra.npz,
                             dtv_snr_summary.csv (merged sweep summary)
    ~/paper/results_bundle/  an extracted results_chime-pilots_* bundle
    ~/paper/out/             everything these scripts write
    <repo>/analysis/         this directory (PP_REPO defaults to its parent)

Set up the bundle once:

    mkdir -p ~/paper/results_bundle
    tar -xzf ~/paper/bundles/results_bundle_*.tar.gz \
        -C ~/paper/results_bundle --strip-components=1

## CANFAR-side (need ~/pilot_proxy_runs/chime-pilots/_per_pilot)

| script | writes |
|---|---|
| dump_perframe.py | ~/paper/dumps/perframe.npz (~4 MB) |
| dump_power.py | ~/paper/dumps/power.npz (~3 MB) |
| dump_spectra.py | ~/paper/dumps/all_spectra.npz (~5 MB) |
| fulldepth_and_subsets.py | h0_fulldepth.csv + event_presence_signatures.npz |

(The P_d sweep itself is `scripts/run_pd_sweep.sh`; copy its merged
`dtv_snr_summary.csv` into ~/paper/dumps/.)

## Analysis (anywhere with the repo + dumps)

Run order — (1) first, the rest in any order:

1. `zero_point_study.py` — measured zero points -> **empirical_zero_points.csv**
   (+ fig_empirical_zero_points, fig_diurnal_mask_fraction). Everything below
   reads this CSV from PP_OUT.
2. `tail_decomposition.py` — common-mode evidence (fig_tail_decomposition)
3. `aggressiveness_study.py` — one-sided-depth falsification
   (fig_aggressive_masking_tradeoff + aggressive_masking.csv)
4. `seasonal_propagation.py` — detrended seasonality + secular transmitter
   history (fig_seasonal_propagation, fig_secular_rates)
5. `threearm_fulldepth.py` — ceiling/floor/power-veto ladder at full depth
   (threearm_fulldepth.csv + fig_threearm_veto); needs power.npz
6. `build_empirical_thresholds.py` — exact-integer kernel constants
   (empirical_thresholds.csv), verified frame-for-frame vs the float rule
7. `pfb_zero_point_prediction.py` — the falsified option-(b) physics
   prediction (fig_pfb_zero_point_prediction); uses <repo>/weights
8. `fig3_publication.py [sweep.csv] [label]` — publication Fig. 3 + item-2
   acceptance for threshold and positive-excess rules
9. bundle-driven: `build_deliverables.py` (Table 3 + appendix tables +
   zero-point/full-depth figs), `build_spectra_all23.py`,
   `build_histograms_all23.py`, `build_excess_histograms.py`,
   `build_fig1.py`, `build_fig2.py`, `build_concept_fig.py`

## Notes

* Scripts import `pilot_proxy` from `$PP_REPO/src` via `_paths.py`; an
  editable install works too.
* Untrusted zero points (ch24, ch30) are handled inside the scripts —
  analytic constants, one-sided ceiling — matching the adopted operating
  point everywhere.
* Regeneration is deterministic: same dumps + bundle -> byte-identical CSVs.

## Where things go (path conventions, recorded 2026-07-20)

    ~/data/<name>/            datatrawl inventories (one dir per survey name)
    ~/pilot_proxy_runs/<name> scan products (chime-pilots, chime-controls, ...)
    ~/paper/                  analysis data home: dumps/, out/, results_bundle/
    ~/archive/                transfer tarballs and completed-run bundles
                              (generate_results.py writes bundles here)
    <repo>/generated/         generated waveforms and testbench captures
    <repo>/data/provenance/   paper-grade provenance, committed (incl. the
                              event_presence_keys.csv.gz the survey scripts
                              read by default)
    pd_*, *smoke*             disposable scratch, delete freely

Sweep protocol: every evaluate-snr command pins --num-input-streams 4
and --noise-source explicitly, and every new host is qualified with a
100-trial anchor smoke before bulk runs.
