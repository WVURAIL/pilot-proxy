# Paper analysis chain

The scripts in this directory turn retained scan products into the paper's
tables, figures, and audit CSVs. They do not download CHIME data or run the
detector. We keep that boundary explicit so a paper artifact can be traced to
either a stored product or a deterministic post-processing step.

For the minimum PilotProxy result, the priority artifacts are the F-statistic
histograms and the integrated before/after spectra. Scripts that fit thresholds
from CANFAR-measured means or add a common-mode power veto are retained as
exploratory work for the supplement and later development. They do not define
the current production mask, which is the fixed weight-norm comparison
`F > mu0`.

## Shared paths

Most analysis scripts import `_paths.py`. Its defaults are:

```text
~/paper/dumps/           perframe.npz, power.npz, all_spectra.npz,
                         dtv_snr_summary.csv
~/paper/results_bundle/  extracted results_chime-pilots_* bundle
~/paper/out/             generated figures, tables, and audit CSVs
<repo>/analysis/         this directory; PP_REPO defaults to its parent
```

Copy `analysis/env.example.sh`, edit it for the local checkout, and source it
before running the chain. `_paths.py` recognizes `PP_REPO`, `PP_OUT`,
`PP_DUMPS`, `PP_PERFRAME`, `PP_POWER`, `PP_SPECTRA`, `PP_RESULTS`, and
`PP_SWEEP_CSV`. `survey_composition.py` additionally uses `PP_INVENTORY` and
`PP_EVENTKEYS`.

The three `dump_*.py` scripts and `fulldepth_and_subsets.py` predate the shared
path module. They use positional input/output arguments with the defaults
shown below.

Extract a results bundle once:

```bash
mkdir -p ~/paper/results_bundle
tar -xzf ~/paper/bundles/results_bundle_*.tar.gz \
  -C ~/paper/results_bundle --strip-components=1
```

The scripts use NumPy, Matplotlib, and, for the Gaussian-model figures,
SciPy. They also import PilotProxy from `$PP_REPO/src`; an editable install is
optional.

## Build the compact dumps on CANFAR

By default, these commands read
`~/pilot_proxy_runs/chime-pilots/_per_pilot/*.npz`:

| Script | Default output | Purpose |
|---|---|---|
| `dump_perframe.py` | `~/paper/dumps/perframe.npz` | Per-frame target/reference powers, masks, frame-unit indices, and channel scalars |
| `dump_power.py` | `~/paper/dumps/power.npz` | Per-frame baseband power and target power |
| `dump_spectra.py` | `~/paper/dumps/all_spectra.npz` | Full-depth before/after integrated spectra |
| `fulldepth_and_subsets.py` | Current directory: `h0_fulldepth.csv`, `event_presence_signatures.npz` | Full-depth zero-point table and exact common-event subset search |

Each dump script accepts `<input_dir> <output.npz>`. The publication detection
sweep is generated separately by `scripts/run_pd_sweep.sh`; place its merged
`dtv_snr_summary.csv` in `PP_DUMPS` or set `PP_SWEEP_CSV`.

## Run the analysis

Run `zero_point_study.py` first. It writes
`empirical_zero_points.csv`, `fig_empirical_zero_points`, and
`fig_diurnal_mask_fraction`. The tail, threshold, propagation, and
full-depth-mask studies read that CSV.

The remaining measured-data analyses include exploratory studies. In
particular, `threearm_fulldepth.py` and `build_empirical_thresholds.py` are
deferred from the minimum detector result:

| Script | Result |
|---|---|
| `tail_decomposition.py` | Separates high- and low-F tail behavior and writes the common-mode diagnostics |
| `aggressiveness_study.py` | Tests one-sided threshold depth and writes `aggressive_masking.csv` plus its tradeoff figure |
| `seasonal_propagation.py` | Measures seasonal and secular tail behavior |
| `threearm_fulldepth.py` | Exploratory: evaluates the ceiling, band-floor, and common-mode power-veto ladder; requires `power.npz` |
| `build_empirical_thresholds.py` | Exploratory: builds CANFAR-derived exact-integer thresholds and checks them frame by frame against the float rules |
| `pfb_zero_point_prediction.py` | Tests the reference-PFB explanation against measured zero points |
| `ch30_offair_minority.py` | Records the two-component channel-30 case |
| `survey_composition.py` | Stratifies quarterly rates by archive trigger class; requires inventory and event-presence inputs |

`fig3_publication.py [sweep.csv] [label]` builds the synthetic detection-rate
figure and applies the item-2 checks. The following scripts build paper or
supplementary artifacts from the shared dumps and extracted bundle:

- `build_deliverables.py`
- `build_spectra_all23.py`
- `build_excess_histograms.py`
- `build_fig1.py`
- `build_fig2.py`
- `build_concept_fig.py`

## Interpretation and reproducibility

The zero-point trust decision is computed from the retained distributions.
In the current products, channels 24 and 30 fail that gate; downstream scripts
use their analytic constants and a one-sided ceiling where documented.

For the same inputs, software revision, and environment, the numerical CSV
outputs are deterministic. Figure files can still differ in embedded metadata
or rendering backend. Retain the input dump hashes, repository commit, and
environment with every publication bundle rather than relying on byte
identity alone.
