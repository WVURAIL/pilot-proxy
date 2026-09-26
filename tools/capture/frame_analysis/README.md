# Capture frame policies

`frame_policy_v1.py` applies the frozen current-era calibration thresholds from
the September 9 coarse release to individual capture frames. It uses exact
integer comparisons and joins the detector and correlation products by FPGA
sample identity within each event, with the source event and UTC/FPGA origins
checked before selection. Missing pilot data, incompatible banks,
missing timing metadata, and unmatched frames refuse the affected comparison.

```sh
python frame_policy_v1.py --root "${WORKSPACE_ROOT:?}" \
  --output "$WORKSPACE_ROOT/output/new-frame-policy-release"
python -m unittest -v test_frame_policy_v1
```

`WORKSPACE_ROOT` is the local workspace root, the directory that holds
`output/`, `results/`, `products/` and `datasets/`. It is not committed; the
command stops if it is unset.

The output directory must not already exist. The four evaluated policies are
the three saved calibration quantiles and keep-all. The quantiles are not
refitted to the captures or recomputed across the full archive. The nominal
channel-33 bank remains a separate diagnostic; its off-target response cannot
support an optimal-detector claim. A corrected-bank policy needs its own
calibration population, thresholds, and disjoint evaluation records. The
detector loader can validate an explicit bank contract for a separate replay;
that caller must establish the contract and threshold provenance. The default
command always uses the archived nominal bank.

The completed nominal capture release is
`output/dissertation-implementation-2026-09-19/frame-policies/all-channels-v2`
under the workspace root; its compact import and audit are in
`frame-policies/guard-validation-v2`. Nineteen regression tests pass. The
separate `corrected-ch33-calibration-v1` and `corrected-ch33-replay-v1`
releases use 11,784 matched calibration records and retain 0/0/33 of 150
capture frames under q0.1/q0.5/q0.9. They are exploratory: the inherited
nominal-bank era still needs qualification, and the capture used to choose
the corrected bank is not held-out evidence.

`complex_moments.csv` retains the frequency, baseline class, polarization,
event, and policy. Complex means are in quantized voltage-product units per
nominal time sample, averaged over the reducer's redundant products. The
reducer stores `conj(x_i) * x_j` for input order `i <= j`. Native Kotekan's
upper-product convention is `x_i * conj(x_j)`, so a comparison must conjugate
one side and verify the baseline ordering. This replay does not establish
native equivalence.

The sample covariance describes fluctuations among the retained frames in that
capture. It is not the covariance of a survey mean. The signed cross-frame
product and lag-one covariance are diagnostics; temporal independence and a
thermal-only reference are not assumed. No floor, neighboring-bin baseline,
mean phase, or downstream filter credit is subtracted.

The optional normalized mean uses measured total input power, averaged over
inputs whose all-frame mean power exceeds 0.5. This is not calibrated thermal
power or the Fisher residual template. The selected live-input power does not
repair the original stack's inclusion of other inputs. Stacked products cannot
recover individual-product calibration or arbitrary per-input masks.

`count` in the source reduction is redundant-product multiplicity. The export
keeps that quantity separate from nominal retained samples. Actual joint-valid
sample counts are unavailable and remain blank. Keep-all does not depend on
pilot validity; threshold policies require a valid reference denominator.

The captures are short, separated records. Summed retained duration is not a
contiguous integration; none is relabeled as a measured 10-second visibility.
The inverse retained fraction is explicitly a uniform-information arithmetic
diagnostic, not a validated observing-time prediction. The producer reports
every coarse bin intersecting the six-megahertz allocation, including partial
edge overlaps, and lists missing frequency coverage. Coverage counts only bins
whose products and frame joins passed validation. Present but refused files are
reported separately, and a refused bin contributes no partial moment rows.

The receipt hashes the exact threshold files, detector files, configuration,
input manifests, and the consumed arrays of visibility products. Large unused
NPZ members and raw voltage payloads are not rehashed. Raw HDF5 headers and input
maps establish the single-segment frame origins; their payload identity still
depends on the original capture record.

The other scripts in this directory reproduce scalar screening scenarios.
Their output labels are not scientific channel rulings. Use
`screen_export_v1.py SOURCE.csv NEW.csv` when exporting those scenarios: it
prefixes every scenario field with `screen_`, keeps scientific rulings
undetermined, and records the source digest. The authoritative dissertation
decision register must separately qualify the physical transfer, uncertainty,
information loss, and comparison class. A failed finite policy screen and the
lowest dump average are not lower bounds on every possible frame mask.

## Inputs of `table_of_record.py`

Apart from the per-dump products, `table_of_record.py` reads four inputs. They are CHIME collaboration products
and are not published here, so a reviewer has to get them through the collaboration.

| key | variable | name under `TABLE_OF_RECORD_INPUTS` | content |
|---|---|---|---|
| `channels=` | `TOR_CHANNELS_CSV` | `channels.csv` | tolerance without credit per channel (forecast-convergence release, 2026-09-09) |
| `worlds=` | `TOR_WORLDS_CSV` | `coarse-world-sensitivity.csv` | deployed-world tolerance and suppression (CANFAR reanalysis, 2026-09-09) |
| `board=` | `TOR_BOARD_CSV` | `board_final.csv` | archive gain bound and tau bound per channel (channel ruling, 2026-09-14) |
| `archive=` | `TOR_ARCHIVE_DIR` | `_per_pilot/` | archive per-pilot detector products `<freq_id>.npz` (archive rebuild, 2026-08-29) |

Each input is taken from its `key=` argument first, then from its variable, then from the directory named by
`TABLE_OF_RECORD_INPUTS`. For example:

```sh
TABLE_OF_RECORD_INPUTS=/path/to/inputs python table_of_record.py table_of_record.csv \
  <label>=<frame_residual csv>:<chime-run dir> ... gain=<cadence_tau csv> lowbound=<csv>,... bao=<gain_bao csv>
(cd /path/to/inputs && sha256sum -c table_of_record_inputs.sha256)
```

`table_of_record_inputs.sha256` lists the copies of record. Every run writes `<out csv without extension>.inputs.json`
next to the table. It records the sha256, size, modification time, resolved path and provenance of every file read,
covering these inputs, the dump inputs and the script itself. The script differs from the frozen copy
(sha256 `f2372bf677095b32e840ad5b075dedb0e186a5198620bb304b4ba99475ba709c`) only in how it finds and records its
inputs. On 2026-09-25, a three-dump run gave the same inputs to both versions, and they wrote the same table byte for
byte.
