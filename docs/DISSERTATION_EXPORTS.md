# Dissertation data-export boundary

The dissertation and the PilotProxy repository have deliberately different
responsibilities:

- **PilotProxy owns scientific computation, validation, and provenance.**
- **The dissertation owns typography, layout, and explanatory graphics.**

The dissertation must not import PilotProxy or depend on a mutable checkout at
build time. Instead, PilotProxy emits a small, versioned data export that the
dissertation freezes, hashes, and renders in its own visual style.

## Create an export

From the repository root:

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --output-dir exports/dissertation/v1
```

The exporter must run from a clean PilotProxy Git checkout. It refuses staged
or unstaged changes to tracked files, and it checks again after reading the
inputs so a manifest cannot present a dirty worktree as a clean `HEAD`.
`--source-commit` is an optional assertion for automation: the supplied
revision must resolve to the current `HEAD`; it cannot override the checkout's
actual provenance. Generated optional CSV inputs may remain outside Git because
their normalized exported bytes are recorded by SHA-256 in the manifest.

The default export is intentionally **partial**. It always includes:

- the inclusive 500-mile transmitter-census envelope derived from
  `data/census/census.csv`, preserving its `schema_version` and per-row
  `evidence_status` so licence-only candidates are not presented as observed
  carriers;
- the inner-120-mile subset used for the detailed map panel;
- the current epoch-operating-point summary;
- the current 23-channel evidence-status matrix; and
- the current channel-33 policy comparison.

The default summary is the explicit, versioned
`data/provenance/dissertation_summary_v3.json`; no mutable duplicate or
`current` copy is maintained. Pass `--summary` only when deliberately
reproducing an older frozen draft.

Large archive products and Fisher-forecast curves are included only when their
authoritative CSV exports are supplied explicitly:

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --output-dir exports/dissertation/v1 \
  --census-psd /path/to/census_psd.csv \
  --worked-example-spectra /path/to/worked_example_spectra.csv \
  --bao-time-vs-masking /path/to/bao_time_vs_masking.csv \
  --bao-convergence /path/to/bao_convergence.csv \
  --bao-two-walls /path/to/bao_two_walls.csv \
  --require-complete
```

A missing table is recorded as `pending` in `export_manifest.json`. The exporter
never substitutes digitized artwork, inferred values, or synthetic data.

## Generate the optional tables

`tools/make_dissertation_tables.py` produces two of the optional tables
directly from their owning pipelines:

```bash
PYTHONPATH=src python tools/make_dissertation_tables.py \
  --products /path/to/per_pilot_products
```

- `census_psd.csv` — the archive-averaged spectrum within ±15 kHz of each
  synthesized pilot, read from every per-pilot product's stored integrated
  before-mask spectrum. Offsets are reported in the transmitted-frequency
  sense (the receiver's spectral inversion is undone), so off-nominal carriers
  appear at their measured RF displacement.
- `worked_example_spectra.csv` — the worked example's two archived frames
  (generated only behind `--worked-example`): the frames are named
  explicitly by UTC day and F/mu0 ratio, and rows are emitted only after
  the digits the dissertation quotes reproduce from the product.
- `bao_time_vs_masking.csv` — observing-time-versus-masked-fraction curves
  for the survey-amplitude, worst-bin-amplitude, and worst-bin-dilation
  targets, computed with the released `baonoise` package (bao-noise-tolerance
  must be installed; pass `--skip-forecast` to omit).

Tables are written to `exports/dissertation/inputs/` (ignored by git) and are
then supplied to the exporter through `--census-psd`,
`--worked-example-spectra`, and `--bao-time-vs-masking`.

`tools/make_chain_table.py` generates the per-channel residual-chain table
(the dissertation's Table 9.6 and its lower-band extension) from the same
products via the released `baonoise` residual machinery. Its built-in
self-test reproduces the published first-measured-block constants from raw
products and aborts on any drift: the table's provenance is the
reproduction, not a remembered analysis. The remaining optional tables
(`bao_convergence`, `bao_two_walls`) are deliberately not generated here:
they depend on the `_Pres` bias-response bank and the dissertation draft's
fine-credit and floor-basis conventions. They remain `pending` until those
calculations are reproduced under their own conventions.

## Verify before import

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --verify exports/dissertation/v1
```

Use `--require-complete` for a final archival snapshot. Verification checks the
schema, completion flag, file paths, row counts, and SHA-256 hashes.

## Export contents

Each export contains:

- schema-normalized CSV tables;
- `export_manifest.json`, with source repository/commit, ownership, authority,
  row counts, and file hashes;
- `SHA256SUMS`; and
- a generated `README.md` explaining available and pending tables.

The export contains **data only**. Plotting code remains in the dissertation so
that dissertation figures share one font, canvas, palette, and annotation
grammar. Article and software-documentation figures remain in this repository
because they serve different documents and may use different layouts.

## Curated summary snapshot

`data/provenance/dissertation_summary_v3.json` (snapshot
`dissertation-draft-2026-08-19`) is the current snapshot and the exporter
default. It covers all 23 ATSC allocations, including channels 14 and 15, and
carries the corrected channel-33 policy comparison on the common
`acquisitions >= 8` population. It is intentionally labeled
`curated-dissertation-snapshot` in the export manifest. Updating the default to
a later numbered snapshot is a scientific change and must be accompanied by
the analysis evidence that justifies the new values or classifications.

`dissertation_summary_v1.json` and `dissertation_summary_v2.json` remain as
immutable historical inputs for reproducing the 2026-08-12 and 2026-08-18
drafts. Select one explicitly with `--summary` only for that purpose; neither is
the current export source.

## Large products

The exporter does not move raw captures, result bundles, NPZ dumps, or Fisher
workspaces into git. Those remain in the archive channel. Their small tabular
exports can be supplied to the exporter and are then fingerprinted in the
resulting dissertation export.

## Frozen dissertation inputs still requiring replacement

The dissertation bundle can render every current figure reproducibly, but some
plots still consume audited frozen intermediate tables rather than direct
exports from the owning scientific pipeline. The census power spectra and the
observing-time-versus-masking curves now have direct generators (above). Before
the archival dissertation release, replace the remaining frozen inputs: the
worked detector example, the BAO convergence and two-walls curves, and any
digitized introductory cosmology curves. The manifest must continue to label
these as frozen bridges until their upstream generators emit hash-pinned tables
directly.
