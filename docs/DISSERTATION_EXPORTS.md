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

The default export is intentionally **partial**. It always includes:

- the complete 500-mile transmitter census derived from
  `data/census/census.csv`;
- the inner-120-mile subset used for the detailed map panel;
- the current epoch-operating-point summary;
- the current 23-channel evidence-status matrix; and
- the current channel-33 policy comparison.

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
- `bao_time_vs_masking.csv` — observing-time-versus-masked-fraction curves
  for the survey-amplitude, worst-bin-amplitude, and worst-bin-dilation
  targets, computed with the released `baonoise` package (bao-noise-tolerance
  must be installed; pass `--skip-forecast` to omit).

Tables are written to `exports/dissertation/inputs/` (ignored by git) and are
then supplied to the exporter through `--census-psd` and
`--bao-time-vs-masking`. The remaining optional tables
(`worked_example_spectra`, `bao_convergence`, `bao_two_walls`) are deliberately
not generated here: the worked example annotates two specific archived frames
whose identities must be named explicitly, and the two bias tables depend on
the `_Pres` bias-response bank and the dissertation draft's fine-credit and
floor-basis conventions. They remain `pending` until those calculations are
reproduced under their own conventions.

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

`data/provenance/dissertation_summary_v1.json` is the small, reviewable snapshot
used to emit the epoch, status, and channel-33 policy tables. It is intentionally
labeled `curated-dissertation-snapshot` in the manifest. Updating it is a
scientific change and should be accompanied by the analysis evidence that
justifies the new values or classification.

`dissertation_summary_v2.json` (snapshot `dissertation-draft-2026-08-18`) is
the current snapshot: it extends the status matrix to the 21 channels with
archive products at the August 2026 trawl completion, adds the epoch pairs of
the dated sign-off channels (19, 20, 26, 27) with their measured survey-rule
fractions and off-epoch shelf floors noted, and leaves the lower band's
residual-chain and tolerance rows pending. Select it with `--summary`.

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
