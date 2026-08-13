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

## Large products

The exporter does not move raw captures, result bundles, NPZ dumps, or Fisher
workspaces into git. Those remain in the archive channel. Their small tabular
exports can be supplied to the exporter and are then fingerprinted in the
resulting dissertation export.
