# Dissertation figure tables (pilot-proxy side)

Frozen snapshots of tables this repository generates; the figures in
`../figures.py` render from them and the dissertation bundle vendors the
resulting PDFs. Regenerate against a products snapshot with:

| table | regenerate with |
|---|---|
| census_psd.csv | `tools/make_dissertation_tables.py --products DIR` (archive-averaged pilot-region spectra, all 21 measured channels) |
| worked_example_spectra.csv | `tools/make_dissertation_tables.py --products DIR --worked-example` (frames located and digit-verified against the published values) |
| epoch_operating_points.csv | `tools/export_dissertation_data.py --summary data/provenance/dissertation_summary_v2.json` (curated summary snapshot) |
| channel_status.csv | same export; the 23-channel evidence-status matrix |

Current snapshots derive from the complete-21 August 2026 survey products
and summary snapshot `dissertation-draft-2026-08-18`.
