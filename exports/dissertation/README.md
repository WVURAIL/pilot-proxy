# Generated dissertation exports

Run the versioned exporter from the repository root:

```bash
PYTHONPATH=src python tools/export_dissertation_data.py \
  --output-dir exports/dissertation/v1
```

Generated export directories are intentionally ignored by git. The dissertation
bundle freezes the exact export it uses, together with the producing commit and
SHA-256 manifest. The export contract and field definitions are documented in
`docs/DISSERTATION_EXPORTS.md`.
