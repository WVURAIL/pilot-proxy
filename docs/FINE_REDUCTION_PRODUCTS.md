# Fine-reduction product fields

These fields extend the current per-pilot product described in
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md). They are measurements and diagnostics;
the product's active rejection decision remains the coarse norm-corrected
positive-excess rule recorded by `decision_contract_json`.

## Product-wide fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_status` | str | `enabled`, `disabled_by_option`, or an explicit capability-failure status. `fine_products=on` fails instead of degrading. |
| `fine_num_bins` | int64 | Padded fine-bin count. |
| `fine_pad_factor` | int64 | Zero-padding factor of the window-axis transform. |
| `fine_designated_bins` | int64[] | Designated transmitter bins used by the diagnostic analysis. |
| `fine_census_excluded_bins` | int64[] | Fine bins excluded from the diagnostic null bulk. |
| `fine_guard_fine_bins` | int64 | Guard margin around designated bins. |
| `fine_p_fa` | float64 | Requested diagnostic tail probability. |

## Per-frame fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_power_ratio` | float32 `(N, fine_num_bins)` | Fine-bin local-reference power ratio; NaN rows when unavailable. |
| `fine_cfar_location` | float64 `(N, 1)` | Diagnostic null-bulk location. |
| `fine_cfar_scale` | float64 `(N, 1)` | Diagnostic null-bulk scale. |
| `fine_cfar_threshold` | float64 `(N, 1)` | Diagnostic threshold. |
| `fine_cfar_mode` | uint8 `(N, 1)` | Encoded diagnostic threshold-estimation method. |
| `fine_null_bulk_exceedance_fraction` | float64 `(N, 1)` | In-sample fraction of the same independent null-bulk bins used to estimate the frame threshold that exceed that threshold. This is not an independently measured false-alarm rate. |
| `fine_threshold_exceedance_count` | int32 `(N, 1)` | Number of diagnostic threshold exceedances in the frame. |

The ragged exceedance list uses `fine_threshold_exceedance_frame` and
`fine_threshold_exceedance_bin`. Rows are ordered by frame, with

```python
fine_threshold_exceedance_frame == np.repeat(np.arange(N), fine_threshold_exceedance_count)
```

The exact integer sum of squared matched-filter row projections reproduces the
coarse power terms bit-for-bit. The padded fixed-point transform is separately
gated by its Parseval identity and golden vectors. These identities connect the
coarse and fine measurements without making the diagnostic fine threshold the
active rejection policy.
