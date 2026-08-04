# Per-pilot detector product schema (v3)

`schema_version = "pilotproxy_detector_datatrawl_v3"`

v3 is v2 (see `product_schema_v2.md`) plus the time-coherent fine-reduction
products introduced with kernel core 2.0.0 and `pilot_proxy.fine_reduction`.
All v2 arrays are unchanged. This document lists the v3 additions and the
invariants the test suite enforces on them.

## Fine-reduction fields

Scalars (product-wide):

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_status` | str | `enabled`, or `kernel_library_lacks_row_sums` when the loaded library has no v2 row-sum front end (`fine_products=auto`). `fine_products=on` hard-fails instead of degrading. |
| `fine_num_bins` | int64 | Padded fine-bin count: `FINE_PAD_FACTOR * windows_per_stream` (256 for the 16384/128 geometry). |
| `fine_pad_factor` | int64 | Zero-padding factor of the window-axis FFT (2). |
| `fine_designated_bins` | int64[] | Designated transmitter bins (default `[0]`, the nominal pilot). |
| `fine_census_excluded_bins` | int64[] | Bins excluded from the CFAR null bulk by census knowledge (default empty). |
| `fine_guard_fine_bins` | int64 | Guard margin around designated bins (1). |
| `fine_p_fa` | float64 | Pre-registered per-bin false-alarm probability (1e-3). |

Per frame (first dimension = `n_frames`, aligned with `frame_index`):

| Field | Type | Meaning |
| --- | --- | --- |
| `fstat_fine` | float32 `(n, fine_num_bins)` | Fine-bin statistic `F2[b]`; NaN rows for invalid frames or when fine products are off. |
| `fine_cfar_location` | float64 | Null-bulk location (median, or P25 in fallback mode). |
| `fine_cfar_scale` | float64 | Null-bulk scale (left-side spread). |
| `fine_cfar_threshold` | float64 | Detection threshold; finite exactly on valid frames, and `threshold > location`. |
| `fine_cfar_mode` | uint8 | 0 = none, 1 = `median_left_side_scale`, 2 = `quantile_fallback`. |
| `fine_nd_flag_rate` | float64 | Fraction of independent non-designated bins flagged; the live false-alarm monitor. On DTV-occupied channels an off-nominal real pilot lives in non-designated bins, so an elevated rate can be signal, not miscalibration. |
| `fine_detected_count` | int32 | Number of detected bins in this frame (any-bin rule). |

Ragged detection list (row dimension = `sum(fine_detected_count)`):

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_detected_frame` | int64 | Global frame index of each detection row. |
| `fine_detected_bin` | int64 | Detected fine bin of each detection row, in `[0, fine_num_bins)`. |

**Invariant (enforced):** the rows are ordered by frame, and

```python
fine_detected_frame == np.repeat(np.arange(n_frames), fine_detected_count)
```

`fine_detected_count` is the authoritative partition. A product whose
detections are stamped with the unit's first frame index instead is exactly
repairable in place with `tools/repair_fine_frame_labels.py` (idempotent,
rewrites only this column).

## Runtime identities (not stored, enforced at production time)

The exact int64 v1 marginal identity ties the fine path to the v1 statistic
per frame: the integer sum over rows of `|z|^2` from the v2 row-sum front end
must reproduce the deployed v1 power kernel bit-for-bit, and the float
Parseval identity gates the FFT path at ULP tolerance
(`tests/kernel/test_row_sums_gpu.py`). `fstat_raw` therefore remains the v1
all-rows statistic, with `fstat_fine` its coherent refinement.

## `detector_version` token semantics

```text
pilot-proxy/<version> source=<sha256> kernel=<core> kernel_sha256=<sha256>
pilotproxy_detector_datatrawl_v3 K=<detector_window_samples>
```

`source=` hashes every `*.py` under `src/pilot_proxy` (path-relative), is
memoized per process (one stamp per scan process even if the tree changes on
disk mid-run), and is build provenance only: resume treats a source-only
difference as forgivable when the kernel, K, and schema tokens match, and
stamps the product at each channel's `begin()`. Map stamps back to commits
with `python -c "from pilot_proxy.provenance import package_source_sha256 as
h; print(h())"` at the relevant checkout.
