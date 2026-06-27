# Per-Pilot Detector Product Schema (v2)

The `pilot-proxy-detector` datatrawl analyzer fans out one **authoritative per-pilot
product** per CHIME coarse channel: `<freq_id>.npz` (under the scan's
`_per_pilot/` directory). Every reporting artifact — the combined products in
`DATA_PRODUCTS.md`, spectrograms, F-statistic distributions, LST folds — is
**derived** from these files. They are the source of truth; nothing downstream
recovers information they do not contain.

`schema_version = "pilotproxy_detector_datatrawl_v2"`.

The combined products (`chime_detector_outputs.npz` etc.) keep the legacy field
name `mask` for byte-compatibility with the standalone runner; the per-pilot
product below uses the clearer `reject_mask`. They hold identical values.

---

## What changed from v1

- `mask` → **`reject_mask`** (`1 = discard`; same positive-excess values).
- Added **`integrated_spectrum_before_mask` / `_after_mask`** (per-bin power).
- Added a **per-unit absolute-time axis** + per-frame unit tags (below).
- Added provenance: **`weights_hash`, `detector_version`, `mask_rule`**.

The schema bump is load-bearing: `resume()` refuses to extend a v1 product, so a
run started under the old schema can never be silently mixed with v2 output. A
v1 → v2 transition starts the product fresh.

`N` = number of frames (one per `nfft`-sample chunk). `U` = number of units
(input files) consumed into this product.

---

## Identity and geometry

| Array                  | Shape  | Dtype     | Meaning                              |
| ---------------------- | ------ | --------- | ------------------------------------ |
| `freq_id`              | `(1,)` | `int64`   | CHIME coarse-channel id              |
| `physical_channel`     | `(1,)` | `int32`   | ATSC physical channel                |
| `pilot_in_band`        | `(1,)` | `uint8`   | `1` if the ATSC pilot is in band     |
| `pilot_frequency_hz`   | `(1,)` | `float64` | ATSC pilot RF frequency              |
| `chime_frequency_hz`   | `(1,)` | `float64` | CHIME coarse-channel center          |
| `nfft`                 | scalar | `int64`   | FFT / frame length (16384)           |
| `detector_window_samples` | scalar | `int64` | Kernel window `K` (128)            |
| `num_input_streams`    | scalar | `int64`   | Feed/polarization streams summed     |
| `sense`                | scalar | `int64`   | Spectral sense (`+1` / `-1`)         |

When `pilot_in_band == 0` the channel carries no in-band pilot: every frame is
emitted `valid = 0`, `reject_mask = 0`, with zeroed powers, and **both integrated
spectra stay zero** (no valid frame enters them).

## Per-frame detector output (length `N`)

| Array            | Shape    | Dtype     | Meaning                                  |
| ---------------- | -------- | --------- | ---------------------------------------- |
| `frame_index`    | `(N,)`   | `int64`   | 0-based positional frame counter         |
| `p_target_u64`   | `(N, 1)` | `uint64`  | Target-bin power (raw fixed point)       |
| `p_ref_sum_u64`  | `(N, 1)` | `uint64`  | Lower + upper reference power            |
| `fstat_raw`      | `(N, 1)` | `float64` | `2*p_target / p_ref_sum`                 |
| `fstat_level_db` | `(N, 1)` | `float64` | `10*log10(F)`                            |
| `pnr_bin_db`     | `(N, 1)` | `float64` | One-bin pilot-excess PNR                 |
| `snr_shelf_db`   | `(N, 1)` | `float64` | Estimated ATSC data-shelf SNR            |
| `valid`          | `(N, 1)` | `uint8`   | `p_ref_sum != 0`                         |
| `reject_mask`    | `(N, 1)` | `uint8`   | `1 = discard` (positive excess)          |
| `baseband_power_linear` | `(N, 1)` | `float64` | Per-frame non-coherent baseband power |

`p_target_u64` / `p_ref_sum_u64` are the **raw num/den** kept verbatim: any
alternative threshold or dB recalibration is a post-hoc recompute against these,
never a re-run.

### Mask convention

`reject_mask` is parameter-free positive excess — discard a frame when the pilot
bin exceeds the reference mean:

```text
reject_mask = valid && (2 * p_target_u64 > p_ref_sum_u64)
            = valid && (p_target_u64 > (p_ref_sum_u64 >> 1))     # mask_rule
```

The multiply-ready **`keep_mask = 1 - reject_mask`** is derived in reporting
(e.g. a spectrogram is `power ⊙ keep_mask`); it is not stored.

## Integrated power spectra

Rectangular-window `|FFT|^2` of each frame's raw samples, summed over feeds and
accumulated over frames. The window is permanently rectangular because the stored
quantity is the accumulated `|FFT|^2`, not per-frame complex spectra.

| Array                             | Shape     | Dtype     | Meaning                          |
| --------------------------------- | --------- | --------- | -------------------------------- |
| `integrated_spectrum_before_mask` | `(nfft,)` | `float64` | Σ over **valid** frames          |
| `integrated_spectrum_after_mask`  | `(nfft,)` | `float64` | Σ over **kept** frames           |

`before` integrates valid frames; `after` integrates valid **and** not-rejected
frames, so `before - after` is exactly the spectrum the mask removed. Stored
raw-accumulated; per-feed / per-frame normalization is a reporting choice. Bin
`k` is baseband frequency `((k + nfft//2) % nfft - nfft//2) * fs / nfft`, with
`fs = 1 / unit_delta_time`.

The FFT runs on the GPU (cupy) on a CANFAR node and on numpy off-GPU; the
arithmetic is identical (float64 accumulation of the feed sum).

## Absolute-time axis

Per-unit values read from each file's HDF5 root attrs (aligned 1:1 with
`unit_order`), plus per-frame tags that locate each frame in that axis. This
avoids storing a timestamp per frame.

| Array               | Shape  | Dtype     | Meaning                                  |
| ------------------- | ------ | --------- | ---------------------------------------- |
| `unit_time0_ctime`  | `(U,)` | `float64` | File start time (UNIX), NaN if absent    |
| `unit_time0_fpga`   | `(U,)` | `uint64`  | FPGA frame counter at start, 0 if absent |
| `unit_event_id`     | `(U,)` | `int64`   | CHIME event id, `-1` if absent           |
| `unit_delta_time`   | `(U,)` | `float64` | Sample period (s), NaN if absent         |
| `archive_version`   | `(U,)` | `str`     | CHIME archive version, `""` if absent    |
| `frame_unit_index`  | `(N,)` | `int32`   | Unit `u` each frame belongs to           |
| `frame_in_unit`     | `(N,)` | `int32`   | Frame's time position within its unit    |

Per-frame wall time, with `u = frame_unit_index[f]`:

```python
t[f] = unit_time0_ctime[u] + frame_in_unit[f] * nfft * unit_delta_time[u]
```

A synthetic file carries only `freq`; the time axis then degrades to
`NaN / 0 / -1 / ""` rather than failing the run.

## Provenance and calibration

| Array               | Shape  | Dtype     | Meaning                                  |
| ------------------- | ------ | --------- | ---------------------------------------- |
| `weights_hash`      | scalar | `str`     | SHA-256 of the selected detector weights |
| `weight_bank_sha256` | scalar | `str`    | SHA-256 of the complete weight bank      |
| `weight_manifest_sha256` | scalar | `str` | SHA-256 of the adjacent manifest         |
| `detector_version`  | scalar | `str`     | Package, source-tree, and kernel identity |
| `mask_rule`         | scalar | `str`     | The positive-excess rule (above)         |
| `reference_placement_json` | scalar | `str` | Auditable placement summary (JSON)   |
| `rational_overflow_count` | scalar | `uint64` | Kernel fixed-point overflow tally   |
| `max_chunks_per_file` | scalar | `int64` | Per-file cap, or `-1`                    |
| `detector_contract_json` | scalar | `str` | Full detector contract (JSON)           |
| `pilot_below_data_db`, `bin_enbw_hz`, `dtv_bandwidth_hz`, `pilot_capture_efficiency` | scalar | `float64` | dB-calibration constants |

Plus resume/alignment keys: `unit_keys`, `unit_order`, `source_event_keys`.
A resume is accepted only when these configuration/provenance fields match the
current analyzer exactly; otherwise the run must use a clean output directory.

---

## Derived in reporting (not stored)

Computed downstream from the fields above; never baked into the run so the choice
of binning / normalization / threshold stays free:

- `keep_mask = 1 - reject_mask` (the `power ⊙ keep_mask` array).
- Carrier spread from `integrated_spectrum_before_mask` (peak bin, width, sidebands).
- 23-channel baseband before/after from `baseband_power_linear` gated on `reject_mask`.
- F-statistic PDF / histogram / **survival** from raw `p_target_u64` / `p_ref_sum_u64`.
- Masked fraction from `reject_mask` / `valid`.
- Spectrum before-vs-after from the two integrated spectra.
- LST / time dependence from the absolute-time axis at CHIME's longitude.

Full 6 MHz DTV-channel mask expansion is intentionally **not** in this run; it is
a separate, larger pass once the pilot F-statistic is trusted.
