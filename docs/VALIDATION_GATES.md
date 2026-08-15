# Validation gates for the chime-pilots relaunch

This is the complete gate set for restarting the production scan after the
2026-07-29 baseband frame correction and the kernel 2.0.0 (v2 fine
reduction) upgrade. Every gate has a command and a pass criterion. The scan
does not relaunch until every gate in both sections has passed; the
container-verified section records the evidence already in hand, and the
A100 section is the remaining mandatory work on production infrastructure.

Ground-truth inputs used throughout: CHIME/FRB baseband raw event
`68399317.h5` (freq_id 844, DTV channel 14) and the first-epoch
`chime-pilots` per-channel products (now the *legacy-halfband epoch*; see
PAPER_PLAN Amendment A1).

## Section 1 - verified in-container (2026-07-29)

1. **Frame audit on the rebuilt bank.**
   `PYTHONPATH=src python3 tools/framing_audit.py <event.h5> --chunks 4`
   Pass: `VERDICT: ALIGNED`, exit 0. Measured: target lobe +3056.4 Hz,
   miss +0.52 kHz (equal to the strongest transmitter's -517 Hz offset),
   deployed target response at the line -0.4 dB.

2. **Deployed-configuration recovery of the buried detection.** The numpy
   replica of the production configuration (2048 streams, per-stream
   3-sigma int4 quantization, spectral flip, shipped int4 weights) on the
   event file. Pass: per-chunk v1 F far above the mistuned baseline.
   Measured: F = 12.85-13.99 over 12 chunks (mean excess +1228%), versus
   0.997 +/- 0.002 with the archived legacy bank.

3. **Full test tree.**
   `PYTHONPATH=src python3 -m pytest tests -q`
   Pass: zero failures. Measured: 204 passed, 16 skipped (GPU- and
   datatrawl-dependent tests skip off-infrastructure by design).

4. **CPU kernel-reference self-test (v2 semantics).**
   `make -C cuda test_ref`
   Pass: builds with g++ and exits 0. Covers exact row sums against an
   independently ordered brute force and the bit-exact all-bin v1 marginal
   identity.

5. **Legacy reproducibility.** Profiles without a declared
   `baseband_frame` reproduce the historical generation bit-for-bit and
   carry `baseband_frame_warning` in their manifests
   (`tests/core/test_integration_contract.py`,
   `tests/test_cli.py::test_make_weights_reports_adaptive_reference_on_stderr`
   retain the legacy reference profile as a regression fixture).

## Section 2 - mandatory on the A100 before relaunch

Run in order on the production node (`cupy-gpu`); stop at the first
failure.

1. **Rebuild the kernel library.**
   `make -C cuda clean && make -C cuda`
   Pass: builds; `FStat_GetVersion` reports 2.1.x
   (`python3 -c "from pilot_proxy.kernel import FStatKernel; print(FStatKernel().version.as_string())"`).

2. **CUDA regression + exact v2 parity.**
   `make -C cuda test_cuda`
   Pass: exit 0. Includes `test_matched_filter_row_projections_exact` and
   `test_matched_filter_row_projections_batch_exact`: GPU row sums equal the CPU reference
   exactly (integers, no tolerance) across grid-stride row counts and
   batches, and the GPU marginal reproduces `Compute_Powers_U64`
   bit-for-bit.

3. **GPU pytest gates.**
   `PYTHONPATH=src python3 -m pytest tests/kernel -q`
   Pass: zero failures, zero skips. `test_matched_filter_row_projections_gpu.py` enforces the
   numpy-reference equality, the marginal identity, and the pre-registered
   ULP gate (cupy complex64 fine reduction within 5e-6 relative of the
   float64 prototype). Note: this test *fails* rather than skips if the
   loaded library predates 2.0.0.

4. **Frame parity - CLOSED by census (2026-07-29).** The legacy-epoch
   integrated spectra (salvaged before the CANFAR teardown) settle the
   odd/even question without new baseband: across all 23 channels, every
   detectable transmitter flock sits at the center-at-DC position (even
   freq_ids: 10 of 13 with strong flocks, peaks 41x-8880x over floor; odd
   freq_ids: 8 of 10, peaks 5.9x-693x) and the half-band alternative is
   consistent with noise on every channel of both parities (max 1.3x). The
   five quiet channels show nothing at either hypothesis and carry no
   evidence. `channel_center_normalized_odd_channels` is therefore NOT
   set; the profile and bank stand as shipped. Evidence:
   `docs/evidence/frame_parity_census.csv` / `.png`.
   Independent control from the same census: freq_id 598's pilot offset
   (+96.81 kHz) lies 847 Hz from fs/4, where the half-band error
   self-cancels; its legacy lobe missed the true pilot by only 1.69 kHz
   and the channel ran at mean excess +1043% (F ~ 23) through the entire
   legacy scan - the one accidentally-tuned channel detected at full
   strength while the other 22 sat at the bias floor.
   Optional belt-and-braces on the A100: one odd-freq_id baseband file
   through `tools/framing_audit.py` (expect ALIGNED).

5. **End-to-end smoke on real infrastructure.** One short datatrawl run
   (a handful of files, one even and one odd freq_id) with the current
   analyzer:
   pass criteria, all from the produced npz:
   - `schema_version == pilotproxy_per_pilot_product_v1`;
   - `detector_version` embeds the installed `pilot-proxy/<version>`,
     kernel 2.1.x, the library sha, and the profile hash via the weight
     bank;
   - `fine_status == enabled`, `fine_power_ratio.shape == (n_frames, 256)`;
   - no v1-marginal identity assertion fired (the run raises on
     mismatch);
   - `fine_null_bulk_exceedance_fraction` is bounded to `[0, 1]` on valid
     frames and interpreted as an in-sample threshold diagnostic rather than
     an independent false-alarm-rate measurement;
   - resume test: interrupt after a checkpoint, resume, and verify the
     frame count and `unit_order` continue without duplication (run this
     half in an output directory without a chunk cap: a capped product
     refuses completion under a different cap, by design).

6. **Fine-spectrum sanity against the walkthrough.** Run the smoke
   product for freq_id 844 over the archived event and confirm the fine
   spectrum shows the known line forest (strongest lines near +143 Hz and
   +524 Hz envelope with the +524 line's F capped near ~130 by reference
   leak, per PAPER_PLAN A1.4).

7. **Combine compatibility.** `chime-combine` over two smoke products.
   Pass: succeeds on matching v3 products; refuses a mixed v2/v3 pair
   loudly (schema_version is a combine invariant). Note: combine currently
   carries the v1 field set; fine products live in the per-channel npz.

## Relaunch configuration

Unchanged from the first epoch except: rebuilt bank
(`weights/chime_dtv_weights_k128.bin`, manifest hash
`135bb194cfa58f67...`), `--checkpoint-every 50`, `--name chime-pilots-v2`
(new name; do not resume legacy-epoch products - the detector_version
invariant will refuse, by design). `fine_products=auto` is the default;
set `fine_products=on` to hard-fail if the library on the node is stale.
