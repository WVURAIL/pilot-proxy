# Archive re-run parameter register

This is the scientific and product authority for the CHIME archive re-run. The
pass/fail gates are in [`VALIDATION_GATES.md`](VALIDATION_GATES.md), the sole
production command is in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md), and the
actual launch values belong in the local run ledger. The CANFAR guide is an
alternate bounded remote workflow and cannot override this chain.

The run is historical estimation and sufficient-statistic reprocessing. The
coarse positive-excess flag is retained as a bootstrap diagnostic; the fine
rank/eta decision is inactive and no calibrated detection policy is applied.
The checked-in profiles, weight manifest, and product contract remain
authoritative; this register ties them together and makes unresolved choices
visible.

Scientific and design choices use one of four justification classes:

- **fixed by instrument**: imposed by the receiver or recorded data format;
- **provable**: follows exactly from locked values or an arithmetic bound;
- **literature**: comes from a cited standard or statistical method;
- **shown reasonably optimal**: selected by a recorded comparison or measurement.

`Open` means the repository does not yet support one of those four claims. A
compiled default is not evidence that a choice is optimal. Implementation pins
and operational controls are listed separately and do not receive a scientific
justification label.

## CHIME scientific and receiver parameters

| Parameter | Production value | Status | Justification class | Basis |
|---|---:|---|---|---|
| RF band | 400--800 MHz | Locked | fixed by instrument | Receiver profile |
| ADC sample rate | 800 MHz | Locked | fixed by instrument | Receiver profile |
| PFB transform length | 2048 | Locked | fixed by instrument | Receiver profile |
| PFB taps | 4 | Locked | fixed by instrument | Receiver profile |
| Coarse channels | 1024 | Locked | fixed by instrument | Receiver profile |
| Coarse-channel width and output rate | 390625 Hz | Locked | fixed by instrument | Receiver profile |
| Frequency-axis order and sense | descending RF, inverted | Locked | fixed by instrument | Verified receiver profile |
| Baseband channel center | DFT DC | Locked | fixed by instrument | Verified receiver profile |
| Archive-adapter detector-window reversal | on | Locked | fixed by instrument | Verified archive frame convention |
| Input streams | 2048 feed-polarization streams | Locked | fixed by instrument | Archive format |
| Native sample encoding | offset-binary 4+4-bit complex in one byte | Locked | fixed by instrument | Packed archive reader |
| Detector repack | lossless offset-binary to signed int4 | Locked | provable | Encoding conversion; no scale estimate or requantization |
| Detector window, `K` | 128 samples | Locked | provable | Exact tiling, measured span/reference bracket, fixed-point closure, and kernel geometry |
| Expected frame size, `nfft` | 16384 channelized samples | Verify at launch | fixed by instrument | Profile value and measured pre-flight value; the profile permits an override |
| Detector-cell width | 3051.7578125 Hz | Locked with `K = 128` | provable | 390625 / 128 |
| Expected frame time | 41.94304 ms | Verify at launch | provable | 16384 / 390625 |
| Expected windows per stream | 128 | Verify at launch | provable | 16384 / 128 |
| Expected detector rows per frame | 262144 | Verify at launch | provable | 2048 streams multiplied by 128 windows |
| Weight terms | target, lower reference, upper reference | Locked | shown reasonably optimal | A symmetric pair cancels odd background terms; added references cost 33% each for little idealized narrowing in the measured regime |
| Skipped guard cells | 1 | Locked | provable | A two-cell reference offset leaves exactly one intervening cell |
| Reference offset | 2 detector cells | Locked | shown reasonably optimal | A one-cell offset admits 40.5% worst-case pilot leakage; two cells reduce it to 4.5% while retaining locality |
| Coarse statistic | `F = 2 P_target / (P_ref_lower + P_ref_upper)` | Locked | provable | Two-reference statistic definition |
| Null scale | `mu0 = 2 target_norm_sq / reference_norm_sum_sq` | Locked per weight row | provable | Exact packed-weight norms |
| Recorded coarse bootstrap flag | valid and `F > mu0` | Locked diagnostic recording policy | Open | Exact and calibration-free, but not a tuned or final detection policy |
| Television channel width | 6000000 Hz | Locked | literature | ATSC 1.0 channel plan |
| Standard pilot offset from lower channel edge | `177/572` MHz, about 309440.559 Hz | Locked | literature | ATSC 8-VSB pilot placement |
| Nominal pilot below data shelf | 11.3 dB | Locked only as the standard conversion | literature | ATSC synthetic convention |
| Pilot capture efficiency | 1.0 | Conditional | Open | Replace when a receiver measurement or model supplies a correction |
| Fine zero-padding factor | 2 | Locked measurement method | Open | Nonblocking; the exact terms are retained while the fine decision is inactive |
| Diagnostic fine false-alarm probability | 0.001 per bin | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Diagnostic fine guard | 1 independent bin | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Predicted-line half-width | 30 padded bins, about 357.63 Hz | Locked diagnostic default | shown reasonably optimal | Covers the measured station offsets with margin |
| Ordinary null location and scale | median; median minus P15.87 | Locked diagnostic method | literature | Robust center and one-standard-deviation left spread |
| Fallback null location and scale | P25; `(P25 - P2.275) / 2` | Locked diagnostic method | literature | Lower-quantile robust estimator |
| Fallback trigger fraction | 0.2 | Not used by this acquisition | Open | Nonblocking; selection belongs to later detection calibration |
| Science-budget factor, `zeta` | Not used by this acquisition | Deferred | literature | Approval belongs to later detection calibration and does not block sufficient-statistic acquisition |

The production pre-flight must resolve `nfft = 16384`. Any other value changes
frame time, windows, rows, fine-bin count, and fine-bin width. It should fail the
current production gate instead of silently inheriting the expected values above.

Sources: [`chime_dtv_fengine.json`](../configs/receiver_profiles/chime_dtv_fengine.json),
[`pilotproxy_cuda_local_reference_power_ratio.json`](../configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json),
[`detector.py`](../src/pilot_proxy/datatrawl_plugins/detector.py),
[`METHOD_SPEC.md`](METHOD_SPEC.md), and
[`fine_reduction.py`](../src/pilot_proxy/fine_reduction.py).

## Cross-instrument design invariant

These are profile values, not settings for the CHIME archive launch. The CHORD
profiles still require verification against an operational data product.

| Parameter | CHIME | CHORD | CHORD Pathfinder | Basis |
|---|---:|---:|---:|---|
| Channelized sample rate | 390625 Hz | 195312.5 Hz | 195312.5 Hz | fixed by instrument |
| Profile frame samples | 16384 | 8192 | 8192 | fixed by instrument |
| Detector window, `K` | 128 | 64 | 64 | common detector-cell design |
| Profile frame time | 41.94304 ms | 41.94304 ms | 41.94304 ms | provable |
| Profile windows per stream | 128 | 128 | 128 | provable |
| Input streams | 2048 | 1024 | 128 | fixed by instrument |
| Profile detector rows | 262144 | 131072 | 16384 | provable |

The design invariant is `frame_size_samples / K = 128`. The supported compiled
values of `K` are 64 and 128. `K = 256` is not supported.

Sources: the CHIME profile above,
[`chord_dtv_fengine.json`](../configs/receiver_profiles/chord_dtv_fengine.json), and
[`chord_pathfinder_dtv_fengine.json`](../configs/receiver_profiles/chord_pathfinder_dtv_fengine.json).

## Implementation and product pins

These values reproduce the current method. They are implementation or provenance
pins, not claims that a scientific choice is optimal.

| Pin | Required value or rule | Authority |
|---|---|---|
| Product schema | `pilotproxy_per_pilot_product_v5` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Source-event key schema | `pilotproxy_namespaced_source_event_key_v1` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Recorded coarse bootstrap flag | coarse norm-corrected positive excess; diagnostic only | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Scan-time fine decision | inactive; calibration status `pending_campaign` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Shipped detector channel range | physical channels 14 through 36 | [`chime_dtv_weights_k128.bin.manifest.json`](../weights/chime_dtv_weights_k128.bin.manifest.json) |
| Implemented pilot offset | 309441 Hz above the lower channel edge | [`atsc_channels.py`](../src/pilot_proxy/atsc_channels.py) |
| Fine transform input | 128 complex row sums, padded to 256 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Fine transform arithmetic | radix-2 DIT, signed Q15 twiddles, natural-order output | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Butterfly rounding | add 16384, then arithmetic shift by 15 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Stage scaling | none | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Input component bound | absolute value at most `2^20` | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Master twiddle length | 2048 | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Master twiddle SHA-256 | `fec7f22309f4689a1a4a26258dc562487a02e33698a5e0341e2db1928f58d197` | [`fxfft.py`](../src/pilot_proxy/fxfft.py) |
| Fine terms | `[frames, 3, 256]` unsigned 64-bit integers | [`PER_PILOT_PRODUCT_FIELDS.md`](PER_PILOT_PRODUCT_FIELDS.md) |
| Stored diagnostic fine designated bins | predicted pilot residual with 30-bin half-width; exact array stored per product | [`detector.py`](../src/pilot_proxy/datatrawl_plugins/detector.py) |
| Fine census exclusions | empty unless explicitly supplied; exact array stored per product | [`fine_reduction.py`](../src/pilot_proxy/fine_reduction.py) |
| Per-frame spectrum | `[frames, 16384]` signed 16-bit codes when the required `nfft` passes | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Spectrum encoding | 0.01 dB per code about the per-frame reference; invalid code -32768 | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Rail and fill diagnostics | separate per-frame counts; total components `2 * nfft * streams` | [`product_contract.py`](../src/pilot_proxy/product_contract.py) |
| Per-channel target, references, norms, and adaptation | exact rows in the shipped manifest; channel 14 is the sole adaptive wrap case | [`chime_dtv_weights_k128.bin.manifest.json`](../weights/chime_dtv_weights_k128.bin.manifest.json) |
| Receiver-profile SHA-256 | `bc59e77442a4c15f74c716d14eaeea4f10a69517d3bfb8c88ce10a7a42ea1e15` | Checked-in file digest |
| Weight-bank SHA-256 | `1383c6d0ca521a26b317d008feb6e09eb41427155bda9a320f70bca62e0e6259` | Shipped manifest and checked-in file digest |
| Weight-manifest SHA-256 | `d0ccc8162a350e9d3266e6acf3b38d2fe5982c474b73ef0715b8b838954e81a7` | Checked-in file digest |

Do not duplicate the 23 per-channel weight rows here. Pin the manifest path and
digest in the run ledger, then check the resolved first product against it.

## Archive launch controls

| Control | Production setting | State |
|---|---|---|
| Instrument | `chime` | Locked |
| Source | `cadc-datatrail` | Locked |
| Reader | `chime-baseband-packed` | Locked |
| Analyzer | `pilot-proxy-detector` | Locked |
| Pilot selection | 506, 521, 537, 552, 568, 583, 598, 614, 629, 644, 660, 675, 690, 706, 721, 736, 752, 767, 783, 798, 813, 829, 844 | Locked |
| Frozen inventory path and SHA-256 | `/home/djg/rail/archive_inputs/chime-pilots-v5/inventory.jsonl`; `b2cfef752a6f2cf88317141a974161439299668a37a5464d710b3800b9a872d8` | Prepared; verify before launch |
| Source inventory accounting | 170,377 units and 9,214 events | Verified against the source survey and supplemental resolution |
| Frozen inventory accounting | 165,682 usable units and 8,983 events; 4,695 exclusions | Approved and generated |
| Exclusion ledger SHA-256 | `d0ab78db9dec847fb5270fb94e3fc45efd6bbbc71adc376830fb71467693079a` | Locked |
| Pending-event resolution | 3 events, 69 selected objects absent, no errors or sub-floor objects | Verified by authenticated metadata checks on 2026-08-25 |
| Pending-event resolution SHA-256 | `077915779d234ad9cca1e9b52171e28025a60474f20fad7aff5684f55ce0c4c7` | Locked |
| Inventory manifest SHA-256 | `5695e1cc9c007cb2c79ad39535cf9ed2fb20848ad00d3cf0215933670d0707e5` | Locked |
| Resolved `nfft` | 16384 | Must pass pre-flight |
| Execution host | Local WSL workstation: 24-core host, 32 WSL virtual CPUs, 62 GiB WSL memory, one 16 GiB RTX 5000 Ada | Locked; verify current free space at launch |
| Source revision | Exact pushed commit recorded in the run ledger | Freeze after all code and documentation changes |
| Source state | Clean and equal to the pushed revision | Mandatory at every launch and resume |
| Package-source SHA-256 | Exact digest recorded in the run ledger and embedded products | Freeze with the final source |
| Download workers | 4 | Best tested starting value; 13.053 MiB/s aggregate across forward and reverse sequences, though two workers won the reverse sequence |
| Maximum staged files | 8 | Ordered prefetch; frozen-inventory bound about 8.35 GiB |
| Staging permissions | `umask 077` in every launch and resume shell | Locked |
| File cap | none | Locked |
| Chunk cap | none | Locked |
| Partial-run acknowledgement | off | Locked; any new quarantine or failure stops acceptance |
| Fine products | on | Locked; retains exact fine powers and does not enable fine detection |
| Checkpoint interval | 250 units | Reduces cumulative product rewrites; a stop can repeat 249 analyzed files plus uncheckpointed prefetched downloads |
| Real-data GPU gate | One 2048-stream file, one full chunk, peak VRAM below 13,900 MiB, product validation passes | Must repeat on the final frozen source; the 2026-08-25 result used an earlier package digest |
| Production-profile rehearsal | Archive source, four workers, eight slots, full chunks, forced checkpoint and identical-command resume in a separate capped output | Must repeat on the final frozen source; the 2026-08-25 result used an earlier package digest |
| Resume invariant | Same source revision, inventory, output, staging, selection, execution settings, and preserved library | Locked |
| Output directory | fresh path outside the source checkout | Fill at launch |
| Receiver profile path and SHA-256 | Checked-in CHIME profile and locked digest above | Verify at launch |
| Weight bank and manifest SHA-256 | Checked-in bank, manifest, and locked digests above | Verify at launch |
| Detector library path, version, and SHA-256 | `cuda/libfstatistic-2.3.0-sm89-e48ffa59bb592be8.so`; core 2.3.0; `e48ffa59bb592be839218dfb6f920c8f9e9653b10abab97e856372cdcfa3bc8b` | Preserved digest verified; final-source CUDA, kernel, and real-file gates pending |
| Terminal product | all 23 per-pilot v5 products; channel subsets are derived only | Locked |

The 23 pilot identifiers are the ATSC 14--36 pilot locations mapped onto the
verified CHIME grid. No production file or chunk cap is allowed.

Kernel core 2.3.0 is selected for the fresh re-run. Use the preserved SM89
artifact named above, complete the source-build GPU and parity gates, and
complete the preserved-artifact capability and real-file gates. Use its
recorded digest for every restart. Rebuilding changes the pinned byte identity
and requires a new approval. The preserved 2.1.0 binary remains historical
cohort evidence and is not the production selection.

Source: [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md).

## Approved launch decisions

- The frozen inventory excludes the exact 4,695 known-unusable source units.
  The original inventory and historical quarantine remain unchanged beside it.
- Three pending survey events were resolved by authenticated metadata checks.
  All 69 selected objects were absent, so no inventory rows were added.
- All 23 per-pilot v5 products are the authoritative terminal deliverable.
  Any common-event channel subset is derived after the scan.
- The historical reprocessing is unblinded. A future frozen epoch is reserved
  for blinded validation.
- Combined rail and fill fields do not block launch because combined products
  are not terminal products.

## Deferred calibration decisions

| Parameter or decision | Current state | Required evidence or decision |
|---|---|---|
| Fine decision anchor and width | Implemented as runtime data but not calibrated for activation | Measure per-channel and per-era pilot locations |
| Fine decision testbench rank, `rho` | 64 is the recorded candidate; the runtime-bundle exporter's pending block remains unset | Check in the supporting measurement and recalibrate before activation |
| Fine decision rank and multiplier | No active production values | Calibrate on verified null epochs and record product hashes and quantiles |

These decisions do not block archive acquisition. The scan records the exact
fine terms while the fine decision remains inactive.

## Operational setup before archive launch

| Item | Current state | Required action |
|---|---|---|
| Archive certificate | Present locally, owner-only, expires 2026-09-03 | Renew before launch; at least 72 hours before each expiry, stop after a checkpoint, renew, and resume with the identical command |
| Local archive environment and client | Earlier-source environment built 2026-08-25; isolated archive suite passed | Rebuild or refresh on the final frozen revision and never record secrets in the ledger |
| Real-data GPU capacity | Earlier gate passed with 2,904 MiB peak and zero validation errors | Repeat on the final frozen source and record the new evidence path and embedded digests |
| Production-profile resume | Earlier gate passed with eight unique units and empty staging | Repeat on the final frozen source and record the new evidence path and embedded digests |

Do not promote the recorded testbench rank to deployed runtime data. An exported
pending bundle leaves anchor, rank, and multiplier unset while calibration is
pending.

## Separate radio transfer study

This study is not an archive-launch gate. It validates estimator transfer
through the radio path and remains separate from the historical archive
acquisition.

Fill these values after a real receiver capture exists:

| Parameter | Required value |
|---|---|
| Capture | Fresh headerless complex64 file outside the source checkout |
| Capture SHA-256 | Record before audit |
| Sample rate | True receiver rate in Hz; greater than 6 MHz with edge margin |
| Tuning | Center the selected 6 MHz channel at DC |
| Physical channel | One shipped channel from 14 through 36 |
| Expected audit pilot | about -2690559.441 Hz in centered baseband |
| Minimum samples | `ceil(((((16384 + 4 - 1) * 2048) - 1) * rate / 800000000)) + 1` |
| Minimum at 10762237.762237763 Hz | 451485 complex samples, or 3611880 bytes |
| Audit command | `PYTHONPATH=src "$PYTHON_BIN" -m pilot_proxy.testbench.audit_atsc_signal --input-iq "$CAPTURE" --sample-rate-hz "$SAMPLE_RATE_HZ" --output-json "$AUDIT_JSON"` |
| Evaluator sample rate | Pass `--iq-sample-rate-hz "$SAMPLE_RATE_HZ"` with the same true receiver rate used by the audit |
| Evaluator audit attachment | Pass `--waveform-audit-json "$AUDIT_JSON"` so the report embeds the capture audit rather than the synthetic default |
| Pilot below data shelf | Pass `audit.measured_pilot_below_data_db` explicitly as `--pilot-below-data-db` |
| Evaluator frame samples | 16384 |
| Evaluator streams | 4 |
| SNR grid | -60 through 0 dB in 2 dB steps |
| Trials | 60 per point |
| Seed | 20260824 for direct comparison with the synthetic curve |
| Detector backend | CUDA |
| Synthesis backend | CPU for direct comparison with the synthetic curve |
| Noise source | GNU Radio through `/usr/bin/python3` |
| Quantization clip sigma | `--clip-sigma 3`; leave `--scale` unset so the evaluator estimates it |
| Channel gain and phase | `--channel-gain-db 0 --channel-phase-deg 0` |
| Threshold override | Leave `--threshold-data-shelf-snr-db` unset |
| Spectral sense | normal |
| Reference archive phase | off; pass `--no-reference-archive-phase` |
| Added frequency offset | 0 Hz |
| Output directory | fresh path outside the source checkout |

Four evaluator streams repeat the same captured waveform with independent added
noise; they are not four independent receiver feeds. The sweep treats receiver
noise and multipath already present in the capture as part of the clean signal.
The signal audit runs first. The evaluator startup guard then checks the clean
capture through the detector statistic and refuses a mistuned or reversed signal.

The archive adapter's detector-window reversal is part of reading the recorded
CHIME frames. The evaluator's reference archive phase is a separate testbench
transform and must remain off for this transfer curve.

Preserve the exact audit and evaluator commands. The evaluator report embeds the
audit JSON, but it does not independently bind that file to the capture or record
every command-line value. `--waveform-audit-json` is provenance-only and does not
replace the evaluator's nominal 11.3 dB conversion; pass the measured value
explicitly.

## Run ledger

Copy [`LOCAL_ARCHIVE_RUN_LEDGER_TEMPLATE.md`](LOCAL_ARCHIVE_RUN_LEDGER_TEMPLATE.md)
beside the run output before launch and record:

- UTC start time and operator;
- complete command as executed;
- source revision and dirty-state result;
- resolved `nfft`, frame time, stream count, and fine geometry;
- detector library version, path, and SHA-256 digest;
- receiver-profile path and SHA-256 digest;
- weight-bank and manifest paths and SHA-256 digests;
- inventory path, SHA-256 digest, and selected pilot list;
- runtime image, Python, NumPy, CuPy, driver, and toolkit versions;
- host and WSL CPU counts, WSL memory and swap, ext4 free space, and measured
  peak VRAM for the 2048-stream smoke test;
- output directory, delivery-order controls, and checkpoint interval;
- production-profile rehearsal command, interruption point, resume result, and
  validation result;
- certificate expiry, each planned checkpoint stop and renewal, and each exact
  resume command;
- terminal-product decision;
- input capture and signal-audit paths and SHA-256 digests, when applicable;
- every quarantine-ledger entry and its disposition;
- first-product tripwire result, including the stored designated and census arrays;
- final per-pilot audit results and canonical validation when terminal combine
  succeeds.

Do not store credentials or generated products in the source checkout.
