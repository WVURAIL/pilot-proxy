# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

This is the development line for PilotProxy 2.0. It deliberately removes
pre-2.0 Python aliases, duplicated command entry points, and receiver-specific
names from receiver-neutral contracts. Existing products must be regenerated;
the stricter validators do not reinterpret incomplete legacy products.

### Added

- `pilot-proxy-control`, a datatrawl analyzer for non-pilot control freq_ids
  (`pilotproxy_control_product_v1`): per-frame `baseband_power_linear` in the
  detector's native units, a per-frame K=128 rectangular coarse power marginal
  (the unit-modulus analogue of the deployed statistic, so
  `F(b) = 2*S[b] / (S[b-2] + S[b+2])` is recoverable offline at any bin,
  including a virtual-pilot bin on channels with no transmitter), one
  full-resolution integrated spectrum, and the same freq_id-stripped
  `source_event_keys` the combine step joins on. CPU-only by default
  (`--set gpu=1` optional and excluded from the resume fingerprint); the
  deployed detector remains the only mask authority.

### Changed

- Reference-placement diagnostics now say what the `edge_wrapped` bookkeeping
  flag means physically (`frame_origin_description`: under a center-at-DC
  profile the wrap crosses the coarse-channel center/DC, not the channel edge)
  and report references requested beyond +-fs/2 of the channel center
  (`*_crosses_channel_edge`, `channel_edge_notes`); `list-channels` prints the
  corresponding `NOTE`s, including for manifests written before these fields
  existed. Weight values and placements are unchanged; regenerating a manifest
  changes its hash only.
- Tightened receiver, stream-map, detector-core, runtime-bundle, and product
  compatibility validation.
- Replaced the receiver-specific CHIME detector-contract identity with the
  receiver-neutral `pilotproxy_detector_contract_v1` contract.
- Centralized duplicated detector preprocessing and schema identities.
- Updated generated specifications and operational documentation to describe
  the current product contract.
- Separated historical provenance from supported runtime behavior.

### Removed

- The receiver-specific CHIME detector-contract token and its public builder
  aliases. Use `pilotproxy_detector_contract_v1`, `build_detector_contract`,
  and `validate_detector_contract`.
- Result-summary helpers from the `pilot_proxy.testbench` package root. Import
  them from `pilot_proxy.testbench.summarize_results`.
- The duplicate `scripts/fulldepth_and_subsets.py` analysis entry point. The
  historical paper provenance remains archived, but it is not a supported
  runtime command.
- Ambiguous SNR-summary aliases `num_positive_excess_frames` and
  `positive_excess_fraction`. The replacement columns state that they count
  finite data-shelf SNR estimates among detector-valid frames.

### Fixed

- Propagated CUDA API failures to Python callers.
- Prevented debug CUDA builds from being reused as release artifacts.
- Preserved receiver-specific sample rates throughout combined products.

## 1.0.0

- Initial public release.
