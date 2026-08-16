# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

This is the development line for PilotProxy 2.0. It deliberately removes
pre-2.0 Python aliases, duplicated command entry points, and receiver-specific
names from receiver-neutral contracts. Existing products must be regenerated;
the stricter validators do not reinterpret incomplete legacy products.

### Changed

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
