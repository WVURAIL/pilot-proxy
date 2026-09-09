# Stationary waveform preparation and saturation reconciliation

The large historical synthetic saturation discrepancy is explained by the startup-containing standards-chain input. The current CPU channelizer and independently recounted saved trial powers support that explanation. This does not measure the telescope's received pilot-to-allocation transfer.

## Recount and current-waveform audit

```bash
python tools/audit_settled_waveform.py --output /new/audit
```

The audit checks the weight-bank binary and manifest against the archived reports before channelization. It records all current source/input hashes. The clean one-stream normalized coarse ratios are 120.870(float)/123.944(int4) for the startup-containing waveform and 286.031/268.319 for the settled waveform. Recounted four-input GPU controls at +60 dB give −0.115 dB and +2.604 dB respectively. The independent verifier in the result release reconstructs powers and checks all 18 historical pooled estimates; the maximum numerical discrepancy is 4.31e−15 dB.

The effective diagnostic A/(Q−1) is 0.550 for settled packed samples/weights and 0.516 for ideal floating weights. It includes finite waveform, pilot capture, channelizer, representation and the saved pilot-power convention. It is not an independent physical skirt measurement or the earlier fitted b=1.106. Finite-window Welch estimates vary materially with segment length and are published as sensitivity diagnostics, not a confidence interval.

## New waveform preparation

```bash
python tools/generate_settled_atsc.py --output-dir /new/waveform
```

This new wrapper leaves the legacy generator and all historical products unchanged. It requires an empty directory, freezes source/settings/runtime identities, generates 800,000 complex64 samples, discards exactly 200,000, and retains 600,000. It hashes the raw IQ, exact trimmed IQ, MPEG transport pattern and audit, and records contiguous 65,536-sample pilot/total-power diagnostics. The discard is an explicit experiment setting; passing the broad waveform audit alone does not establish universal stationarity.

Two current runs with the same seed and source/runtime identity differ by 2.96e−7 relative RMS. The first new settled file and surviving historical settled file differ by 2.39e−7 relative RMS. Preserve the exact output IQ and its hash for replay: regeneration is numerically repeatable here, not byte-identical. The historical settled-generation sidecar and original generating-source revision remain unavailable; the new receipt does not reconstruct them.

The September 9 evidence is under `results/calibration_progress_2026-09-09/waveform` in the parent workspace, including raw/settled generation pairs, plans, receipts, independent proof and the archived arithmetic audit. None of these newly generated samples entered the separately frozen residual experiment.

An OTA experiment additionally needs a specified RF setup, frequency/power constraints, measured power conversion, immutable capture records, sequential acceptance guards and an audited hardware adapter. Antenna connection and a device probe alone do not close those requirements.
