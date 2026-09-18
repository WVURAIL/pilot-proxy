#!/usr/bin/env python3
"""Build the predeclared single-input bench protocol spec and its evidence files.

Writes, next to this script: tx_payload_2s.cfile (the exact 2 s tone payload the
unchanged capture helper commands), stationarity_audit.json (computed on that
digital payload), power_bound_declaration.json (a declared conducted bound, not
a meter reading) and protocol_spec.json. Never opens a radio.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/home/djg/rail/pilot-proxy")
sys.path.insert(0, str(REPO/"src"))
from pilot_proxy.testbench.sdr_protocol import SCHEMA, digest, settings_identity  # noqa: E402

RATE = 2_000_000.
RECORD_SAMPLES = 4_000_000          # 2 s: the geometry the unchanged analyzer accepts
LADDER_DB = [0, 6, 12, 18, 24, 30, 36]
BUILD = "/home/djg/rail/results/sdr_reference_transport_2026-09-09/build-v1/build.json"


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def payload():
    """Bit-identical to tools/lime_reference_capture_v1.py prepare(mode='tone')."""
    waveform = np.zeros(RECORD_SAMPLES+600000, dtype=np.complex64)
    n = RECORD_SAMPLES+400000
    tone = .005*np.exp(2j*np.pi*100000*np.arange(n)/RATE)
    ramp = 2000
    tone[:ramp] *= np.sin(np.linspace(0, math.pi/2, ramp))**2
    tone[-ramp:] *= np.sin(np.linspace(math.pi/2, 0, ramp))**2
    waveform[100000:100000+n] = tone.astype(np.complex64)
    return waveform


def audit(wave_path, wave, start, samples):
    segments, count = [], 7
    edges = np.linspace(start, start+samples, count+1).astype(int)
    powers = []
    for a, b in zip(edges[:-1], edges[1:]):
        block = wave[a:b].astype(np.complex128)
        power = float(np.mean(np.abs(block)**2))
        spectrum = np.abs(np.fft.fft(block))
        peak_hz = float(np.fft.fftfreq(block.size, 1/RATE)[int(np.argmax(spectrum))])
        powers.append(power)
        segments.append({"start_sample": int(a), "stop_sample": int(b), "mean_power": power,
                         "peak_hz": peak_hz, "passed": abs(peak_hz-100000.) <= RATE/block.size})
    mean = float(np.mean(powers))
    for seg in segments:
        seg["passed"] = bool(seg["passed"] and abs(seg["mean_power"]-mean)/mean <= 1e-6)
    return {"schema": "pilotproxy-stationary-input-audit-v1", "waveform_sha256": digest(wave_path),
            "start_sample": start, "samples": samples, "source_startup_discard_samples": 300000,
            "stationarity_passed": all(s["passed"] for s in segments),
            "method": "Digital command payload only: seven disjoint chronological segments covering the declared interval; per-segment mean power and FFT peak frequency. This audits the commanded waveform, not any RF measurement.",
            "thresholds": {"relative_mean_power_deviation_max": 1e-6, "peak_frequency_hz": 100000., "peak_tolerance_hz": "one FFT bin of the segment"},
            "segments": segments}


def slots():
    order = LADDER_DB + LADDER_DB[::-1] + LADDER_DB   # pass A up, pass B down, pass C up
    plan = [("tx_zero", "txzero", None, "terminated TX-zero before"),
            ("signal_only", "tone", 0, "cabled tone, cabling check"),
            ("noise_only", "noise", None, "terminated null before")]
    plan += [("mixture", "tone", db, f"ladder pass {'ABC'[i//len(LADDER_DB)]} rung {db} dB") for i, db in enumerate(order)]
    plan += [("noise_only", "noise", None, "terminated null after"),
             ("tx_zero", "txzero", None, "terminated TX-zero after")]
    result = []
    for index, (kind, mode, db, label) in enumerate(plan):
        terminated = db is None
        result.append({"slot_index": index, "kind": kind, "seed": 1000+index,
                       "data_shelf_snr_db": (-float(db) if kind == "mixture" else None),
                       "helper_mode": mode, "bench_label": label,
                       "commanded_attenuation_db": db,
                       "rx_port": "50 ohm terminator" if terminated else "attenuator ladder output via SMA cable C2",
                       "tx_port": "50 ohm terminator" if terminated else "SMA cable C1 to attenuator ladder input",
                       "record_samples": RECORD_SAMPLES, "record_seconds": 2.0})
    return result


def main():
    wave_path = HERE/"tx_payload_2s.cfile"
    wave = payload()
    wave.tofile(wave_path)
    start, samples = 200000, 4_200_000       # inside the steady tone (102000 .. 4498000)
    audit_path = HERE/"stationarity_audit.json"
    audit_path.write_text(json.dumps(audit(wave_path, wave, start, samples), indent=2)+"\n")
    build = json.loads(Path(BUILD).read_text())
    sources = [REPO/"tools/lime_reference_capture_v1.py", REPO/"tools/lime_reference_worker_v1.cpp",
               REPO/"tools/analyze_sdr_antenna_controls_v1.py", REPO/"tools/run_sdr_distribution_reference_v1.py",
               REPO/"src/pilot_proxy/integration/sdr_upgrade_adapter.py", Path(BUILD), Path(build["worker"])]
    spec = {
        "schema": SCHEMA, "run_id": "SDR-BENCH-2026-09-17-single-input-reference-v1",
        "bench_scope": "Conducted single-input reference on the LimeSDR Mini: terminated nulls and a cabled tone through a step-attenuator ladder. Closes the physical single-input reference only; no sensitivity, ROC, telescope deployment or absolute dBm claim.",
        "waveform": {**ref(wave_path), "start_sample": start, "samples": samples,
                     "source_startup_discard_samples": 300000},
        "implementation_sources": [ref(p) for p in sources],
        "noise_model": "independent_complex_gaussian",
        "radio": {"frequency_hz": 500_000_000., "sample_rate_hz": RATE,
                  "emission_bandwidth_hz": 2_000_000., "radio_filter_bandwidth_hz": 1_500_000.,
                  "tx_bandwidth_hz": 5_000_000., "tx_rms": .005,
                  "device_serial": "1D423D9108F273", "device": "LimeSDR Mini",
                  "antenna_separation_cm": 30., "antenna_separation_meaning": "no antennas: nominal SMA cable run length, conducted bench",
                  "antenna_polarization": "not applicable, conducted",
                  "clock_source": "LimeSDR Mini internal VCTCXO",
                  "native_tx_gain_db": 50, "tx_gain_setting_db": 38.,
                  "native_rx_gain_db": 30, "rx_gain_setting_db": 18.,
                  "rx_path": "LNAW", "active_tx_path": "TX2",
                  "tx_antenna": "none: SMA cable C1 to attenuator ladder, or 50 ohm terminator",
                  "rx_antenna": "none: attenuator ladder output via SMA cable C2, or 50 ohm terminator"},
        "capture": {"settle_samples": 100000, "marker_guard_samples": 100000, "capture_samples": RECORD_SAMPLES},
        "guards": {"clip_level": .98, "rms_min": 1e-5, "rms_max": .5, "clip_fraction_max": 0., "marker_correlation_min": .15},
        "slots": slots(),
        "ladder": {"commanded_attenuation_db": LADDER_DB, "passes": ["A ascending", "B descending", "C ascending"],
                   "repeats_per_rung": 3, "record_seconds": 2.0, "record_samples": RECORD_SAMPLES,
                   "capture_maximum_seconds": 4.0, "helper": "tools/lime_reference_capture_v1.py, unchanged",
                   "analysis": "tools/analyze_sdr_antenna_controls_v1.py analyze_capture, unchanged: mix -100 kHz, 25/128 resampler, K=L=128, bins 0/-2/+2, sample scale 500"},
        "rf_limits": {"frequency_low_hz": 499_000_000., "frequency_high_hz": 501_000_000.,
                      "power_metric": "average_conducted_dbm", "maximum_power_dbm": 0.,
                      "provided_by": "Author's conducted-bench constraint of 2026-09-17: TX port into a 50 ohm attenuator ladder or terminator only, no antenna, no radiated emission, RX port input kept at or below 0 dBm. Not a jurisdictional authorization."},
    }
    power = {"schema": "pilotproxy-measured-rf-power-v1", "settings_sha256": settings_identity(spec),
             "power_metric": "average_conducted_dbm", "covers_all_slots_markers_transitions": True,
             "measurement_id": "BENCH-2026-09-17-declared-conducted-bound",
             "method": "Declared upper bound, not a meter reading: LimeSDR Mini specified maximum TX output near +10 dBm at full digital scale and full gain; the 0.005 tone is 46 dB below full scale; bound stated at -30 dBm with 12 dB uncertainty covering gain-table and device spread. Applies to every slot, the zero-IQ slots and the ramps, whose amplitude never exceeds the tone. Absolute dBm is outside this protocol's claims.",
             "conducted_power_dbm": -30., "total_uncertainty_db": 12.}
    power_path = HERE/"power_bound_declaration.json"
    power_path.write_text(json.dumps(power, indent=2)+"\n")
    spec["stationarity_audit"] = ref(audit_path)
    spec["power_conversion"] = ref(power_path)
    (HERE/"protocol_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"slots": len(spec["slots"]), "waveform_sha256": spec["waveform"]["sha256"]}))


if __name__ == "__main__":
    main()
