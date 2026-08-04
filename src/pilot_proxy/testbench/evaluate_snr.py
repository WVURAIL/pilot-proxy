#!/usr/bin/env python3
# coding=utf-8
"""Evaluate DTV data-shelf SNR estimates from the CUDA F-statistic kernel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pilot_proxy.detector_geometry import (  # noqa: E402
    apply_spectral_sense_to_detector_matrix,
    build_stream_map,
    flatten_feed_channel_streams,
    input_layout_metadata,
    stream_time_block_to_detector_matrix,
)
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz  # noqa: E402
from pilot_proxy.detector_contract import (
    norm_corrected_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (  # noqa: E402
    REFERENCE_LOWER_TERM_INDEX,
    REFERENCE_TARGET_TERM_INDEX,
    REFERENCE_UPPER_TERM_INDEX,
    fstat_cpu_reference,
    fstat_cpu_reference_packed,
)
from pilot_proxy.detector_weights import DetectorWeightBank  # noqa: E402
from pilot_proxy.dtv_units import (  # noqa: E402
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    NO_PILOT_EXCESS_FSTAT,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    composite_to_data_shelf_snr_correction_db,
    fstat_raw_to_pnr_bin_db,
    fstat_num_den_to_fstat_level_db,
    fstat_num_den_to_pilot_excess_linear,
    fstat_num_den_to_pnr_bin_db,
    fstat_num_den_to_raw,
    pilot_to_data_power_ratio,
    pnr_bin_db_to_snr_shelf_db,
    pnr_bin_to_snr_shelf_metadata,
    snr_shelf_threshold_fields,
)
from pilot_proxy.json_utils import write_json_strict  # noqa: E402
from pilot_proxy.kernel import FStatKernel  # noqa: E402
from pilot_proxy.integration import QUANTIZATION_SCALE_MODE_GLOBAL  # noqa: E402
from pilot_proxy.integration.packing import (  # noqa: E402
    pack_channelized_streams_for_detector,
)
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH  # noqa: E402
from pilot_proxy.reference_channelizer import (  # noqa: E402
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    apply_reference_archive_phase_convention,
    channelize_real_blocks_to_reference_channels,
    complex_envelope_to_real_adc_blocks,
    nearest_reference_channel_index,
    sinc_hamming_pfb_response,
)
from pilot_proxy.result_schema import (  # noqa: E402
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    RESULT_SCHEMA_VERSION,
    result_schema_object,
)
from pilot_proxy.testbench.quantize import (  # noqa: E402
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    DEFAULT_DTV_PILOT_HZ,
    DEFAULT_FRAME_SIZE_SAMPLES,
    GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    LOCKED_BITS_PER_COMPONENT,
    LOCKED_DETECTOR_WINDOW_SAMPLES,
)

HZ_PER_MHZ = 1.0e6
HALF_SCALE = 2.0
COMPLEX_COMPONENT_COUNT = 2.0
DEFAULT_EVALUATOR_SEED = 12345
DEFAULT_NOISE_TRIALS = 3
DEFAULT_NUM_INPUT_STREAMS = 4  # deployment-standard trial: 4 streams x 128 rows = 512
# (was 1; a silent 1-stream default cost a debugging night on 2026-07-19)
DEFAULT_GNURADIO_PYTHON = "/usr/bin/python3"
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_SNR_SWEEP_MIN_DB = -60.0
DEFAULT_SNR_SWEEP_MAX_DB = 0.0
DEFAULT_SNR_SWEEP_STEP_DB = 3.0
DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ = 0.0
STANDARD_FREQUENCY_OFFSET_SWEEP_HZ = (-1_000.0, 0.0, 1_000.0)
DEFAULT_CHANNEL_GAIN_DB = 0.0
DEFAULT_CHANNEL_PHASE_DEG = 0.0
DB_AMPLITUDE_FACTOR = 20.0
DEGREES_PER_HALF_TURN = 180.0
TWO_PI = 2.0 * math.pi
SNR_RANGE_EPSILON = 1e-12
CSV_SNR_LABEL_PRECISION = 3


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanstd_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _nanmax_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def required_iq_samples(
    *,
    iq_sample_rate_hz: float,
    adc_sample_rate_hz: float,
    num_output_samples: int,
    pfb_taps: int = REFERENCE_PFB_TAPS,
    pfb_fft_size: int = REFERENCE_PFB_FFT_SIZE,
) -> int:
    """Return input IQ samples needed for the requested channelizer output."""
    n_blocks = int(num_output_samples) + int(pfb_taps) - 1
    total_adc_samples = n_blocks * int(pfb_fft_size)
    last_source_position = (
        (total_adc_samples - 1) * float(iq_sample_rate_hz) / float(adc_sample_rate_hz)
    )
    return int(math.ceil(last_source_position)) + 1


def _signal_and_noise_power_for_snr(
    signal: np.ndarray,
    *,
    snr_db: float,
    sample_rate_hz: float | None = None,
    snr_bandwidth_hz: float | None = None,
) -> tuple[np.ndarray, float, float]:
    clean = np.asarray(signal, dtype=np.complex64)
    clean_signal_power = float(np.mean(np.abs(clean.astype(np.complex64)) ** 2))
    if not np.isfinite(clean_signal_power) or clean_signal_power <= 0.0:
        raise ValueError("signal power must be positive and finite.")
    noise_power = clean_signal_power / float(
        DB_LINEAR_BASE ** (float(snr_db) / DB_POWER_FACTOR)
    )
    if sample_rate_hz is not None or snr_bandwidth_hz is not None:
        if sample_rate_hz is None or snr_bandwidth_hz is None:
            raise ValueError(
                "sample_rate_hz and snr_bandwidth_hz must be provided together."
            )
        if sample_rate_hz <= 0.0 or snr_bandwidth_hz <= 0.0:
            raise ValueError("sample_rate_hz and snr_bandwidth_hz must be positive.")
        noise_power *= float(sample_rate_hz) / float(snr_bandwidth_hz)
    return clean, clean_signal_power, noise_power


def add_complex_awgn_for_snr(
    signal: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator,
    sample_rate_hz: float | None = None,
    snr_bandwidth_hz: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Add complex AWGN to match a requested signal/noise-band power ratio."""
    clean, clean_signal_power, noise_power = _signal_and_noise_power_for_snr(
        signal,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        snr_bandwidth_hz=snr_bandwidth_hz,
    )
    component_sigma = math.sqrt(noise_power / COMPLEX_COMPONENT_COUNT)
    noise = rng.normal(0.0, component_sigma, clean.shape) + 1j * rng.normal(
        0.0,
        component_sigma,
        clean.shape,
    )
    return (
        np.asarray(clean + noise.astype(np.complex64), dtype=np.complex64),
        clean_signal_power,
        noise_power,
    )


def add_gnuradio_awgn_for_snr(
    signal: np.ndarray,
    *,
    input_iq_path: Path,
    output_iq_path: Path,
    snr_db: float,
    seed: int,
    gnuradio_python: str,
    sample_rate_hz: float,
    snr_bandwidth_hz: float,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Add AWGN with GNU Radio analog.noise_source_c in a helper process."""
    clean, clean_signal_power, noise_power = _signal_and_noise_power_for_snr(
        signal,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        snr_bandwidth_hz=snr_bandwidth_hz,
    )
    metadata_path = output_iq_path.with_suffix(output_iq_path.suffix + ".json")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not existing_pythonpath
        else str(SRC_ROOT) + os.pathsep + existing_pythonpath
    )
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [
        str(gnuradio_python),
        "-m",
        "pilot_proxy.testbench.add_awgn",
        "--input-iq",
        str(input_iq_path),
        "--output-iq",
        str(output_iq_path),
        "--metadata-json",
        str(metadata_path),
        "--num-samples",
        str(clean.size),
        "--snr-db",
        str(float(snr_db)),
        "--sample-rate-hz",
        str(float(sample_rate_hz)),
        "--snr-bandwidth-hz",
        str(float(snr_bandwidth_hz)),
        "--seed",
        str(int(seed)),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "GNU Radio AWGN helper failed. "
            f"Command: {' '.join(cmd)}\n{details}"
        )
    noisy = np.fromfile(output_iq_path, dtype=np.complex64)
    if noisy.size != clean.size:
        raise RuntimeError(
            "GNU Radio AWGN helper wrote "
            f"{noisy.size} samples; expected {clean.size}."
        )
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return np.ascontiguousarray(noisy), clean_signal_power, noise_power, metadata


def apply_channel_impairments(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    frequency_offset_hz: float = DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ,
    gain_db: float = DEFAULT_CHANNEL_GAIN_DB,
    phase_deg: float = DEFAULT_CHANNEL_PHASE_DEG,
) -> np.ndarray:
    """Apply deterministic lightweight channel effects before AWGN injection."""
    clean = np.asarray(signal, dtype=np.complex64)
    sample_rate = float(sample_rate_hz)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    gain = DB_LINEAR_BASE ** (float(gain_db) / DB_AMPLITUDE_FACTOR)
    phase_rad = float(phase_deg) * math.pi / DEGREES_PER_HALF_TURN
    if (
        float(frequency_offset_hz) == DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ
        and float(gain_db) == DEFAULT_CHANNEL_GAIN_DB
        and float(phase_deg) == DEFAULT_CHANNEL_PHASE_DEG
    ):
        return np.ascontiguousarray(clean)
    sample_index = np.arange(clean.size, dtype=np.float64)
    phase = (
        TWO_PI * float(frequency_offset_hz) * sample_index / sample_rate
        + phase_rad
    )
    rotation = np.exp(1j * phase).astype(np.complex64)
    return np.ascontiguousarray(clean * np.complex64(gain) * rotation)


def estimate_quantization_scale(
    streams: np.ndarray,
    *,
    bits: int,
    clip_sigma: float,
) -> float:
    """Estimate int4 quantization scale from complex stream statistics."""
    values = np.asarray(streams)
    sigma = float(np.std(values.real))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values.imag))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(np.abs(values)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Could not estimate a positive quantization scale.")
    max_int = (1 << (int(bits) - 1)) - 1
    return float(max_int) / (float(clip_sigma) * sigma)


def _resolve_rf_center_hz(args: argparse.Namespace) -> float:
    if args.rf_center_mhz is not None:
        return float(args.rf_center_mhz) * HZ_PER_MHZ
    pilot_hz = float(args.dtv_pilot_mhz) * HZ_PER_MHZ
    return pilot_hz + (
        ATSC_CHANNEL_WIDTH_HZ / HALF_SCALE - float(args.atsc_pilot_offset_hz)
    )


def _requested_snr_shelf_values(args: argparse.Namespace) -> list[float]:
    values = [float(v) for v in args.requested_snr_shelf_db or []]
    if args.snr_start_db is not None:
        if args.snr_stop_db is None or args.snr_step_db is None:
            raise SystemExit(
                "--snr-start-db requires --snr-stop-db and --snr-step-db."
            )
        if args.snr_step_db == 0.0:
            raise SystemExit("--snr-step-db must be non-zero.")
        current = float(args.snr_start_db)
        stop = float(args.snr_stop_db)
        step = float(args.snr_step_db)
        compare = (
            (lambda a, b: a <= b + SNR_RANGE_EPSILON)
            if step > 0
            else (
                lambda a, b: a >= b - SNR_RANGE_EPSILON
            )
        )
        while compare(current, stop):
            values.append(float(current))
            current += step
    if not values:
        current = DEFAULT_SNR_SWEEP_MIN_DB
        while current <= DEFAULT_SNR_SWEEP_MAX_DB + SNR_RANGE_EPSILON:
            values.append(float(current))
            current += DEFAULT_SNR_SWEEP_STEP_DB
    for value in values:
        if (
            value < DEFAULT_SNR_SWEEP_MIN_DB - SNR_RANGE_EPSILON
            or value > DEFAULT_SNR_SWEEP_MAX_DB + SNR_RANGE_EPSILON
        ):
            raise SystemExit(
                "Testbench SNR values must be in the public validation range "
                f"[{DEFAULT_SNR_SWEEP_MIN_DB:g}, {DEFAULT_SNR_SWEEP_MAX_DB:g}] dB; "
                f"got {value:g} dB."
            )
    return values


def _frequency_offset_values(args: argparse.Namespace) -> list[float]:
    values = [float(v) for v in args.frequency_offset_hz or []]
    if args.standard_frequency_offset_sweep:
        values.extend(float(v) for v in STANDARD_FREQUENCY_OFFSET_SWEEP_HZ)
    if not values:
        values = [DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ]
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 9)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(value))
    return unique


def _ideal_float_weights_from_layout(
    selected_weight_layout: dict[str, Any],
    *,
    detector_window_samples: int,
) -> np.ndarray:
    """Build unquantized complex DFT weights from manifest target frequencies."""
    keys = [
        "target_normalized_frequency",
        "lower_reference_normalized_frequency",
        "upper_reference_normalized_frequency",
    ]
    if not all(key in selected_weight_layout for key in keys):
        raise ValueError("selected weight layout lacks normalized frequencies.")
    sample_index = np.arange(int(detector_window_samples), dtype=np.float64)
    rows = [
        np.exp(-1j * TWO_PI * float(selected_weight_layout[key]) * sample_index)
        for key in keys
    ]
    return np.ascontiguousarray(np.stack(rows).astype(np.complex128))


def _float_streams_for_reference(
    streams: np.ndarray,
    *,
    samples_per_block: int,
    detector_window_samples: int,
    spectral_sense: str,
) -> np.ndarray:
    """Convert unquantized streams to detector rows for the CPU float path."""
    matrix = stream_time_block_to_detector_matrix(
        np.asarray(streams)[:, : int(samples_per_block)],
        detector_window_samples=int(detector_window_samples),
    )
    return apply_spectral_sense_to_detector_matrix(
        matrix,
        spectral_sense=spectral_sense,
    )


def _pack_streams_for_kernel(
    streams: np.ndarray,
    *,
    samples_per_block: int,
    detector_window_samples: int,
    bits: int,
    scale: float,
    spectral_sense: str,
) -> np.ndarray:
    feed_channel_streams = np.asarray(streams)[:, np.newaxis, :]
    packed_input = pack_channelized_streams_for_detector(
        feed_channel_streams,
        frame_size_samples=int(samples_per_block),
        detector_window_samples=int(detector_window_samples),
        spectral_sense=spectral_sense,
        quantization_scale_mode=QUANTIZATION_SCALE_MODE_GLOBAL,
        clip_sigma=DEFAULT_CLIP_SIGMA,
        bits_per_component=int(bits),
        scale=float(scale),
    )
    return np.ascontiguousarray(packed_input.packed[0])


def _kernel_measurements(
    *,
    cp: Any,
    kernel: FStatKernel,
    packed: np.ndarray,
    weights: np.ndarray,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run diagnostic float and exact uint64 power readbacks."""
    d_in = cp.asarray(packed)
    d_diag = cp.zeros(1, dtype=cp.float32)
    d_powers = cp.zeros(int(kernel.specs.N), dtype=cp.uint64)
    d_mask_num = cp.zeros(1, dtype=cp.uint64)
    d_mask_den = cp.zeros(1, dtype=cp.uint64)
    d_mask = cp.zeros(1, dtype=cp.uint8)
    d_overflow = cp.zeros(1, dtype=cp.uint32)
    handle = kernel.create_raw(d_in.shape[0], d_in.data.ptr, d_diag.data.ptr)
    try:
        kernel.compute_diagnostic_float(handle, weights.ctypes.data)
        cp.cuda.Device().synchronize()
        diagnostic_float = float(d_diag[0].get())

        kernel.compute_powers_u64(handle, weights.ctypes.data, d_powers.data.ptr)
        cp.cuda.Device().synchronize()
        powers = cp.asnumpy(d_powers).astype(np.uint64, copy=False)

        mask = 0
        overflow = 0
        if threshold is not None:
            if getattr(kernel, "_has_numden_mask_rational_half_checked", False):
                kernel.compute_numden_mask_rational_half_checked(
                    handle,
                    weights.ctypes.data,
                    int(threshold["threshold_half_num"]),
                    int(threshold["threshold_half_den"]),
                    d_mask_num.data.ptr,
                    d_mask_den.data.ptr,
                    d_mask.data.ptr,
                    d_overflow.data.ptr,
                )
                cp.cuda.Device().synchronize()
                overflow = int(cp.asnumpy(d_overflow)[0])
            else:
                kernel.compute_numden_mask_rational_half(
                    handle,
                    weights.ctypes.data,
                    int(threshold["threshold_half_num"]),
                    int(threshold["threshold_half_den"]),
                    d_mask_num.data.ptr,
                    d_mask_den.data.ptr,
                    d_mask.data.ptr,
                )
                cp.cuda.Device().synchronize()
                overflow = 0
            mask = int(cp.asnumpy(d_mask)[0])
    finally:
        kernel.destroy(handle)

    return _measurements_from_powers(
        diagnostic_float=diagnostic_float,
        p_target=int(powers[REFERENCE_TARGET_TERM_INDEX]),
        p_ref_lower=int(powers[REFERENCE_LOWER_TERM_INDEX]),
        p_ref_upper=int(powers[REFERENCE_UPPER_TERM_INDEX]),
        weights=weights,
        pilot_below_data_db=pilot_below_data_db,
        bin_enbw_hz=bin_enbw_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        threshold=threshold,
        mask=mask,
        overflow=overflow,
    )


def _cpu_reference_measurements(
    *,
    packed: np.ndarray,
    weights: np.ndarray,
    bits: int,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU exact-integer sibling of ``_kernel_measurements``.

    Uses the validated CPU reference (``fstat_cpu_reference_packed``) for the
    power sums and Python-integer arithmetic for the rational-half threshold
    decision, so the fields are the kernel's semantics without a GPU. The
    kernel <-> reference equivalence itself is CI-gated by the kernel parity
    suite; a small same-seed GPU spot check ties a CPU-produced sweep to the
    deployed kernel (see docs/PUBLICATION_VALIDATION.md, item 2).
    """
    fstat, sums = fstat_cpu_reference_packed(packed, weights, int(bits))
    p_target = int(round(float(sums[0])))
    p_ref_lower = int(round(float(sums[1])))
    p_ref_upper = int(round(float(sums[2])))
    diagnostic_float = float(np.float32(fstat))
    mask = 0
    overflow = 0
    if threshold is not None:
        p_ref_sum = p_ref_lower + p_ref_upper
        mask = int(
            p_ref_sum != 0
            and p_target * int(threshold["threshold_half_den"])
            > int(threshold["threshold_half_num"]) * p_ref_sum
        )
    return _measurements_from_powers(
        diagnostic_float=diagnostic_float,
        p_target=p_target,
        p_ref_lower=p_ref_lower,
        p_ref_upper=p_ref_upper,
        weights=weights,
        pilot_below_data_db=pilot_below_data_db,
        bin_enbw_hz=bin_enbw_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        threshold=threshold,
        mask=mask,
        overflow=overflow,
    )


def _measurements_from_powers(
    *,
    diagnostic_float: float,
    p_target: int,
    p_ref_lower: int,
    p_ref_upper: int,
    weights: np.ndarray,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None,
    mask: int,
    overflow: int,
) -> dict[str, Any]:
    """Backend-agnostic measurement fields from the three integer powers."""
    p_ref_sum = int(p_ref_lower + p_ref_upper)
    fstat_raw = float(fstat_num_den_to_raw(p_target, p_ref_sum))
    fstat_level_db = float(fstat_num_den_to_fstat_level_db(p_target, p_ref_sum))
    pilot_excess = float(fstat_num_den_to_pilot_excess_linear(p_target, p_ref_sum))
    pnr_bin_db = float(fstat_num_den_to_pnr_bin_db(p_target, p_ref_sum))
    snr_shelf_db = float(
        pnr_bin_db_to_snr_shelf_db(
            pnr_bin_db,
            pilot_below_data_db=float(pilot_below_data_db),
            bin_enbw_hz=float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
            pilot_capture_efficiency=float(pilot_capture_efficiency),
        )
    )
    _nt, _nl, _nu = weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    out = {
        "diagnostic_raw_float32": diagnostic_float,
        "diagnostic_level_db_float32": _positive_to_db(diagnostic_float),
        "p_target_u64": p_target,
        "p_ref_lower_u64": p_ref_lower,
        "p_ref_upper_u64": p_ref_upper,
        "p_ref_sum_u64": p_ref_sum,
        "fstat_raw": fstat_raw,
        "fstat_level_db": fstat_level_db,
        "pilot_excess_linear": pilot_excess,
        "pnr_bin_db": pnr_bin_db,
        "estimated_snr_shelf_db": snr_shelf_db,
        "positive_excess": norm_corrected_positive_excess(
            p_target,
            p_ref_sum,
            target_norm_sq=_nt,
            ref_norm_sum_sq=_nl + _nu,
        ),
    }
    if threshold is not None:
        out["mask"] = int(mask)
        out["rational_overflow_count"] = int(overflow)
    return out


def _positive_to_db(value: float) -> float:
    """Convert a positive linear value to dB, preserving non-positive as -inf."""
    value = float(value)
    if value <= 0.0:
        return float("-inf")
    return float(DB_POWER_FACTOR * math.log10(value))


def _safe_snr_label(snr_db: float) -> str:
    return (
        f"{float(snr_db):+.{CSV_SNR_LABEL_PRECISION}f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


def _load_waveform_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _command_or_path_exists(command: str) -> bool:
    command_text = str(command)
    if not command_text:
        return False
    if Path(command_text).exists():
        return True
    has_path_separator = any(
        separator and separator in command_text for separator in (os.sep, os.altsep)
    )
    if has_path_separator:
        return False
    return any(
        (Path(directory) / command_text).exists()
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
    )


def _evaluate_one_trial(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    cp: Any,
    kernel: FStatKernel,
    weights: np.ndarray,
    cpu_float_weights: np.ndarray,
    clean_iq: np.ndarray,
    gnuradio_input_iq_path: Path,
    output_dir: Path,
    noisy_iq_dir: Path,
    rf_center_hz: float,
    band_lower_hz: float,
    response: np.ndarray,
    spec: ReferenceChannelizerSpec,
    channel_index: int,
    n_blocks: int,
    threshold: dict[str, Any] | None,
    requested_snr_shelf_db: float,
    requested_composite_atsc_snr_db: float,
    pilot_data_ratio: float,
    frequency_offset_hz: float,
    trial: int,
) -> dict[str, Any]:
    """Evaluate one SNR/frequency-offset/noise trial."""
    feed_noisy_iq: list[np.ndarray] = []
    feed_signal_powers: list[float] = []
    feed_requested_noise_powers: list[float] = []
    feed_noise_seeds: list[int] = []
    feed_noise_iq_paths: list[str] = []
    feed_gnuradio_blocks: list[str] = []

    for feed_index in range(int(args.num_input_streams)):
        feed_noisy_sample: np.ndarray | None = None
        feed_signal_power: float | None = None
        feed_requested_noise_power: float | None = None
        feed_gnuradio_metadata: dict[str, Any] = {}
        if args.noise_source == "gnuradio":
            noise_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            feed_noise_seeds.append(noise_seed)
            if args.save_noisy_iq:
                noise_iq_file = noisy_iq_dir / (
                    "snr_shelf_"
                    f"{_safe_snr_label(float(requested_snr_shelf_db))}_"
                    "freq_offset_"
                    f"{_safe_snr_label(float(frequency_offset_hz))}_hz_"
                    f"trial_{trial:04d}_feed_{feed_index:04d}_"
                    f"seed_{noise_seed}.cfile"
                )
                (
                    feed_noisy_sample,
                    feed_signal_power,
                    feed_requested_noise_power,
                    feed_gnuradio_metadata,
                ) = add_gnuradio_awgn_for_snr(
                    clean_iq,
                    input_iq_path=gnuradio_input_iq_path,
                    output_iq_path=noise_iq_file,
                    snr_db=requested_composite_atsc_snr_db,
                    seed=noise_seed,
                    gnuradio_python=str(args.gnuradio_python),
                    sample_rate_hz=float(args.iq_sample_rate_hz),
                    snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
                )
                feed_noise_iq_paths.append(str(noise_iq_file))
            else:
                with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmp_dir:
                    noise_iq_file = Path(tmp_dir) / "noisy_iq.cfile"
                    (
                        feed_noisy_sample,
                        feed_signal_power,
                        feed_requested_noise_power,
                        feed_gnuradio_metadata,
                    ) = add_gnuradio_awgn_for_snr(
                        clean_iq,
                        input_iq_path=gnuradio_input_iq_path,
                        output_iq_path=noise_iq_file,
                        snr_db=requested_composite_atsc_snr_db,
                        seed=noise_seed,
                        gnuradio_python=str(args.gnuradio_python),
                        sample_rate_hz=float(args.iq_sample_rate_hz),
                        snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
                    )
            feed_gnuradio_blocks.append(
                str(feed_gnuradio_metadata.get("gnuradio_block", ""))
            )
        else:
            (
                feed_noisy_sample,
                feed_signal_power,
                feed_requested_noise_power,
            ) = add_complex_awgn_for_snr(
                clean_iq,
                snr_db=requested_composite_atsc_snr_db,
                rng=rng,
                sample_rate_hz=float(args.iq_sample_rate_hz),
                snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
            )

        if (
            feed_noisy_sample is None
            or feed_signal_power is None
            or feed_requested_noise_power is None
        ):
            raise RuntimeError("noise generation did not produce feed samples")
        feed_noisy_iq.append(feed_noisy_sample)
        feed_signal_powers.append(float(feed_signal_power))
        feed_requested_noise_powers.append(float(feed_requested_noise_power))

    mean_signal_power = float(np.mean(feed_signal_powers))
    mean_requested_noise_power = float(np.mean(feed_requested_noise_powers))
    bandwidth_ratio = float(args.dtv_bandwidth_hz) / float(args.iq_sample_rate_hz)
    requested_in_band_noise_power = mean_requested_noise_power * bandwidth_ratio
    requested_combined_in_band_noise_power = (
        requested_in_band_noise_power * int(args.num_input_streams)
    )
    realized_noise_power_by_feed = [
        float(np.mean(np.abs(noisy.astype(np.complex64) - clean_iq) ** 2))
        for noisy in feed_noisy_iq
    ]
    realized_in_band_noise_power_by_feed = [
        float(value * bandwidth_ratio)
        for value in realized_noise_power_by_feed
    ]
    realized_noise_power = float(np.sum(realized_noise_power_by_feed))
    realized_in_band_noise_power = float(np.sum(realized_in_band_noise_power_by_feed))
    combined_signal_power = float(mean_signal_power * int(args.num_input_streams))
    measured_truth_composite_atsc_snr_db = _positive_to_db(
        combined_signal_power / realized_in_band_noise_power
    )
    measured_data_shelf_power = float(
        combined_signal_power / (NO_PILOT_EXCESS_FSTAT + pilot_data_ratio)
    )
    measured_truth_snr_shelf_db = _positive_to_db(
        measured_data_shelf_power / realized_in_band_noise_power
    )

    feed_channel_streams = []
    for noisy_iq in feed_noisy_iq:
        raw_blocks = complex_envelope_to_real_adc_blocks(
            noisy_iq,
            iq_sample_rate_hz=float(args.iq_sample_rate_hz),
            rf_center_hz=rf_center_hz,
            adc_sample_rate_hz=float(args.adc_sample_rate_hz),
            band_lower_hz=band_lower_hz,
            n_blocks=n_blocks,
            block_size=REFERENCE_PFB_FFT_SIZE,
        )
        channel_streams = channelize_real_blocks_to_reference_channels(
            raw_blocks,
            channel_indices=[channel_index],
            response=response,
            spec=spec,
        )
        if args.reference_archive_phase:
            channel_streams = apply_reference_archive_phase_convention(
                channel_streams
            )
        feed_channel_streams.append(channel_streams)
    streams = flatten_feed_channel_streams(np.stack(feed_channel_streams, axis=0))

    cpu_float_rows = _float_streams_for_reference(
        streams,
        samples_per_block=int(args.samples_per_block),
        detector_window_samples=int(args.detector_window_samples),
        spectral_sense=str(args.spectral_sense),
    )
    cpu_float_fstat, cpu_float_powers = fstat_cpu_reference(
        cpu_float_rows,
        cpu_float_weights,
    )
    cpu_float_pnr_bin_db = float(fstat_raw_to_pnr_bin_db(cpu_float_fstat))
    cpu_float_estimated_snr_shelf_db = float(
        pnr_bin_db_to_snr_shelf_db(
            cpu_float_pnr_bin_db,
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
        )
    )

    scale = (
        float(args.scale)
        if args.scale is not None
        else estimate_quantization_scale(
            streams,
            bits=int(args.bits),
            clip_sigma=float(args.clip_sigma),
        )
    )
    packed = _pack_streams_for_kernel(
        streams,
        samples_per_block=int(args.samples_per_block),
        detector_window_samples=int(args.detector_window_samples),
        bits=int(args.bits),
        scale=scale,
        spectral_sense=str(args.spectral_sense),
    )
    cpu_packed_fstat, _ = fstat_cpu_reference_packed(packed, weights, int(args.bits))
    if str(getattr(args, "detector_backend", "cuda")) == "cpu-reference":
        gpu = _cpu_reference_measurements(
            packed=packed,
            weights=weights,
            bits=int(args.bits),
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            threshold=threshold,
        )
    else:
        gpu = _kernel_measurements(
            cp=cp,
            kernel=kernel,
            packed=packed,
            weights=weights,
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            threshold=threshold,
        )
    estimated_snr_shelf_db = float(gpu["estimated_snr_shelf_db"])
    row = {
        "detector_backend": str(getattr(args, "detector_backend", "cuda")),
        "requested_snr_shelf_db": float(requested_snr_shelf_db),
        "requested_composite_atsc_snr_db": float(requested_composite_atsc_snr_db),
        "frequency_offset_hz": float(frequency_offset_hz),
        "channel_gain_db": float(args.channel_gain_db),
        "channel_phase_deg": float(args.channel_phase_deg),
        "measured_truth_snr_shelf_db": measured_truth_snr_shelf_db,
        "measured_truth_composite_atsc_snr_db": measured_truth_composite_atsc_snr_db,
        "measured_data_shelf_power": measured_data_shelf_power,
        "measured_composite_atsc_power": float(combined_signal_power),
        "measured_noise_power": float(realized_noise_power),
        "measured_noise_power_per_feed_mean": float(
            np.mean(realized_noise_power_by_feed)
        ),
        "measured_in_band_noise_power": float(realized_in_band_noise_power),
        "measured_in_band_noise_power_per_feed_mean": float(
            np.mean(realized_in_band_noise_power_by_feed)
        ),
        "fstat_raw": float(gpu["fstat_raw"]),
        "fstat_level_db": float(gpu["fstat_level_db"]),
        "pilot_excess_linear": float(gpu["pilot_excess_linear"]),
        "pnr_bin_db": float(gpu["pnr_bin_db"]),
        "estimated_snr_shelf_db": estimated_snr_shelf_db,
        "gpu_estimated_snr_shelf_db": estimated_snr_shelf_db,
        "snr_error_db": estimated_snr_shelf_db - measured_truth_snr_shelf_db,
        "p_target_u64": int(gpu["p_target_u64"]),
        "p_ref_lower_u64": int(gpu["p_ref_lower_u64"]),
        "p_ref_upper_u64": int(gpu["p_ref_upper_u64"]),
        "p_ref_sum_u64": int(gpu["p_ref_sum_u64"]),
        "positive_excess": int(gpu["positive_excess"]),
        "diagnostic_raw_float32": float(gpu["diagnostic_raw_float32"]),
        "diagnostic_level_db_float32": float(gpu["diagnostic_level_db_float32"]),
        "cpu_float_fstat_raw": float(cpu_float_fstat),
        "cpu_float_p_target": float(cpu_float_powers[REFERENCE_TARGET_TERM_INDEX]),
        "cpu_float_p_ref_lower": float(cpu_float_powers[REFERENCE_LOWER_TERM_INDEX]),
        "cpu_float_p_ref_upper": float(cpu_float_powers[REFERENCE_UPPER_TERM_INDEX]),
        "cpu_float_pnr_bin_db": cpu_float_pnr_bin_db,
        "cpu_float_estimated_snr_shelf_db": cpu_float_estimated_snr_shelf_db,
        "cpu_float_snr_error_db": (
            cpu_float_estimated_snr_shelf_db - measured_truth_snr_shelf_db
        ),
        "trial": int(trial),
        "num_feeds": int(args.num_input_streams),
        "num_input_streams": int(args.num_input_streams),
        "num_selected_channels": 1,
        "detector_rows_per_block": int(packed.shape[0]),
        "noise_source": str(args.noise_source),
        "noise_seed": (
            "" if not feed_noise_seeds else ";".join(str(seed) for seed in feed_noise_seeds)
        ),
        "noise_seeds": [int(seed) for seed in feed_noise_seeds],
        "noise_iq_path": ";".join(feed_noise_iq_paths),
        "noise_iq_paths": [str(path) for path in feed_noise_iq_paths],
        "gnuradio_block": ";".join(
            sorted({block for block in feed_gnuradio_blocks if block})
        ),
        "requested_noise_power": float(mean_requested_noise_power),
        "requested_in_band_noise_power": float(requested_in_band_noise_power),
        "requested_combined_in_band_noise_power": float(
            requested_combined_in_band_noise_power
        ),
        "noise_amplitude": float(math.sqrt(mean_requested_noise_power)),
        "noise_component_sigma": float(
            math.sqrt(mean_requested_noise_power / COMPLEX_COMPONENT_COUNT)
        ),
        "quantization_scale": float(scale),
        "cpu_fstat_raw": float(cpu_packed_fstat),
        "cpu_packed_fstat_raw": float(cpu_packed_fstat),
        "cpu_gpu_abs_diff": abs(float(gpu["fstat_raw"]) - float(cpu_packed_fstat)),
        "cpu_float_gpu_snr_diff_db": (
            cpu_float_estimated_snr_shelf_db - estimated_snr_shelf_db
        ),
    }
    if threshold is not None:
        row.update(
            {
                "mask": int(gpu["mask"]),
                "threshold_snr_shelf_db": float(threshold["threshold_snr_shelf_db"]),
                "threshold_pnr_bin_db": float(threshold["threshold_pnr_bin_db"]),
                "threshold_fstat_raw": float(threshold["threshold_fstat_raw"]),
                "threshold_half_num": int(threshold["threshold_half_num"]),
                "threshold_half_den": int(threshold["threshold_half_den"]),
                "rational_overflow_count": int(gpu["rational_overflow_count"]),
            }
        )
    return row


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Detection-rate points on publication curves should carry these bounds;
    they stay meaningful at rates near 0 or 1 where the normal approximation
    fails. Returns (lo, hi); (nan, nan) when trials == 0.
    """
    n = int(trials)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / n
    if successes <= 0:
        p = 0.0
    elif successes >= n:
        p = 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / denom
    lo = 0.0 if successes <= 0 else max(0.0, center - half)
    hi = 1.0 if successes >= n else min(1.0, center + half)
    return (lo, hi)


def _detection_rate_fields(group: list[dict]) -> dict:
    """Detection-rate summary fields with Wilson 95% bounds for one group."""
    fields: dict = {}
    n = len(group)
    if n and all("positive_excess" in row for row in group):
        detected = sum(int(row["positive_excess"]) for row in group)
        lo, hi = wilson_interval(detected, n)
        fields["positive_excess_detection_rate"] = detected / n
        fields["positive_excess_detection_rate_wilson95_lo"] = lo
        fields["positive_excess_detection_rate_wilson95_hi"] = hi
    if n and all("mask" in row for row in group):
        detected = sum(int(row["mask"]) for row in group)
        lo, hi = wilson_interval(detected, n)
        fields["threshold_detection_rate"] = detected / n
        fields["threshold_detection_rate_wilson95_lo"] = lo
        fields["threshold_detection_rate_wilson95_hi"] = hi
    return fields


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    requested_values: list[float],
    frequency_offset_values: list[float],
    composite_to_shelf_db: float,
    num_input_streams: int,
) -> list[dict[str, Any]]:
    """Summarize validation rows by requested SNR and frequency offset."""
    summary_rows: list[dict[str, Any]] = []
    for frequency_offset_hz in frequency_offset_values:
        for requested_snr_shelf_db in requested_values:
            group = [
                row
                for row in rows
                if row["requested_snr_shelf_db"] == float(requested_snr_shelf_db)
                and row["frequency_offset_hz"] == float(frequency_offset_hz)
            ]
            if not group:
                continue
            estimates = np.asarray(
                [row["estimated_snr_shelf_db"] for row in group],
                dtype=np.float64,
            )
            errors = np.asarray(
                [row["snr_error_db"] for row in group],
                dtype=np.float64,
            )
            fstats = np.asarray([row["fstat_raw"] for row in group], dtype=np.float64)
            fstat_levels = np.asarray(
                [row["fstat_level_db"] for row in group],
                dtype=np.float64,
            )
            pnr_bin = np.asarray([row["pnr_bin_db"] for row in group], dtype=np.float64)
            truth_shelf = np.asarray(
                [row["measured_truth_snr_shelf_db"] for row in group],
                dtype=np.float64,
            )
            truth_composite = np.asarray(
                [row["measured_truth_composite_atsc_snr_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_estimates = np.asarray(
                [row["cpu_float_estimated_snr_shelf_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_errors = np.asarray(
                [row["cpu_float_snr_error_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_fstats = np.asarray(
                [row["cpu_float_fstat_raw"] for row in group],
                dtype=np.float64,
            )
            diffs = np.asarray(
                [row["cpu_gpu_abs_diff"] for row in group],
                dtype=np.float64,
            )
            cpu_gpu_snr_diff = np.asarray(
                [row["cpu_float_gpu_snr_diff_db"] for row in group],
                dtype=np.float64,
            )
            summary_rows.append(
                {
                    "requested_snr_shelf_db": float(requested_snr_shelf_db),
                    "frequency_offset_hz": float(frequency_offset_hz),
                    "channel_gain_db": float(group[0]["channel_gain_db"]),
                    "channel_phase_deg": float(group[0]["channel_phase_deg"]),
                    "requested_composite_atsc_snr_db": float(
                        float(requested_snr_shelf_db) - composite_to_shelf_db
                    ),
                    "measured_truth_snr_shelf_db_mean": _nanmean_or_nan(
                        truth_shelf
                    ),
                    "measured_truth_composite_atsc_snr_db_mean": _nanmean_or_nan(
                        truth_composite
                    ),
                    "fstat_raw_mean": _nanmean_or_nan(fstats),
                    "fstat_level_db_mean": _nanmean_or_nan(fstat_levels),
                    "pnr_bin_db_mean": _nanmean_or_nan(pnr_bin),
                    "estimated_snr_shelf_db_mean": _nanmean_or_nan(estimates),
                    "estimated_snr_shelf_db_std": _nanstd_or_nan(estimates),
                    "snr_error_db_mean": _nanmean_or_nan(errors),
                    "snr_error_db_std": _nanstd_or_nan(errors),
                    "cpu_float_fstat_raw_mean": _nanmean_or_nan(cpu_float_fstats),
                    "cpu_float_estimated_snr_shelf_db_mean": _nanmean_or_nan(
                        cpu_float_estimates
                    ),
                    "cpu_float_estimated_snr_shelf_db_std": _nanstd_or_nan(
                        cpu_float_estimates
                    ),
                    "cpu_float_snr_error_db_mean": _nanmean_or_nan(
                        cpu_float_errors
                    ),
                    "cpu_float_snr_error_db_std": _nanstd_or_nan(cpu_float_errors),
                    "cpu_float_gpu_snr_diff_db_mean": _nanmean_or_nan(
                        cpu_gpu_snr_diff
                    ),
                    "trials": int(len(group)),
                    **_detection_rate_fields(group),
                    "num_feeds": int(num_input_streams),
                    "num_input_streams": int(num_input_streams),
                    "cpu_gpu_abs_diff_max": _nanmax_or_nan(diffs),
                }
            )
    return summary_rows


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description=(
            "Inject AWGN into a clean GNU Radio ATSC IQ waveform, run the "
            "reference-channelizer/4+4-bit pipeline and CUDA kernel, and "
            "compare the F-statistic-derived data-shelf SNR to measured truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-iq", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "generated" / "dtv_snr_eval",
    )
    parser.add_argument(
        "--requested-snr-shelf-db",
        type=float,
        action="append",
        default=None,
        help=(
            "Requested ATSC data-shelf SNR relative to non-DTV noise power "
            "integrated over dtv_bandwidth_hz."
        ),
    )
    parser.add_argument("--snr-start-db", type=float, default=None)
    parser.add_argument("--snr-stop-db", type=float, default=None)
    parser.add_argument("--snr-step-db", type=float, default=None)
    parser.add_argument(
        "--frequency-offset-hz",
        type=float,
        action="append",
        default=None,
        help=(
            "Apply a baseband frequency offset before AWGN. Repeat to sweep "
            "multiple offsets."
        ),
    )
    parser.add_argument(
        "--standard-frequency-offset-sweep",
        action="store_true",
        help="Evaluate the built-in -1 kHz, 0 Hz, +1 kHz offset sweep.",
    )
    parser.add_argument(
        "--channel-gain-db",
        type=float,
        default=DEFAULT_CHANNEL_GAIN_DB,
        help="Static gain applied before noise injection.",
    )
    parser.add_argument(
        "--channel-phase-deg",
        type=float,
        default=DEFAULT_CHANNEL_PHASE_DEG,
        help="Static phase rotation applied before noise injection.",
    )
    parser.add_argument(
        "--detector-backend",
        choices=("cuda", "cpu-reference"),
        default="cuda",
        help=(
            "Which detector computes the primary fields. 'cuda' runs the "
            "compiled kernel (requires a GPU). 'cpu-reference' uses the "
            "validated exact-integer CPU reference, so full publication "
            "sweeps run without a GPU; tie the result to the deployed "
            "kernel with a small same-seed GPU spot check afterwards."
        ),
    )
    parser.add_argument(
        "--noise-trials",
        type=int,
        default=DEFAULT_NOISE_TRIALS,
        help=(
            "Independent noise realizations per (offset, SNR) point. The "
            "default is sized for quick sweeps; publication-grade "
            "detection-rate curves need >= 100-1000 trials per point (the "
            "summary reports Wilson 95%% intervals sized by this count)."
        ),
    )
    parser.add_argument(
        "--noise-source",
        choices=["gnuradio", "python"],
        default="gnuradio",
        help=(
            "Source used to add AWGN. The GNU Radio mode uses "
            "analog.noise_source_c plus blocks.add_cc in a helper process."
        ),
    )
    parser.add_argument(
        "--gnuradio-python",
        default=DEFAULT_GNURADIO_PYTHON,
        help="Python executable that can import GNU Radio.",
    )
    parser.add_argument(
        "--save-noisy-iq",
        action="store_true",
        help="Keep per-trial noisy IQ files generated by the GNU Radio helper.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATOR_SEED)
    parser.add_argument(
        "--waveform-audit-json",
        type=Path,
        default=REPO_ROOT / "generated" / "atsc" / "atsc_waveform_audit.json",
        help="Optional waveform-audit JSON to embed in the validation report.",
    )
    parser.add_argument(
        "--iq-sample-rate-hz",
        type=float,
        default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    )
    parser.add_argument(
        "--adc-sample-rate-hz",
        type=float,
        default=REFERENCE_ADC_SAMPLE_RATE_HZ,
    )
    parser.add_argument(
        "--band-lower-mhz",
        type=float,
        default=REFERENCE_BAND_LOWER_HZ / HZ_PER_MHZ,
    )
    parser.add_argument(
        "--dtv-pilot-mhz",
        type=float,
        default=DEFAULT_DTV_PILOT_HZ / HZ_PER_MHZ,
    )
    parser.add_argument("--physical-channel", type=int, default=None)
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument(
        "--pilot-below-data-db",
        dest="pilot_below_data_db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
        help="Positive dB offset: ATSC pilot power below average data-shelf power.",
    )
    parser.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    parser.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    parser.add_argument("--threshold-snr-shelf-db", type=float, default=None)
    parser.add_argument(
        "--max-denominator",
        type=int,
        default=DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    )
    parser.add_argument("--rf-center-mhz", type=float, default=None)
    parser.add_argument(
        "--atsc-pilot-offset-hz",
        type=float,
        default=ATSC_PILOT_OFFSET_HZ,
    )
    parser.add_argument("--channel-index", type=int, default=None)
    parser.add_argument(
        "--frame-size-samples",
        dest="samples_per_block",
        type=int,
        default=DEFAULT_FRAME_SIZE_SAMPLES,
        help="Frame size, in channelized samples, to evaluate per trial.",
    )
    parser.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
        help=(
            "Number of independent input streams/feeds to combine into one "
            "detector decision. Each stream receives the same clean ATSC "
            "waveform and an independent AWGN realization."
        ),
    )
    parser.add_argument(
        "--experimental-detector-window-samples",
        dest="detector_window_samples",
        type=int,
        default=LOCKED_DETECTOR_WINDOW_SAMPLES,
        help="Advanced: v0.1 only accepts the locked value 128.",
    )
    parser.add_argument(
        "--experimental-bits",
        dest="bits",
        type=int,
        default=LOCKED_BITS_PER_COMPONENT,
        help="Advanced: v0.1 only accepts the locked 4+4 bit format.",
    )
    parser.add_argument("--clip-sigma", type=float, default=DEFAULT_CLIP_SIGMA)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--spectral-sense",
        choices=["normal", "inverted"],
        default="normal",
    )
    parser.add_argument(
        "--reference-archive-phase",
        dest="reference_archive_phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the reference channel phase convention expected by weights.",
    )
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    return run(parser.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Run the evaluator from a parsed namespace (shared with the CLI)."""
    if args.bits != LOCKED_BITS_PER_COMPONENT:
        raise SystemExit("This evaluator is intended for locked 4+4 bit input.")
    if args.detector_window_samples != LOCKED_DETECTOR_WINDOW_SAMPLES:
        raise SystemExit(
            "This evaluator is intended for the locked 128-sample detector "
            "window used by the shipped kernel and weights."
        )
    if args.physical_channel is not None:
        args.dtv_pilot_mhz = physical_channel_to_pilot_hz(
            int(args.physical_channel)
        ) / HZ_PER_MHZ
    if args.noise_trials <= 0:
        raise SystemExit("--noise-trials must be positive.")
    if args.num_input_streams <= 0:
        raise SystemExit("--num-input-streams must be positive.")
    if args.samples_per_block % args.detector_window_samples != 0:
        raise SystemExit(
            "--frame-size-samples must be an integer multiple of the locked "
            "128-sample detector window."
        )
    if args.noise_source == "gnuradio":
        gnuradio_python = str(args.gnuradio_python)
        if not _command_or_path_exists(gnuradio_python):
            raise SystemExit(
                f"Could not find GNU Radio Python executable: {gnuradio_python}"
            )

    if str(args.detector_backend) == "cpu-reference":
        cp = None
        kernel = None
    else:
        import cupy as cp
        kernel = FStatKernel(args.lib_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    noisy_iq_dir = output_dir / "noisy_iq"
    if args.save_noisy_iq:
        noisy_iq_dir.mkdir(parents=True, exist_ok=True)

    num_output_samples = int(args.samples_per_block)
    required = required_iq_samples(
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        num_output_samples=num_output_samples,
    )
    clean_iq = np.fromfile(args.input_iq, dtype=np.complex64)
    if clean_iq.size < required:
        raise SystemExit(
            f"Input IQ is too short: need {required} samples, got {clean_iq.size}."
        )
    clean_iq = np.ascontiguousarray(clean_iq[:required])

    band_lower_hz = float(args.band_lower_mhz) * HZ_PER_MHZ
    rf_center_hz = _resolve_rf_center_hz(args)
    spec = ReferenceChannelizerSpec(
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
    )
    channel_index = (
        int(args.channel_index)
        if args.channel_index is not None
        else nearest_reference_channel_index(float(args.dtv_pilot_mhz) * HZ_PER_MHZ, spec)
    )
    response = sinc_hamming_pfb_response(REFERENCE_PFB_TAPS, REFERENCE_PFB_FFT_SIZE)
    n_blocks = num_output_samples + REFERENCE_PFB_TAPS - 1

    weights_bank = DetectorWeightBank(
        explicit_path=args.weights_path,
        expected_kernel=(kernel.specs if kernel is not None else None),
    )
    selected_weight_layout = weights_bank.layout_for_pilot_frequency(
        float(args.dtv_pilot_mhz)
    )
    weights, valid = weights_bank.get_weights_for_pilot_frequency(
        float(args.dtv_pilot_mhz)
    )
    if weights is None or not valid:
        raise SystemExit(
            "No valid detector weights for DTV pilot "
            f"{float(args.dtv_pilot_mhz):.6f} MHz."
        )
    if kernel is None and int(weights.shape[1]) != int(args.detector_window_samples):
        raise SystemExit(
            "cpu-reference backend: weight bank K "
            f"({int(weights.shape[1])}) does not match "
            f"--detector-window-samples ({int(args.detector_window_samples)})."
        )

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    requested_values = _requested_snr_shelf_values(args)
    frequency_offset_values = _frequency_offset_values(args)
    composite_to_shelf_db = composite_to_data_shelf_snr_correction_db(
        pilot_below_data_db=float(args.pilot_below_data_db)
    )
    pilot_data_ratio = pilot_to_data_power_ratio(
        pilot_below_data_db=float(args.pilot_below_data_db)
    )
    threshold = None
    if args.threshold_snr_shelf_db is not None:
        threshold = snr_shelf_threshold_fields(
            float(args.threshold_snr_shelf_db),
            max_denominator=int(args.max_denominator),
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
        )
    cpu_float_weights = _ideal_float_weights_from_layout(
        selected_weight_layout,
        detector_window_samples=int(args.detector_window_samples),
    )

    for frequency_offset_hz in frequency_offset_values:
        channel_clean_iq = apply_channel_impairments(
            clean_iq,
            sample_rate_hz=float(args.iq_sample_rate_hz),
            frequency_offset_hz=float(frequency_offset_hz),
            gain_db=float(args.channel_gain_db),
            phase_deg=float(args.channel_phase_deg),
        )
        channel_effects_active = not (
            math.isclose(float(frequency_offset_hz), DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ)
            and math.isclose(float(args.channel_gain_db), DEFAULT_CHANNEL_GAIN_DB)
            and math.isclose(float(args.channel_phase_deg), DEFAULT_CHANNEL_PHASE_DEG)
        )
        channel_input_temp: tempfile.TemporaryDirectory[str] | None = None
        channel_input_iq_path = args.input_iq
        if channel_effects_active:
            channel_temp = tempfile.TemporaryDirectory(dir=str(output_dir))
            channel_input_temp = channel_temp
            channel_input_iq_path = Path(channel_temp.name) / "channel_iq.cfile"
            channel_clean_iq.tofile(channel_input_iq_path)
        try:
            for requested_snr_shelf_db in requested_values:
                requested_composite_atsc_snr_db = (
                    float(requested_snr_shelf_db) - composite_to_shelf_db
                )
                for trial in range(int(args.noise_trials)):
                    row = _evaluate_one_trial(
                        args=args,
                        rng=rng,
                        cp=cp,
                        kernel=kernel,
                        weights=weights,
                        cpu_float_weights=cpu_float_weights,
                        clean_iq=channel_clean_iq,
                        gnuradio_input_iq_path=channel_input_iq_path,
                        output_dir=output_dir,
                        noisy_iq_dir=noisy_iq_dir,
                        rf_center_hz=rf_center_hz,
                        band_lower_hz=band_lower_hz,
                        response=response,
                        spec=spec,
                        channel_index=channel_index,
                        n_blocks=n_blocks,
                        threshold=threshold,
                        requested_snr_shelf_db=float(requested_snr_shelf_db),
                        requested_composite_atsc_snr_db=(
                            requested_composite_atsc_snr_db
                        ),
                        pilot_data_ratio=pilot_data_ratio,
                        frequency_offset_hz=float(frequency_offset_hz),
                        trial=int(trial),
                    )
                    rows.append(row)
        finally:
            if channel_input_temp is not None:
                channel_input_temp.cleanup()

    if not rows:
        raise SystemExit("No validation rows were produced.")

    csv_path = output_dir / "dtv_snr_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = _summarize_rows(
        rows,
        requested_values=requested_values,
        frequency_offset_values=frequency_offset_values,
        composite_to_shelf_db=composite_to_shelf_db,
        num_input_streams=int(args.num_input_streams),
    )

    summary_csv_path = output_dir / "dtv_snr_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    conversion_metadata = pnr_bin_to_snr_shelf_metadata()
    conversion_metadata.update(
        {
            "bin_enbw_hz": float(args.bin_enbw_hz),
            "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
            "pilot_below_data_db": float(args.pilot_below_data_db),
            "pilot_to_data_power_db": float(-args.pilot_below_data_db),
            "pilot_capture_efficiency": float(args.pilot_capture_efficiency),
        }
    )
    audit = _load_waveform_audit(args.waveform_audit_json)
    measured_pilot_to_data_power_db = _optional_float(
        None if audit is None else audit.get("measured_pilot_to_data_power_db")
    )
    measured_pilot_below_data_db = _optional_float(
        None if audit is None else audit.get("measured_pilot_below_data_db")
    )
    if (
        measured_pilot_below_data_db is None
        and measured_pilot_to_data_power_db is not None
    ):
        measured_pilot_below_data_db = -float(measured_pilot_to_data_power_db)
    all_errors = np.asarray([row["snr_error_db"] for row in rows], dtype=np.float64)
    input_layout = input_layout_metadata(
        frame_size_samples=int(args.samples_per_block),
        detector_window_samples=int(args.detector_window_samples),
        num_feeds=int(args.num_input_streams),
        num_selected_channels=1,
    )
    stream_map = build_stream_map(
        num_feeds=int(args.num_input_streams),
        selected_channel_indices=[int(channel_index)],
        physical_channel=(
            None if args.physical_channel is None else int(args.physical_channel)
        ),
    )
    detector_geometry = {
        "input_layout": input_layout,
        "stream_map": stream_map,
        "detector_window_samples": int(args.detector_window_samples),
        "reference_offset_bins": int(weights_bank.reference_offset_bins),
        "nominal_reference_offset_bins": int(weights_bank.reference_offset_bins),
        "selected_lower_reference_offset_bins": selected_weight_layout.get(
            "lower_reference_offset_bins"
        ),
        "selected_upper_reference_offset_bins": selected_weight_layout.get(
            "upper_reference_offset_bins"
        ),
        "bin_enbw_hz": float(args.bin_enbw_hz),
        "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
        "pilot_capture_efficiency": float(args.pilot_capture_efficiency),
        "frame_size_samples": int(args.samples_per_block),
        "samples_per_block": int(args.samples_per_block),
        "windows_per_stream": int(input_layout["windows_per_stream"]),
        "windows_per_feed": int(input_layout["windows_per_feed"]),
        "num_feeds": int(args.num_input_streams),
        "num_selected_channels": 1,
        "num_input_streams": int(input_layout["num_input_streams"]),
        "num_streams": int(input_layout["num_streams"]),
        "detector_rows_per_block": int(input_layout["detector_rows_per_block"]),
        "combine_mode": str(input_layout["combine_mode"]),
        "stable_combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        "power_sum_rule": (
            "sum target/reference powers over all detector rows before forming F"
        ),
        "selected_channel_index": int(channel_index),
        "dtv_pilot_hz": float(args.dtv_pilot_mhz) * HZ_PER_MHZ,
        "rf_center_hz": float(rf_center_hz),
        "reference_archive_phase": bool(args.reference_archive_phase),
        "spectral_sense": str(args.spectral_sense),
    }
    summary = {
        "schema_version": "pilot_proxy_validation_report_v1",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "project_identity": (
            "Standalone CUDA F-statistic DTV pilot detector and GNU Radio "
            "ATSC 1.0 testbench."
        ),
        "input_iq": str(args.input_iq),
        "output_dir": str(output_dir),
        "atsc_waveform_audit": audit,
        "selected_weight_layout": selected_weight_layout,
        "result_schema": result_schema_object(
            frame_size_samples=int(args.samples_per_block),
            num_input_streams=int(args.num_input_streams),
            detector_window_samples=int(args.detector_window_samples),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_below_data_db=float(args.pilot_below_data_db),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            threshold=threshold,
            reference_offset_bins=int(weights_bank.reference_offset_bins),
            num_selected_channels=1,
        ),
        "detector_geometry": detector_geometry,
        "truth": {
            "snr_shelf_definition": (
                "ATSC data-shelf power relative to non-DTV noise power "
                "integrated over dtv_bandwidth_hz; pilot power is excluded "
                "from the shelf truth."
            ),
            "requested_snr_shelf_db_values": [float(v) for v in requested_values],
            "requested_snr_shelf_db_min": float(DEFAULT_SNR_SWEEP_MIN_DB),
            "requested_snr_shelf_db_max": float(DEFAULT_SNR_SWEEP_MAX_DB),
            "composite_to_shelf_snr_correction_db": float(composite_to_shelf_db),
            "pilot_to_data_power_ratio": float(pilot_data_ratio),
        },
        "calibration": {
            "mode": "standard"
            if (
                math.isclose(float(args.pilot_below_data_db), PILOT_BELOW_DATA_DB)
                and math.isclose(float(args.bin_enbw_hz), float(EFFECTIVE_BIN_BW_HZ))
                and math.isclose(
                    float(args.pilot_capture_efficiency),
                    PILOT_CAPTURE_EFFICIENCY,
                )
            )
            else "calibrated",
            "pilot_below_data_db_assumed": float(args.pilot_below_data_db),
            "measured_pilot_to_data_power_db": measured_pilot_to_data_power_db,
            "measured_pilot_below_data_db": measured_pilot_below_data_db,
            "bin_enbw_hz_assumed": float(args.bin_enbw_hz),
            "pilot_capture_efficiency_assumed": float(args.pilot_capture_efficiency),
            "snr_bias_db_mean": float(np.nanmean(all_errors)),
            "snr_bias_db_std": float(np.nanstd(all_errors)),
        },
        "detector_output": {
            "fstat_definition": "F = 2*P_target/(P_ref_lower + P_ref_upper)",
            "fstat_level_db_definition": "10*log10(F)",
            "pilot_excess_linear_definition": "rho = F - 1",
            "pnr_bin_db_definition": "10*log10(F - 1)",
            "estimated_snr_shelf_db_definition": (
                "pnr_bin_db - 10*log10(dtv_bandwidth_hz / bin_enbw_hz) "
                "+ pilot_below_data_db - 10*log10(pilot_capture_efficiency)"
            ),
            "uses_exact_uint64_powers": True,
            "cpu_float_reference": (
                "Unquantized channelized detector rows with ideal complex DFT "
                "weights from the selected manifest layout."
            ),
            "cpu_packed_reference": (
                "NumPy reference using packed int4 samples and packed int4 "
                "weights, useful for CPU/GPU fixed-point agreement checks."
            ),
        },
        "threshold": threshold,
        "testbench": {
            "noise_source": str(args.noise_source),
            "gnuradio_python": str(args.gnuradio_python),
            "save_noisy_iq": bool(args.save_noisy_iq),
            "iq_samples_used": int(clean_iq.size),
            "snr_sweep": {
                "default_min_db": float(DEFAULT_SNR_SWEEP_MIN_DB),
                "default_max_db": float(DEFAULT_SNR_SWEEP_MAX_DB),
                "default_step_db": float(DEFAULT_SNR_SWEEP_STEP_DB),
                "requested_values_db": [float(v) for v in requested_values],
            },
            "channel_effects": {
                "frequency_offset_hz_values": [
                    float(v) for v in frequency_offset_values
                ],
                "standard_frequency_offset_sweep_hz": [
                    float(v) for v in STANDARD_FREQUENCY_OFFSET_SWEEP_HZ
                ],
                "channel_gain_db": float(args.channel_gain_db),
                "channel_phase_deg": float(args.channel_phase_deg),
            },
            "kernel_version": (
                kernel.version.as_string() if kernel is not None
                else "cpu-reference"
            ),
            "kernel_specs": (
                kernel.specs.as_descriptive_dict() if kernel is not None
                else None
            ),
            "cuda_device": (
                cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if kernel is not None
                else None
            ),
            "conversion_metadata": conversion_metadata,
        },
        "csv_columns": {
            "requested_snr_shelf_db": (
                "Requested ATSC data-shelf SNR relative to non-DTV noise."
            ),
            "frequency_offset_hz": (
                "Baseband frequency offset applied before AWGN injection."
            ),
            "measured_truth_snr_shelf_db": (
                "Measured data-shelf truth from clean composite power, "
                "pilot/data correction, and realized in-band noise."
            ),
            "measured_truth_composite_atsc_snr_db": (
                "Measured clean composite ATSC IQ power over realized in-band noise."
            ),
            "fstat_raw": "2*P_target/(P_ref_lower + P_ref_upper).",
            "fstat_level_db": "10*log10(fstat_raw).",
            "pilot_excess_linear": "fstat_raw - 1.",
            "pnr_bin_db": "10*log10(pilot_excess_linear).",
            "estimated_snr_shelf_db": "DTV shelf SNR inferred from pnr_bin_db.",
            "snr_error_db": (
                "estimated_snr_shelf_db minus measured_truth_snr_shelf_db."
            ),
            "cpu_float_estimated_snr_shelf_db": (
                "DTV shelf SNR from the unquantized CPU float reference."
            ),
            "cpu_float_snr_error_db": (
                "cpu_float_estimated_snr_shelf_db minus measured truth."
            ),
            "cpu_fstat_raw": (
                "Packed NumPy CPU reference F-statistic for fixed-point "
                "CPU/GPU agreement diagnostics."
            ),
        },
        "summary": summary_rows,
        "results": rows,
    }
    json_path = output_dir / "dtv_snr_eval.json"
    write_json_strict(json_path, summary, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {json_path}")
    print(
        "frequency_offset_hz, requested_snr_shelf_db, "
        "measured_truth_snr_shelf_db, cpu_float_snr_shelf_db, "
        "gpu_snr_shelf_db, gpu_snr_error_db"
    )
    for row in summary_rows:
        print(
            f"{row['frequency_offset_hz']:10.3f}, "
            f"{row['requested_snr_shelf_db']:8.3f}, "
            f"{row['measured_truth_snr_shelf_db_mean']:8.3f}, "
            f"{row['cpu_float_estimated_snr_shelf_db_mean']:8.3f}, "
            f"{row['estimated_snr_shelf_db_mean']:8.3f}, "
            f"{row['snr_error_db_mean']:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
