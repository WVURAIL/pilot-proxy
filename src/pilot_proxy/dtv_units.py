# coding=utf-8
"""DTV pilot-detector interpretation coordinates.

The public API keeps the raw F-statistic separate from the physical pilot excess.
The F-statistic level is the base-10 logarithmic power level of F. The one-bin
pilot-excess PNR applies the same logarithmic scale to F minus one.

The data-shelf SNR is referenced to the full 6 MHz ATSC channel allocation:
it is the average DTV data-shelf PSD over the allocation relative to the
non-DTV noise-floor PSD in the same band (``DTV_BANDWIDTH_HZ = 6 MHz``), not
an SNR in the occupied ~5.38 MHz symbol bandwidth.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

# Included reference receiver geometry: 400 MHz band split into 1024 coarse
# channels, then a 128-sample fine detector window inside one coarse channel.
REFERENCE_BANDWIDTH_HZ = 400.0e6
REFERENCE_NUM_CHANNELS = 1024
REFERENCE_CHANNEL_WIDTH_HZ = REFERENCE_BANDWIDTH_HZ / REFERENCE_NUM_CHANNELS

# ATSC 1.0 occupies a nominal 6 MHz broadcast channel.
DTV_BANDWIDTH_HZ = 6.0e6

DETECTOR_WINDOW_SAMPLES = 128
# Default equivalent-noise-bandwidth correction for the shipped fine-bin model.
ENBW = 1.0

# Power quantities use 10*log10 by definition.
DB_LINEAR_BASE = 10.0
DB_POWER_FACTOR = 10.0
RAW_FSTAT_REFERENCE_SCALE = 2.0
NO_PILOT_EXCESS_FSTAT = 1.0
HALF_THRESHOLD_DIVISOR = 2.0
DEFAULT_THRESHOLD_MAX_DENOMINATOR = 2**32

# ATSC A/53 pilot relationship: pilot power is about 11.3 dB below the
# average data-shelf power. Keep this as a positive offset by convention.
PILOT_BELOW_DATA_DB = 11.3
PILOT_CAPTURE_EFFICIENCY = 1.0

FINE_BIN_WIDTH_HZ = REFERENCE_CHANNEL_WIDTH_HZ / DETECTOR_WINDOW_SAMPLES
EFFECTIVE_BIN_BW_HZ = ENBW * FINE_BIN_WIDTH_HZ
N_SHELF_BINS = DTV_BANDWIDTH_HZ / EFFECTIVE_BIN_BW_HZ
SPREADING_LOSS_DB = DB_POWER_FACTOR * np.log10(N_SHELF_BINS)


def spreading_loss_db_from_bin_enbw_hz(
    bin_enbw_hz: float,
    *,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
) -> float:
    """Return ``10*log10(dtv_bandwidth_hz / bin_enbw_hz)``."""
    bin_width = float(bin_enbw_hz)
    bandwidth = float(dtv_bandwidth_hz)
    if bin_width <= 0.0 or bandwidth <= 0.0:
        raise ValueError("bin_enbw_hz and dtv_bandwidth_hz must be positive.")
    return float(DB_POWER_FACTOR * np.log10(bandwidth / bin_width))


def pilot_capture_efficiency_db(
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> float:
    """Return the capture-efficiency correction in dB for the target estimator."""
    eta = float(pilot_capture_efficiency)
    if eta <= 0.0 or not np.isfinite(eta):
        raise ValueError("pilot_capture_efficiency must be positive and finite.")
    return float(DB_POWER_FACTOR * np.log10(eta))


PNR_BIN_TO_SNR_SHELF_OFFSET_DB = (
    PILOT_BELOW_DATA_DB
    - SPREADING_LOSS_DB
    - pilot_capture_efficiency_db(PILOT_CAPTURE_EFFICIENCY)
)


def pnr_bin_db_to_snr_shelf_db(
    pnr_bin_db,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
):
    """Convert one-bin pilot-excess PNR [dB] to DTV data-shelf SNR [dB]."""
    spreading = (
        spreading_loss_db_from_bin_enbw_hz(
            float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
        if bin_enbw_hz is not None
        else float(spreading_loss_db)
    )
    return (
        np.asarray(pnr_bin_db)
        + float(pilot_below_data_db)
        - spreading
        - pilot_capture_efficiency_db(pilot_capture_efficiency)
    )


def snr_shelf_db_to_pnr_bin_db(
    snr_shelf_db,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
):
    """Convert DTV data-shelf SNR [dB] to one-bin pilot-excess PNR [dB]."""
    spreading = (
        spreading_loss_db_from_bin_enbw_hz(
            float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
        if bin_enbw_hz is not None
        else float(spreading_loss_db)
    )
    return (
        np.asarray(snr_shelf_db)
        - float(pilot_below_data_db)
        + spreading
        + pilot_capture_efficiency_db(pilot_capture_efficiency)
    )


def fstat_raw_to_fstat_level_db(fstat_raw):
    """Convert raw F values to the F-statistic level in dB."""
    fstat = np.asarray(fstat_raw, dtype=np.float64)
    out = np.full(fstat.shape, np.nan, dtype=np.float64)
    valid = fstat > 0.0
    out[valid] = DB_POWER_FACTOR * np.log10(fstat[valid])
    if np.isscalar(fstat_raw):
        return float(np.asarray(out).reshape(()))
    return out


def fstat_raw_to_pilot_excess_linear(fstat_raw):
    """Convert raw F values to the linear pilot-excess ratio."""
    out = np.asarray(fstat_raw, dtype=np.float64) - NO_PILOT_EXCESS_FSTAT
    if np.isscalar(fstat_raw):
        return float(np.asarray(out).reshape(()))
    return out


def fstat_raw_to_pnr_bin_db(fstat_raw):
    """Convert raw F values to the one-bin pilot-excess PNR in dB."""
    fstat = np.asarray(fstat_raw, dtype=np.float64)
    out = np.full(fstat.shape, np.nan, dtype=np.float64)
    valid = fstat > NO_PILOT_EXCESS_FSTAT
    out[valid] = DB_POWER_FACTOR * np.log10(
        fstat[valid] - NO_PILOT_EXCESS_FSTAT
    )
    if np.isscalar(fstat_raw):
        return float(np.asarray(out).reshape(()))
    return out


def fstat_num_den_to_raw(num, den):
    """Convert deployed kernel NumDen outputs to raw F = 2*num/den."""
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape))
    np.divide(
        RAW_FSTAT_REFERENCE_SCALE * numerator,
        denominator,
        out=out,
        where=denominator != 0.0,
    )
    if np.isscalar(num) and np.isscalar(den):
        return float(np.asarray(out).reshape(()))
    return out


def fstat_num_den_to_fstat_level_db(num, den):
    """Convert deployed NumDen outputs to F-statistic level [dB]."""
    return fstat_raw_to_fstat_level_db(fstat_num_den_to_raw(num, den))


def fstat_num_den_to_pnr_bin_db(num, den):
    """Convert deployed NumDen outputs to one-bin pilot-excess PNR [dB]."""
    return fstat_raw_to_pnr_bin_db(fstat_num_den_to_raw(num, den))


def fstat_num_den_to_pilot_excess_linear(num, den):
    """Convert deployed NumDen outputs to the linear pilot excess.

    A zero denominator is an invalid reference floor and maps to 0.0 to match
    the C helper and deployed mask policy.
    """
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    raw = fstat_num_den_to_raw(numerator, denominator)
    out = np.asarray(raw, dtype=np.float64) - NO_PILOT_EXCESS_FSTAT
    out = np.where(denominator != 0.0, out, 0.0)
    if np.isscalar(num) and np.isscalar(den):
        return float(np.asarray(out).reshape(()))
    return out


def pilot_to_data_power_ratio(
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
) -> float:
    """Return pilot-power/data-shelf-power ratio from the ATSC pilot offset."""
    return float(DB_LINEAR_BASE ** (-float(pilot_below_data_db) / DB_POWER_FACTOR))


def composite_to_data_shelf_snr_correction_db(
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
) -> float:
    """Return the correction from composite clean-ATSC SNR to data-shelf SNR.

    The generated clean ATSC waveform contains a data shelf, sync content, and
    the pilot. The correction subtracts the pilot total-power contribution from
    the composite ATSC SNR.
    """
    ratio = pilot_to_data_power_ratio(pilot_below_data_db=pilot_below_data_db)
    return float(-DB_POWER_FACTOR * np.log10(NO_PILOT_EXCESS_FSTAT + ratio))


def pnr_bin_db_to_fstat_raw_threshold(pnr_bin_db_threshold):
    """Convert a one-bin pilot-excess PNR threshold [dB] to a raw-F threshold."""
    pnr = np.asarray(pnr_bin_db_threshold, dtype=np.float64)
    out = NO_PILOT_EXCESS_FSTAT + DB_LINEAR_BASE ** (pnr / DB_POWER_FACTOR)
    if np.isscalar(pnr_bin_db_threshold):
        return float(np.asarray(out).reshape(()))
    return out


def snr_shelf_db_to_fstat_raw_threshold(
    snr_shelf_db: float,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> float:
    """Convert a DTV shelf-SNR threshold [dB] to a raw F-statistic threshold."""
    pnr_bin_db = snr_shelf_db_to_pnr_bin_db(
        snr_shelf_db,
        pilot_below_data_db=pilot_below_data_db,
        spreading_loss_db=spreading_loss_db,
        bin_enbw_hz=bin_enbw_hz,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
    )
    return float(
        NO_PILOT_EXCESS_FSTAT
        + DB_LINEAR_BASE ** (float(pnr_bin_db) / DB_POWER_FACTOR)
    )


def fstat_raw_threshold_to_half_threshold_rational(
    fstat_raw_threshold: float,
    *,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
) -> tuple[int, int]:
    """Convert a full raw-F threshold to a rational half-threshold for the kernel."""
    raw = float(fstat_raw_threshold)
    max_den = int(max_denominator)
    if raw < 0.0 or not np.isfinite(raw):
        raise ValueError("fstat_raw_threshold must be non-negative and finite.")
    if max_den <= 0:
        raise ValueError("max_denominator must be positive.")
    half = Fraction(raw / HALF_THRESHOLD_DIVISOR).limit_denominator(max_den)
    return int(half.numerator), int(half.denominator)


def snr_shelf_db_to_half_threshold_rational(
    snr_shelf_db: float,
    *,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> tuple[int, int]:
    """Convert DTV shelf-SNR threshold to kernel rational half-threshold."""
    fstat_raw = snr_shelf_db_to_fstat_raw_threshold(
        snr_shelf_db,
        pilot_below_data_db=pilot_below_data_db,
        spreading_loss_db=spreading_loss_db,
        bin_enbw_hz=bin_enbw_hz,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
    )
    return fstat_raw_threshold_to_half_threshold_rational(
        fstat_raw,
        max_denominator=max_denominator,
    )


def snr_shelf_threshold_fields(
    snr_shelf_db: float,
    *,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> dict[str, float | int]:
    """Return all public and kernel threshold coordinates for shelf SNR."""
    pnr_bin_db = float(
        snr_shelf_db_to_pnr_bin_db(
            snr_shelf_db,
            pilot_below_data_db=pilot_below_data_db,
            spreading_loss_db=spreading_loss_db,
            bin_enbw_hz=bin_enbw_hz,
            dtv_bandwidth_hz=dtv_bandwidth_hz,
            pilot_capture_efficiency=pilot_capture_efficiency,
        )
    )
    fstat_raw = float(
        NO_PILOT_EXCESS_FSTAT + DB_LINEAR_BASE ** (pnr_bin_db / DB_POWER_FACTOR)
    )
    half_num, half_den = fstat_raw_threshold_to_half_threshold_rational(
        fstat_raw,
        max_denominator=max_denominator,
    )
    spreading = (
        spreading_loss_db_from_bin_enbw_hz(
            float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
        if bin_enbw_hz is not None
        else float(spreading_loss_db)
    )
    return {
        "threshold_snr_shelf_db": float(snr_shelf_db),
        "threshold_pnr_bin_db": pnr_bin_db,
        "threshold_pilot_excess_linear": float(
            DB_LINEAR_BASE ** (pnr_bin_db / DB_POWER_FACTOR)
        ),
        "threshold_fstat_raw": fstat_raw,
        "threshold_half_num": int(half_num),
        "threshold_half_den": int(half_den),
        "threshold_half_float": float(half_num / half_den),
        "max_denominator": int(max_denominator),
        "pilot_below_data_db": float(pilot_below_data_db),
        "spreading_loss_db": float(spreading),
        "bin_enbw_hz": float(EFFECTIVE_BIN_BW_HZ if bin_enbw_hz is None else bin_enbw_hz),
        "dtv_bandwidth_hz": float(dtv_bandwidth_hz),
        "pilot_capture_efficiency": float(pilot_capture_efficiency),
    }


def pnr_bin_to_snr_shelf_metadata() -> dict[str, float]:
    """Return one-bin pilot-excess PNR to DTV shelf-SNR constants."""
    return {
        "reference_bandwidth_hz": float(REFERENCE_BANDWIDTH_HZ),
        "reference_num_channels": int(REFERENCE_NUM_CHANNELS),
        "channel_width_hz": float(REFERENCE_CHANNEL_WIDTH_HZ),
        "detector_window_samples": int(DETECTOR_WINDOW_SAMPLES),
        "fine_bin_width_hz": float(FINE_BIN_WIDTH_HZ),
        "detector_bin_enbw": float(ENBW),
        "bin_enbw_hz": float(EFFECTIVE_BIN_BW_HZ),
        "dtv_bandwidth_hz": float(DTV_BANDWIDTH_HZ),
        "n_shelf_bins": float(N_SHELF_BINS),
        "spreading_loss_db": float(SPREADING_LOSS_DB),
        "pilot_below_data_db": float(PILOT_BELOW_DATA_DB),
        "pilot_to_data_power_db": float(-PILOT_BELOW_DATA_DB),
        "pilot_capture_efficiency": float(PILOT_CAPTURE_EFFICIENCY),
        "pilot_capture_efficiency_db": float(
            pilot_capture_efficiency_db(PILOT_CAPTURE_EFFICIENCY)
        ),
        "pilot_to_data_power_ratio": float(pilot_to_data_power_ratio()),
        "composite_to_data_shelf_snr_correction_db": float(
            composite_to_data_shelf_snr_correction_db()
        ),
        "pnr_bin_to_snr_shelf_offset_db": float(PNR_BIN_TO_SNR_SHELF_OFFSET_DB),
    }


def coordinate_convention_metadata() -> dict[str, str]:
    """Return the detector/display coordinate convention used in real-data figures."""
    return {
        "raw_detector_quantity": "fstat_raw",
        "level_coordinate": "fstat_level_db",
        "pilot_excess_coordinate": "pnr_bin_db",
        "derived_coordinate": "snr_shelf_db",
        "detector_mask_rule": "mask = fstat_raw >= fstat_threshold_raw",
    }


def threshold_coordinate_fields(
    pnr_bin_db,
    *,
    fstat_threshold_raw: float | None = None,
) -> dict[str, Any]:
    """Return canonical threshold coordinate fields for one PNR-bin value."""
    pnr_bin = float(pnr_bin_db)
    shelf = pnr_bin_db_to_snr_shelf_db(pnr_bin)
    if fstat_threshold_raw is None:
        raw = pnr_bin_db_to_fstat_raw_threshold(pnr_bin)
    else:
        raw = float(fstat_threshold_raw)
    return {
        "threshold_pnr_bin_db": pnr_bin,
        "threshold_snr_shelf_db": float(np.asarray(shelf).reshape(())),
        "fstat_threshold_raw": raw,
        "pilot_below_data_db": float(PILOT_BELOW_DATA_DB),
        "spreading_loss_db": float(SPREADING_LOSS_DB),
        "pilot_capture_efficiency": float(PILOT_CAPTURE_EFFICIENCY),
        "pnr_bin_to_snr_shelf_offset_db": float(PNR_BIN_TO_SNR_SHELF_OFFSET_DB),
    }


def add_snr_shelf_secondary_axis(ax):
    """Add a top x-axis mapping one-bin pilot-excess PNR to shelf SNR."""
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            pnr_bin_db_to_snr_shelf_db,
            snr_shelf_db_to_pnr_bin_db,
        ),
    )
    secax.set_xlabel("DTV data-shelf SNR [dB]")
    return secax
