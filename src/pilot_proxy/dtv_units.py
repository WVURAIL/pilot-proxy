# coding=utf-8
"""Detector power-ratio and DTV pilot-excess coordinates.

The deployed coarse detector stores

``R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)``.

Exact squared norms of the packed target and reference weights set the
flat-noise null ratio

``R_null = 2 * target_norm_sq / reference_norm_sum_sq``.

Scientific pilot excess is therefore ``R_coarse / R_null - 1``.  The raw
quantity ``R_coarse - 1`` is retained only as an explicitly named diagnostic.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

REFERENCE_BANDWIDTH_HZ = 400.0e6
REFERENCE_NUM_CHANNELS = 1024
REFERENCE_CHANNEL_WIDTH_HZ = REFERENCE_BANDWIDTH_HZ / REFERENCE_NUM_CHANNELS
DTV_BANDWIDTH_HZ = 6.0e6
DETECTOR_WINDOW_SAMPLES = 128
ENBW = 1.0

DB_LINEAR_BASE = 10.0
DB_POWER_FACTOR = 10.0
COARSE_POWER_RATIO_SCALE = 2.0
UNIT_NORMALIZED_POWER_RATIO = 1.0
UNIT_DATA_SHELF_POWER = 1.0
DEFAULT_THRESHOLD_MAX_DENOMINATOR = 2**32

PILOT_BELOW_DATA_DB = 11.3
PILOT_CAPTURE_EFFICIENCY = 1.0

FINE_BIN_WIDTH_HZ = REFERENCE_CHANNEL_WIDTH_HZ / DETECTOR_WINDOW_SAMPLES
EFFECTIVE_BIN_BW_HZ = ENBW * FINE_BIN_WIDTH_HZ
N_SHELF_BINS = DTV_BANDWIDTH_HZ / EFFECTIVE_BIN_BW_HZ
SPREADING_LOSS_DB = DB_POWER_FACTOR * np.log10(N_SHELF_BINS)


def _scalar_if_scalar(value, *inputs):
    if all(np.isscalar(item) for item in inputs):
        return float(np.asarray(value).reshape(()))
    return value


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
    """Return the target-estimator capture fraction in dB."""
    fraction = float(pilot_capture_efficiency)
    if fraction <= 0.0 or not np.isfinite(fraction):
        raise ValueError("pilot_capture_efficiency must be positive and finite.")
    return float(DB_POWER_FACTOR * np.log10(fraction))


PILOT_EXCESS_TO_DATA_SHELF_SNR_OFFSET_DB = (
    PILOT_BELOW_DATA_DB
    - SPREADING_LOSS_DB
    - pilot_capture_efficiency_db(PILOT_CAPTURE_EFFICIENCY)
)


def pilot_excess_db_to_data_shelf_snr_db(
    pilot_excess_db,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
):
    """Convert one-bin normalized pilot excess [dB] to data-shelf SNR [dB]."""
    spreading = (
        spreading_loss_db_from_bin_enbw_hz(
            float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
        if bin_enbw_hz is not None
        else float(spreading_loss_db)
    )
    return (
        np.asarray(pilot_excess_db)
        + float(pilot_below_data_db)
        - spreading
        - pilot_capture_efficiency_db(pilot_capture_efficiency)
    )


def data_shelf_snr_db_to_pilot_excess_db(
    data_shelf_snr_db,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
):
    """Convert data-shelf SNR [dB] to one-bin normalized pilot excess [dB]."""
    spreading = (
        spreading_loss_db_from_bin_enbw_hz(
            float(bin_enbw_hz),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
        if bin_enbw_hz is not None
        else float(spreading_loss_db)
    )
    return (
        np.asarray(data_shelf_snr_db)
        - float(pilot_below_data_db)
        + spreading
        + pilot_capture_efficiency_db(pilot_capture_efficiency)
    )


def coarse_power_ratio_to_db(coarse_power_ratio):
    """Convert positive raw coarse power ratios to dB."""
    ratio = np.asarray(coarse_power_ratio, dtype=np.float64)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = ratio > 0.0
    out[valid] = DB_POWER_FACTOR * np.log10(ratio[valid])
    return _scalar_if_scalar(out, coarse_power_ratio)


def normalize_coarse_power_ratio(coarse_power_ratio, null_power_ratio):
    """Return ``coarse_power_ratio / null_power_ratio``."""
    ratio = np.asarray(coarse_power_ratio, dtype=np.float64)
    null = np.asarray(null_power_ratio, dtype=np.float64)
    if np.any(~np.isfinite(null)) or np.any(null <= 0.0):
        raise ValueError("null_power_ratio must be positive and finite.")
    out = np.asarray(ratio / null, dtype=np.float64)
    return _scalar_if_scalar(out, coarse_power_ratio, null_power_ratio)


def coarse_power_ratio_to_normalized_pilot_excess(
    coarse_power_ratio,
    null_power_ratio,
):
    """Return ``coarse_power_ratio/null_power_ratio - 1``."""
    return normalized_coarse_power_ratio_to_pilot_excess(
        normalize_coarse_power_ratio(coarse_power_ratio, null_power_ratio)
    )


def normalized_coarse_power_ratio_to_db(normalized_coarse_power_ratio):
    """Convert positive norm-corrected coarse power ratios to dB."""
    ratio = np.asarray(normalized_coarse_power_ratio, dtype=np.float64)
    out = np.full(ratio.shape, np.nan, dtype=np.float64)
    valid = ratio > 0.0
    out[valid] = DB_POWER_FACTOR * np.log10(ratio[valid])
    return _scalar_if_scalar(out, normalized_coarse_power_ratio)


def normalized_coarse_power_ratio_to_pilot_excess(
    normalized_coarse_power_ratio,
):
    """Return ``normalized_coarse_power_ratio - 1``."""
    out = (
        np.asarray(normalized_coarse_power_ratio, dtype=np.float64)
        - UNIT_NORMALIZED_POWER_RATIO
    )
    return _scalar_if_scalar(out, normalized_coarse_power_ratio)


def normalized_pilot_excess_to_db(normalized_pilot_excess):
    """Convert positive normalized pilot excess to dB."""
    excess = np.asarray(normalized_pilot_excess, dtype=np.float64)
    out = np.full(excess.shape, np.nan, dtype=np.float64)
    valid = excess > 0.0
    out[valid] = DB_POWER_FACTOR * np.log10(excess[valid])
    return _scalar_if_scalar(out, normalized_pilot_excess)


def coarse_power_ratio_to_raw_pilot_excess(coarse_power_ratio):
    """Return the diagnostic quantity ``R_coarse - 1``."""
    out = np.asarray(coarse_power_ratio, dtype=np.float64) - 1.0
    return _scalar_if_scalar(out, coarse_power_ratio)


def coarse_power_ratio_to_raw_pilot_excess_db(coarse_power_ratio):
    """Return ``10*log10(R_coarse - 1)`` without null normalization."""
    return normalized_pilot_excess_to_db(
        coarse_power_ratio_to_raw_pilot_excess(coarse_power_ratio)
    )


def power_terms_to_coarse_power_ratio(num, den):
    """Return the raw ratio ``2*num/den``; zero reference power maps to NaN."""
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    out = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        COARSE_POWER_RATIO_SCALE * numerator,
        denominator,
        out=out,
        where=denominator > 0.0,
    )
    return _scalar_if_scalar(out, num, den)


def power_terms_to_normalized_coarse_power_ratio(
    num,
    den,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
):
    """Return ``num*reference_norm_sum_sq/(den*target_norm_sq)``."""
    target_norm = int(target_norm_sq)
    reference_norm = int(reference_norm_sum_sq)
    if target_norm <= 0:
        raise ValueError("target_norm_sq must be positive.")
    if reference_norm <= 0:
        raise ValueError("reference_norm_sum_sq must be positive.")
    numerator = np.asarray(num, dtype=np.float64)
    denominator = np.asarray(den, dtype=np.float64)
    out = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape),
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        numerator * float(reference_norm),
        denominator * float(target_norm),
        out=out,
        where=denominator > 0.0,
    )
    return _scalar_if_scalar(out, num, den)


def power_terms_to_normalized_coarse_power_ratio_db(
    num,
    den,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
):
    """Return ``10*log10(Q_coarse)`` from exact power and norm coordinates."""
    return normalized_coarse_power_ratio_to_db(
        power_terms_to_normalized_coarse_power_ratio(
            num,
            den,
            target_norm_sq=target_norm_sq,
            reference_norm_sum_sq=reference_norm_sum_sq,
        )
    )


def power_terms_to_normalized_pilot_excess(
    num,
    den,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
):
    """Return ``num*reference_norm_sum_sq/(den*target_norm_sq) - 1``."""
    return normalized_coarse_power_ratio_to_pilot_excess(
        power_terms_to_normalized_coarse_power_ratio(
            num,
            den,
            target_norm_sq=target_norm_sq,
            reference_norm_sum_sq=reference_norm_sum_sq,
        )
    )


def power_terms_to_pilot_excess_db(
    num,
    den,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
):
    """Return normalized one-bin pilot excess in dB."""
    return normalized_pilot_excess_to_db(
        power_terms_to_normalized_pilot_excess(
            num,
            den,
            target_norm_sq=target_norm_sq,
            reference_norm_sum_sq=reference_norm_sum_sq,
        )
    )


def power_terms_to_raw_pilot_excess(num, den):
    """Return the diagnostic quantity ``2*num/den - 1``."""
    return coarse_power_ratio_to_raw_pilot_excess(
        power_terms_to_coarse_power_ratio(num, den)
    )


def power_terms_to_raw_pilot_excess_db(num, den):
    """Return diagnostic raw pilot excess in dB without norm correction."""
    return coarse_power_ratio_to_raw_pilot_excess_db(
        power_terms_to_coarse_power_ratio(num, den)
    )


def pilot_to_data_power_ratio(
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
) -> float:
    """Return pilot-power/data-shelf-power ratio."""
    return float(
        DB_LINEAR_BASE
        ** (-float(pilot_below_data_db) / DB_POWER_FACTOR)
    )


def composite_to_data_shelf_snr_correction_db(
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
) -> float:
    """Return the correction from composite clean-ATSC SNR to data-shelf SNR."""
    ratio = pilot_to_data_power_ratio(
        pilot_below_data_db=pilot_below_data_db
    )
    return float(
        -DB_POWER_FACTOR * np.log10(UNIT_DATA_SHELF_POWER + ratio)
    )


def pilot_excess_db_to_normalized_power_ratio_threshold(
    pilot_excess_db_threshold,
):
    """Convert pilot-excess dB to ``Q_threshold = 1 + rho_threshold``."""
    value = np.asarray(pilot_excess_db_threshold, dtype=np.float64)
    out = UNIT_NORMALIZED_POWER_RATIO + DB_LINEAR_BASE ** (
        value / DB_POWER_FACTOR
    )
    return _scalar_if_scalar(out, pilot_excess_db_threshold)


def data_shelf_snr_db_to_normalized_power_ratio_threshold(
    data_shelf_snr_db: float,
    *,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> float:
    """Convert data-shelf SNR to a normalized coarse-ratio threshold."""
    pilot_excess_db = data_shelf_snr_db_to_pilot_excess_db(
        data_shelf_snr_db,
        pilot_below_data_db=pilot_below_data_db,
        spreading_loss_db=spreading_loss_db,
        bin_enbw_hz=bin_enbw_hz,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
    )
    return float(
        pilot_excess_db_to_normalized_power_ratio_threshold(pilot_excess_db)
    )


def normalized_power_ratio_threshold_to_half_rational(
    normalized_power_ratio_threshold: float,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
) -> tuple[int, int]:
    """Convert ``Q`` threshold to the kernel's ``P_target/P_ref_sum`` ratio."""
    threshold = float(normalized_power_ratio_threshold)
    target_norm = int(target_norm_sq)
    reference_norm = int(reference_norm_sum_sq)
    max_den = int(max_denominator)
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError(
            "normalized_power_ratio_threshold must be non-negative and finite."
        )
    if target_norm <= 0:
        raise ValueError("target_norm_sq must be positive.")
    if reference_norm <= 0:
        raise ValueError("reference_norm_sum_sq must be positive.")
    if max_den <= 0:
        raise ValueError("max_denominator must be positive.")
    half = Fraction(
        threshold * float(target_norm) / float(reference_norm)
    ).limit_denominator(max_den)
    return int(half.numerator), int(half.denominator)


def data_shelf_snr_db_to_half_threshold_rational(
    data_shelf_snr_db: float,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> tuple[int, int]:
    """Convert data-shelf SNR to the exact packed-weight kernel threshold."""
    normalized_threshold = data_shelf_snr_db_to_normalized_power_ratio_threshold(
        data_shelf_snr_db,
        pilot_below_data_db=pilot_below_data_db,
        spreading_loss_db=spreading_loss_db,
        bin_enbw_hz=bin_enbw_hz,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
    )
    return normalized_power_ratio_threshold_to_half_rational(
        normalized_threshold,
        target_norm_sq=target_norm_sq,
        reference_norm_sum_sq=reference_norm_sum_sq,
        max_denominator=max_denominator,
    )


def data_shelf_snr_threshold_fields(
    data_shelf_snr_db: float,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
    max_denominator: int = DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    spreading_loss_db: float = SPREADING_LOSS_DB,
    bin_enbw_hz: float | None = None,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
) -> dict[str, float | int]:
    """Return public and kernel thresholds in the same norm-corrected coordinate."""
    pilot_excess_db = float(
        data_shelf_snr_db_to_pilot_excess_db(
            data_shelf_snr_db,
            pilot_below_data_db=pilot_below_data_db,
            spreading_loss_db=spreading_loss_db,
            bin_enbw_hz=bin_enbw_hz,
            dtv_bandwidth_hz=dtv_bandwidth_hz,
            pilot_capture_efficiency=pilot_capture_efficiency,
        )
    )
    normalized_pilot_excess = float(
        DB_LINEAR_BASE ** (pilot_excess_db / DB_POWER_FACTOR)
    )
    normalized_threshold = float(
        UNIT_NORMALIZED_POWER_RATIO + normalized_pilot_excess
    )
    null_power_ratio = float(
        COARSE_POWER_RATIO_SCALE
        * float(target_norm_sq)
        / float(reference_norm_sum_sq)
    )
    coarse_threshold = float(null_power_ratio * normalized_threshold)
    half_num, half_den = normalized_power_ratio_threshold_to_half_rational(
        normalized_threshold,
        target_norm_sq=target_norm_sq,
        reference_norm_sum_sq=reference_norm_sum_sq,
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
        "threshold_data_shelf_snr_db": float(data_shelf_snr_db),
        "threshold_pilot_excess_db": pilot_excess_db,
        "threshold_normalized_pilot_excess": normalized_pilot_excess,
        "threshold_normalized_power_ratio": normalized_threshold,
        "threshold_coarse_power_ratio": coarse_threshold,
        "threshold_half_num": int(half_num),
        "threshold_half_den": int(half_den),
        "threshold_half_float": float(half_num / half_den),
        "target_norm_sq": int(target_norm_sq),
        "reference_norm_sum_sq": int(reference_norm_sum_sq),
        "null_power_ratio": null_power_ratio,
        "max_denominator": int(max_denominator),
        "pilot_below_data_db": float(
            pilot_below_data_db
        ),
        "spreading_loss_db": float(spreading),
        "bin_enbw_hz": float(
            EFFECTIVE_BIN_BW_HZ if bin_enbw_hz is None else bin_enbw_hz
        ),
        "dtv_bandwidth_hz": float(dtv_bandwidth_hz),
        "pilot_capture_efficiency": float(pilot_capture_efficiency),
    }


def pilot_excess_to_data_shelf_metadata() -> dict[str, float]:
    """Return normalized pilot-excess to data-shelf SNR constants."""
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
        "pilot_below_data_db": float(
            PILOT_BELOW_DATA_DB
        ),
        "pilot_to_data_power_db": float(-PILOT_BELOW_DATA_DB),
        "pilot_capture_efficiency": float(PILOT_CAPTURE_EFFICIENCY),
        "pilot_capture_efficiency_db": float(
            pilot_capture_efficiency_db(PILOT_CAPTURE_EFFICIENCY)
        ),
        "pilot_to_data_power_ratio": float(pilot_to_data_power_ratio()),
        "composite_to_data_shelf_snr_correction_db": float(
            composite_to_data_shelf_snr_correction_db()
        ),
        "pilot_excess_to_data_shelf_snr_offset_db": float(
            PILOT_EXCESS_TO_DATA_SHELF_SNR_OFFSET_DB
        ),
    }


def coordinate_convention_metadata() -> dict[str, str]:
    """Return the current detector and display coordinate convention."""
    return {
        "raw_detector_quantity": "coarse_power_ratio",
        "null_quantity": "null_power_ratio",
        "normalized_level_coordinate": "normalized_coarse_power_ratio_db",
        "normalized_pilot_excess_coordinate": "normalized_pilot_excess",
        "pilot_excess_db_coordinate": "pilot_excess_db",
        "derived_coordinate": "estimated_data_shelf_snr_db",
        "detector_mask_rule": (
            "p_target*reference_norm_sum_sq > "
            "p_ref_sum*target_norm_sq"
        ),
    }


def add_data_shelf_snr_secondary_axis(ax):
    """Add a top axis mapping pilot-excess dB to data-shelf SNR."""
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            pilot_excess_db_to_data_shelf_snr_db,
            data_shelf_snr_db_to_pilot_excess_db,
        ),
    )
    secax.set_xlabel("DTV data-shelf SNR [dB]")
    return secax
