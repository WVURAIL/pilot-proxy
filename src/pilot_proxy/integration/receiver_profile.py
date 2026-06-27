# coding=utf-8
"""Receiver/channelizer profile contract for external integrations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pilot_proxy.detector_geometry import (
    SPECTRAL_SENSE_NORMAL,
    normalize_spectral_sense,
    spectral_sense_requires_time_reversal,
)
from pilot_proxy.dtv_units import (
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
)
from pilot_proxy.reference_channelizer import (
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_BANDWIDTH_HZ,
    REFERENCE_NUM_CHANNELS,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
)

from .schemas import RECEIVER_PROFILE_SCHEMA_VERSION

DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_NUM_INPUT_STREAMS = 1
DEFAULT_DETECTOR_WINDOW_SAMPLES = 128
DEFAULT_BITS_PER_COMPONENT = 4
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_CHANNELIZER_TYPE = "pfb"
DEFAULT_CHANNELIZER_RESPONSE = "sinc_hamming"
DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT = "complex64"
DEFAULT_ADAPTER_OUTPUT_FORMAT = "complex_int4_packed_int8"
DEFAULT_QUANTIZATION_SCALE_MODE = "global"
REFERENCE_PROFILE_NAME = "reference_800mhz_pfb"
DEFAULT_COARSE_CHANNEL_CENTER_OFFSET_MULTIPLIER = 1.0
PROFILE_HASH_HEX_CHARS = 64
FREQUENCY_ORDER_ASCENDING_RF = "ascending_rf"
FREQUENCY_ORDER_DESCENDING_RF = "descending_rf"
SUPPORTED_FREQUENCY_ORDER = frozenset(
    {FREQUENCY_ORDER_ASCENDING_RF, FREQUENCY_ORDER_DESCENDING_RF}
)


def _is_nested_receiver_profile(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("rf_band", "framing", "input_streams"))


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"receiver profile requires object field {key!r}.")
    return dict(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"receiver profile requires integer field {field_name!r}.")
    return int(value)


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"receiver profile requires numeric field {field_name!r}.")
    return float(value)


@dataclass(frozen=True)
class ChannelizerProfile:
    """Neutral channelizer description embedded in a receiver profile."""

    type: str = DEFAULT_CHANNELIZER_TYPE
    fft_size: int = REFERENCE_PFB_FFT_SIZE
    taps: int = REFERENCE_PFB_TAPS
    response: str = DEFAULT_CHANNELIZER_RESPONSE

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChannelizerProfile":
        data = _mapping_or_empty(data)
        return cls(
            type=str(data.get("type", DEFAULT_CHANNELIZER_TYPE)),
            fft_size=_as_int(
                data.get(
                    "fft_size",
                    data.get("pfb_fft_size", REFERENCE_PFB_FFT_SIZE),
                ),
                "channelizer.fft_size",
            ),
            taps=_as_int(data.get("taps", REFERENCE_PFB_TAPS), "channelizer.taps"),
            response=str(data.get("response", DEFAULT_CHANNELIZER_RESPONSE)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "fft_size": int(self.fft_size),
            "taps": int(self.taps),
            "response": str(self.response),
        }


@dataclass(frozen=True)
class ReceiverProfile:
    """Neutral receiver profile used to map RF frequencies to detector streams."""

    schema_version: str
    name: str
    sample_rate_hz: float
    band_lower_hz: float
    band_upper_hz: float
    num_coarse_channels: int
    frame_size_samples: int
    num_input_streams: int
    spectral_sense: str = SPECTRAL_SENSE_NORMAL
    frequency_order: str = FREQUENCY_ORDER_ASCENDING_RF
    channelizer: ChannelizerProfile = field(default_factory=ChannelizerProfile)
    bin_enbw_hz: float = EFFECTIVE_BIN_BW_HZ
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY
    coarse_channel_center_offset_hz: float | None = None
    instrument_name: str | None = None
    profile_status: str | None = None
    detector_window_samples: int = DEFAULT_DETECTOR_WINDOW_SAMPLES
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB
    native_channelized_sample_format: str = DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT
    adapter_output_format: str = DEFAULT_ADAPTER_OUTPUT_FORMAT
    bits_per_component: int = DEFAULT_BITS_PER_COMPONENT
    quantization_scale_mode_default: str = DEFAULT_QUANTIZATION_SCALE_MODE
    clip_sigma_default: float = DEFAULT_CLIP_SIGMA
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RECEIVER_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported receiver profile schema_version: "
                f"{self.schema_version!r}"
            )
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.band_upper_hz <= self.band_lower_hz:
            raise ValueError("band_upper_hz must be greater than band_lower_hz.")
        if self.num_coarse_channels <= 0:
            raise ValueError("num_coarse_channels must be positive.")
        if self.frame_size_samples <= 0:
            raise ValueError("frame_size_samples must be positive.")
        if self.num_input_streams <= 0:
            raise ValueError("num_input_streams must be positive.")
        if self.bin_enbw_hz <= 0.0:
            raise ValueError("bin_enbw_hz must be positive.")
        if self.pilot_capture_efficiency <= 0.0:
            raise ValueError("pilot_capture_efficiency must be positive.")
        object.__setattr__(
            self,
            "spectral_sense",
            normalize_spectral_sense(self.spectral_sense),
        )
        frequency_order = str(self.frequency_order).strip().lower()
        if frequency_order not in SUPPORTED_FREQUENCY_ORDER:
            raise ValueError(
                "frequency_order must be one of "
                f"{sorted(SUPPORTED_FREQUENCY_ORDER)}; got "
                f"{self.frequency_order!r}."
            )
        object.__setattr__(self, "frequency_order", frequency_order)
        if self.detector_window_samples <= 0:
            raise ValueError("detector_window_samples must be positive.")
        if self.dtv_bandwidth_hz <= 0.0:
            raise ValueError("dtv_bandwidth_hz must be positive.")
        if self.bits_per_component <= 0:
            raise ValueError("bits_per_component must be positive.")
        if self.clip_sigma_default <= 0.0:
            raise ValueError("clip_sigma_default must be positive.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def bandwidth_hz(self) -> float:
        return float(self.band_upper_hz - self.band_lower_hz)

    @property
    def coarse_channel_width_hz(self) -> float:
        return float(self.bandwidth_hz / int(self.num_coarse_channels))

    @property
    def center_offset_hz(self) -> float:
        if self.coarse_channel_center_offset_hz is not None:
            return float(self.coarse_channel_center_offset_hz)
        return float(
            DEFAULT_COARSE_CHANNEL_CENTER_OFFSET_MULTIPLIER
            * self.coarse_channel_width_hz
        )

    def coarse_channel_center_hz(self, index: int) -> float:
        idx = int(index)
        if idx < 0 or idx >= int(self.num_coarse_channels):
            raise ValueError(
                "coarse channel index out of range: "
                f"{idx}, valid 0-{int(self.num_coarse_channels) - 1}"
            )
        if self.frequency_order == FREQUENCY_ORDER_DESCENDING_RF:
            return float(
                self.band_upper_hz
                - self.center_offset_hz
                - idx * self.coarse_channel_width_hz
            )
        return float(
            self.band_lower_hz
            + self.center_offset_hz
            + idx * self.coarse_channel_width_hz
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReceiverProfile":
        raw = dict(data)
        if _is_nested_receiver_profile(raw):
            return cls._from_nested_dict(raw)
        band_lower_hz = float(raw["band_lower_hz"])
        if "band_upper_hz" in raw:
            band_upper_hz = float(raw["band_upper_hz"])
        else:
            band_upper_hz = band_lower_hz + float(raw["bandwidth_hz"])
        profile = cls(
            schema_version=str(
                raw.get("schema_version", RECEIVER_PROFILE_SCHEMA_VERSION)
            ),
            name=str(raw["name"]),
            sample_rate_hz=float(raw["sample_rate_hz"]),
            band_lower_hz=band_lower_hz,
            band_upper_hz=band_upper_hz,
            num_coarse_channels=int(raw["num_coarse_channels"]),
            frame_size_samples=int(raw["frame_size_samples"]),
            num_input_streams=int(raw.get("num_input_streams", 1)),
            spectral_sense=str(raw.get("spectral_sense", SPECTRAL_SENSE_NORMAL)),
            frequency_order=str(
                raw.get("frequency_order", FREQUENCY_ORDER_ASCENDING_RF)
            ),
            channelizer=ChannelizerProfile.from_dict(
                _mapping_or_none(raw.get("channelizer"))
            ),
            bin_enbw_hz=_as_float(
                raw.get("bin_enbw_hz", EFFECTIVE_BIN_BW_HZ),
                "bin_enbw_hz",
            ),
            pilot_capture_efficiency=_as_float(
                raw.get("pilot_capture_efficiency", PILOT_CAPTURE_EFFICIENCY),
                "pilot_capture_efficiency",
            ),
            coarse_channel_center_offset_hz=(
                None
                if raw.get("coarse_channel_center_offset_hz") is None
                else float(raw["coarse_channel_center_offset_hz"])
            ),
            instrument_name=_optional_str(raw.get("instrument_name")),
            profile_status=_optional_str(raw.get("profile_status")),
            detector_window_samples=int(
                raw.get("detector_window_samples", DEFAULT_DETECTOR_WINDOW_SAMPLES)
            ),
            dtv_bandwidth_hz=float(raw.get("dtv_bandwidth_hz", DTV_BANDWIDTH_HZ)),
            pilot_below_data_db=float(
                raw.get("pilot_below_data_db", PILOT_BELOW_DATA_DB)
            ),
            native_channelized_sample_format=str(
                raw.get(
                    "native_channelized_sample_format",
                    DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT,
                )
            ),
            adapter_output_format=str(
                raw.get("adapter_output_format", DEFAULT_ADAPTER_OUTPUT_FORMAT)
            ),
            bits_per_component=int(
                raw.get("bits_per_component", DEFAULT_BITS_PER_COMPONENT)
            ),
            quantization_scale_mode_default=str(
                raw.get(
                    "quantization_scale_mode_default",
                    DEFAULT_QUANTIZATION_SCALE_MODE,
                )
            ),
            clip_sigma_default=float(raw.get("clip_sigma_default", DEFAULT_CLIP_SIGMA)),
            metadata=_mapping_or_empty(raw.get("metadata")),
        )
        if "coarse_channel_width_hz" in raw and not math.isclose(
            float(raw["coarse_channel_width_hz"]),
            profile.coarse_channel_width_hz,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "coarse_channel_width_hz does not match band/num_coarse_channels: "
                f"{raw['coarse_channel_width_hz']} vs "
                f"{profile.coarse_channel_width_hz}"
            )
        return profile

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "receiver_profile_id": self.name,
            "sample_rate_hz": float(self.sample_rate_hz),
            "band_lower_hz": float(self.band_lower_hz),
            "band_upper_hz": float(self.band_upper_hz),
            "num_coarse_channels": int(self.num_coarse_channels),
            "coarse_channel_width_hz": float(self.coarse_channel_width_hz),
            "frame_size_samples": int(self.frame_size_samples),
            "num_input_streams": int(self.num_input_streams),
            "spectral_sense": self.spectral_sense,
            "frequency_order": self.frequency_order,
            "channelizer": self.channelizer.to_dict(),
            "bin_enbw_hz": float(self.bin_enbw_hz),
            "pilot_capture_efficiency": float(self.pilot_capture_efficiency),
            "coarse_channel_center_offset_hz": float(self.center_offset_hz),
            "detector_window_samples": int(self.detector_window_samples),
            "dtv_bandwidth_hz": float(self.dtv_bandwidth_hz),
            "pilot_below_data_db": float(self.pilot_below_data_db),
            "native_channelized_sample_format": self.native_channelized_sample_format,
            "adapter_output_format": self.adapter_output_format,
            "bits_per_component": int(self.bits_per_component),
            "quantization_scale_mode_default": self.quantization_scale_mode_default,
            "clip_sigma_default": float(self.clip_sigma_default),
        }
        if self.instrument_name is not None:
            out["instrument_name"] = self.instrument_name
        if self.profile_status is not None:
            out["profile_status"] = self.profile_status
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out

    @classmethod
    def _from_nested_dict(cls, data: dict[str, Any]) -> "ReceiverProfile":
        raw = dict(data)
        rf_band = _require_mapping(raw, "rf_band")
        channelizer = _require_mapping(raw, "channelizer")
        framing = _require_mapping(raw, "framing")
        input_streams = _require_mapping(raw, "input_streams")
        quantization = _mapping_or_empty(raw.get("quantization"))
        adapter = _mapping_or_empty(raw.get("detector_adapter"))
        frequency_axis = _mapping_or_empty(channelizer.get("frequency_axis"))
        digitizer = _mapping_or_empty(raw.get("digitizer"))

        band_lower_hz = float(rf_band["lower_hz"])
        if "upper_hz" in rf_band:
            band_upper_hz = float(rf_band["upper_hz"])
        else:
            band_upper_hz = band_lower_hz + float(rf_band["bandwidth_hz"])
        sample_rate_hz = _as_float(
            channelizer.get(
                "input_sample_rate_hz",
                raw.get("sample_rate_hz", digitizer.get("adc_sample_rate_hz")),
            ),
            "channelizer.input_sample_rate_hz",
        )
        spectral_sense = str(
            frequency_axis.get(
                "spectral_sense",
                channelizer.get("spectral_sense", SPECTRAL_SENSE_NORMAL),
            )
        )
        frequency_order = str(
            frequency_axis.get("order", channelizer.get("frequency_order", FREQUENCY_ORDER_ASCENDING_RF))
        )
        name = str(
            raw.get(
                "receiver_profile_id",
                raw.get("name", raw.get("instrument_name", REFERENCE_PROFILE_NAME)),
            )
        )
        nested_metadata = {
            "rf_band": {
                key: value
                for key, value in rf_band.items()
                if key not in {"lower_hz", "upper_hz", "bandwidth_hz"}
            },
            "channelizer_frequency_axis": frequency_axis,
        }
        nested_metadata = {
            key: value for key, value in nested_metadata.items() if value
        }
        nested_metadata.update(_mapping_or_empty(raw.get("metadata")))

        return cls(
            schema_version=str(
                raw.get("schema_version", RECEIVER_PROFILE_SCHEMA_VERSION)
            ),
            name=name,
            sample_rate_hz=sample_rate_hz,
            band_lower_hz=band_lower_hz,
            band_upper_hz=band_upper_hz,
            num_coarse_channels=int(channelizer["num_coarse_channels"]),
            frame_size_samples=int(framing["frame_size_samples"]),
            num_input_streams=int(input_streams["num_input_streams"]),
            spectral_sense=spectral_sense,
            frequency_order=frequency_order,
            channelizer=ChannelizerProfile.from_dict(channelizer),
            bin_enbw_hz=_as_float(
                adapter.get(
                    "fine_bin_enbw_hz",
                    adapter.get("bin_enbw_hz", EFFECTIVE_BIN_BW_HZ),
                ),
                "detector_adapter.fine_bin_enbw_hz",
            ),
            pilot_capture_efficiency=float(
                adapter.get("pilot_capture_efficiency", PILOT_CAPTURE_EFFICIENCY)
            ),
            coarse_channel_center_offset_hz=(
                None
                if channelizer.get("coarse_channel_center_offset_hz") is None
                else float(channelizer["coarse_channel_center_offset_hz"])
            ),
            instrument_name=raw.get("instrument_name"),
            profile_status=raw.get("profile_status"),
            detector_window_samples=int(
                adapter.get(
                    "detector_window_samples",
                    DEFAULT_DETECTOR_WINDOW_SAMPLES,
                )
            ),
            dtv_bandwidth_hz=float(
                adapter.get("dtv_bandwidth_hz", DTV_BANDWIDTH_HZ)
            ),
            pilot_below_data_db=float(
                adapter.get("pilot_below_data_db", PILOT_BELOW_DATA_DB)
            ),
            native_channelized_sample_format=str(
                quantization.get(
                    "native_channelized_sample_format",
                    DEFAULT_NATIVE_CHANNELIZED_SAMPLE_FORMAT,
                )
            ),
            adapter_output_format=str(
                quantization.get(
                    "adapter_output_format",
                    DEFAULT_ADAPTER_OUTPUT_FORMAT,
                )
            ),
            bits_per_component=int(
                quantization.get("bits_per_component", DEFAULT_BITS_PER_COMPONENT)
            ),
            quantization_scale_mode_default=str(
                quantization.get("scale_mode_default", DEFAULT_QUANTIZATION_SCALE_MODE)
            ),
            clip_sigma_default=float(
                quantization.get("clip_sigma_default", DEFAULT_CLIP_SIGMA)
            ),
            metadata=nested_metadata,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ReceiverProfile":
        """Load a receiver profile from JSON."""
        return load_receiver_profile(path)

    def to_nested_dict(self) -> dict[str, Any]:
        """Return the preferred public nested JSON representation."""
        frequency_axis = {
            "spectral_sense": self.spectral_sense,
            "order": self.frequency_order,
        }
        metadata_frequency_axis = self.metadata.get("channelizer_frequency_axis")
        if isinstance(metadata_frequency_axis, Mapping):
            frequency_axis.update(dict(metadata_frequency_axis))
        channelizer = {
            "type": self.channelizer.type,
            "input_sample_rate_hz": float(self.sample_rate_hz),
            "pfb_fft_size": int(self.channelizer.fft_size),
            "taps": int(self.channelizer.taps),
            "num_coarse_channels": int(self.num_coarse_channels),
            "coarse_channel_width_hz": float(self.coarse_channel_width_hz),
            "coarse_channel_center_offset_hz": float(self.center_offset_hz),
            "output_sample_rate_hz": float(self.coarse_channel_width_hz),
            "spectral_sense": self.spectral_sense,
            "response": self.channelizer.response,
            "frequency_axis": frequency_axis,
        }
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "receiver_profile_id": self.name,
            "instrument_name": self.instrument_name or self.name,
            "rf_band": {
                "lower_hz": float(self.band_lower_hz),
                "upper_hz": float(self.band_upper_hz),
                "bandwidth_hz": float(self.bandwidth_hz),
            },
            "channelizer": channelizer,
            "framing": {
                "frame_size_samples": int(self.frame_size_samples),
                "frame_size_unit": "channelized complex samples per input stream",
            },
            "input_streams": {
                "num_input_streams": int(self.num_input_streams),
                "stream_unit": "input_stream",
                "stream_map_required": bool(self.num_input_streams > 1),
                "combine_default": "incoherent_power_sum_over_streams",
            },
            "quantization": {
                "native_channelized_sample_format": (
                    self.native_channelized_sample_format
                ),
                "adapter_output_format": self.adapter_output_format,
                "bits_per_component": int(self.bits_per_component),
                "scale_mode_default": self.quantization_scale_mode_default,
                "clip_sigma_default": float(self.clip_sigma_default),
                "record_scale_by_stream": True,
                "record_clip_fraction_by_stream": True,
            },
            "detector_adapter": {
                "compatible_detector_core_id": "pilotproxy_cuda_fstat_v1",
                "detector_window_samples": int(self.detector_window_samples),
                "windows_per_stream_formula": (
                    "frame_size_samples / detector_window_samples"
                ),
                "detector_rows_per_frame_formula": (
                    "num_input_streams * num_selected_channels * windows_per_stream"
                ),
                "fine_bin_width_hz": float(self.bin_enbw_hz),
                "fine_bin_enbw_hz": float(self.bin_enbw_hz),
                "dtv_bandwidth_hz": float(self.dtv_bandwidth_hz),
                "pilot_below_data_db": float(self.pilot_below_data_db),
                "pilot_capture_efficiency": float(self.pilot_capture_efficiency),
            },
        }
        if self.profile_status is not None:
            out["profile_status"] = self.profile_status
        return out


@dataclass(frozen=True)
class ChannelSelection:
    """Mapping from an RF target frequency to a receiver coarse channel."""

    rf_hz: float
    coarse_channel_index: int
    coarse_channel_center_hz: float
    fine_bin_offset_hz: float
    spectral_sense: str
    requires_time_reversal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rf_hz": float(self.rf_hz),
            "coarse_channel_index": int(self.coarse_channel_index),
            "coarse_channel_center_hz": float(self.coarse_channel_center_hz),
            "fine_bin_offset_hz": float(self.fine_bin_offset_hz),
            "spectral_sense": str(self.spectral_sense),
            "requires_time_reversal": bool(self.requires_time_reversal),
        }


def default_reference_receiver_profile(
    *,
    frame_size_samples: int = DEFAULT_FRAME_SIZE_SAMPLES,
    num_input_streams: int = DEFAULT_NUM_INPUT_STREAMS,
) -> ReceiverProfile:
    """Return the shipped 800 MS/s, 400-800 MHz reference receiver profile."""
    return ReceiverProfile(
        schema_version=RECEIVER_PROFILE_SCHEMA_VERSION,
        name=REFERENCE_PROFILE_NAME,
        sample_rate_hz=REFERENCE_ADC_SAMPLE_RATE_HZ,
        band_lower_hz=REFERENCE_BAND_LOWER_HZ,
        band_upper_hz=REFERENCE_BAND_LOWER_HZ + REFERENCE_BANDWIDTH_HZ,
        num_coarse_channels=REFERENCE_NUM_CHANNELS,
        frame_size_samples=int(frame_size_samples),
        num_input_streams=int(num_input_streams),
        spectral_sense=SPECTRAL_SENSE_NORMAL,
        channelizer=ChannelizerProfile(),
        bin_enbw_hz=EFFECTIVE_BIN_BW_HZ,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
    )


def load_receiver_profile(path: str | Path) -> ReceiverProfile:
    """Load a receiver profile JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("receiver profile JSON must contain an object.")
    return ReceiverProfile.from_dict(data)


def receiver_profile_hash(profile: ReceiverProfile | dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a receiver profile."""
    payload = profile.to_dict() if isinstance(profile, ReceiverProfile) else dict(profile)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if len(digest) != PROFILE_HASH_HEX_CHARS:
        raise RuntimeError("unexpected SHA-256 digest length.")
    return digest


def receiver_frequency_to_channel(
    rf_hz: float,
    profile: ReceiverProfile,
) -> ChannelSelection:
    """Map an RF frequency to the nearest receiver coarse-channel selection."""
    rf = float(rf_hz)
    if profile.frequency_order == FREQUENCY_ORDER_DESCENDING_RF:
        first_center = profile.band_upper_hz - profile.center_offset_hz
        idx = int(round((first_center - rf) / profile.coarse_channel_width_hz))
    else:
        first_center = profile.band_lower_hz + profile.center_offset_hz
        idx = int(round((rf - first_center) / profile.coarse_channel_width_hz))
    if idx < 0 or idx >= int(profile.num_coarse_channels):
        raise ValueError(
            f"rf_hz={rf:.3f} is outside receiver profile band/channel centers."
        )
    center = profile.coarse_channel_center_hz(idx)
    rf_offset = rf - center
    fine_offset = -rf_offset if spectral_sense_requires_time_reversal(
        profile.spectral_sense
    ) else rf_offset
    return ChannelSelection(
        rf_hz=rf,
        coarse_channel_index=idx,
        coarse_channel_center_hz=center,
        fine_bin_offset_hz=fine_offset,
        spectral_sense=profile.spectral_sense,
        requires_time_reversal=spectral_sense_requires_time_reversal(
            profile.spectral_sense
        ),
    )


def validate_weight_manifest_profile_hash(
    manifest: dict[str, Any],
    profile: ReceiverProfile,
    *,
    allow_missing: bool = False,
) -> bool:
    """Validate a manifest receiver-profile hash when present."""
    expected = receiver_profile_hash(profile)
    got = manifest.get("receiver_profile_hash", manifest.get("profile_hash"))
    if got is None:
        if allow_missing:
            return False
        raise ValueError("weight manifest does not contain receiver_profile_hash.")
    if str(got) != expected:
        raise ValueError(
            "weight manifest receiver profile hash does not match: "
            f"manifest={got}, expected={expected}"
        )
    return True


__all__ = [
    "ChannelSelection",
    "ChannelizerProfile",
    "ReceiverProfile",
    "default_reference_receiver_profile",
    "FREQUENCY_ORDER_ASCENDING_RF",
    "FREQUENCY_ORDER_DESCENDING_RF",
    "load_receiver_profile",
    "receiver_frequency_to_channel",
    "receiver_profile_hash",
    "validate_weight_manifest_profile_hash",
]
