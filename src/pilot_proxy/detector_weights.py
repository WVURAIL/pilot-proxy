# coding=utf-8
"""Loader for the packed ATSC reference detector weight ROM."""

from __future__ import annotations

import math
import os
import struct
import zlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import numpy as np

from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_RAW_INPUT,
    input_coordinate_system_for_weight_coordinate,
    normalize_weight_coordinate_system,
)
from pilot_proxy.provenance import file_sha256, git_blob_sha1

from .atsc_channels import (
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
    physical_channel_to_pilot_hz,
)

WEIGHT_MAGIC = b"FSTATWGT1"
WEIGHT_VERSION = 3
HEADER_FIXED_FMT = "<9sIIIIIIIddHH"
HEADER_FIXED_SIZE = struct.calcsize(HEADER_FIXED_FMT)
CRC_SIZE = 4
# zlib.crc32 returns a signed-looking Python int on some platforms; mask to the
# unsigned 32-bit field stored in the weight-file trailer.
CRC32_UNSIGNED_MASK = 0xFFFFFFFF
# Physical ATSC pilot matching should be exact to the manifest, with a small
# tolerance for decimal MHz/Hertz roundoff in command-line inputs.
DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0
HZ_PER_MHZ = 1.0e6
WEIGHT_MANIFEST_SCHEMA_VERSION = "fstat_weight_manifest_v2"
_REFERENCE_FIELD_PART = "reference"
_OLD_GAP_FIELD_PART = "guard"
_OFFSET_FIELD_PART = "offset"
_BINS_FIELD_PART = "bins"
_NOMINAL_FIELD_PART = "nominal"
_REQUESTED_FIELD_PART = "requested"
_SELECTED_FIELD_PART = "selected"
_MIN_EMPIRICAL_FIELD_PART = "min_empirical"
DEPRECATED_DETECTOR_SPACING_FIELDS = frozenset(
    {
        "_".join((_REFERENCE_FIELD_PART, _OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _REQUESTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _SELECTED_FIELD_PART,
            )
        ),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OLD_GAP_FIELD_PART,
                _BINS_FIELD_PART,
                _MIN_EMPIRICAL_FIELD_PART,
            )
        ),
        "_".join((_OLD_GAP_FIELD_PART, _BINS_FIELD_PART)),
        "_".join(
            (
                _REFERENCE_FIELD_PART,
                _OFFSET_FIELD_PART,
                _BINS_FIELD_PART,
                _NOMINAL_FIELD_PART,
            )
        ),
    }
)

# The shipped weight ROM is built for the included 400-800 MHz reference band.
REFERENCE_BAND_LOWER_MHZ = 400.0
REFERENCE_BANDWIDTH_MHZ = 400.0
ATSC_CHANNEL_WIDTH_MHZ = ATSC_CHANNEL_WIDTH_HZ / HZ_PER_MHZ
ATSC_UHF_CHANNEL_14_PILOT_MHZ = (
    ATSC_UHF_CHANNEL_14_LOWER_EDGE_HZ + ATSC_PILOT_OFFSET_HZ
) / HZ_PER_MHZ


@dataclass(frozen=True)
class WeightHeader:
    """Parsed header metadata for a serialized weight file."""

    magic: str
    version: int
    header_size: int
    detector_window_samples: int
    num_weight_terms: int
    reference_offset_bins: int
    component_bits: int
    num_channels: int
    doppler_tol_hz: Optional[float]
    fine_bin_width_hz: float
    reference_name: str
    profile_name: str
    crc32: int

    @property
    def K(self) -> int:
        return self.detector_window_samples

    @property
    def N(self) -> int:
        return self.num_weight_terms

    @property
    def n_channels(self) -> int:
        return self.num_channels


class InvalidWeightHeaderError(ValueError):
    """Raised when a weight file lacks the expected header."""


def _packed_dtype_for_component_bits(component_bits: int) -> np.dtype:
    bits = int(component_bits)
    if bits == 4:
        return np.dtype(np.int8)
    if bits == 8:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported packed component bit depth: {bits}")


def _extract_kernel_specs(expected_kernel) -> Optional[tuple[int, int, int, int]]:
    if expected_kernel is None:
        return None
    if isinstance(expected_kernel, (tuple, list)):
        if len(expected_kernel) < 4:
            raise TypeError(
                "expected_kernel must include K, N, bits, reference_offset_bins"
            )
        return (
            int(expected_kernel[0]),
            int(expected_kernel[1]),
            int(expected_kernel[2]),
            int(expected_kernel[3]),
        )
    return (
        int(getattr(expected_kernel, "K")),
        int(getattr(expected_kernel, "N")),
        int(getattr(expected_kernel, "bits")),
        int(getattr(expected_kernel, "reference_offset_bins")),
    )


def read_header_and_weights(path: Path) -> tuple[WeightHeader, np.ndarray]:
    """Read and validate a detector weight file header plus payload."""
    path = Path(path)
    with path.open("rb") as f:
        fixed = f.read(HEADER_FIXED_SIZE)
        if len(fixed) != HEADER_FIXED_SIZE:
            raise ValueError(f"Weight file too small for header: {path}")

        (
            magic,
            version,
            header_size,
            detector_window_samples,
            num_weight_terms,
            reference_offset_bins,
            component_bits,
            num_channels,
            doppler_val,
            fine_bin_width_hz,
            reference_name_len,
            profile_name_len,
        ) = struct.unpack(HEADER_FIXED_FMT, fixed)

        if magic != WEIGHT_MAGIC:
            raise InvalidWeightHeaderError(
                "Weight file is missing the expected FSTATWGT1 header."
            )
        if version != WEIGHT_VERSION:
            raise ValueError(
                f"Unsupported weight version {version}; expected {WEIGHT_VERSION}."
            )

        expected_header_size = (
            HEADER_FIXED_SIZE
            + int(reference_name_len)
            + int(profile_name_len)
            + CRC_SIZE
        )
        if header_size != expected_header_size:
            raise ValueError(
                f"Header size mismatch: expected {expected_header_size}, "
                f"got {header_size}."
            )

        rest = f.read(header_size - HEADER_FIXED_SIZE)
        if len(rest) != header_size - HEADER_FIXED_SIZE:
            raise ValueError(f"Incomplete header data in weight file: {path}")

        reference_name_bytes = rest[:reference_name_len]
        profile_name_bytes = rest[
            reference_name_len : reference_name_len + profile_name_len
        ]
        crc = struct.unpack("<I", rest[-CRC_SIZE:])[0]
        weights_bytes = f.read()

    header_no_crc = fixed + reference_name_bytes + profile_name_bytes + struct.pack(
        "<I", 0
    )
    calc_crc = zlib.crc32(header_no_crc)
    calc_crc = zlib.crc32(weights_bytes, calc_crc) & CRC32_UNSIGNED_MASK
    if calc_crc != crc:
        raise ValueError(
            f"Weight file CRC mismatch: expected 0x{crc:08X}, "
            f"got 0x{calc_crc:08X}."
        )

    packed_dtype = _packed_dtype_for_component_bits(component_bits)
    weights = np.frombuffer(weights_bytes, dtype=packed_dtype).copy()
    doppler_tol_hz = None if math.isnan(doppler_val) else float(doppler_val)
    header = WeightHeader(
        magic=magic.decode("utf-8", errors="replace"),
        version=int(version),
        header_size=int(header_size),
        detector_window_samples=int(detector_window_samples),
        num_weight_terms=int(num_weight_terms),
        reference_offset_bins=int(reference_offset_bins),
        component_bits=int(component_bits),
        num_channels=int(num_channels),
        doppler_tol_hz=doppler_tol_hz,
        fine_bin_width_hz=float(fine_bin_width_hz),
        reference_name=reference_name_bytes.decode(errors="replace"),
        profile_name=profile_name_bytes.decode(errors="replace"),
        crc32=int(crc),
    )
    return header, weights


class DetectorWeightBank:
    """Load a prebuilt packed detector weight ROM and expose channel weights."""

    def __init__(
        self,
        profile: str = "atsc_reference",
        profile_name: str = "dtv",
        K: Optional[int] = None,
        N: int = 3,
        reference_offset_bins: Optional[int] = None,
        explicit_path: str | os.PathLike | None = None,
        expected_kernel: object | None = None,
    ) -> None:
        if explicit_path is None:
            raise ValueError("DetectorWeightBank requires explicit_path")

        self.path = Path(explicit_path)
        self.filename = str(self.path)
        self.header, flat = read_header_and_weights(self.path)
        self.manifest = _read_adjacent_manifest(self.path)
        _validate_manifest_spacing_schema(self.manifest)
        _validate_manifest_weight_binding(self.path, self.manifest)
        self.K = int(self.header.K)
        self.N = int(self.header.N)
        header_reference_offset_bins = self.header.reference_offset_bins
        if header_reference_offset_bins is None:
            raise ValueError("Weight header is missing reference_offset_bins.")
        self.reference_offset_bins = int(header_reference_offset_bins)
        self.component_bits = int(self.header.component_bits)
        self.bits = self.component_bits
        self.profile = str(profile)
        self.profile_name = str(profile_name)

        expected = _extract_kernel_specs(expected_kernel)
        if expected is not None:
            exp_k, exp_n, exp_bits, exp_offset = expected
            actual = (
                self.K,
                self.N,
                self.component_bits,
                self.reference_offset_bins,
            )
            if actual != (exp_k, exp_n, exp_bits, exp_offset):
                raise ValueError(
                    "Weight ROM does not match kernel specs: "
                    f"weights={actual}, kernel={(exp_k, exp_n, exp_bits, exp_offset)}"
                )
        if K is not None and int(K) != self.K:
            raise ValueError(f"Weight K={self.K} does not match requested K={K}")
        if int(N) != self.N:
            raise ValueError(f"Weight N={self.N} does not match requested N={N}")
        if (
            reference_offset_bins is not None
            and int(reference_offset_bins) != self.reference_offset_bins
        ):
            raise ValueError(
                f"Weight reference_offset_bins={self.reference_offset_bins} "
                "does not match requested "
                f"reference_offset_bins={reference_offset_bins}"
            )

        expected_size = self.header.num_channels * self.N * self.K
        if flat.size != expected_size:
            raise ValueError(
                f"Weight payload has {flat.size} values; expected {expected_size}."
            )
        self.rom_table = np.ascontiguousarray(
            flat.reshape(self.header.num_channels, self.N, self.K)
        )

        channel_width_mhz = REFERENCE_BANDWIDTH_MHZ / self.header.num_channels
        self.reference_freqs = (
            REFERENCE_BAND_LOWER_MHZ
            + channel_width_mhz * (np.arange(self.header.num_channels) + 1)
        )
        self.detector_profile = {
            "name": "atsc_reference",
            "num_channels": int(self.header.num_channels),
            "bandwidth_mhz": float(REFERENCE_BANDWIDTH_MHZ),
        }
        self._known_layout = _known_layout_from_manifest(self.manifest)
        self.known_pilot_frequencies_mhz = [
            float(cast(float, row["target_frequency_mhz"]))
            for row in self._known_layout
        ]
        self._layout_by_physical_channel = {
            _physical_channel_from_pilot_mhz(
                float(cast(float, row["target_frequency_mhz"]))
            ): row
            for row in self._known_layout
        }

    def _weights_for_channel_index(self, channel_index: int) -> tuple[np.ndarray | None, bool]:
        idx = int(channel_index)
        if idx < 0 or idx >= self.rom_table.shape[0]:
            raise ValueError(
                f"coarse channel index must be in [0, {self.rom_table.shape[0] - 1}], "
                f"got {idx}."
            )
        weights = np.ascontiguousarray(self.rom_table[idx])
        if np.any(weights):
            return weights, True
        return None, False

    def get_weights(self, freq_mhz: float) -> tuple[np.ndarray | None, bool]:
        """Return packed weights for the nearest reference-channel entry.

        This is an advanced nearest-channel lookup. Public detector paths should
        use get_weights_for_pilot_frequency or get_weights_for_physical_channel
        so the requested pilot is
        validated against the shipped weight manifest.
        """
        chan_idx = int(np.argmin(np.abs(self.reference_freqs - float(freq_mhz))))
        return self._weights_for_channel_index(chan_idx)

    def get_weights_for_pilot_frequency(
        self,
        freq_mhz: float,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> tuple[np.ndarray | None, bool]:
        """Return weights only when the requested MHz value matches a known pilot."""
        layout = self.layout_for_pilot_frequency(
            freq_mhz,
            tolerance_hz=tolerance_hz,
        )
        if not layout:
            raise ValueError(
                "Pilot-frequency and physical-channel lookups require the adjacent "
                "weight manifest; use get_weights() explicitly for an expert nearest-"
                "coarse-channel lookup."
            )
        coarse_index = int(cast(int, layout["coarse_channel_index"]))
        return self._weights_for_channel_index(coarse_index)

    def get_weights_for_physical_channel(
        self,
        channel: int,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> tuple[np.ndarray | None, bool]:
        """Return weights for an ATSC UHF physical channel."""
        pilot_mhz = physical_channel_to_pilot_hz(int(channel)) / HZ_PER_MHZ
        return self.get_weights_for_pilot_frequency(
            pilot_mhz,
            tolerance_hz=tolerance_hz,
        )

    def layout_for_pilot_frequency(
        self,
        freq_mhz: float,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> dict[str, object]:
        """Return the manifest target/reference layout for a known pilot."""
        if not self._known_layout:
            return {}
        requested = float(freq_mhz)
        known = np.asarray(self.known_pilot_frequencies_mhz, dtype=np.float64)
        nearest_idx = int(np.argmin(np.abs(known - requested)))
        delta_hz = abs(float(known[nearest_idx]) - requested) * HZ_PER_MHZ
        if delta_hz > float(tolerance_hz):
            raise ValueError(
                "Requested DTV pilot frequency is not in the weight manifest: "
                f"{requested:.6f} MHz, nearest known "
                f"{float(known[nearest_idx]):.6f} MHz, delta={delta_hz:.3f} Hz."
            )
        return dict(self._known_layout[nearest_idx])

    def layout_for_physical_channel(
        self,
        channel: int,
        *,
        tolerance_hz: float = DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    ) -> dict[str, object]:
        """Return the manifest target/reference layout for a physical channel."""
        pilot_mhz = physical_channel_to_pilot_hz(int(channel)) / HZ_PER_MHZ
        return self.layout_for_pilot_frequency(
            pilot_mhz,
            tolerance_hz=tolerance_hz,
        )

    def supported_physical_channels(self) -> list[int]:
        """Return physical channels covered by this weight manifest."""
        return sorted(int(ch) for ch in self._layout_by_physical_channel)


def _read_adjacent_manifest(path: Path) -> dict[str, object]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _known_layout_from_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    layout = manifest.get("target_reference_layout", [])
    if not isinstance(layout, list):
        return []
    out: list[dict[str, object]] = []
    for row in layout:
        if not isinstance(row, dict):
            continue
        if "target_frequency_mhz" not in row or "coarse_channel_index" not in row:
            continue
        out.append(row)
    return out


def _validate_manifest_weight_binding(
    path: Path,
    manifest: dict[str, object],
) -> None:
    """Verify that an adjacent manifest names the exact binary being loaded."""
    if not manifest:
        return
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Weight manifest requires an artifacts object.")

    expected_sha256 = artifacts.get("weights_sha256")
    if expected_sha256 is not None:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != str(expected_sha256):
            raise ValueError(
                "Weight manifest/binary SHA256 mismatch: "
                f"{actual_sha256!r} != {str(expected_sha256)!r}."
            )
        return

    # The checked-in legacy bank predates the SHA256 field. Its Git blob id still
    # binds the complete file; newly generated banks always emit weights_sha256.
    expected_blob = artifacts.get("weights_git_blob_sha1")
    if expected_blob is not None:
        actual_blob = git_blob_sha1(path)
        if actual_blob != str(expected_blob):
            raise ValueError(
                "Weight manifest/binary Git-blob mismatch: "
                f"{actual_blob!r} != {str(expected_blob)!r}."
            )
        return

    raise ValueError(
        "Weight manifest must bind the binary with artifacts.weights_sha256 "
        "or artifacts.weights_git_blob_sha1."
    )


def _validate_manifest_spacing_schema(manifest: dict[str, object]) -> None:
    if not manifest:
        return
    schema_version = manifest.get("schema_version")
    if schema_version != WEIGHT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported weight manifest schema_version: "
            f"{schema_version!r}; expected {WEIGHT_MANIFEST_SCHEMA_VERSION!r}."
        )
    coordinate = manifest.get("weight_coordinate_system")
    if coordinate is None:
        raise ValueError(
            "Weight manifest schema v2 requires weight_coordinate_system."
        )
    coordinate_system = normalize_weight_coordinate_system(coordinate)
    input_coordinate = manifest.get("input_coordinate_system")
    if input_coordinate is None:
        raise ValueError(
            "Weight manifest schema v2 requires input_coordinate_system."
        )
    expected_input_coordinate = input_coordinate_system_for_weight_coordinate(
        coordinate_system
    )
    if str(input_coordinate) != expected_input_coordinate:
        raise ValueError(
            "Weight manifest input_coordinate_system does not match "
            f"weight_coordinate_system: {input_coordinate!r} != "
            f"{expected_input_coordinate!r}."
        )
    input_preprocessing = manifest.get("input_preprocessing")
    if not isinstance(input_preprocessing, dict):
        raise ValueError("Weight manifest schema v2 requires input_preprocessing.")
    if "time_reverse_detector_windows_before_kernel" not in input_preprocessing:
        raise ValueError(
            "Weight manifest input_preprocessing requires "
            "time_reverse_detector_windows_before_kernel."
        )
    if (
        coordinate_system == WEIGHT_COORDINATE_RAW_INPUT
        and bool(input_preprocessing["time_reverse_detector_windows_before_kernel"])
    ):
        raise ValueError(
            "Raw input-coordinate weights must not request detector-window "
            "time reversal before the kernel."
        )
    _reject_deprecated_spacing_fields(manifest)
    kernel_spec = manifest.get("kernel_spec")
    if isinstance(kernel_spec, dict):
        _reject_deprecated_spacing_fields(kernel_spec)
    layout = manifest.get("target_reference_layout")
    if isinstance(layout, list):
        for row in layout:
            if isinstance(row, dict):
                _reject_deprecated_spacing_fields(row)


def _reject_deprecated_spacing_fields(data: dict[str, object]) -> None:
    for key in sorted(DEPRECATED_DETECTOR_SPACING_FIELDS):
        if key in data:
            raise ValueError(
                f"Deprecated detector-spacing field found: {key}. "
                "Use skipped_guard_bins or reference_offset_bins."
            )


def _physical_channel_from_pilot_mhz(pilot_mhz: float) -> int:
    channel_offset = (
        float(pilot_mhz) - ATSC_UHF_CHANNEL_14_PILOT_MHZ
    ) / ATSC_CHANNEL_WIDTH_MHZ
    return int(round(channel_offset)) + int(ATSC_UHF_MIN_PHYSICAL_CHANNEL)


__all__ = [
    "CRC32_UNSIGNED_MASK",
    "CRC_SIZE",
    "DetectorWeightBank",
    "HEADER_FIXED_FMT",
    "HEADER_FIXED_SIZE",
    "InvalidWeightHeaderError",
    "WEIGHT_MAGIC",
    "WEIGHT_VERSION",
    "WeightHeader",
    "read_header_and_weights",
]
