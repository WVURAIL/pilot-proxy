# coding=utf-8
"""Kernel library interface: specs, features, handle management, config parsing."""

from __future__ import annotations

import ctypes
import os
import re
from typing import Any, Optional

from .paths import DEFAULT_CONFIG_H, DEFAULT_LIB_PATH


# =============================================================================
# Data Classes
# =============================================================================


from dataclasses import dataclass


@dataclass
class KernelSpecs:
    """Compile-time kernel parameters."""

    K: int          # detector_window_samples
    N: int          # num_weight_terms
    bits: int       # sample_bits_per_component
    reference_offset_bins: int

    @property
    def detector_window_samples(self) -> int:
        """Time samples per detector weight/window."""
        return int(self.K)

    @property
    def num_weight_terms(self) -> int:
        """Number of detector weight terms, currently target/lower/upper."""
        return int(self.N)

    @property
    def sample_bits_per_component(self) -> int:
        """Packed complex bits per real/imaginary component."""
        return int(self.bits)

    def as_descriptive_dict(self) -> dict[str, int]:
        """Serialize using non-ambiguous public names."""
        return {
            "detector_window_samples": self.detector_window_samples,
            "num_weight_terms": self.num_weight_terms,
            "sample_bits_per_component": self.sample_bits_per_component,
            "reference_offset_bins": int(self.reference_offset_bins),
        }


@dataclass(frozen=True)
class KernelFeatures:
    """Compiled implementation feature switches."""

    use_dp4a: bool
    use_uint64_power_accumulation: bool
    block_threads: int
    use_constant_weight_lanes: bool = False
    use_shared_weight_lanes: bool = False
    grid_max_blocks: int = 0

    @property
    def dot_product_path(self) -> str:
        """Short label for the compiled dot-product implementation."""
        return "dp4a_packed_int8_lanes" if self.use_dp4a else "scalar_integer"

    @property
    def power_accumulator(self) -> str:
        """Short label for the compiled block-power accumulator."""
        return (
            "uint64_integer"
            if self.use_uint64_power_accumulation
            else "float_diagnostic"
        )

    def as_dict(self) -> dict[str, object]:
        """Serialize compiled implementation features."""
        return {
            "dot_product_path": self.dot_product_path,
            "use_dp4a": bool(self.use_dp4a),
            "power_accumulator": self.power_accumulator,
            "use_uint64_power_accumulation": bool(
                self.use_uint64_power_accumulation
            ),
            "block_threads": int(self.block_threads),
            "use_constant_weight_lanes": bool(self.use_constant_weight_lanes),
            "use_shared_weight_lanes": bool(self.use_shared_weight_lanes),
            "grid_max_blocks": int(self.grid_max_blocks),
        }


@dataclass(frozen=True)
class KernelVersion:
    """Locked kernel core version."""

    major: int
    minor: int
    patch: int

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.major), int(self.minor), int(self.patch)

    def as_string(self) -> str:
        major, minor, patch = self.as_tuple()
        return f"{major}.{minor}.{patch}"


# =============================================================================
# Kernel Interface
# =============================================================================


def _has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    # ctypes.CDLL never raises AttributeError for missing symbols — the only
    # reliable check is inspecting the underlying function pointer address.
    try:
        fn = getattr(lib, name)
        return bool(ctypes.cast(fn, ctypes.c_void_p).value)
    except (AttributeError, TypeError, ValueError):
        return False


class FStatKernel:
    """Wrapper for the F-statistic CUDA kernel library."""

    def __init__(self, lib_path: str | os.PathLike[str] = DEFAULT_LIB_PATH):
        """Load the kernel shared library and initialize metadata."""
        lib_path_text = str(lib_path)
        if not os.path.exists(lib_path_text):
            raise FileNotFoundError(f"Kernel library not found: {lib_path_text}")

        self._lib = ctypes.CDLL(lib_path_text)
        self._setup_signatures()
        self.specs = self._get_specs()
        self.features = self._get_features()
        self.version = self._get_version()

    def _setup_signatures(self):
        """Configure ctypes function signatures."""
        if not _has_symbol(self._lib, "FStat_GetSpecs"):
            raise RuntimeError("Kernel library does not expose FStat_GetSpecs.")
        self._lib.FStat_GetSpecs.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
        self._has_features = _has_symbol(self._lib, "FStat_GetFeatures")
        if self._has_features:
            self._lib.FStat_GetFeatures.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ] * 3
        self._has_optimization_features = _has_symbol(
            self._lib, "FStat_GetOptimizationFeatures"
        )
        if self._has_optimization_features:
            self._lib.FStat_GetOptimizationFeatures.argtypes = [
                ctypes.POINTER(ctypes.c_int)
            ] * 3
        if not _has_symbol(self._lib, "FStat_GetVersion"):
            raise RuntimeError("Kernel library does not expose FStat_GetVersion.")
        self._lib.FStat_GetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        self._has_last_error = _has_symbol(self._lib, "FStat_LastError")
        if self._has_last_error:
            self._lib.FStat_LastError.argtypes = []
            self._lib.FStat_LastError.restype = ctypes.c_char_p

        if not _has_symbol(self._lib, "FStat_Create"):
            raise RuntimeError("Kernel library does not expose FStat_Create.")
        self._lib.FStat_Create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._lib.FStat_Create.restype = ctypes.c_void_p

        self._has_batch_create = _has_symbol(self._lib, "FStat_Create_Batch")
        if self._has_batch_create:
            self._lib.FStat_Create_Batch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self._lib.FStat_Create_Batch.restype = ctypes.c_void_p

        if not _has_symbol(self._lib, "FStat_Compute_DiagnosticFloat"):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_DiagnosticFloat."
            )
        self._lib.FStat_Compute_DiagnosticFloat.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # weights pointer
        ]

        self._lib.FStat_Destroy.argtypes = [ctypes.c_void_p]

        self._has_powers = _has_symbol(self._lib, "FStat_Compute_Powers")
        if self._has_powers:
            self._lib.FStat_Compute_Powers.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_void_p,  # weights pointer
            ]
        self._has_powers_u64 = _has_symbol(self._lib, "FStat_Compute_Powers_U64")
        if self._has_powers_u64:
            self._lib.FStat_Compute_Powers_U64.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_void_p,  # weights pointer
                ctypes.c_void_p,  # uint64 device output pointer
            ]

        self._has_numden_mask_rational_half = _has_symbol(
            self._lib, "FStat_Compute_NumDen_Mask_RationalHalf"
        )
        if self._has_numden_mask_rational_half:
            self._lib.FStat_Compute_NumDen_Mask_RationalHalf.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
        self._has_numden_mask_rational_half_checked = _has_symbol(
            self._lib, "FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount"
        )
        if self._has_numden_mask_rational_half_checked:
            checked = (
                self._lib.FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount
            )
            checked.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_ulonglong,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]

    def _get_specs(self) -> KernelSpecs:
        """Query compile-time kernel parameters."""
        c_k = ctypes.c_int()
        c_n = ctypes.c_int()
        c_b = ctypes.c_int()
        c_reference_offset_bins = ctypes.c_int()

        self._lib.FStat_GetSpecs(
            ctypes.byref(c_k),
            ctypes.byref(c_n),
            ctypes.byref(c_b),
            ctypes.byref(c_reference_offset_bins),
        )

        return KernelSpecs(
            K=c_k.value,
            N=c_n.value,
            bits=c_b.value,
            reference_offset_bins=c_reference_offset_bins.value,
        )

    def _get_features(self) -> KernelFeatures:
        """Query compiled implementation features when supported by the library."""
        if not getattr(self, "_has_features", False):
            return KernelFeatures(
                use_dp4a=False,
                use_uint64_power_accumulation=False,
                block_threads=0,
                use_constant_weight_lanes=False,
                use_shared_weight_lanes=False,
                grid_max_blocks=0,
            )

        c_dp4a = ctypes.c_int()
        c_uint64_power = ctypes.c_int()
        c_threads = ctypes.c_int()
        c_constant_weight_lanes = ctypes.c_int()
        c_shared_weight_lanes = ctypes.c_int()
        c_grid_max_blocks = ctypes.c_int()
        self._lib.FStat_GetFeatures(
            ctypes.byref(c_dp4a),
            ctypes.byref(c_uint64_power),
            ctypes.byref(c_threads),
        )
        if getattr(self, "_has_optimization_features", False):
            self._lib.FStat_GetOptimizationFeatures(
                ctypes.byref(c_constant_weight_lanes),
                ctypes.byref(c_shared_weight_lanes),
                ctypes.byref(c_grid_max_blocks),
            )
        return KernelFeatures(
            use_dp4a=bool(c_dp4a.value),
            use_uint64_power_accumulation=bool(c_uint64_power.value),
            block_threads=int(c_threads.value),
            use_constant_weight_lanes=bool(c_constant_weight_lanes.value),
            use_shared_weight_lanes=bool(c_shared_weight_lanes.value),
            grid_max_blocks=int(c_grid_max_blocks.value),
        )

    def _get_version(self) -> KernelVersion:
        """Query the locked core version."""
        c_major = ctypes.c_int()
        c_minor = ctypes.c_int()
        c_patch = ctypes.c_int()

        self._lib.FStat_GetVersion(
            ctypes.byref(c_major),
            ctypes.byref(c_minor),
            ctypes.byref(c_patch),
        )

        return KernelVersion(
            major=int(c_major.value),
            minor=int(c_minor.value),
            patch=int(c_patch.value),
        )

    def create_handle(self, M: int, d_in: Any, d_out: Any):
        """Create a kernel handle (use as a context manager)."""
        return _KernelHandle(self._lib, M, d_in, d_out)

    def create_detector_matrix_handle(
        self, detector_rows_per_block: int, d_in: Any, d_out: Any
    ):
        """Create a handle using descriptive detector-matrix terminology."""
        return self.create_handle(int(detector_rows_per_block), d_in, d_out)

    def create_batch_handle(self, M: int, batch: int, d_in: Any, d_out: Any):
        """Create a batched kernel handle (use as a context manager)."""
        return _KernelHandle(self._lib, M, d_in, d_out, batch=batch)

    def create_detector_matrix_batch_handle(
        self,
        detector_rows_per_block: int,
        batch: int,
        d_in: Any,
        d_out: Any,
    ):
        """Create a batched handle using descriptive detector-matrix terminology."""
        return self.create_batch_handle(
            int(detector_rows_per_block), batch, d_in, d_out
        )

    def create_raw(self, M: int, in_ptr: int, out_ptr: int):
        """Create a handle from raw pointers."""
        handle = self._lib.FStat_Create(in_ptr, out_ptr, M)
        if not handle:
            raise RuntimeError(
                self.last_error() or "FStat_Create returned NULL."
            )
        return handle

    def create_raw_batch(self, M: int, batch: int, in_ptr: int, out_ptr: int):
        """Create a batched handle from raw pointers."""
        if not getattr(self, "_has_batch_create", False):
            raise RuntimeError("Kernel library does not expose FStat_Create_Batch.")
        handle = self._lib.FStat_Create_Batch(in_ptr, out_ptr, M, batch)
        if not handle:
            raise RuntimeError(
                self.last_error() or "FStat_Create_Batch returned NULL."
            )
        return handle

    def last_error(self) -> str:
        """Return the last CUDA/API error reported by the kernel library."""
        if not getattr(self, "_has_last_error", False):
            return ""
        raw = self._lib.FStat_LastError()
        if not raw:
            return ""
        return raw.decode(errors="replace")

    @property
    def supports_batch(self) -> bool:
        """Return True if the kernel library supports batched handles."""
        return bool(getattr(self, "_has_batch_create", False))

    def compute_diagnostic_float(self, handle, weights_ptr: int):
        """Execute the diagnostic floating-point F-statistic path."""
        self._lib.FStat_Compute_DiagnosticFloat(handle, weights_ptr)

    def compute_powers(self, handle, weights_ptr: int):
        """Execute the kernel and write per-weight power terms to d_out."""
        if not getattr(self, "_has_powers", False):
            raise RuntimeError("Kernel library does not expose FStat_Compute_Powers.")
        self._lib.FStat_Compute_Powers(handle, weights_ptr)

    def compute_powers_u64(self, handle, weights_ptr: int, powers_ptr: int):
        """Execute the kernel and write exact uint64 power terms to powers_ptr."""
        if not getattr(self, "_has_powers_u64", False):
            raise RuntimeError(
                "Kernel library does not expose FStat_Compute_Powers_U64."
            )
        self._lib.FStat_Compute_Powers_U64(handle, weights_ptr, powers_ptr)

    def compute_numden_mask_rational_half(
        self,
        handle,
        weights_ptr: int,
        threshold_half_numerator: int,
        threshold_half_denominator: int,
        numerator_ptr: int,
        denominator_ptr: int,
        mask_ptr: int,
    ):
        """Execute the half-threshold path and write numerator/denominator/mask."""
        if not getattr(self, "_has_numden_mask_rational_half", False):
            raise RuntimeError(
                "Kernel library does not expose "
                "FStat_Compute_NumDen_Mask_RationalHalf."
            )
        if int(threshold_half_denominator) <= 0:
            raise ValueError("threshold_half_denominator must be positive.")
        if int(threshold_half_numerator) < 0:
            raise ValueError("threshold_half_numerator must be non-negative.")
        self._lib.FStat_Compute_NumDen_Mask_RationalHalf(
            handle,
            weights_ptr,
            ctypes.c_ulonglong(int(threshold_half_numerator)),
            ctypes.c_ulonglong(int(threshold_half_denominator)),
            numerator_ptr,
            denominator_ptr,
            mask_ptr,
        )

    def compute_numden_mask_rational_half_checked(
        self,
        handle,
        weights_ptr: int,
        threshold_half_numerator: int,
        threshold_half_denominator: int,
        numerator_ptr: int,
        denominator_ptr: int,
        mask_ptr: int,
        rational_overflow_count_ptr: int,
    ):
        """Execute half-threshold num/den/mask and write an overflow counter."""
        if not getattr(self, "_has_numden_mask_rational_half_checked", False):
            raise RuntimeError(
                "Kernel library does not expose "
                "FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount."
            )
        if int(threshold_half_denominator) <= 0:
            raise ValueError("threshold_half_denominator must be positive.")
        if int(threshold_half_numerator) < 0:
            raise ValueError("threshold_half_numerator must be non-negative.")
        self._lib.FStat_Compute_NumDen_Mask_RationalHalf_WithOverflowCount(
            handle,
            weights_ptr,
            ctypes.c_ulonglong(int(threshold_half_numerator)),
            ctypes.c_ulonglong(int(threshold_half_denominator)),
            numerator_ptr,
            denominator_ptr,
            mask_ptr,
            rational_overflow_count_ptr,
        )

    def destroy(self, handle):
        """Destroy a kernel handle."""
        self._lib.FStat_Destroy(handle)


class _KernelHandle:
    """Context manager for kernel handles."""

    def __init__(
        self,
        lib,
        M: int,
        d_in: Any,
        d_out: Any,
        *,
        batch: int = 1,
    ):
        """Create a kernel handle for the provided device buffers."""
        self._lib = lib
        if batch < 1:
            raise ValueError("batch must be >= 1.")
        if batch > 1:
            create_batch = getattr(lib, "FStat_Create_Batch", None)
            if create_batch is None:
                raise RuntimeError("Kernel library does not expose FStat_Create_Batch.")
            self._handle = create_batch(d_in.data.ptr, d_out.data.ptr, M, batch)
        else:
            self._handle = lib.FStat_Create(
                d_in.data.ptr,
                d_out.data.ptr,
                M,
            )
        if not self._handle:
            last_error = ""
            last_error_fn = getattr(lib, "FStat_LastError", None)
            if last_error_fn is not None:
                raw = last_error_fn()
                if raw:
                    last_error = raw.decode(errors="replace")
            raise RuntimeError(
                last_error or "F-statistic kernel handle creation failed."
            )

    def __enter__(self):
        """Return the underlying handle for use in a context manager."""
        return self._handle

    def __exit__(self, *args):
        """Destroy the underlying handle when exiting the context."""
        self._lib.FStat_Destroy(self._handle)


# =============================================================================
# Config Parsing
# =============================================================================


def read_kernel_config_h(
    config_h_path: str | os.PathLike = DEFAULT_CONFIG_H,
) -> Optional[dict[str, int]]:
    """Parse detector-window, weight-term, and reference offset from config.h."""
    from pathlib import Path

    try:
        text = Path(config_h_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    def find_define(name: str) -> Optional[int]:
        match = re.search(
            rf"^\s*#\s*define\s+{re.escape(name)}\s+(\d+)",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            return None
        return int(match.group(1))

    k_val = find_define("FSTAT_DETECTOR_WINDOW_SAMPLES")
    n_val = find_define("FSTAT_NUM_WEIGHT_TERMS")
    reference_offset_bins = find_define("FSTAT_REFERENCE_BIN_OFFSET")
    if k_val is None or reference_offset_bins is None:
        return None
    return {
        "detector_window_samples": k_val,
        "num_weight_terms": n_val if n_val is not None else 0,
        "reference_offset_bins": reference_offset_bins,
        "K": k_val,
        "N": n_val if n_val is not None else 0,
    }
