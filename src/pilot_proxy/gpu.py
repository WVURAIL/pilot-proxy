# coding=utf-8
"""GPU availability checks, memory queries, and bandwidth constants."""

from __future__ import annotations

import ctypes
import warnings
from typing import Any, Callable, Optional, Tuple, cast

try:
    import cupy as cp

    _CUPY_IMPORT_ERROR: Exception | None = None
except Exception as cupy_import_error:
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = cupy_import_error

# Batch-size guardrails for host-side auto-tuning. These limit memory pressure
# and launch latency without changing detector math.
DEFAULT_CHUNK_SIZE = 128
MIN_CHUNK_SIZE = 1
MAX_CHUNK_SIZE = 4096
# Conservative scratch/output byte estimate per detector row for memory sizing.
DEFAULT_BYTES_PER_SAMPLE = 24
GPU_MEMORY_SAFETY_FRACTION = 0.8
NVIDIA_SMI_TIMEOUT_SECONDS = 5
CUDA_DRIVER_SUCCESS = 0


# Approximate device memory bandwidths [GB/s] for quick throughput estimates.
GPU_BANDWIDTHS = {
    # Laptop variants FIRST (more specific matches)
    "RTX 5000 Ada Generation Laptop": 448.0,
    "RTX 4000 Ada Generation Laptop": 288.0,
    "RTX 4090 Laptop": 576.0,
    "RTX 4080 Laptop": 432.0,
    "RTX 3080 Laptop": 448.0,
    "RTX 3070 Laptop": 384.0,
    # Data center / workstation
    "A100": 1555.0,
    "A40": 696.0,
    "A6000": 768.0,
    "V100": 900.0,
    "T4": 320.0,
    "L40": 864.0,
    # Desktop
    "RTX 3090": 936.0,
    "RTX 4090": 1008.0,
    # Workstation (desktop form factor)
    "RTX 5000 Ada": 576.0,
    "RTX 4000 Ada": 360.0,
}


def _cupy_unavailable_reason() -> str:
    if _CUPY_IMPORT_ERROR is None:
        return "CuPy import unavailable"
    return f"CuPy import failed: {_CUPY_IMPORT_ERROR}"


def get_gpu_info() -> Tuple[str, Optional[float]]:
    """Return (gpu_name, peak_bandwidth_gbps), or ("Unknown", None) on failure."""
    if cp is None:
        return "Unknown", None
    try:
        runtime = cp.cuda.runtime
        get_props = getattr(runtime, "get_device_properties", None)
        if get_props is None:
            get_props = getattr(runtime, "getDeviceProperties", None)
        if get_props is None:
            raise AttributeError("CuPy runtime lacks device property accessor")
        props = cast(Callable[[int], dict[str, Any]], get_props)(cp.cuda.Device().id)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()

        peak_bw = None
        for gpu_key, bw in GPU_BANDWIDTHS.items():
            if gpu_key.lower() in name.lower():
                peak_bw = bw
                break

        return name, peak_bw
    except Exception as gpu_exc:
        import logging

        logging.debug(f"Failed to get GPU info: {gpu_exc}")
        return "Unknown", None


def cuda_available() -> Tuple[bool, str]:
    """Return availability plus a reason string when no GPU is found."""
    if cp is None:
        return False, _cupy_unavailable_reason()
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as cuda_runtime_error:
        import logging

        logging.debug(f"CUDA availability check failed: {cuda_runtime_error}")
        return False, (
            f"CuPy runtime check failed: {cuda_runtime_error}; "
            f"{_nvidia_smi_summary()}; {_cuda_driver_summary()}"
        )
    if count < 1:
        return False, (
            "CuPy runtime found no devices; "
            f"{_nvidia_smi_summary()}; {_cuda_driver_summary()}"
        )
    return True, ""


def _nvidia_smi_summary() -> str:
    """Return a short NVML visibility diagnostic via nvidia-smi."""
    try:
        import subprocess

        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except Exception as nvidia_smi_error:
        return f"nvidia-smi diagnostic failed: {nvidia_smi_error}"

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        detail = f": {stderr}" if stderr else ""
        return f"nvidia-smi failed with exit={proc.returncode}{detail}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return "nvidia-smi reports no GPUs"
    return "nvidia-smi sees " + "; ".join(lines)


def _cuda_driver_summary() -> str:
    # Intentionally separate from the CuPy runtime gate. On WSL and mixed
    # toolkit installs, NVML/driver visibility can succeed while the CUDA
    # runtime library selected by CuPy fails to initialize.
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        version = ctypes.c_int()
        version_err = int(lib.cuDriverGetVersion(ctypes.byref(version)))
        init_err = int(lib.cuInit(0))
        count = ctypes.c_int()
        count_err = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    except Exception as cuda_driver_error:
        return f"CUDA driver API diagnostic failed: {cuda_driver_error}"

    if (
        version_err == CUDA_DRIVER_SUCCESS
        and init_err == CUDA_DRIVER_SUCCESS
        and count_err == CUDA_DRIVER_SUCCESS
    ):
        return (
            "CUDA driver API sees "
            f"{int(count.value)} device(s), driver_version={int(version.value)}"
        )
    return (
        "CUDA driver API errors: "
        f"cuDriverGetVersion={version_err}, cuInit={init_err}, "
        f"cuDeviceGetCount={count_err}"
    )


def require_cuda(context: str = "This script") -> None:
    """Exit with a clear message if no CUDA-capable GPU is available."""
    ok, reason = cuda_available()
    if ok:
        return
    msg = f"{context} requires a CUDA-capable GPU"
    if reason:
        msg = f"{msg}: {reason}"
    raise SystemExit(msg)


def get_optimal_chunk_size(
    M: int,
    K: int,
    bytes_per_sample: int = DEFAULT_BYTES_PER_SAMPLE,
) -> int:
    """Return a batch size (1–4096) based on available GPU memory."""
    if cp is None:
        return DEFAULT_CHUNK_SIZE
    try:
        free_mem, _ = cp.cuda.Device().mem_info
    except Exception as size_exc:
        warnings.warn(
            f"Could not query GPU memory: {size_exc}", RuntimeWarning, stacklevel=2
        )
        return DEFAULT_CHUNK_SIZE

    bytes_per_trial = M * K * bytes_per_sample
    if bytes_per_trial <= 0:
        return MIN_CHUNK_SIZE
    optimal = int((free_mem * GPU_MEMORY_SAFETY_FRACTION) // bytes_per_trial)
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, optimal))
