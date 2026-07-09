# coding=utf-8
"""datatrawl Analyzer: CHIME DTV F-statistic detector.

The detector functionality of ``pilot_proxy.chime.runner.run_chime_analysis`` as a
datatrawl Analyzer. It reimplements no DSP: it wraps
PilotProxy's own ``pack_chime_block_for_detector`` -> ``detector_fn`` (the CUDA kernel
via ``detect_packed_for_positive_excess`` by default) -> ``_append_detection_rows``
maths, so the per-frame product matches the runner by construction.

Data path (per coarse channel / pilot, fanned out one ``<channel>.npz`` each):

* the ``chime-baseband-packed`` reader yields one raw ``uint8 [nfft, n_feeds]``
  frame per chunk (native offset-binary 4+4-bit);
* the analyzer reshapes it to PilotProxy's normalised ``(n_feeds, 1, nfft)`` block and
  calls ``pack_chime_block_for_detector`` with ``sample_encoding`` =
  native-offset-binary, which takes the LOSSLESS repack route (no calibration
  scale / requantisation -- the native int4 grid passes straight through);
* ``detector_fn(packed=packed.packed, weights=weights, kernel=kernel)`` yields
  per-frame target/reference powers; the positive-excess mask + dB metrics come
  from ``dtv_units`` exactly as ``_append_detection_rows`` computes them.

The CUDA kernel is GPU-only, so the real kernel-level + real-data parity is a
CANFAR/GPU step. ``detector_fn`` / ``kernel`` / ``weights`` are injectable (via
``ctx.options``), mirroring ``run_chime_analysis``, so a CPU reference can drive
a GPU-free plumbing parity test in the same way PilotProxy's own runner test does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from datatrawl.instruments import nyquist_sign
from datatrawl.interfaces import Analyzer, RunContext, PluginInfo, EXPERIMENTAL

try:
    from datatrawl.registry import analyzer as _register_analyzer
except Exception:  # pragma: no cover - registry shape guard
    def _register_analyzer(cls):  # type: ignore[no-redef]
        return cls

from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
from pilot_proxy.chime.hdf5_input import (
    CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
    nearest_atsc_physical_channel,
)
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED, SPECTRAL_SENSE_NORMAL
from pilot_proxy.dtv_units import (
    DETECTOR_WINDOW_SAMPLES,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    fstat_num_den_to_fstat_level_db,
    fstat_num_den_to_pnr_bin_db,
    fstat_num_den_to_raw,
    pnr_bin_db_to_snr_shelf_db,
)
from pilot_proxy.detector_contract import (
    POSITIVE_EXCESS_MASK_RULE,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
    build_chime_detector_contract,
    norm_corrected_mu0,
    normalize_weight_coordinate_system,
    weight_term_norms_sq,
)
from pilot_proxy.provenance import (
    file_sha256,
    package_source_sha256,
    sidecar_manifest_path,
)
import json
import hashlib

from datatrawl import accel

from ._chime_coarse import chime_freq_id_from_hz, source_event_key

import warnings

# v2 adds: integrated spectra (before/after mask), a per-unit absolute time axis,
# per-frame unit tags, and provenance (weights_hash / detector_version /
# mask_rule); and renames the per-frame ``mask`` -> ``reject_mask`` (1 = discard).
# The bump is deliberate: ``resume()`` refuses to extend a v1 product, so a run
# started on the old schema is never silently mixed with v2 output.
_SCHEMA_VERSION = "pilotproxy_detector_datatrawl_v2"

# Native CHIME baseband packing: one uint8 per complex sample, high nibble = real,
# low nibble = imag, each a 4-bit offset-binary value (stored = signed + 8). This
# is exactly datatrawl ``_baseband_format.unpack_4bit``, kept xp-generic here so
# the integrated-spectrum FFT runs on cupy (GPU) in production and numpy in tests
# with identical arithmetic.
_INT4_OFFSET = np.float32(8.0)


def _unpack_4bit_xp(xp, packed):
    """offset-binary uint8 [..] -> complex64 [..], on numpy or cupy (`xp`)."""
    real = (packed >> 4).astype(xp.float32) - _INT4_OFFSET
    imag = (packed & np.uint8(0x0F)).astype(xp.float32) - _INT4_OFFSET
    return (real + 1j * imag).astype(xp.complex64)


def _to_host(a) -> np.ndarray:
    """Bring an accumulator to host numpy (cupy ndarray -> .get(); numpy -> asarray)."""
    get = getattr(a, "get", None)
    return np.asarray(get() if callable(get) else a)


def _detector_fft_backend():
    """Return cupy only when its CUDA runtime is actually usable.

    Some CANFAR / CI images have CuPy importable through system-site packages even
    when the visible node has no compatible CUDA driver.  ``accel.import_cupy()``
    can therefore return the module, but the first allocation later fails with a
    CUDA runtime error.  The detector's integrated-spectrum FFT is only a
    reporting-side accumulator and already has a NumPy path, so fail closed to
    NumPy unless a tiny runtime probe succeeds.
    """
    try:
        cp = accel.import_cupy()
    except Exception:
        return np
    if cp is None:
        return np
    try:
        runtime = getattr(getattr(cp, "cuda", None), "runtime", None)
        get_count = getattr(runtime, "getDeviceCount", None)
        if callable(get_count) and int(get_count()) <= 0:
            return np
        probe = cp.zeros(1, dtype=cp.float32)
        # Touch the array so lazy runtime failures surface during backend
        # selection instead of later, after the analyzer has committed to cupy.
        if hasattr(probe, "sum"):
            probe.sum()
        return cp
    except Exception:
        return np


_DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0


@_register_analyzer
class PilotProxyDetectorAnalyzer(Analyzer):
    """Per-pilot CHIME F-statistic detector, parity with PilotProxy's batch runner."""

    requires_in_order = True

    info = PluginInfo(
        name="pilot-proxy-detector",
        kind="analyzer",
        summary="CHIME DTV F-statistic detector (PilotProxy parity; per-channel "
                "fixed-point pilot detection + positive-excess mask).",
        status=EXPERIMENTAL,  # real kernel-level + real-data parity is a CANFAR/GPU step
        instruments=("chime", "kko", "gbo", "hco"),
        produces="<channel>.npz (chime_detector_outputs schema)",
        requires=("h5py", "pilot-proxy", "GPU+libfstatistic.so (CUDA kernel)"),
        notes="Use with the 'chime-baseband-packed' reader. Wraps "
              "pack_chime_block_for_detector + detect_packed_for_positive_excess; "
              "detector_fn/kernel/weights injectable via options (CPU ref for tests).",
    )

    def __init__(self) -> None:
        self._nfft = 0
        self._K = int(DETECTOR_WINDOW_SAMPLES)
        self._spectral_sense = SPECTRAL_SENSE_NORMAL
        self._physical_channel = -1
        self._freq_id = -1
        self._f0_hz = 800_000_000.0
        self._pilot_in_band = True
        self._pilot_rf_hz = float("nan")
        self._coarse_center_hz = float("nan")
        # injected detector pieces (defaults resolved lazily in begin)
        self._detector_fn = None
        self._kernel = None
        self._weights: np.ndarray | None = None
        # dB-calibration constants (overridable)
        self._pilot_below_data_db = float(PILOT_BELOW_DATA_DB)
        self._bin_enbw_hz = float(EFFECTIVE_BIN_BW_HZ)
        self._dtv_bandwidth_hz = float(DTV_BANDWIDTH_HZ)
        self._pilot_capture_efficiency = float(PILOT_CAPTURE_EFFICIENCY)
        self._max_chunks_per_file: int | None = None
        self._num_input_streams = 0
        self._detector_contract: dict[str, Any] = {}
        self._detector_contract_json = "{}"
        self._weights_hash = ""
        self._weight_bank_sha256 = ""
        self._weight_manifest_sha256 = ""
        self._reference_placement: dict[str, Any] = {}
        self._detector_version = ""
        # exact integer weight-norm zero-point (set with the weights in begin())
        self._target_norm_sq = 0
        self._ref_norm_sum_sq = 0
        self._mu0 = float("nan")
        self._resumed_provenance: dict[str, Any] | None = None
        # accumulators
        self._p_target: list[int] = []
        self._p_ref_sum: list[int] = []
        self._fstat_raw: list[float] = []
        self._fstat_level_db: list[float] = []
        self._pnr_bin_db: list[float] = []
        self._snr_shelf_db: list[float] = []
        self._pilot_excess_corrected: list[float] = []  # F/mu0 - 1 (NaN if invalid)
        self._reject_mask: list[int] = []  # 1 = discard frame (positive excess)
        self._valid: list[int] = []
        self._baseband_power: list[float] = []
        self._overflow = 0
        self._n_frames = 0
        self._keys: set = set()
        self._unit_order: list[str] = []  # consumption order, for alignment keys
        # integrated power spectra (sum over valid frames of |FFT|^2 summed over
        # feeds); rectangular window. Allocated in begin() once nfft + xp are known
        # (None here so resume() can seed them before begin() moves them onto xp).
        self._xp = np  # replaced in begin() with cupy when a GPU is present
        self._spec_before = None  # xp.float64[nfft]; valid frames
        self._spec_after = None   # xp.float64[nfft]; valid AND kept (not rejected)
        # per-frame unit tags -> absolute time: t[f] = unit_time0_ctime[u]
        #   + frame_in_unit[f] * nfft * unit_delta_time[u],  u = frame_unit_index[f]
        self._frame_unit_index: list[int] = []
        self._frame_in_unit: list[int] = []
        # per-unit (per-file) time axis + provenance, aligned to _unit_order
        self._unit_time0_ctime: list[float] = []
        self._unit_time0_fpga: list[int] = []
        self._unit_event_id: list[int] = []
        self._unit_delta_time: list[float] = []
        self._unit_archive_version: list[str] = []
        # run provenance (set in begin())
        self._weights_hash = ""
        self._detector_version = ""

    # -- selection / fan-out (per CHIME coarse channel / freq_id) -----------
    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        # --select is CHIME freq_id (coarse-channel indices) -- the namespace the
        # CADC inventory and filenames key on. One freq_id is one pilot; the
        # product is labelled with the ATSC channel that pilot falls in.
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)):
            return [int(s) for s in spec]
        out: list[int] = []
        for part in str(spec).split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
        return out

    def plan_runs(self, ctx: RunContext, spec: Any) -> list:
        sel = self.resolve_selection(ctx, spec)
        if not sel:
            raise ValueError(
                "pilot-proxy-detector requires an explicit freq_id selection "
                "(e.g. select=400 or select='399,400'); it has no 'all' mode, "
                "because one product holds exactly one coarse channel (one pilot) "
                "and an unscoped run would accumulate several channels under the "
                "first file's label."
            )
        dupes = sorted({fid for fid in sel if list(sel).count(fid) > 1})
        if dupes:
            raise ValueError(
                f"pilot-proxy-detector: duplicate freq_id(s) in --select: {dupes}. Each "
                f"coarse channel must appear at most once."
            )
        return [[int(fid)] for fid in sel]

    def resume(self, path: str, ctx: RunContext) -> bool:
        """Reload a checkpointed product so a killed run continues instead of
        restarting from scratch.

        Returns False when there is no prior product (fresh build). Raises
        SystemExit when a prior product exists but was built with incompatible
        parameters -- so a capped smoke product is never silently "completed"
        by a full run, and a different channel's product is never resumed into
        this one.
        """
        p = Path(path)
        if not p.exists():
            return False
        try:
            with np.load(p, allow_pickle=False) as npz:
                data = {k: npz[k] for k in npz.files}
        except Exception as exc:  # unreadable / corrupt checkpoint
            raise SystemExit(
                f"pilot-proxy-detector: cannot read existing product {path} for "
                f"resume ({type(exc).__name__}: {exc}). Remove it to rebuild "
                f"from scratch, or point --output-dir at a clean directory."
            )

        # -- compatibility guards (refuse; never silently overwrite/complete) --
        saved_schema = str(data["schema_version"].item())
        if saved_schema != _SCHEMA_VERSION:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} has schema_version "
                f"{saved_schema!r}, but this build writes {_SCHEMA_VERSION!r}. "
                f"Remove it to rebuild."
            )
        saved_cap = int(data["max_chunks_per_file"])
        req = (ctx.options or {}).get("max_chunks_per_file", None)
        req_cap = -1 if req is None else int(req)
        if saved_cap != req_cap:
            _s = saved_cap if saved_cap >= 0 else None
            _r = req_cap if req_cap >= 0 else None
            raise SystemExit(
                f"pilot-proxy-detector: product {path} was built with "
                f"max_chunks_per_file={_s}, but this run requests {_r}. A capped "
                f"product cannot be completed by a different cap -- use a clean "
                f"--output-dir."
            )
        saved_fid = int(data["freq_id"][0])
        sel = list(getattr(ctx, "selection", None) or [])
        if sel and int(sel[0]) != saved_fid:
            raise SystemExit(
                f"pilot-proxy-detector: product {path} is freq_id {saved_fid}, but "
                f"--select requests {int(sel[0])}. Refusing to resume a "
                f"different channel's product into this one."
            )

        # -- restore identity / calibration (begin() re-derives, must agree) ---
        self._freq_id = saved_fid
        self._physical_channel = int(data["physical_channel"][0])
        self._pilot_in_band = bool(int(data["pilot_in_band"][0]))
        self._pilot_rf_hz = float(data["pilot_frequency_hz"][0])
        self._coarse_center_hz = float(data["chime_frequency_hz"][0])
        self._nfft = int(data["nfft"])
        self._K = int(data["detector_window_samples"])
        self._spectral_sense = (
            SPECTRAL_SENSE_INVERTED if int(data["sense"]) == -1
            else SPECTRAL_SENSE_NORMAL
        )
        self._pilot_below_data_db = float(data["pilot_below_data_db"])
        self._bin_enbw_hz = float(data["bin_enbw_hz"])
        self._dtv_bandwidth_hz = float(data["dtv_bandwidth_hz"])
        self._pilot_capture_efficiency = float(data["pilot_capture_efficiency"])
        self._max_chunks_per_file = None if saved_cap < 0 else saved_cap

        required_provenance = (
            "num_input_streams",
            "weights_hash",
            "weight_bank_sha256",
            "weight_manifest_sha256",
            "detector_version",
            "mask_rule",
            "detector_contract_json",
            "reference_placement_json",
        )
        missing = [name for name in required_provenance if name not in data]
        if missing:
            raise SystemExit(
                "pilot-proxy-detector: existing product lacks resume-critical provenance "
                f"fields {missing}; remove it and rebuild with the current version."
            )

        self._num_input_streams = int(
            np.asarray(data["num_input_streams"]).reshape(()).item()
        )
        if self._num_input_streams <= 0:
            raise SystemExit(
                "pilot-proxy-detector: existing product has an invalid num_input_streams; "
                "remove it and rebuild."
            )

        def _text(name: str) -> str:
            return str(np.asarray(data[name]).reshape(()).item())

        try:
            saved_contract = json.loads(_text("detector_contract_json"))
            saved_reference = json.loads(_text("reference_placement_json"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"pilot-proxy-detector: invalid resume provenance JSON in {path}: {exc}"
            ) from exc
        self._resumed_provenance = {
            "weights_hash": _text("weights_hash"),
            "detector_version": _text("detector_version"),
            "mask_rule": _text("mask_rule"),
            "target_norm_sq": int(np.asarray(data["target_norm_sq"]).reshape(-1)[0])
            if "target_norm_sq" in data else None,
            "ref_norm_sum_sq": int(np.asarray(data["ref_norm_sum_sq"]).reshape(-1)[0])
            if "ref_norm_sum_sq" in data else None,
            "detector_contract": saved_contract,
            "weight_bank_sha256": _text("weight_bank_sha256"),
            "weight_manifest_sha256": _text("weight_manifest_sha256"),
            "reference_placement": saved_reference,
        }

        # -- restore the per-frame accumulator, in stored order ----------------
        def _col(name: str) -> np.ndarray:
            return np.asarray(data[name]).reshape(-1)
        self._p_target = [int(x) for x in _col("p_target_u64")]
        self._p_ref_sum = [int(x) for x in _col("p_ref_sum_u64")]
        self._fstat_raw = [float(x) for x in _col("fstat_raw")]
        self._fstat_level_db = [float(x) for x in _col("fstat_level_db")]
        self._pnr_bin_db = [float(x) for x in _col("pnr_bin_db")]
        self._snr_shelf_db = [float(x) for x in _col("snr_shelf_db")]
        self._pilot_excess_corrected = (
            [float(x) for x in _col("pilot_excess_corrected")]
            if "pilot_excess_corrected" in data
            else [float("nan")] * len(self._snr_shelf_db)
        )
        self._reject_mask = [int(x) for x in _col("reject_mask")]
        self._valid = [int(x) for x in _col("valid")]
        self._baseband_power = [float(x) for x in _col("baseband_power_linear")]
        self._frame_unit_index = [int(x) for x in _col("frame_unit_index")]
        self._frame_in_unit = [int(x) for x in _col("frame_in_unit")]
        self._n_frames = len(self._p_target)
        self._overflow = int(data["rational_overflow_count"])
        # integrated spectra: restore on host (1-D float64); begin() moves them onto
        # the FFT backend (cupy/numpy) and resumes accumulation there.
        self._spec_before = np.asarray(
            data["integrated_spectrum_before_mask"], dtype=np.float64
        ).reshape(-1)
        self._spec_after = np.asarray(
            data["integrated_spectrum_after_mask"], dtype=np.float64
        ).reshape(-1)

        # -- restore processed keys + consumption order ------------------------
        self._keys = {str(k) for k in _col("unit_keys")}
        if "unit_order" in data:
            self._unit_order = [str(k) for k in _col("unit_order")]
        else:  # products written before unit_order was persisted
            self._unit_order = sorted(self._keys)
        # -- restore the per-unit time axis (aligned to unit_order) ------------
        self._unit_time0_ctime = [float(x) for x in _col("unit_time0_ctime")]
        self._unit_time0_fpga = [int(x) for x in _col("unit_time0_fpga")]
        self._unit_event_id = [int(x) for x in _col("unit_event_id")]
        self._unit_delta_time = [float(x) for x in _col("unit_delta_time")]
        self._unit_archive_version = [str(x) for x in _col("archive_version")]

        # let begin() verify the new input's identity matches what we restored
        self._resumed_identity = (
            self._freq_id, int(self._nfft), int(self._K), self._spectral_sense,
            self._physical_channel, self._pilot_below_data_db,
            self._bin_enbw_hz, self._dtv_bandwidth_hz,
            self._pilot_capture_efficiency,
        )
        return True

    def processed_keys(self) -> set:
        return set(self._keys)

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        """Report detector runtime problems before a long scan starts."""
        import importlib.util
        problems: list[str] = []
        opts = dict(ctx.options or {})
        # A CPU/test detector_fn replaces the GPU kernel, so it needs no cupy.
        if "detector_fn" not in opts and importlib.util.find_spec("cupy") is None:
            problems.append(
                "cupy/CUDA is not importable; pilot-proxy-detector requires a GPU runtime"
            )
        try:
            from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH
            lib_path = Path(opts.get("lib_path", DEFAULT_LIB_PATH))
            weights_path = Path(opts.get("weights_path", DEFAULT_WEIGHTS_PATH))
            if "kernel" not in opts and not lib_path.exists():
                problems.append(f"CUDA detector library not found: {lib_path}")
            if (
                "weights" not in opts
                and "weights_by_channel" not in opts
                and not weights_path.exists()
            ):
                problems.append(f"detector weight bank not found: {weights_path}")
        except Exception as exc:  # noqa: BLE001 - preflight should report, not crash.
            problems.append(f"detector preflight failed: {type(exc).__name__}: {exc}")
        return (not problems), problems

    # -- lifecycle ----------------------------------------------------------
    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        inst = ctx.instrument
        opts = dict(ctx.options or {})
        self._nfft = int(getattr(inst, "nfft", 0) or first_meta.get("nfft") or 0)
        if self._nfft <= 0:
            raise ValueError("detector analyzer: nfft must come from the instrument")
        # datatrawl exposes the spectral sense as nyquist_zone (1=normal, 2=inverted),
        # not a "sense" attribute. nyquist_sign maps that to +1/-1.
        sense = int(nyquist_sign(int(getattr(inst, "nyquist_zone", 1) or 1)))
        self._spectral_sense = (
            SPECTRAL_SENSE_INVERTED if sense == -1 else SPECTRAL_SENSE_NORMAL
        )

        # pilot / channel geometry from the channel-centre frequency
        f_center_hz = float(first_meta["f_center_hz"])
        self._coarse_center_hz = f_center_hz
        meta_streams = first_meta.get("num_input_streams")
        if meta_streams is not None:
            meta_count = int(meta_streams)
            if meta_count <= 0:
                raise ValueError("detector analyzer: num_input_streams must be positive")
            if self._num_input_streams not in (0, meta_count):
                raise SystemExit(
                    "pilot-proxy-detector: resumed product input-stream count does not "
                    f"match new input ({self._num_input_streams} != {meta_count})."
                )
            self._num_input_streams = meta_count
        physical_channel = nearest_atsc_physical_channel(f_center_hz)
        if physical_channel is None:
            raise ValueError(
                f"detector analyzer: coarse centre {f_center_hz:.0f} Hz is not near "
                "any ATSC pilot"
            )
        self._physical_channel = int(physical_channel)
        self._f0_hz = float(getattr(inst, "f0_mhz", 800.0)) * 1e6
        self._freq_id = chime_freq_id_from_hz(f_center_hz, self._f0_hz)
        self._pilot_rf_hz = float(physical_channel_to_pilot_hz(self._physical_channel))
        # Validate the *first* file against the requested freq_id -- begin() fixes
        # the product identity here, and the per-file guard only protects later
        # files against this. A mislabelled filename / wrong inventory record would
        # otherwise silently redefine the product's freq_id.
        sel = list(getattr(ctx, "selection", None) or [])
        if sel:
            requested = int(sel[0])
            if self._freq_id != requested:
                raise ValueError(
                    f"detector analyzer: --select requested freq_id {requested}, but "
                    f"the first file's centre {f_center_hz:.0f} Hz implies freq_id "
                    f"{self._freq_id}. Refusing to build a {self._freq_id} product "
                    f"under a {requested} request -- check the file naming / inventory."
                )
        # In-band test: the detector's weights target the pilot bin *inside* this
        # coarse channel. If the pilot is more than fs/2 from centre it lives in a
        # different coarse channel (a wrong --select freq_id), so the detection is
        # meaningless. Flag it; consume_file then emits an explicitly invalid
        # product (mask=0, valid=0) instead of fabricating detections.
        _coarse_width = float(getattr(inst, "fs_hz", 0.0)) or (400e6 / 1024.0)
        _expected_offset = self._pilot_rf_hz - f_center_hz
        self._pilot_in_band = abs(_expected_offset) < (_coarse_width / 2.0)
        if not self._pilot_in_band:
            warnings.warn(
                f"detector analyzer: freq_id {self._freq_id} (centre "
                f"{f_center_hz / 1e6:.4f} MHz) does not contain ATSC ch"
                f"{self._physical_channel}'s pilot -- nominal offset "
                f"{_expected_offset / 1e3:.0f} kHz exceeds +/-{_coarse_width / 2e3:.0f} "
                f"kHz. Emitting an all-invalid product (no in-band pilot); pick the "
                f"freq_id whose centre is within fs/2 of the pilot.",
                RuntimeWarning,
                stacklevel=2,
            )

        # detector backend: injectable, mirroring run_chime_analysis
        self._detector_fn = opts.get("detector_fn")
        if self._detector_fn is None:
            from pilot_proxy.chime.runner import detect_packed_for_positive_excess
            self._detector_fn = detect_packed_for_positive_excess

        self._kernel = opts.get("kernel")
        if self._kernel is None:
            from pilot_proxy.kernel import FStatKernel
            from pilot_proxy.paths import DEFAULT_LIB_PATH
            self._kernel = FStatKernel(opts.get("lib_path", DEFAULT_LIB_PATH))
        self._K = int(getattr(getattr(self._kernel, "specs", None), "K",
                              DETECTOR_WINDOW_SAMPLES))
        if int(self._nfft) % int(self._K) != 0:
            raise ValueError("detector analyzer: nfft must be divisible by kernel K")

        bank = None
        weights = opts.get("weights")
        if weights is None:
            by_channel = opts.get("weights_by_channel")
            if by_channel is not None:
                weights = by_channel.get(self._physical_channel)
        if weights is None:
            from pilot_proxy.detector_weights import DetectorWeightBank
            from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH
            bank = DetectorWeightBank(
                explicit_path=opts.get("weights_path", DEFAULT_WEIGHTS_PATH),
                expected_kernel=getattr(self._kernel, "specs", None),
            )
            weights, valid = bank.get_weights_for_physical_channel(
                self._physical_channel,
                tolerance_hz=float(
                    opts.get("pilot_frequency_tolerance_hz",
                             _DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ)
                ),
            )
            if weights is None or not valid:
                raise ValueError(
                    f"detector analyzer: no valid weights for channel "
                    f"{self._physical_channel}"
                )
        self._weights = np.ascontiguousarray(weights)
        # Exact integer squared norms of the three weight terms. mu0 =
        # 2*nt/(nl+nu) is the flat-floor H0 zero-point of F that int4 weight
        # quantization shifts away from 1; the mask rule and the corrected
        # pilot excess divide it out (see detector_contract).
        _nt, _nl, _nu = weight_term_norms_sq(self._weights)
        self._target_norm_sq = int(_nt)
        self._ref_norm_sum_sq = int(_nl + _nu)
        self._mu0 = (
            norm_corrected_mu0(self._target_norm_sq, self._ref_norm_sum_sq)
            if self._ref_norm_sum_sq > 0
            else float("nan")
        )

        time_reverse = self._spectral_sense == SPECTRAL_SENSE_INVERTED
        default_wc = (
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE if time_reverse
            else WEIGHT_COORDINATE_RAW_INPUT
        )
        if bank is not None:
            # Reuse the standalone runner's manifest/runtime coordinate validation.
            from pilot_proxy.chime.runner import (
                _reference_placement_summary,
                _weight_coordinate_metadata,
            )
            coordinate_metadata = _weight_coordinate_metadata(
                weight_bank=bank,
                input_spectral_sense=self._spectral_sense,
            )
            contract_weight_coordinate = str(
                coordinate_metadata["effective_weight_coordinate_system"]
            )
            if "weight_coordinate_system" in opts:
                requested_coordinate = normalize_weight_coordinate_system(
                    opts["weight_coordinate_system"]
                )
                if requested_coordinate != contract_weight_coordinate:
                    raise ValueError(
                        "Requested weight_coordinate_system disagrees with the "
                        f"weight manifest: {requested_coordinate!r} != "
                        f"{contract_weight_coordinate!r}."
                    )
            self._reference_placement = (
                _reference_placement_summary(
                    getattr(bank, "manifest", None),
                    [self._physical_channel],
                )
                or {}
            )
            self._weight_bank_sha256 = file_sha256(bank.path) or ""
            manifest_path = sidecar_manifest_path(bank.path)
            self._weight_manifest_sha256 = file_sha256(manifest_path) or ""
        else:
            contract_weight_coordinate = normalize_weight_coordinate_system(
                opts.get("weight_coordinate_system", default_wc)
            )
            self._reference_placement = {}
            self._weight_bank_sha256 = ""
            self._weight_manifest_sha256 = ""

        # dB-calibration constants (overridable)
        self._pilot_below_data_db = float(opts.get("pilot_below_data_db", PILOT_BELOW_DATA_DB))
        self._bin_enbw_hz = float(opts.get("bin_enbw_hz", EFFECTIVE_BIN_BW_HZ))
        self._dtv_bandwidth_hz = float(opts.get("dtv_bandwidth_hz", DTV_BANDWIDTH_HZ))
        self._pilot_capture_efficiency = float(
            opts.get("pilot_capture_efficiency", PILOT_CAPTURE_EFFICIENCY)
        )
        self._max_chunks_per_file = opts.get("max_chunks_per_file", None)

        # Detector contract, so the combine step can emit run_config/stats that
        # validate-products accepts. Built from the kernel specs + spectral sense
        # the same way run_chime_analysis builds it; the weight coordinate system
        # defaults to raw-input and is overridable via options.
        specs = getattr(self._kernel, "specs", None)
        spec_dict: dict = {}
        ser = getattr(specs, "as_descriptive_dict", None)
        if callable(ser):
            raw = ser()
            if isinstance(raw, Mapping):
                spec_dict = dict(raw)

        def _spec(name: str, attr: str, default: int) -> int:
            if name in spec_dict and spec_dict[name] is not None:
                return int(spec_dict[name])
            return int(getattr(specs, attr, default) or default)

        ref_off = _spec("reference_offset_bins", "reference_offset_bins", 2)
        self._detector_contract = build_chime_detector_contract(
            detector_window_samples=int(self._K),
            skipped_guard_bins=max(0, ref_off - 1),
            reference_offset_bins=ref_off,
            num_weight_terms=_spec("num_weight_terms", "N", 3),
            sample_bits_per_component=_spec("sample_bits_per_component", "bits", 4),
            weight_coordinate_system=contract_weight_coordinate,
            time_reverse_detector_windows_before_kernel=time_reverse,
        )

        # If this run resumed a prior product, the identity begin() just derived
        # from the first NEW file must match the restored one -- otherwise new
        # frames would be appended under a different channel/kernel geometry (or
        # dB calibration) than the existing ones, silently corrupting the
        # product. Refuse rather than mix.
        prev = getattr(self, "_resumed_identity", None)
        if prev is not None:
            now = (
                self._freq_id, int(self._nfft), int(self._K),
                self._spectral_sense, self._physical_channel,
                self._pilot_below_data_db, self._bin_enbw_hz,
                self._dtv_bandwidth_hz, self._pilot_capture_efficiency,
            )
            if now != prev:
                raise SystemExit(
                    "pilot-proxy-detector: resumed product identity does not match the "
                    "identity derived from new input (instrument / kernel / "
                    "--select / dB-calibration changed between runs). Use a clean "
                    f"--output-dir to rebuild. restored={prev} new={now}"
                )

        # -- integrated-spectrum backend + accumulators ------------------------
        # cupy when a GPU runtime is present (keeps the per-frame FFT inside the
        # download-idle GPU time on CANFAR); numpy otherwise and in tests, with
        # identical arithmetic. A fresh run allocates zeros; a resumed run has
        # had host spectra restored by resume() -- move them onto the backend.
        self._xp = _detector_fft_backend()
        if self._spec_before is None:
            self._spec_before = self._xp.zeros(int(self._nfft), dtype=self._xp.float64)
            self._spec_after = self._xp.zeros(int(self._nfft), dtype=self._xp.float64)
        else:  # resumed: arrays restored on host -> move onto the backend
            self._spec_before = self._xp.asarray(self._spec_before)
            self._spec_after = self._xp.asarray(self._spec_after)

        # -- provenance: which weights + implementation produced this product --
        self._refresh_provenance(opts)
        self._validate_resumed_provenance()

    def _refresh_provenance(self, opts: Mapping[str, Any]) -> None:
        if self._weights is None:
            raise RuntimeError("detector analyzer: weights are not initialized")
        self._weights_hash = hashlib.sha256(
            np.ascontiguousarray(self._weights).tobytes()
        ).hexdigest()
        self._detector_contract_json = json.dumps(
            self._detector_contract,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            import pilot_proxy as _pkg
            version = getattr(_pkg, "__version__", "unknown")
        except Exception:  # pragma: no cover
            version = "unknown"
        kernel_version_obj = getattr(self._kernel, "version", None)
        serializer = getattr(kernel_version_obj, "as_string", None)
        kernel_version = (
            str(serializer()) if callable(serializer) else str(kernel_version_obj or "unknown")
        )
        if "kernel" in opts:
            kernel_sha256 = "injected"
        else:
            from pilot_proxy.paths import DEFAULT_LIB_PATH
            kernel_sha256 = (
                file_sha256(opts.get("lib_path", DEFAULT_LIB_PATH)) or "unavailable"
            )
        self._detector_version = (
            f"pilot-proxy/{version} source={package_source_sha256()} "
            f"kernel={kernel_version} kernel_sha256={kernel_sha256} "
            f"{_SCHEMA_VERSION} K={int(self._K)}"
        )

    def _validate_resumed_provenance(self) -> None:
        saved = self._resumed_provenance
        if saved is None:
            return
        current = {
            "weights_hash": self._weights_hash,
            "detector_version": self._detector_version,
            "mask_rule": POSITIVE_EXCESS_MASK_RULE,
            "target_norm_sq": int(self._target_norm_sq),
            "ref_norm_sum_sq": int(self._ref_norm_sum_sq),
            "detector_contract": self._detector_contract,
            "weight_bank_sha256": self._weight_bank_sha256,
            "weight_manifest_sha256": self._weight_manifest_sha256,
            "reference_placement": self._reference_placement,
        }
        mismatches = [
            key for key in current
            if saved.get(key) != current.get(key)
        ]
        # detector_version gets token-aware treatment: its `source=` token is
        # build provenance (patches applied mid-survey change the tree hash
        # without touching detector math); the kernel hash, K, and schema
        # tokens are what resume correctness needs. Same policy as combine's
        # _check_invariants.
        if "detector_version" in mismatches:
            def _geom(v):
                return tuple(t for t in str(v).split()
                             if not t.startswith("source="))
            def _src(v):
                for t in str(v).split():
                    if t.startswith("source="):
                        return t[len("source="):][:12]
                return "?"
            sv, cv = saved.get("detector_version"), current.get("detector_version")
            if _geom(sv) == _geom(cv):
                mismatches.remove("detector_version")
                print(
                    "[resume] provenance: source build changed "
                    f"({_src(sv)} -> {_src(cv)}); detector geometry "
                    "identical, continuing.",
                    flush=True,
                )
        if mismatches:
            details = "; ".join(
                f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
                for key in mismatches
            )
            raise SystemExit(
                "pilot-proxy-detector: resumed product provenance does not match the "
                f"current detector configuration ({details}). Use a clean "
                "--output-dir to rebuild."
            )

    def _check_file_meta(self, meta: Mapping[str, Any]) -> None:
        """Reject any file whose channel/nfft does not match this product.

        begin() fixes the product identity from the first file; every later file
        must agree, or we would silently accumulate multiple channels (or a
        different FFT length) into one product. This turns that into a hard error,
        which is the real guard against a wrong --select or a filename/freq mismatch.
        """
        fc = meta.get("f_center_hz")
        if fc is not None:
            fid = chime_freq_id_from_hz(float(fc), self._f0_hz)
            if fid != self._freq_id:
                raise ValueError(
                    f"detector analyzer: file centre {float(fc):.0f} Hz is coarse "
                    f"channel freq_id {fid}, but this product is freq_id "
                    f"{self._freq_id} (ATSC ch{self._physical_channel}). Refusing "
                    f"to mix coarse channels in one product -- check the --select "
                    f"freq_id and the input file naming."
                )
        mnfft = meta.get("nfft")
        if mnfft is not None and int(mnfft) != int(self._nfft):
            raise ValueError(
                f"detector analyzer: file nfft {int(mnfft)} != product nfft "
                f"{int(self._nfft)}."
            )
        streams = meta.get("num_input_streams")
        if (
            streams is not None
            and self._num_input_streams > 0
            and int(streams) != self._num_input_streams
        ):
            raise ValueError(
                "detector analyzer: file input-stream count "
                f"{int(streams)} != product count {self._num_input_streams}."
            )

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        if self._weights is None or self._detector_fn is None:
            raise RuntimeError("detector analyzer: begin() was not called")
        self._check_file_meta(meta)
        n = 0
        # this file's position in the per-unit time axis (the unit row is appended
        # at the tail below, so len(_unit_order) is this unit's index); frame_in_unit
        # is the chunk's 0-based *time* position in the file. The packed reader yields
        # contiguous full nfft chunks, so chunk position == time position, giving each
        # frame an absolute time without storing one timestamp per frame.
        unit_idx = len(self._unit_order)
        chunk_in_unit = 0
        for chunk in arrays:
            if self._max_chunks_per_file is not None and n >= int(
                self._max_chunks_per_file
            ):
                break
            arr = np.asarray(chunk)
            if arr.ndim != 2 or arr.shape[0] != self._nfft:
                continue  # ragged/partial frame -> skip
            if arr.dtype != np.uint8:
                raise ValueError(
                    "detector analyzer requires raw uint8 chunks from the "
                    "'chime-baseband-packed' reader; got dtype "
                    f"{arr.dtype!s}. This usually means raw `datatrawl scan` "
                    "inferred the telescope's canonical 'chime-baseband' "
                    "reader from inventory metadata. Use `pilot-proxy "
                    "chime-scan`, or pass `--reader chime-baseband-packed` "
                    "when driving datatrawl directly."
                )
            chunk_streams = int(arr.shape[1])
            if self._num_input_streams == 0:
                self._num_input_streams = chunk_streams
            elif chunk_streams != self._num_input_streams:
                raise ValueError(
                    "detector analyzer: input-stream count changed within a product: "
                    f"{chunk_streams} != {self._num_input_streams}."
                )
            if not self._pilot_in_band:
                # No in-band pilot in this coarse channel: emit an explicitly
                # invalid frame (reject_mask=0, valid=0, zero powers) and skip the
                # kernel, so the product cannot be read as a real detection.
                # valid=0 keeps the frame out of both integrated spectra.
                self._p_target.append(0)
                self._p_ref_sum.append(0)
                self._fstat_raw.append(float("nan"))
                self._fstat_level_db.append(float("nan"))
                self._pnr_bin_db.append(float("nan"))
                self._snr_shelf_db.append(float("nan"))
                self._pilot_excess_corrected.append(float("nan"))
                self._reject_mask.append(0)
                self._valid.append(0)
                self._baseband_power.append(float("nan"))
                self._frame_unit_index.append(unit_idx)
                self._frame_in_unit.append(chunk_in_unit)
                n += 1
                chunk_in_unit += 1
                continue
            # reader yields native uint8 [nfft, n_feeds]; fstat wants (streams, 1, time)
            native_block = np.ascontiguousarray(arr.T)[:, np.newaxis, :]
            packed = pack_chime_block_for_detector(
                native_block,
                frame_size_samples=self._nfft,
                detector_window_samples=self._K,
                spectral_sense=self._spectral_sense,
                frames_in_chunk=1,
                sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
                selected_coarse_channel=0,   # unused on the native (lossless) path
                physical_channel=self._physical_channel,
            )
            detection = self._detector_fn(
                packed=packed.packed,
                weights=self._weights,
                kernel=self._kernel,
            )
            self._overflow += int(detection.get("rational_overflow_count", 0))
            baseband_power = np.asarray(packed.baseband_power_linear, dtype=np.float64)
            results = detection["results"]
            # The integrated spectrum (one FFT per nfft chunk) and the absolute-time
            # axis (one time per chunk) both assume one detector result per chunk --
            # which is exactly what the packed reader + frames_in_chunk=1 produce.
            # Guard it so a future packer change can't silently misalign the spectrum
            # / time axis against the per-frame arrays.
            if len(results) != 1:
                raise ValueError(
                    "detector analyzer: integrated-spectrum + time axis assume one "
                    f"detector result per nfft chunk, got {len(results)}. The "
                    "chime-baseband-packed reader yields one frame per chunk."
                )
            for local_index, row in enumerate(results):
                num = int(row.get("p_target_u64", 0))
                den = int(row.get("p_ref_sum_u64", 0))
                self._p_target.append(num)
                self._p_ref_sum.append(den)
                self._fstat_raw.append(float(fstat_num_den_to_raw(num, den)))
                self._fstat_level_db.append(
                    float(fstat_num_den_to_fstat_level_db(num, den))
                )
                pnr = float(fstat_num_den_to_pnr_bin_db(num, den))
                self._pnr_bin_db.append(pnr)
                self._snr_shelf_db.append(
                    float(pnr_bin_db_to_snr_shelf_db(
                        pnr,
                        pilot_below_data_db=self._pilot_below_data_db,
                        bin_enbw_hz=self._bin_enbw_hz,
                        dtv_bandwidth_hz=self._dtv_bandwidth_hz,
                        pilot_capture_efficiency=self._pilot_capture_efficiency,
                    ))
                )
                self._pilot_excess_corrected.append(
                    self._fstat_raw[-1] / self._mu0 - 1.0
                    if den > 0
                    else float("nan")
                )
                self._reject_mask.append(int(row.get("mask", 0)))
                self._valid.append(1 if den > 0 else 0)
                bp = (
                    float(baseband_power[local_index])
                    if local_index < baseband_power.size
                    else float("nan")
                )
                self._baseband_power.append(bp)
                self._frame_unit_index.append(unit_idx)
                self._frame_in_unit.append(chunk_in_unit)
                n += 1
            # Integrated power spectrum: one FFT (rectangular window) of this chunk's
            # raw samples, |.|^2 summed over feeds, accumulated for valid frames only.
            # spec_before = every valid frame; spec_after = valid AND kept (not
            # rejected), so (before - after) is the spectrum the mask removed. Runs on
            # the kernel's GPU (cupy) in production and numpy in tests with identical
            # arithmetic; skipped for invalid frames (which enter neither spectrum).
            if self._valid[-1]:
                a = self._xp.asarray(arr)
                X = self._xp.fft.fft(_unpack_4bit_xp(self._xp, a), axis=0)
                psd = (self._xp.abs(X) ** 2).sum(axis=1, dtype=self._xp.float64)
                self._spec_before += psd
                if not self._reject_mask[-1]:
                    self._spec_after += psd
            chunk_in_unit += 1
        self._n_frames += n
        key = meta.get("unit_key")
        if key is not None:
            self._keys.add(key)
            self._unit_order.append(str(key))
            # per-unit absolute-time axis + provenance, aligned 1:1 with _unit_order.
            # The reader's probe surfaces these from the file root attrs (NaN/0/""
            # when absent), so a missing attr degrades to an unusable-but-not-fatal
            # time rather than crashing the run.
            ev = meta.get("event_id")
            self._unit_time0_ctime.append(float(meta.get("time0_ctime", float("nan"))))
            self._unit_time0_fpga.append(int(meta.get("time0_fpga_count") or 0))
            self._unit_event_id.append(int(ev) if ev is not None else -1)
            self._unit_delta_time.append(float(meta.get("delta_time", float("nan"))))
            self._unit_archive_version.append(str(meta.get("archive_version", "")))
        return n

    def save(self, path: str) -> None:
        n = int(self._n_frames)
        frame_index = np.arange(n, dtype=np.int64)
        col_f = lambda lst: np.asarray(lst, dtype=np.float64).reshape(n, 1)
        col_u = lambda lst, dt: np.asarray(lst, dtype=dt).reshape(n, 1)
        tmp = str(path) + ".tmp.npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            # --- chime_detector_outputs schema (single pilot) ---
            physical_channel=np.asarray([self._physical_channel], dtype=np.int32),
            freq_id=np.asarray([self._freq_id], dtype=np.int64),
            pilot_in_band=np.asarray([1 if self._pilot_in_band else 0], dtype=np.uint8),
            pilot_frequency_hz=np.asarray([self._pilot_rf_hz], dtype=np.float64),
            chime_frequency_hz=np.asarray([self._coarse_center_hz], dtype=np.float64),
            frame_index=frame_index,
            p_target_u64=col_u(self._p_target, np.uint64),
            p_ref_sum_u64=col_u(self._p_ref_sum, np.uint64),
            fstat_raw=col_f(self._fstat_raw),
            fstat_level_db=col_f(self._fstat_level_db),
            pnr_bin_db=col_f(self._pnr_bin_db),
            snr_shelf_db=col_f(self._snr_shelf_db),
            pilot_excess_corrected=col_f(self._pilot_excess_corrected),
            reject_mask=col_u(self._reject_mask, np.uint8),
            valid=col_u(self._valid, np.uint8),
            # --- per-frame power + integrated spectra (rectangular window) ---
            baseband_power_linear=col_f(self._baseband_power),
            integrated_spectrum_before_mask=(
                _to_host(self._spec_before) if self._spec_before is not None
                else np.zeros(int(self._nfft), dtype=np.float64)
            ),
            integrated_spectrum_after_mask=(
                _to_host(self._spec_after) if self._spec_after is not None
                else np.zeros(int(self._nfft), dtype=np.float64)
            ),
            # --- per-frame -> per-unit time tags (see unit_* axis below) ---
            frame_unit_index=np.asarray(self._frame_unit_index, dtype=np.int32),
            frame_in_unit=np.asarray(self._frame_in_unit, dtype=np.int32),
            rational_overflow_count=np.asarray(self._overflow, dtype=np.uint64),
            # --- datatrawl provenance / resume keys ---
            schema_version=np.asarray(_SCHEMA_VERSION),
            nfft=np.asarray(int(self._nfft), dtype=np.int64),
            detector_window_samples=np.asarray(int(self._K), dtype=np.int64),
            num_input_streams=np.asarray(int(self._num_input_streams), dtype=np.int64),
            sense=np.asarray(
                -1 if self._spectral_sense == SPECTRAL_SENSE_INVERTED else 1,
                dtype=np.int64,
            ),
            unit_keys=np.asarray(sorted(str(k) for k in self._keys)),
            unit_order=np.asarray([str(k) for k in self._unit_order]),
            source_event_keys=np.asarray(
                [source_event_key(k, self._freq_id) for k in self._unit_order]
            ),
            # --- per-unit absolute-time axis (aligned to unit_order) -----------
            # per-frame time = unit_time0_ctime[u] + frame_in_unit[f]*nfft
            #                  * unit_delta_time[u],  u = frame_unit_index[f]
            unit_time0_ctime=np.asarray(self._unit_time0_ctime, dtype=np.float64),
            unit_time0_fpga=np.asarray(self._unit_time0_fpga, dtype=np.uint64),
            unit_event_id=np.asarray(self._unit_event_id, dtype=np.int64),
            unit_delta_time=np.asarray(self._unit_delta_time, dtype=np.float64),
            archive_version=np.asarray([str(s) for s in self._unit_archive_version]),
            max_chunks_per_file=np.asarray(
                -1 if self._max_chunks_per_file is None
                else int(self._max_chunks_per_file),
                dtype=np.int64,
            ),
            detector_contract_json=np.asarray(self._detector_contract_json),
            reference_placement_json=np.asarray(
                json.dumps(self._reference_placement, sort_keys=True, separators=(",", ":"))
            ),
            # dB-calibration constants -- recorded so a callable combine can refuse
            # to stack products reduced with different snr_shelf calibration.
            pilot_below_data_db=np.asarray(self._pilot_below_data_db, dtype=np.float64),
            bin_enbw_hz=np.asarray(self._bin_enbw_hz, dtype=np.float64),
            dtv_bandwidth_hz=np.asarray(self._dtv_bandwidth_hz, dtype=np.float64),
            pilot_capture_efficiency=np.asarray(
                self._pilot_capture_efficiency, dtype=np.float64
            ),
            # --- run provenance: which weights + build + mask rule produced this --
            target_norm_sq=np.asarray([self._target_norm_sq], dtype=np.int64),
            ref_norm_sum_sq=np.asarray([self._ref_norm_sum_sq], dtype=np.int64),
            mu0=np.asarray([self._mu0], dtype=np.float64),
            weights_hash=np.asarray(self._weights_hash),
            weight_bank_sha256=np.asarray(self._weight_bank_sha256),
            weight_manifest_sha256=np.asarray(self._weight_manifest_sha256),
            detector_version=np.asarray(self._detector_version),
            mask_rule=np.asarray(POSITIVE_EXCESS_MASK_RULE),
        )
        Path(tmp).replace(path)

    def summary(self) -> Mapping[str, Any]:
        masked = int(sum(self._reject_mask)) if self._reject_mask else 0
        return {
            "channel": self._physical_channel,
            "frames": self._n_frames,
            "masked_frames": masked,
        }


__all__ = ["PilotProxyDetectorAnalyzer"]
