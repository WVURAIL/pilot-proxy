# coding=utf-8
"""datatrawl Analyzer: CHIME DTV pilot frequency-offset diagnostic.

This is the offset functionality of ``pilot_proxy.chime.frequency_offset`` exposed
as a datatrawl Analyzer. It does NOT reimplement the DSP: it wraps PilotProxy's own
``accumulate_noncoherent_fft_power`` and ``estimate_peak_offset_from_power`` so
the product matches ``run_frequency_offset_diagnostic`` by construction.

Mapping onto datatrawl's contract:

* one *pilot* (one ATSC physical channel / one CHIME coarse channel) is one
  independent per-channel run -> one ``<channel>.npz`` product (``plan_runs``
  fans out, exactly like the bundled spectrum analyzer);
* the ``chime-baseband`` reader yields one ``[nfft, n_feeds]`` complex64 chunk
  per detector frame (it already unpacks offset-binary 4+4-bit with the same
  nibble/offset convention PilotProxy uses), so ``consume_file`` treats each chunk as
  one frame;
* spectral ``sense`` is applied here the way PilotProxy applies it -- as a
  time-reversal of each feed's samples *before* the FFT (sense == -1), NOT as
  the sky-axis relabelling the bundled spectrum analyzer uses. This is what keeps
  the peak offsets identical to fstat.

The expected (nominal) in-channel pilot offset is ``pilot_rf - coarse_center``
in the post-spectral-sense-normalisation RF-offset coordinate, identical to the
run loop in ``frequency_offset.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pilot_proxy.provenance import package_source_sha256
from datatrawl.instruments import nyquist_sign
from datatrawl.interfaces import Analyzer, RunContext, PluginInfo, EXPERIMENTAL

try:  # registration is best-effort; the parity harness instantiates directly too
    from datatrawl.registry import analyzer as _register_analyzer
except Exception:  # pragma: no cover - registry shape guard
    def _register_analyzer(cls):  # type: ignore[no-redef]
        return cls

# Reuse fstat's exact DSP + constants -- do not reimplement.
from pilot_proxy.chime.frequency_offset import (
    accumulate_noncoherent_fft_power,
    estimate_peak_offset_from_power,
    _window,
    COORDINATE_SYSTEM,
    DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ,
    DEFAULT_STREAM_BATCH_SIZE,
    DEFAULT_WINDOW_NAME,
)
from pilot_proxy.chime.hdf5_input import nearest_atsc_physical_channel
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz

from ._chime_coarse import chime_freq_id_from_hz, source_event_key

import warnings

_SCHEMA_VERSION = "pilotproxy_offset_datatrawl_v2"


@_register_analyzer
class PilotProxyOffsetAnalyzer(Analyzer):
    """Per-pilot CHIME frequency-offset diagnostic, parity with fstat."""
    requires_in_order = True

    info = PluginInfo(
        name="pilot-proxy-offset",
        kind="analyzer",
        summary="CHIME DTV pilot frequency-offset diagnostic "
                "(PilotProxy parity; per-channel noncoherent FFT peak offset).",
        status=EXPERIMENTAL,  # promote to READY after real-data parity vs a known run
        instruments=("chime", "kko", "gbo", "hco"),
        produces="<channel>.npz (frequency_offset_outputs schema)",
        requires=("h5py", "pilot-proxy"),
        notes="Wraps pilot_proxy.chime.frequency_offset DSP; sense applied as "
              "PilotProxy does (time-reversal before FFT), not sky-axis relabelling.",
    )

    def __init__(self) -> None:
        # set in begin()
        self._fs = 0.0
        self._nfft = 0
        self._reverse = False
        self._window: np.ndarray | None = None
        self._physical_channel = -1
        self._freq_id = -1
        self._f0_hz = 800_000_000.0
        self._pilot_in_band = True
        self._pilot_rf_hz = float("nan")
        self._coarse_center_hz = float("nan")
        self._expected_offset_hz = float("nan")
        # run options (from ctx.options, fstat defaults)
        self._half_width_hz = float(DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ)
        self._stream_batch_size = int(DEFAULT_STREAM_BATCH_SIZE)
        self._window_name = str(DEFAULT_WINDOW_NAME)
        self._backend = "auto"
        self._min_prom_db: float | None = None
        self._max_chunks_per_file: int | None = None
        self._num_input_streams = 0
        self._analyzer_version = ""
        self._resumed_identity: tuple[Any, ...] | None = None
        self._resumed_analyzer_version: str | None = None
        # accumulators
        self._peak_offset: list[float] = []
        self._freq_offset: list[float] = []
        self._peak_power: list[float] = []
        self._local_floor: list[float] = []
        self._prominence: list[float] = []
        self._valid: list[int] = []
        self._spectrum_sum: np.ndarray | None = None
        self._spectrum_count = 0
        self._n_frames = 0
        self._keys: set = set()
        self._unit_order: list[str] = []  # consumption order, for alignment keys
        self._frame_unit_index: list[int] = []
        self._frame_in_unit: list[int] = []

    # -- selection / fan-out ------------------------------------------------
    def resolve_selection(self, ctx: RunContext, spec: Any) -> Any:
        # interpret --select as explicit CHIME freq_id (coarse-channel indices) --
        # the namespace the CADC file inventory and on-disk filenames key on. One
        # freq_id is one pilot; the product is labelled with the ATSC channel that
        # pilot falls in (derived from the file's centre frequency in begin()).
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
                "pilot-proxy-offset requires an explicit freq_id selection "
                "(e.g. select=400 or select='399,400'); it has no 'all' mode, "
                "because one product holds exactly one coarse channel (one pilot) "
                "and an unscoped run would accumulate several channels under the "
                "first file's label."
            )
        dupes = sorted({fid for fid in sel if list(sel).count(fid) > 1})
        if dupes:
            raise ValueError(
                f"pilot-proxy-offset: duplicate freq_id(s) in --select: {dupes}. Each "
                f"coarse channel must appear at most once."
            )
        return [[int(fid)] for fid in sel]  # one product per coarse channel (freq_id)

    # -- resume -------------------------------------------------------------
    def resume(self, path: str, ctx: RunContext) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            with np.load(p, allow_pickle=False) as npz:
                data = {key: npz[key] for key in npz.files}
        except Exception as exc:
            raise SystemExit(
                f"pilot-proxy-offset: cannot read existing product {path} for resume "
                f"({type(exc).__name__}: {exc}). Remove it to rebuild."
            ) from exc

        saved_schema = str(np.asarray(data["schema_version"]).reshape(()).item())
        if saved_schema != _SCHEMA_VERSION:
            raise SystemExit(
                f"pilot-proxy-offset: product {path} has schema_version {saved_schema!r}, "
                f"but this build writes {_SCHEMA_VERSION!r}. Remove it to rebuild."
            )
        saved_cap = int(np.asarray(data["max_chunks_per_file"]).reshape(()))
        requested = (ctx.options or {}).get("max_chunks_per_file")
        requested_cap = -1 if requested is None else int(requested)
        if saved_cap != requested_cap:
            raise SystemExit(
                "pilot-proxy-offset: existing product and relaunch use different "
                f"max_chunks_per_file values ({saved_cap} != {requested_cap})."
            )
        saved_fid = int(np.asarray(data["freq_id"]).reshape(-1)[0])
        selection = list(getattr(ctx, "selection", None) or [])
        if selection and int(selection[0]) != saved_fid:
            raise SystemExit(
                f"pilot-proxy-offset: product is freq_id {saved_fid}, but the relaunch "
                f"requests {int(selection[0])}."
            )
        required_resume_fields = (
            "analyzer_version",
            "num_input_streams",
            "stream_batch_size",
            "offset_backend",
            "min_peak_prominence_db",
            "frame_unit_index",
            "frame_in_unit",
            "unit_order",
            "time_average_spectrum_sum_linear",
        )
        missing = [name for name in required_resume_fields if name not in data]
        if missing:
            raise SystemExit(
                "pilot-proxy-offset: existing product lacks resume-critical fields "
                f"{missing}; remove it and rebuild with the current version."
            )

        def _col(name: str) -> np.ndarray:
            return np.asarray(data[name]).reshape(-1)

        self._freq_id = saved_fid
        self._physical_channel = int(_col("physical_channel")[0])
        self._pilot_in_band = bool(int(_col("pilot_in_band")[0]))
        self._pilot_rf_hz = float(_col("pilot_frequency_hz")[0])
        self._coarse_center_hz = float(_col("coarse_channel_center_hz")[0])
        self._expected_offset_hz = float(_col("expected_pilot_offset_hz")[0])
        self._nfft = int(np.asarray(data["nfft"]).reshape(()))
        self._fs = float(np.asarray(data["fs_hz"]).reshape(()))
        self._reverse = int(np.asarray(data["sense"]).reshape(())) == -1
        self._half_width_hz = float(np.asarray(data["peak_search_half_width_hz"]).reshape(()))
        self._stream_batch_size = int(np.asarray(data["stream_batch_size"]).reshape(()))
        self._window_name = str(np.asarray(data["window_name"]).reshape(()).item())
        self._backend = str(np.asarray(data["offset_backend"]).reshape(()).item())
        min_prom = float(np.asarray(data["min_peak_prominence_db"]).reshape(()))
        self._min_prom_db = None if np.isnan(min_prom) else min_prom
        self._max_chunks_per_file = None if saved_cap < 0 else saved_cap
        self._num_input_streams = int(np.asarray(data.get("num_input_streams", 0)).reshape(()))

        self._peak_offset = [float(x) for x in _col("peak_offset_hz")]
        self._freq_offset = [float(x) for x in _col("frequency_offset_hz")]
        self._peak_power = [float(x) for x in _col("peak_power_linear")]
        self._local_floor = [float(x) for x in _col("local_floor_power_linear")]
        self._prominence = [float(x) for x in _col("peak_prominence_db")]
        self._valid = [int(x) for x in _col("valid")]
        self._frame_unit_index = [int(x) for x in _col("frame_unit_index")]
        self._frame_in_unit = [int(x) for x in _col("frame_in_unit")]
        self._n_frames = len(self._valid)
        self._spectrum_count = int(_col("time_average_spectrum_count")[0])
        self._spectrum_sum = np.asarray(
            data["time_average_spectrum_sum_linear"], dtype=np.float64
        ).reshape(-1)
        if self._spectrum_sum.shape != (self._nfft,):
            raise SystemExit(
                "pilot-proxy-offset: resumed spectrum sum shape does not match nfft"
            )
        self._keys = {str(x) for x in _col("unit_keys")}
        self._unit_order = (
            [str(x) for x in _col("unit_order")]
            if "unit_order" in data
            else sorted(self._keys)
        )
        self._resumed_identity = (
            self._freq_id, self._physical_channel, self._nfft, self._fs,
            self._reverse, self._half_width_hz, self._stream_batch_size,
            self._window_name, self._backend, self._min_prom_db,
            self._num_input_streams,
        )
        self._resumed_analyzer_version = str(
            np.asarray(data["analyzer_version"]).reshape(()).item()
        )
        return True

    def processed_keys(self) -> set:
        return set(self._keys)

    def preflight(self, ctx: RunContext) -> tuple[bool, list[str]]:
        """Report offset analyzer problems before a long scan starts."""
        import importlib.util
        problems: list[str] = []
        opts = dict(ctx.options or {})
        if importlib.util.find_spec("h5py") is None:
            problems.append("h5py is not importable; CHIME HDF5 input cannot be read")
        window_name = str(opts.get("window_name", DEFAULT_WINDOW_NAME))
        nfft = int(getattr(ctx.instrument, "nfft", 0) or 0)
        if nfft > 0:
            try:
                _window(window_name, nfft)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"invalid offset window {window_name!r}: {exc}")
        return (not problems), problems

    # -- lifecycle ----------------------------------------------------------
    def begin(self, ctx: RunContext, first_meta: Mapping[str, Any]) -> None:
        inst = ctx.instrument
        opts = dict(ctx.options or {})
        self._fs = float(inst.fs_hz)
        self._nfft = int(getattr(inst, "nfft", 0) or first_meta.get("nfft") or 0)
        if self._nfft <= 0:
            raise ValueError("offset analyzer: nfft must come from the instrument")
        # datatrawl exposes the spectral sense as nyquist_zone (1=normal, 2=inverted),
        # not a "sense" attribute. nyquist_sign maps that to +1/-1.
        sense = int(nyquist_sign(int(getattr(inst, "nyquist_zone", 1) or 1)))
        self._reverse = sense == -1  # fstat: inverted spectral sense == time reversal

        self._half_width_hz = float(
            opts.get("peak_search_half_width_hz", DEFAULT_PEAK_SEARCH_HALF_WIDTH_HZ)
        )
        self._stream_batch_size = int(
            opts.get("stream_batch_size", DEFAULT_STREAM_BATCH_SIZE)
        )
        self._window_name = str(opts.get("window_name", DEFAULT_WINDOW_NAME))
        self._backend = str(opts.get("offset_backend", opts.get("backend", "auto")))
        mpp = opts.get("min_peak_prominence_db", None)
        self._min_prom_db = None if mpp is None else float(mpp)
        self._max_chunks_per_file = opts.get("max_chunks_per_file", None)
        self._window = _window(self._window_name, self._nfft)

        f_center_hz = float(first_meta["f_center_hz"])
        self._coarse_center_hz = f_center_hz
        physical_channel = nearest_atsc_physical_channel(f_center_hz)
        if physical_channel is None:
            raise ValueError(
                f"offset analyzer: coarse centre {f_center_hz:.0f} Hz is not near "
                "any ATSC pilot"
            )
        self._physical_channel = int(physical_channel)
        self._f0_hz = float(getattr(inst, "f0_mhz", 800.0)) * 1e6
        self._freq_id = chime_freq_id_from_hz(f_center_hz, self._f0_hz)
        # Validate the *first* file against the requested freq_id. begin() fixes the
        # product identity from this file; if a mislabelled filename, a wrong
        # inventory record, or a source/reader mismatch hands us the wrong coarse
        # channel, the product would silently become a different freq_id than the
        # one --select planned. Guard the initial assignment (the per-file guard
        # only protects subsequent files, against this established identity).
        sel = list(getattr(ctx, "selection", None) or [])
        if sel:
            requested = int(sel[0])
            if self._freq_id != requested:
                raise ValueError(
                    f"offset analyzer: --select requested freq_id {requested}, but "
                    f"the first file's centre {f_center_hz:.0f} Hz implies freq_id "
                    f"{self._freq_id}. Refusing to build a {self._freq_id} product "
                    f"under a {requested} request -- check the file naming / inventory."
                )
        self._pilot_rf_hz = float(physical_channel_to_pilot_hz(self._physical_channel))
        self._expected_offset_hz = self._pilot_rf_hz - f_center_hz
        # The peak search only has FFT bins to look at inside this coarse
        # channel's Nyquist span (+/- fs/2). If the nominal pilot offset exceeds
        # that, the pilot lives in a *different* coarse channel (a wrong --select
        # freq_id), so there is nothing in-band to find: mark it and skip the
        # search per frame rather than letting estimate_peak_offset_from_power
        # raise on a sub-three-bin window.
        self._pilot_in_band = abs(self._expected_offset_hz) < (self._fs / 2.0)
        if not self._pilot_in_band:
            warnings.warn(
                f"offset analyzer: freq_id {self._freq_id} (centre "
                f"{f_center_hz / 1e6:.4f} MHz) does not contain ATSC ch"
                f"{self._physical_channel}'s pilot -- nominal offset "
                f"{self._expected_offset_hz / 1e3:.0f} kHz exceeds +/-{self._fs / 2e3:.0f} "
                f"kHz. Emitting an all-invalid product (no in-band pilot); pick the "
                f"freq_id whose centre is within fs/2 of the pilot.",
                RuntimeWarning,
                stacklevel=2,
            )
        current_identity = (
            self._freq_id, self._physical_channel, self._nfft, self._fs,
            self._reverse, self._half_width_hz, self._stream_batch_size,
            self._window_name, self._backend, self._min_prom_db,
            self._num_input_streams,
        )
        if self._resumed_identity is not None and current_identity != self._resumed_identity:
            raise SystemExit(
                "pilot-proxy-offset: resumed product configuration does not match the "
                f"current run. saved={self._resumed_identity} current={current_identity}"
            )
        try:
            import pilot_proxy as _pkg
            version = getattr(_pkg, "__version__", "unknown")
        except Exception:  # pragma: no cover
            version = "unknown"
        self._analyzer_version = (
            f"pilot-proxy/{version} source={package_source_sha256()} {_SCHEMA_VERSION}"
        )
        if (
            self._resumed_analyzer_version is not None
            and self._resumed_analyzer_version != self._analyzer_version
        ):
            raise SystemExit(
                "pilot-proxy-offset: analyzer implementation changed since the checkpoint; "
                "use a clean output directory."
            )
        if self._spectrum_sum is None:
            self._spectrum_sum = np.zeros(self._nfft, dtype=np.float64)
            self._spectrum_count = 0
        elif self._spectrum_sum.shape != (self._nfft,):
            raise SystemExit("pilot-proxy-offset: resumed spectrum shape does not match nfft")

    def _check_file_meta(self, meta: Mapping[str, Any]) -> None:
        """Reject any file whose channel/nfft does not match this product.

        See the detector analyzer for the rationale: this is the hard guard against
        a wrong selection or a filename/freq mismatch silently mixing channels.
        """
        fc = meta.get("f_center_hz")
        if fc is not None:
            fid = chime_freq_id_from_hz(float(fc), self._f0_hz)
            if fid != self._freq_id:
                raise ValueError(
                    f"offset analyzer: file centre {float(fc):.0f} Hz is coarse "
                    f"channel freq_id {fid}, but this product is freq_id "
                    f"{self._freq_id} (ATSC ch{self._physical_channel}). Refusing "
                    f"to mix coarse channels in one product -- check the --select "
                    f"freq_id and the input file naming."
                )
        mnfft = meta.get("nfft")
        if mnfft is not None and int(mnfft) != int(self._nfft):
            raise ValueError(
                f"offset analyzer: file nfft {int(mnfft)} != product nfft "
                f"{int(self._nfft)}."
            )

    def consume_file(self, arrays: Iterable, meta: Mapping[str, Any]) -> int:
        if self._window is None or self._spectrum_sum is None:
            raise RuntimeError("offset analyzer: begin() was not called")
        self._check_file_meta(meta)
        n = 0
        unit_idx = len(self._unit_order)
        chunk_in_unit = 0
        for chunk in arrays:
            if self._max_chunks_per_file is not None and n >= int(
                self._max_chunks_per_file
            ):
                break
            arr = np.asarray(chunk)
            if arr.ndim != 2 or arr.shape[0] != self._nfft:
                continue  # ragged/partial frame -> skip (engine drops final partial)
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(
                    "offset analyzer requires complex chunks from the "
                    "'chime-baseband' reader; got dtype "
                    f"{arr.dtype!s}. Use `pilot-proxy chime-scan`, or pass "
                    "`--reader chime-baseband` when driving datatrawl directly."
                )
            chunk_streams = int(arr.shape[1])
            if self._num_input_streams == 0:
                self._num_input_streams = chunk_streams
            elif chunk_streams != self._num_input_streams:
                raise ValueError(
                    "offset analyzer: input-stream count changed within a product: "
                    f"{chunk_streams} != {self._num_input_streams}."
                )
            # reader yields [nfft, n_feeds]; fstat DSP wants (streams, time)
            streams = np.ascontiguousarray(arr.T)
            if self._reverse:
                streams = np.ascontiguousarray(streams[:, ::-1])
            power_sum, _ = accumulate_noncoherent_fft_power(
                streams,
                window=self._window,
                stream_batch_size=self._stream_batch_size,
                backend=self._backend,
            )
            if self._pilot_in_band:
                est = estimate_peak_offset_from_power(
                    power_sum,
                    sample_rate_hz=self._fs,
                    expected_offset_hz=self._expected_offset_hz,
                    fft_size=self._nfft,
                    peak_search_half_width_hz=self._half_width_hz,
                )
                fo = float(est["frequency_offset_hz"])
                prom = float(est["peak_prominence_db"])
                valid = bool(np.isfinite(fo))
                if self._min_prom_db is not None:
                    valid = valid and prom >= self._min_prom_db
                self._peak_offset.append(float(est["peak_offset_hz"]))
                self._freq_offset.append(fo)
                self._peak_power.append(float(est["peak_power_linear"]))
                self._local_floor.append(float(est["local_floor_power_linear"]))
                self._prominence.append(prom)
                self._valid.append(1 if valid else 0)
            else:
                # No in-band pilot in this coarse channel: record an invalid frame
                # (NaN offsets, valid=0) but still accumulate the spectrum so the
                # diagnostic shows what *is* in the band.
                self._peak_offset.append(float("nan"))
                self._freq_offset.append(float("nan"))
                self._peak_power.append(float("nan"))
                self._local_floor.append(float("nan"))
                self._prominence.append(float("nan"))
                self._valid.append(0)
            self._spectrum_sum += power_sum
            self._spectrum_count += 1
            self._frame_unit_index.append(unit_idx)
            self._frame_in_unit.append(chunk_in_unit)
            chunk_in_unit += 1
            n += 1
        self._n_frames += n
        key = meta.get("unit_key")
        if key is not None:
            self._keys.add(key)
            self._unit_order.append(str(key))
        return n

    # -- save ---------------------------------------------------------------
    def save(self, path: str) -> None:
        nfft = int(self._nfft)
        n_frames = int(self._n_frames)
        frame_index = np.arange(n_frames, dtype=np.int64)
        relative_time_s = frame_index.astype(np.float64) * float(nfft) / float(self._fs)
        col = lambda lst, dtype: np.asarray(lst, dtype=dtype).reshape(n_frames, 1)
        fft_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(self._fs)))
        if self._spectrum_sum is None or self._spectrum_count == 0:
            spectrum = np.full((1, nfft), np.nan, dtype=np.float64)
        else:
            spectrum = (
                self._spectrum_sum / float(self._spectrum_count)
            ).reshape(1, nfft).astype(np.float64)
        tmp = str(path) + ".tmp.npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            # --- fstat frequency_offset_outputs schema (single pilot) ---
            physical_channel=np.asarray([self._physical_channel], dtype=np.int32),
            freq_id=np.asarray([self._freq_id], dtype=np.int64),
            pilot_in_band=np.asarray(
                [1 if self._pilot_in_band else 0], dtype=np.uint8
            ),
            pilot_frequency_hz=np.asarray([self._pilot_rf_hz], dtype=np.float64),
            chime_frequency_hz=np.asarray([self._coarse_center_hz], dtype=np.float64),
            coarse_channel_center_hz=np.asarray(
                [self._coarse_center_hz], dtype=np.float64
            ),
            expected_pilot_offset_hz=np.asarray(
                [self._expected_offset_hz], dtype=np.float64
            ),
            frame_index=frame_index,
            frame_unit_index=np.asarray(self._frame_unit_index, dtype=np.int32),
            frame_in_unit=np.asarray(self._frame_in_unit, dtype=np.int32),
            relative_time_s=relative_time_s,
            peak_offset_hz=col(self._peak_offset, np.float64),
            frequency_offset_hz=col(self._freq_offset, np.float64),
            peak_power_linear=col(self._peak_power, np.float64),
            local_floor_power_linear=col(self._local_floor, np.float64),
            peak_prominence_db=col(self._prominence, np.float64),
            valid=col(self._valid, np.uint8),
            fft_frequency_axis_hz=fft_axis.astype(np.float64),
            time_average_spectrum_power_linear=spectrum,
            time_average_spectrum_sum_linear=(
                np.zeros((1, nfft), dtype=np.float64)
                if self._spectrum_sum is None
                else np.asarray(self._spectrum_sum, dtype=np.float64).reshape(1, nfft)
            ),
            time_average_spectrum_count=np.asarray(
                [self._spectrum_count], dtype=np.uint64
            ),
            fft_size=np.asarray(nfft, dtype=np.int64),
            fft_bin_width_hz=np.asarray(float(self._fs) / float(nfft), dtype=np.float64),
            sample_rate_hz=np.asarray(float(self._fs), dtype=np.float64),
            window_name=np.asarray(str(self._window_name)),
            peak_search_half_width_hz=np.asarray(
                float(self._half_width_hz), dtype=np.float64
            ),
            coordinate_system=np.asarray(COORDINATE_SYSTEM),
            # --- datatrawl provenance / resume keys ---
            schema_version=np.asarray(_SCHEMA_VERSION),
            nfft=np.asarray(nfft, dtype=np.int64),
            fs_hz=np.asarray(float(self._fs), dtype=np.float64),
            sense=np.asarray(-1 if self._reverse else 1, dtype=np.int64),
            num_input_streams=np.asarray(self._num_input_streams, dtype=np.int64),
            stream_batch_size=np.asarray(self._stream_batch_size, dtype=np.int64),
            offset_backend=np.asarray(self._backend),
            min_peak_prominence_db=np.asarray(
                np.nan if self._min_prom_db is None else self._min_prom_db,
                dtype=np.float64,
            ),
            analyzer_version=np.asarray(self._analyzer_version),
            unit_keys=np.asarray(sorted(str(k) for k in self._keys)),
            unit_order=np.asarray([str(k) for k in self._unit_order]),
            source_event_keys=np.asarray(
                [source_event_key(k, self._freq_id) for k in self._unit_order]
            ),
            max_chunks_per_file=np.asarray(
                -1 if self._max_chunks_per_file is None
                else int(self._max_chunks_per_file),
                dtype=np.int64,
            ),
        )
        Path(tmp).replace(path)

    def summary(self) -> Mapping[str, Any]:
        return {
            "channel": self._physical_channel,
            "frames": self._n_frames,
            "expected_offset_hz": self._expected_offset_hz,
        }


__all__ = ["PilotProxyOffsetAnalyzer"]
