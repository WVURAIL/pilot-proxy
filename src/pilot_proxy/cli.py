# coding=utf-8
"""CLI for the standalone PilotProxy pilot detector testbench."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, cast

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    WEIGHT_COORDINATE_RAW_INPUT,
)
from pilot_proxy.dtv_units import (
    DETECTOR_WINDOW_SAMPLES,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
)
from pilot_proxy.integration import (
    DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    DEFAULT_CHIME_STREAM_MAP,
    DEFAULT_DETECTOR_CORE_PROFILE,
    default_reference_receiver_profile,
    layout_uint64_bound_check,
    load_detector_core_profile,
    load_receiver_profile,
    load_stream_map,
    parse_physical_channel_selection,
    write_weight_bank_from_receiver_profile,
)
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.paths import (
    DEFAULT_LIB_PATH,
    DEFAULT_WEIGHTS_PATH,
    GENERATED_DIR,
)
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# GNU Radio is typically installed into the system Python on Linux/WSL, while
# CUDA/CuPy lives in the active detector environment.
DEFAULT_GNURADIO_PYTHON = "/usr/bin/python3"
DEFAULT_CLEAN_ATSC_IQ_SAMPLES = 600_000
DEFAULT_TS_PACKETS = 4096
# Deterministic defaults keep quickstart output reproducible.
DEFAULT_GENERATOR_SEED = 12345
DEFAULT_EVALUATOR_SEED = 20260529
DEFAULT_NOISE_TRIALS = 10
DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_NUM_INPUT_STREAMS = 1
DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0
HZ_PER_MHZ = 1.0e6


def _run_module(
    module_name: str,
    args: list[str],
    *,
    python: str | None = None,
    env_updates: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not existing_pythonpath
        else str(SRC_ROOT) + os.pathsep + existing_pythonpath
    )
    if env_updates:
        env.update(env_updates)

    cmd = [str(python or sys.executable), "-m", module_name, *args]
    print(f"[run] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit={result.returncode}: {cmd}")


def _cmd_generate_atsc(args: argparse.Namespace) -> None:
    argv = [
        "--output-iq",
        args.output_iq,
        "--output-ts",
        args.output_ts,
        "--num-iq-samples",
        str(args.num_iq_samples),
        "--num-ts-packets",
        str(args.num_ts_packets),
        "--seed",
        str(args.seed),
    ]
    env_updates = {}
    if not args.python_user_site:
        env_updates["PYTHONNOUSERSITE"] = "1"
    _run_module(
        "pilot_proxy.testbench.generate_atsc_signal",
        argv,
        python=args.gnuradio_python,
        env_updates=env_updates,
    )


def _cmd_evaluate_snr(args: argparse.Namespace) -> None:
    from pilot_proxy.testbench.evaluate_snr import run as evaluate_snr_run

    # The subparser inherits the testbench parser via parents=, so the parsed
    # namespace is exactly what the testbench expects: call it directly
    # instead of hand-rebuilding argv (the old pattern silently dropped any
    # option added to the testbench but not mirrored here).
    code = evaluate_snr_run(args)
    if code:
        raise SystemExit(code)


def _cmd_audit_atsc(args: argparse.Namespace) -> None:
    argv = [
        "--input-iq",
        args.input_iq,
        "--output-json",
        args.output_json,
    ]
    _run_module("pilot_proxy.testbench.audit_atsc_signal", argv)


def _cmd_atsc_detector_input(args: argparse.Namespace) -> None:
    argv = [
        "--input-iq",
        args.input_iq,
        "--output-dir",
        args.output_dir,
        "--frame-size-samples",
        str(args.samples_per_block),
        "--num-input-streams",
        str(args.num_input_streams),
        "--experimental-detector-window-samples",
        str(args.detector_window_samples),
    ]
    if args.save_channelized:
        argv.append("--save-channelized")
    if args.physical_channel is not None:
        argv.extend(["--physical-channel", str(args.physical_channel)])
    _run_module("pilot_proxy.testbench.quantize", argv)


def _cmd_detect(args: argparse.Namespace) -> None:
    argv = [
        "--input-detector-matrix",
        args.input_detector_matrix,
        "--output-json",
        args.output_json,
        "--lib-path",
        args.lib_path,
        "--weights-path",
        args.weights_path,
        "--pilot-below-data-db",
        str(args.pilot_below_data_db),
        "--bin-enbw-hz",
        str(args.bin_enbw_hz),
        "--pilot-capture-efficiency",
        str(args.pilot_capture_efficiency),
        "--dtv-bandwidth-hz",
        str(args.dtv_bandwidth_hz),
        "--pilot-frequency-tolerance-hz",
        str(args.pilot_frequency_tolerance_hz),
        "--num-input-streams",
        str(args.num_input_streams),
    ]
    if args.frame_size_samples is not None:
        argv.extend(["--frame-size-samples", str(args.frame_size_samples)])
    if args.physical_channel is not None:
        argv.extend(["--physical-channel", str(args.physical_channel)])
    if args.dtv_pilot_mhz is not None:
        argv.extend(["--dtv-pilot-mhz", str(args.dtv_pilot_mhz)])
    _run_module("pilot_proxy.detect", argv)


def _cmd_plot_results(args: argparse.Namespace) -> None:
    argv = [
        "--input-csv",
        args.input_csv,
        "--output-png",
        args.output_png,
        "--title",
        args.title,
        "--smooth-window",
        str(args.smooth_window),
    ]
    if args.show:
        argv.append("--show")
    _run_module("pilot_proxy.testbench.plot_results", argv)


def _cmd_summarize_results(args: argparse.Namespace) -> None:
    argv = [
        "--input",
        args.input,
        "--output-dir",
        args.output_dir,
        "--bins",
        str(args.bins),
        "--histograms",
        str(args.histograms),
    ]
    _run_module("pilot_proxy.testbench.summarize_results", argv)


def _add_optional_arg(
    argv: list[str],
    flag: str,
    value: object | None,
) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _parse_chime_scan_set_options(values: Sequence[str] | None) -> dict[str, object]:
    """Parse repeated --set key=value options for chime-scan analyzer_options."""
    out: dict[str, object] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        text = raw.strip()
        low = text.lower()
        if low in {"true", "false"}:
            value: object = low == "true"
        elif low in {"none", "null"}:
            value = None
        else:
            for cast in (int, float):
                try:
                    value = cast(text)
                    break
                except ValueError:
                    pass
            else:
                value = text
        if not key:
            raise SystemExit(f"--set key may not be empty in {item!r}")
        out[key] = value
    return out


def _cmd_chime_inspect(args: argparse.Namespace) -> None:
    argv = [str(args.input_dir), "--max-files", str(args.max_files)]
    _add_optional_arg(argv, "--dataset-path", args.dataset_path)
    _add_optional_arg(argv, "--filename-pattern", args.filename_pattern)
    _run_module("pilot_proxy.chime.inspect", argv)


def _cmd_chime_run(args: argparse.Namespace) -> None:
    argv = [
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--receiver-profile",
        str(args.receiver_profile),
        "--weights-path",
        str(args.weights_path),
        "--lib-path",
        str(args.lib_path),
        "--frame-size-samples",
        str(args.frame_size_samples),
        "--frames-per-chunk",
        str(args.frames_per_chunk),
        "--pilot-below-data-db",
        str(args.pilot_below_data_db),
        "--bin-enbw-hz",
        str(args.bin_enbw_hz),
        "--dtv-bandwidth-hz",
        str(args.dtv_bandwidth_hz),
        "--pilot-capture-efficiency",
        str(args.pilot_capture_efficiency),
        "--calibration-seconds",
        str(args.calibration_seconds),
    ]
    if args.stream_map is not None:
        argv.extend(["--stream-map", str(args.stream_map)])
    _add_optional_arg(argv, "--dataset-path", args.dataset_path)
    _add_optional_arg(argv, "--filename-pattern", args.filename_pattern)
    for channel in args.physical_channel or []:
        argv.extend(["--physical-channel", str(channel)])
    _add_optional_arg(argv, "--physical-channel-range", args.physical_channel_range)
    _add_optional_arg(argv, "--max-frames", args.max_frames)
    if args.plot:
        argv.append("--plot")
    _run_module("pilot_proxy.chime.runner", argv)


def _cmd_chime_combine(args: argparse.Namespace) -> None:
    try:
        from pilot_proxy.datatrawl_plugins.combine import combine_detector_products
    except ImportError as exc:
        raise SystemExit(
            "chime-combine needs the datatrawl integration installed "
            "(pip install -e path/to/datatrawl alongside pilot-proxy)."
        ) from exc

    if args.products:
        paths = [Path(p) for p in args.products]
    else:
        paths = sorted(Path(args.work_dir).glob(args.glob))
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"chime-combine: missing product file(s): {missing}")
    if not paths:
        raise SystemExit(
            f"chime-combine: no per-pilot products matched {args.glob!r} under "
            f"{args.work_dir}"
        )
    outputs = combine_detector_products(paths, args.output_dir)
    print(f"Combined {len(paths)} pilot product(s) -> {args.output_dir}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def _cmd_inject_pilot_tone(args: argparse.Namespace) -> None:
    from pilot_proxy.chime.injection import inject_directory

    if args.input.is_dir():
        paths = sorted(args.input.glob(args.glob))
        if not paths:
            raise SystemExit(f"no files matching {args.glob!r} under {args.input}")
    else:
        paths = [args.input]
    entries = inject_directory(
        paths,
        args.output_dir,
        amplitude_lsb=args.amplitude_lsb,
        phase_seed=args.phase_seed,
        baseband_frequency_hz=args.baseband_frequency_hz,
        pilot_frequency_hz=args.pilot_frequency_hz,
        physical_channel=args.physical_channel,
    )
    total_clip = sum(entry["clip_count"] for entry in entries)
    print(
        f"Injected {len(entries)} file(s) -> {args.output_dir} "
        f"(amplitude {args.amplitude_lsb} LSB, total clipped samples {total_clip})"
    )


def _cmd_analyze_injection_recovery(args: argparse.Namespace) -> None:
    from pilot_proxy.chime.injection_recovery import main as recovery_main

    argv: list[str] = []
    for point in args.points:
        argv += ["--point", str(point)]
    for pfa in args.false_alarm_rates or []:
        argv += ["--false-alarm-rate", str(pfa)]
    argv += ["--output-dir", str(args.output_dir)]
    raise SystemExit(recovery_main(argv))


def _cmd_analyze_cleaning_tradeoff(args: argparse.Namespace) -> None:
    from pilot_proxy.chime.cleaning_tradeoff import main as tradeoff_main

    argv: list[str] = ["--run-dir", str(args.run_dir),
                       "--excess-db-start", str(args.excess_db_start),
                       "--excess-db-stop", str(args.excess_db_stop),
                       "--excess-db-step", str(args.excess_db_step)]
    if args.control_run_dir is not None:
        argv += ["--control-run-dir", str(args.control_run_dir)]
    if args.survey_hours is not None:
        argv += ["--survey-hours", str(args.survey_hours)]
    if args.output_dir is not None:
        argv += ["--output-dir", str(args.output_dir)]
    raise SystemExit(tradeoff_main(argv))


def _cmd_chime_plot(args: argparse.Namespace) -> None:
    argv = ["--run-dir", str(args.run_dir)]
    if args.clean_figures:
        argv.append("--clean-figures")
    _run_module("pilot_proxy.chime.plots", argv)


def _cmd_validate_products(args: argparse.Namespace) -> None:
    argv = [
        "--run-dir",
        str(args.run_dir),
    ]
    _add_optional_arg(argv, "--output-json", args.output_json)
    _run_module("pilot_proxy.chime.validate_products", argv)


def _cmd_chime_scan(args: argparse.Namespace) -> None:
    try:
        from pilot_proxy.datatrawl_plugins.scan import run_chime_scan
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "datatrawl":
            raise SystemExit(
                "chime-scan needs datatrawl installed (it is not on PyPI): "
                "pip install -e path/to/datatrawl, then pip install -e . here."
            )
        raise  # a real import error inside pilot-proxy -- don't mask it as 'install datatrawl'
    analyzer_options = _parse_chime_scan_set_options(args.set_option)
    for key, value in {
        "weights_path": args.weights_path,
        "lib_path": args.lib_path,
        "weight_coordinate_system": args.weight_coordinate_system,
        "pilot_frequency_tolerance_hz": args.pilot_frequency_tolerance_hz,
        "pilot_below_data_db": args.pilot_below_data_db,
        "bin_enbw_hz": args.bin_enbw_hz,
        "dtv_bandwidth_hz": args.dtv_bandwidth_hz,
        "pilot_capture_efficiency": args.pilot_capture_efficiency,
    }.items():
        if value is not None:
            analyzer_options[key] = value
    run_chime_scan(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source=args.source,
        analyzer=args.analyzer,
        select=args.select,
        instrument=args.instrument,
        max_files=args.max_files,
        max_chunks_per_file=args.max_chunks_per_file,
        checkpoint_every=args.checkpoint_every,
        inventory=args.inventory,
        inventory_name=args.inventory_name,
        source_root=args.source_root,
        work_dir=args.work_dir,
        source_glob=args.source_glob,
        source_channel_regex=args.source_channel_regex,
        analyzer_options=analyzer_options,
    )


def _cmd_check_layout(args: argparse.Namespace) -> None:
    profile = (
        load_receiver_profile(args.receiver_profile)
        if args.receiver_profile is not None
        else default_reference_receiver_profile(
            frame_size_samples=int(
                DEFAULT_FRAME_SIZE_SAMPLES
                if args.samples_per_block is None
                else args.samples_per_block
            ),
            num_input_streams=int(
                DEFAULT_NUM_INPUT_STREAMS
                if args.num_input_streams is None
                else args.num_input_streams
            ),
        )
    )
    stream_map = None if args.stream_map is None else load_stream_map(args.stream_map)
    frame_size_samples = int(
        profile.frame_size_samples
        if args.samples_per_block is None
        else args.samples_per_block
    )
    num_input_streams = int(
        profile.num_input_streams
        if args.num_input_streams is None
        else args.num_input_streams
    )
    if stream_map is not None:
        num_input_streams = int(stream_map.num_streams)
    check = layout_uint64_bound_check(
        frame_size_samples=frame_size_samples,
        detector_window_samples=int(args.detector_window_samples),
        num_input_streams=num_input_streams,
        num_selected_channels=int(args.num_selected_channels),
    )
    payload = {
        "schema_version": "fstat_layout_check_v1",
        "receiver_profile": profile.to_dict(),
        "receiver_profile_nested": profile.to_nested_dict(),
        "stream_map": None if stream_map is None else stream_map.to_dict(),
        "layout_check": check,
    }
    if args.output_json:
        write_json_strict(Path(args.output_json), payload, indent=2, sort_keys=True)
        print(f"Wrote {args.output_json}")
    print("detector_rows_per_frame, power_sum_fits_uint64, recommended_batching")
    print(
        f"{int(check['detector_rows_per_frame'])}, "
        f"{bool(check['power_sum_fits_uint64'])}, "
        f"{check['recommended_batching']}"
    )


def _cmd_check_profile(args: argparse.Namespace) -> None:
    profile = load_receiver_profile(args.receiver_profile)
    detector_core = (
        None
        if args.detector_core_profile is None
        else load_detector_core_profile(args.detector_core_profile)
    )
    stream_map = None if args.stream_map is None else load_stream_map(args.stream_map)
    payload = {
        "schema_version": "fstat_profile_check_v1",
        "receiver_profile": profile.to_dict(),
        "receiver_profile_nested": profile.to_nested_dict(),
        "detector_core_profile": (
            None if detector_core is None else detector_core.to_dict()
        ),
        "stream_map": None if stream_map is None else stream_map.to_dict(),
    }
    if args.output_json:
        write_json_strict(Path(args.output_json), payload, indent=2, sort_keys=True)
        print(f"Wrote {args.output_json}")
    print("receiver_profile_id, num_input_streams, frame_size_samples, spectral_sense")
    print(
        f"{profile.name}, {int(profile.num_input_streams)}, "
        f"{int(profile.frame_size_samples)}, {profile.spectral_sense}"
    )


def _cmd_make_weights(args: argparse.Namespace) -> None:
    profile = load_receiver_profile(args.receiver_profile)
    detector_core = load_detector_core_profile(args.detector_core_profile)
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    manifest = write_weight_bank_from_receiver_profile(
        output_path=args.output,
        profile=profile,
        core=detector_core,
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output}.manifest.json")
    print("physical_channels, receiver_profile_hash")
    print(
        f"{','.join(str(channel) for channel in physical_channels)}, "
        f"{manifest['receiver_profile_hash']}"
    )


def _cmd_export_runtime_weight_bundle(args: argparse.Namespace) -> None:
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=Path(args.receiver_profile),
        detector_core_profile_path=Path(args.detector_core_profile),
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
        output_dir=Path(args.output_dir),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


def _cmd_validate_runtime_weight_bundle(args: argparse.Namespace) -> None:
    report = validate_runtime_weight_bundle(
        bundle_dir=Path(args.bundle_dir),
        output_json=args.output_json,
    )
    print("valid, num_errors, bundle_dir", flush=True)
    print(
        f"{bool(report['valid'])}, {int(report['num_errors'])}, {report['bundle_dir']}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"ERROR {error['check']}: {error['message']}", flush=True)
    if not report["valid"]:
        raise SystemExit(1)


def _cmd_list_channels(args: argparse.Namespace) -> None:
    weights = DetectorWeightBank(explicit_path=args.weights_path)
    print(
        "physical_channel,pilot_mhz,coarse_channel_index,"
        "adaptive_reference_placement"
    )
    for channel in weights.supported_physical_channels():
        layout = weights.layout_for_physical_channel(channel)
        coarse_channel_index = cast(int, layout["coarse_channel_index"])
        adaptive_reference_placement = cast(
            bool,
            layout.get("adaptive_reference_placement", False),
        )
        print(
            f"{channel},"
            f"{physical_channel_to_pilot_hz(channel) / HZ_PER_MHZ:.6f},"
            f"{int(coarse_channel_index)},"
            f"{bool(adaptive_reference_placement)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone CUDA F-statistic DTV pilot detector testbench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from pilot_proxy import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_command(name: str, summary: str) -> argparse.ArgumentParser:
        # One string serves as both the one-liner in `pilot-proxy --help` and
        # the description at the top of `pilot-proxy <command> --help`.
        return subparsers.add_parser(name, help=summary, description=summary)

    gen = _add_command(
        "generate-atsc",
        "Generate a clean ATSC 8-VSB IQ capture (and transport stream) "
        "with GNU Radio.",
    )
    gen.add_argument(
        "--output-iq",
        default=str(GENERATED_DIR / "atsc" / "atsc_8vsb_complex64.cfile"),
    )
    gen.add_argument(
        "--output-ts",
        default=str(GENERATED_DIR / "atsc" / "atsc_test_pattern.ts"),
    )
    gen.add_argument(
        "--num-iq-samples", type=int, default=DEFAULT_CLEAN_ATSC_IQ_SAMPLES
    )
    gen.add_argument("--num-ts-packets", type=int, default=DEFAULT_TS_PACKETS)
    gen.add_argument("--seed", type=int, default=DEFAULT_GENERATOR_SEED)
    gen.add_argument(
        "--gnuradio-python",
        default=DEFAULT_GNURADIO_PYTHON,
        help="Python executable that can import GNU Radio.",
    )
    gen.add_argument(
        "--python-user-site",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the GNU Radio Python to load user-site packages.",
    )
    gen.set_defaults(func=_cmd_generate_atsc)

    from pilot_proxy.testbench.evaluate_snr import (
        build_parser as _evaluate_snr_parser,
    )

    eval_parser = subparsers.add_parser(
        "evaluate-snr",
        help=(
            "Sweep detection rate versus shelf SNR: inject AWGN into a clean "
            "IQ capture and run the detector at each level."
        ),
        description=(
            "Sweep detection rate versus shelf SNR: inject AWGN into a clean "
            "IQ capture and run the detector at each level. Options are the "
            "testbench evaluator's, inherited directly (single source of "
            "truth)."
        ),
        parents=[_evaluate_snr_parser(add_help=False)],
    )
    eval_parser.set_defaults(func=_cmd_evaluate_snr)

    plot = _add_command(
        "plot-results",
        "Plot detection-rate curves from an evaluate-snr summary CSV.",
    )
    plot.add_argument(
        "--input-csv",
        default=str(GENERATED_DIR / "dtv_snr_eval" / "dtv_snr_summary.csv"),
    )
    plot.add_argument(
        "--output-png",
        default=str(GENERATED_DIR / "dtv_snr_eval" / "dtv_snr_sweep.png"),
    )
    plot.add_argument("--title", default="PilotProxy SNR sweep")
    plot.add_argument("--smooth-window", type=int, default=1)
    plot.add_argument("--show", action="store_true")
    plot.set_defaults(func=_cmd_plot_results)

    summarize = _add_command(
        "summarize-results",
        "Write text and histogram summaries from an evaluate-snr JSON report.",
    )
    summarize.add_argument(
        "--input",
        default=str(GENERATED_DIR / "dtv_snr_eval" / "dtv_snr_eval.json"),
    )
    summarize.add_argument(
        "--output-dir",
        default=str(GENERATED_DIR / "summary"),
    )
    summarize.add_argument("--bins", type=int, default=40)
    summarize.add_argument(
        "--histograms",
        choices=["auto", "always", "never"],
        default="auto",
    )
    summarize.set_defaults(func=_cmd_summarize_results)

    audit = _add_command(
        "audit-atsc",
        "Verify a generated IQ capture: pilot frequency and level, occupied "
        "bandwidth, shelf flatness, edge rolloff.",
    )
    audit.add_argument(
        "--input-iq",
        default=str(GENERATED_DIR / "atsc" / "atsc_8vsb_complex64.cfile"),
    )
    audit.add_argument(
        "--output-json",
        default=str(GENERATED_DIR / "atsc" / "atsc_waveform_audit.json"),
    )
    audit.set_defaults(func=_cmd_audit_atsc)

    quantize = _add_command(
        "quantize",
        "Channelize and pack a clean IQ capture into the int4 "
        "detector-input matrix.",
    )
    quantize.add_argument(
        "--input-iq",
        default=str(GENERATED_DIR / "atsc" / "atsc_8vsb_complex64.cfile"),
    )
    quantize.add_argument(
        "--output-dir",
        default=str(GENERATED_DIR / "detector_input"),
    )
    quantize.add_argument(
        "--frame-size-samples",
        dest="samples_per_block",
        type=int,
        default=DEFAULT_FRAME_SIZE_SAMPLES,
        help="Frame size, in channelized samples, to pack per detector block.",
    )
    quantize.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
        help="Number of input streams/feeds to combine in each detector block.",
    )
    quantize.add_argument(
        "--experimental-detector-window-samples",
        dest="detector_window_samples",
        type=int,
        default=DETECTOR_WINDOW_SAMPLES,
        help="Advanced: v0.1 only accepts the locked value 128.",
    )
    quantize.add_argument("--physical-channel", type=int, default=None)
    quantize.add_argument("--save-channelized", action="store_true")
    quantize.set_defaults(func=_cmd_atsc_detector_input)

    detect = _add_command(
        "detect",
        "Run the CUDA detector once over a packed detector matrix and write "
        "a detections JSON.",
    )
    detect.add_argument(
        "--input-detector-matrix",
        default=str(GENERATED_DIR / "detector_input" / "detector_matrix_i4.npy"),
        help="Packed int4 detector matrix written by `quantize`.",
    )
    detect.add_argument(
        "--output-json",
        default=str(GENERATED_DIR / "detections" / "detect.json"),
    )
    detect.add_argument(
        "--threshold-snr-shelf-db",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    detect.add_argument("--physical-channel", type=int, default=None)
    detect.add_argument("--dtv-pilot-mhz", type=float, default=None)
    detect.add_argument(
        "--pilot-below-data-db",
        dest="pilot_below_data_db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
        help="Positive dB offset: ATSC pilot power below average data-shelf power.",
    )
    detect.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    detect.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    detect.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    detect.add_argument("--frame-size-samples", type=int, default=None)
    detect.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
    )
    detect.add_argument(
        "--max-denominator",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    detect.add_argument(
        "--pilot-frequency-tolerance-hz",
        type=float,
        default=DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    )
    detect.add_argument("--lib-path", default=str(DEFAULT_LIB_PATH))
    detect.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    detect.set_defaults(func=_cmd_detect)

    channels = _add_command(
        "list-channels",
        "List the ATSC channels present in a weight bank (CSV on stdout).",
    )
    channels.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    channels.set_defaults(func=_cmd_list_channels)

    check_profile = _add_command(
        "check-profile",
        "Validate a receiver-profile JSON (and optional stream map) against "
        "the detector-core contract.",
    )
    check_profile.add_argument("--receiver-profile", required=True,
                               help="Receiver-profile JSON to validate.")
    check_profile.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    check_profile.add_argument("--stream-map", default=None)
    check_profile.add_argument("--output-json", default=None)
    check_profile.set_defaults(func=_cmd_check_profile)

    check_layout = _add_command(
        "check-layout",
        "Report the detector row layout and the uint64 accumulator-bound "
        "check for a frame geometry.",
    )
    check_layout.add_argument("--receiver-profile", default=None)
    check_layout.add_argument("--stream-map", default=None)
    check_layout.add_argument(
        "--frame-size-samples",
        dest="samples_per_block",
        type=int,
        default=None,
    )
    check_layout.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=None,
    )
    check_layout.add_argument(
        "--num-selected-channels",
        type=int,
        default=1,
    )
    check_layout.add_argument(
        "--detector-window-samples",
        type=int,
        default=DETECTOR_WINDOW_SAMPLES,
    )
    check_layout.add_argument("--output-json", default=None)
    check_layout.set_defaults(func=_cmd_check_layout)

    make_weights = _add_command(
        "make-weights",
        "Generate a packed int4 detector weight bank from a receiver profile.",
    )
    make_weights.add_argument("--receiver-profile", required=True,
                              help="Receiver-profile JSON describing the "
                                   "channelizer geometry.")
    make_weights.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    make_weights.add_argument(
        "--physical-channel",
        type=int,
        action="append",
        default=None,
        help="ATSC physical channel to include. Repeat as needed.",
    )
    make_weights.add_argument(
        "--physical-channel-range",
        default=None,
        help="Inclusive range like 14:36, comma list, or single channel.",
    )
    make_weights.add_argument(
        "--weight-coordinate-system",
        choices=[
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
        help=(
            "Coordinate convention used to generate the weights. Use "
            "post_spectral_sense_normalization for detector-coordinate weights "
            "after any spectral-sense normalization, or raw_input_frequency_coordinate "
            "for native receiver-coordinate weights."
        ),
    )
    make_weights.add_argument("--output", required=True,
                              help="Output path for the packed weight bank "
                                   "(.bin); a .manifest.json is written "
                                   "beside it.")
    make_weights.set_defaults(func=_cmd_make_weights)

    export_bundle = _add_command(
        "export-runtime-weight-bundle",
        "Export the compact runtime bundle (weights plus manifests) for "
        "deployment handoff.",
    )
    export_bundle.add_argument("--receiver-profile", required=True,
                               help="Receiver-profile JSON describing the "
                                    "channelizer geometry.")
    export_bundle.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    export_bundle.add_argument(
        "--weight-coordinate-system",
        choices=[
            WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
    )
    export_bundle.add_argument(
        "--physical-channel",
        type=int,
        action="append",
        default=None,
        help="ATSC physical channel to include. Repeat as needed.",
    )
    export_bundle.add_argument(
        "--physical-channel-range",
        default=None,
        help="Inclusive range like 14:36, comma list, or single channel.",
    )
    export_bundle.add_argument("--output-dir", required=True,
                               help="Directory to write the runtime bundle "
                                    "into.")
    export_bundle.set_defaults(func=_cmd_export_runtime_weight_bundle)

    validate_bundle = _add_command(
        "validate-runtime-weight-bundle",
        "Re-validate a bundle directory written by "
        "export-runtime-weight-bundle.",
    )
    validate_bundle.add_argument("--bundle-dir", type=Path, required=True,
                                 help="Bundle directory to validate.")
    validate_bundle.add_argument("--output-json", type=Path, default=None)
    validate_bundle.set_defaults(func=_cmd_validate_runtime_weight_bundle)

    chime_inspect = _add_command(
        "chime-inspect",
        "Summarize CHIME baseband HDF5 files in a directory (shapes, dtypes, "
        "frequency metadata).",
    )
    chime_inspect.add_argument("--input-dir", type=Path, required=True,
                               help="Directory of CHIME baseband .h5 files.")
    chime_inspect.add_argument("--max-files", type=int, default=20)
    chime_inspect.add_argument("--dataset-path", default=None)
    chime_inspect.add_argument("--filename-pattern", default=None)
    chime_inspect.set_defaults(func=_cmd_chime_inspect)

    chime_run = _add_command(
        "chime-run",
        "Run the detector over a directory of already-staged CHIME baseband "
        "HDF5 files (batch runner; for archive-scale runs use chime-scan).",
    )
    chime_run.add_argument("--input-dir", type=Path, required=True,
                           help="Directory of CHIME baseband .h5 files.")
    chime_run.add_argument("--output-dir", type=Path, required=True,
                           help="Run directory for products and figures.")
    chime_run.add_argument(
        "--receiver-profile",
        type=Path,
        default=DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
    )
    chime_run.add_argument(
        "--stream-map",
        type=Path,
        default=DEFAULT_CHIME_STREAM_MAP,
    )
    chime_run.add_argument(
        "--weights-path",
        dest="weights_path",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
    )
    chime_run.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    chime_run.add_argument("--dataset-path", default=None)
    chime_run.add_argument("--filename-pattern", default="*.h5")
    chime_run.add_argument(
        "--physical-channel",
        type=int,
        action="append",
        default=None,
    )
    chime_run.add_argument("--physical-channel-range", default=None)
    chime_run.add_argument(
        "--frame-size-samples",
        type=int,
        default=DEFAULT_FRAME_SIZE_SAMPLES,
    )
    chime_run.add_argument("--frames-per-chunk", type=int, default=1)
    chime_run.add_argument("--max-frames", type=int, default=None)
    chime_run.add_argument(
        "--pilot-below-data-db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
    )
    chime_run.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    chime_run.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    chime_run.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    chime_run.add_argument("--calibration-seconds", type=float, default=2.0)
    chime_run.add_argument("--plot", action="store_true")
    chime_run.set_defaults(func=_cmd_chime_run)

    chime_combine = _add_command(
        "chime-combine",
        "Stack per-pilot detector products (work-dir <freq_id>.npz files) "
        "into the canonical combined products. Same combine the scan runs at "
        "completion; usable mid-survey on completed channels or snapshots.",
    )
    combine_source = chime_combine.add_mutually_exclusive_group(required=True)
    combine_source.add_argument("--work-dir", type=Path, default=None,
                                help="Directory of per-pilot <freq_id>.npz products.")
    combine_source.add_argument("--product", type=Path, action="append",
                                dest="products", default=None,
                                help="Explicit per-pilot product path (repeatable).")
    chime_combine.add_argument("--glob", default="*.npz",
                               help="Product glob under --work-dir.")
    chime_combine.add_argument("--output-dir", type=Path, required=True,
                               help="Where the combined canonical products are written.")
    chime_combine.set_defaults(func=_cmd_chime_combine)

    inject = _add_command(
        "inject-pilot-tone",
        "Copy real baseband files with a known pilot tone injected "
        "(integer-domain; amplitude 0 is a byte-identical control).",
    )
    inject.add_argument("--input", type=Path, required=True,
                        help="A baseband .h5 file or a directory of them.")
    inject.add_argument("--glob", default="*.h5",
                        help="Filename glob when --input is a directory.")
    inject.add_argument("--output-dir", type=Path, required=True)
    inject.add_argument("--amplitude-lsb", type=float, required=True,
                        help="Tone amplitude in raw 4-bit LSB units; 0 = identity control.")
    inject_frequency = inject.add_mutually_exclusive_group(required=True)
    inject_frequency.add_argument("--baseband-frequency-hz", type=float, default=None)
    inject_frequency.add_argument("--pilot-frequency-hz", type=float, default=None)
    inject_frequency.add_argument("--physical-channel", type=int, default=None)
    inject.add_argument("--phase-seed", type=int, default=20260701)
    inject.set_defaults(func=_cmd_inject_pilot_tone)

    recovery = _add_command(
        "analyze-injection-recovery",
        "Analyze injection-ladder run products: recovery linearity plus the "
        "F-statistic vs radiometer comparison at matched false-alarm rates.",
    )
    recovery.add_argument("--point", type=Path, action="append", required=True,
                          dest="points",
                          help="Ladder-point run dir (repeat; one must be a=0).")
    recovery.add_argument("--false-alarm-rate", type=float, action="append",
                          dest="false_alarm_rates", default=None)
    recovery.add_argument("--output-dir", type=Path, required=True)
    recovery.set_defaults(func=_cmd_analyze_injection_recovery)

    tradeoff = _add_command(
        "analyze-cleaning-tradeoff",
        "Post-hoc mask-threshold sweep over stored products: operating "
        "curve and recovered-bandwidth headline (exact x=0 anchor).",
    )
    tradeoff.add_argument("--run-dir", type=Path, required=True)
    tradeoff.add_argument("--control-run-dir", type=Path, default=None)
    tradeoff.add_argument("--excess-db-start", type=float, default=0.0)
    tradeoff.add_argument("--excess-db-stop", type=float, default=12.0)
    tradeoff.add_argument("--excess-db-step", type=float, default=0.5)
    tradeoff.add_argument("--survey-hours", type=float, default=None)
    tradeoff.add_argument("--output-dir", type=Path, default=None)
    tradeoff.set_defaults(func=_cmd_analyze_cleaning_tradeoff)

    chime_plot = _add_command(
        "chime-plot",
        "Regenerate the diagnostic figures for a completed CHIME run "
        "directory.",
    )
    chime_plot.add_argument("--run-dir", type=Path, required=True,
                            help="Run directory written by chime-run or "
                                 "chime-scan.")
    chime_plot.add_argument(
        "--clean-figures",
        action="store_true",
        help="Delete known CHIME figures before regenerating plots.",
    )
    chime_plot.set_defaults(func=_cmd_chime_plot)

    validate_products = _add_command(
        "validate-products",
        "Validate a CHIME run directory's products against the v2 product "
        "schema.",
    )
    validate_products.add_argument("--run-dir", type=Path, required=True,
                                   help="Run directory written by chime-run "
                                        "or chime-scan.")
    validate_products.add_argument("--output-json", type=Path, default=None,
                                   help="Also write the validation report as "
                                        "JSON.")
    validate_products.set_defaults(func=_cmd_validate_products)

    chime_scan = _add_command(
        "chime-scan",
        "Stream CHIME data through the datatrawl engine (one resumable "
        "product per pilot), then combine into the canonical products. The "
        "recommended archive-scale entry point; chime-run remains for "
        "pre-staged local directories.",
    )
    chime_scan.add_argument("--input-dir", type=Path, default=None,
                            help="Directory of CHIME baseband .h5 files (required for "
                                 "--source local).")
    chime_scan.add_argument("--output-dir", type=Path, required=True,
                            help="Where the combined canonical products are written.")
    chime_scan.add_argument("--inventory", type=Path, default=None,
                            help="Explicit CADC inventory.jsonl path for "
                                 "--source cadc-datatrail. Overrides --inventory-name.")
    chime_scan.add_argument(
        "--inventory-name", "--name",
        dest="inventory_name",
        default=None,
        help="Named datatrawl inventory under <survey-root>/data/<name>/inventory.jsonl. "
             "Use --source-root to set <survey-root>; default is the current directory.",
    )
    chime_scan.add_argument("--source-root", type=Path, default=None,
                            help="For --source local, an input directory alternative "
                                 "to --input-dir. For --source cadc-datatrail, the "
                                 "datatrawl survey root used with --inventory-name, "
                                 "or the legacy root containing "
                                 "data/<instrument>/inventory.jsonl.")
    chime_scan.add_argument("--source", choices=["local", "cadc-datatrail"],
                            default="local",
                            help="local: files on disk; cadc-datatrail: stream from the archive.")
    chime_scan.add_argument("--analyzer", choices=["pilot-proxy-detector"],
                            default="pilot-proxy-detector")
    chime_scan.add_argument("--select", required=True,
                            help="CHIME freq_id (coarse-channel indices): '844' "
                                 "or '829,844'. This is the namespace the CADC "
                                 "file inventory and on-disk filenames key on; "
                                 "one freq_id is one pilot. Each product is labelled "
                                 "with the ATSC channel that pilot falls in (derived "
                                 "from the file's centre frequency). Required: there "
                                 "is no 'all' mode, because one product holds exactly "
                                 "one coarse channel and an unscoped run would mix "
                                 "several under the first file's label. For the "
                                 "DTV 14-36 pilot set, see README.md or INTEGRATION.md.")
    chime_scan.add_argument("--instrument", default="chime")
    chime_scan.add_argument("--max-files", type=int, default=None,
                            help="Cap files processed per pilot (smoke tests).")
    chime_scan.add_argument("--max-chunks-per-file", type=int, default=None)
    chime_scan.add_argument("--checkpoint-every", type=int, default=None,
                            help="Write the per-pilot product every N files "
                                 "(default 50); resume reloads the last checkpoint. "
                                 "Lower = less redo after a kill, more I/O.")
    chime_scan.add_argument("--work-dir", type=Path, default=None,
                            help="Per-pilot products + staging (default: <output-dir>/_per_pilot).")
    chime_scan.add_argument("--source-glob", default="*.h5")
    chime_scan.add_argument("--source-channel-regex", default=None,
                            help="Override the filename->freq_id regex for --source local.")
    chime_scan.add_argument("--weights-path", type=Path, default=None)
    chime_scan.add_argument("--lib-path", type=Path, default=None)
    chime_scan.add_argument(
        "--weight-coordinate-system",
        choices=[WEIGHT_COORDINATE_POST_SPECTRAL_SENSE, WEIGHT_COORDINATE_RAW_INPUT],
        default=None,
    )
    chime_scan.add_argument(
        "--pilot-frequency-tolerance-hz",
        type=float,
        default=DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    )
    chime_scan.add_argument("--pilot-below-data-db", type=float, default=None)
    chime_scan.add_argument("--bin-enbw-hz", type=float, default=None)
    chime_scan.add_argument("--dtv-bandwidth-hz", type=float, default=None)
    chime_scan.add_argument("--pilot-capture-efficiency", type=float, default=None)
    chime_scan.add_argument(
        "--set", dest="set_option", action="append", default=[], metavar="KEY=VALUE",
        help="Analyzer option passed through to chime-scan ctx.options; repeat as needed.",
    )
    chime_scan.set_defaults(func=_cmd_chime_scan)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
