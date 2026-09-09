#!/usr/bin/env python3
"""Engineering-only CPU/CUDA audit and bounded literal full-geometry benchmark.

Uses one previously prepared deterministic signal template across all weight
profiles solely to exercise arithmetic. It does not estimate a detection curve
or claim that the channel-29 template represents another RF allocation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import shutil
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.testbench.evaluate_snr import _ideal_float_weights_from_layout
from pilot_proxy.testbench.fine_validation_gpu import (
    FineValidationEngine,
    audit_against_cpu,
    draw_noise,
    prepare_null_input,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=REPO / "weights/chime_dtv_weights_k128.bin"
    )
    parser.add_argument("--lib", type=Path, default=REPO / "cuda/libfstatistic.so")
    parser.add_argument("--full-channels", type=int, nargs="+", default=[14, 29, 36])
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--null-group-channels", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("engineering output must be a new directory")
    if args.timing_repeats < 1 or args.timing_repeats > 20:
        raise ValueError("bounded timing repetitions must lie in 1..20")
    import cupy as cp
    import pilot_proxy.testbench.fine_validation_gpu as module

    kernel = FStatKernel(args.lib)
    gpu = SimpleNamespace(cp=cp, kernel=kernel)
    bank = DetectorWeightBank(explicit_path=args.weights)
    with np.load(args.signal_cache, allow_pickle=False) as data:
        signal, tone = data["atsc_rows"], data["ideal_tone_rows"]
    channels = list(bank.supported_physical_channels())
    args.null_group_channels = (
        args.full_channels
        if args.null_group_channels is None
        else args.null_group_channels
    )
    if not set(args.null_group_channels) <= set(channels):
        raise ValueError("unsupported null timing profile")
    if not set(args.full_channels) <= set(channels):
        raise ValueError("unsupported full-audit profile")

    def profile(channel):
        packed, valid = bank.get_weights_for_physical_channel(channel)
        if not valid:
            raise ValueError("invalid packed profile")
        return SimpleNamespace(
            packed_weights=packed,
            ideal_weights=_ideal_float_weights_from_layout(
                bank.layout_for_physical_channel(channel), detector_window_samples=128
            ),
        )

    args.output.mkdir(parents=True)
    plan = {
        "schema": "fine-validation-engineering-preflight-v1",
        "scope": "engineering_only_not_science",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profiles_small_M8": channels,
        "profiles_full_M2048": args.full_channels,
        "null_group_channels": args.null_group_channels,
        "timing_repeats": args.timing_repeats,
        "base_seed": 2026090917000000,
        "template_note": "One channel-29 cached ATSC/tone template across profiles exercises arithmetic only",
        "input_sha256": {
            str(p.resolve()): sha(p)
            for p in (
                args.signal_cache,
                args.weights,
                args.lib,
                Path(__file__),
                Path(module.__file__),
            )
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "kernel_version": kernel.version.as_string(),
            "kernel_specs": kernel.specs.as_descriptive_dict(),
            "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
    }
    snapshot = args.output / "source_snapshots"
    snapshot.mkdir()
    for source in (
        Path(__file__),
        Path(module.__file__),
        REPO / "tests/testbench/test_fine_validation_gpu.py",
    ):
        shutil.copyfile(source, snapshot / source.name)
    write(args.output / "plan.json", plan)
    audits, timings, outputs = [], [], {}
    for streams, selected in ((8, channels), (2048, args.full_channels)):
        for channel in selected:
            engine = FineValidationEngine(
                profile(channel), signal, tone, gpu=gpu, num_streams=streams
            )
            seed = plan["base_seed"] + streams * 100 + channel
            t0 = time.perf_counter()
            result = engine.evaluate(seed, 0.03, audit=True)
            audit = audit_against_cpu(engine, result)
            audit.update(
                channel=channel, wall_seconds_including_cpu=time.perf_counter() - t0
            )
            audits.append(audit)
            print(
                f"CPU/GPU audit M={streams} ch{channel} passed in {audit['wall_seconds_including_cpu']:.3f}s",
                flush=True,
            )
            for stage, values in result["powers_by_stage"].items():
                outputs[f"M{streams}_ch{channel}_fine_{stage}"] = values
            for stage, values in result["coarse_by_stage"].items():
                outputs[f"M{streams}_ch{channel}_coarse_{stage}"] = values
            del result
            if streams == 2048:
                engine.evaluate(seed + 1000000, 0.03)  # explicit engineering warmup
                elapsed = []
                for index in range(args.timing_repeats):
                    cp.cuda.get_current_stream().synchronize()
                    started = time.perf_counter()
                    frame = engine.evaluate(seed + 2000000 + index, 0.03)
                    cp.cuda.get_current_stream().synchronize()
                    elapsed.append(time.perf_counter() - started)
                    del frame
                timings.append(
                    {
                        "channel": channel,
                        "num_streams": streams,
                        "seconds_per_frame": elapsed,
                        "median_seconds": float(np.median(elapsed)),
                        "gpu_memory_pool_used_bytes": cp.get_default_memory_pool().used_bytes(),
                        "gpu_memory_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
                    }
                )
                print(
                    f"Literal six-stage ladder ch{channel}: {np.median(elapsed):.4f}s/frame",
                    flush=True,
                )
            del engine
            cp.get_default_memory_pool().free_all_blocks()
    # Engineering-only paired-profile timings include source generation and all
    # arithmetic. Shared noise is deliberately the same draw across profiles.
    grouped = []
    engines = {
        channel: FineValidationEngine(
            profile(channel), signal, tone, gpu=gpu, num_streams=2048
        )
        for channel in args.null_group_channels
    }
    for index in range(args.timing_repeats):
        seed = plan["base_seed"] + 9000000 + index
        for sharing in ("fresh", "raw_noise", "prepared_null"):
            shared = sharing == "raw_noise"
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            draw = draw_noise(seed, num_streams=2048, cp=cp) if shared else None
            prepared = (
                prepare_null_input(seed, num_streams=2048, cp=cp)
                if sharing == "prepared_null"
                else None
            )
            per_profile = []
            for channel, engine in engines.items():
                t0 = time.perf_counter()
                result = engine.evaluate(seed, 0, noise_draw=draw, null_input=prepared)
                per_profile.append(time.perf_counter() - t0)
                del result
            cp.cuda.get_current_stream().synchronize()
            grouped.append(
                {
                    "seed": seed,
                    "shared_noise": shared,
                    "sharing_mode": sharing,
                    "channels": args.null_group_channels,
                    "total_seconds": time.perf_counter() - started,
                    "per_profile_seconds": per_profile,
                }
            )
            del draw, prepared
    channel = args.full_channels[0]
    engine = engines[channel]
    prepared = prepare_null_input(plan["base_seed"] + 9990000, num_streams=2048, cp=cp)
    from pilot_proxy.fine_reduction import independent_bin_mask

    decision = {
        "anchor_bin": 0,
        "designated_half_width": 2,
        "bulk_mask": independent_bin_mask(256, designated_bins=[0]),
        "cfar_rank": 62,
        "multiplier_q16": 65536,
    }
    result = engine.evaluate(
        prepared.noise_draw.seed, 0, null_input=prepared, decision=decision, audit=True
    )
    audit = audit_against_cpu(engine, result)
    audit.update(
        channel=channel,
        prepared_null_input=True,
        device_q16_checked=result["device_q16_checked"],
    )
    audits.append(audit)
    del result, prepared
    print("Null shared/fresh profile-group timings: " + json.dumps(grouped), flush=True)
    np.savez_compressed(args.output / "powers.npz", **outputs)
    report = {
        "scope": plan["scope"],
        "passed": True,
        "audits": audits,
        "timings": timings,
        "null_profile_group_timings": grouped,
        "plan_sha256": sha(args.output / "plan.json"),
        "powers_sha256": sha(args.output / "powers.npz"),
        "scientific_certification": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write(args.output / "report.json", report)
    print(
        json.dumps(
            {"passed": True, "audits": len(audits), "timings": timings}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
