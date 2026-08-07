#!/usr/bin/env python3
# coding=utf-8
"""Emit the fxfft master twiddle table as C source.

The device and reference implementations must use the same twiddles as the
Python reference, and must do so without runtime trigonometry -- a libm call
on the device would reintroduce exactly the cross-platform float
non-determinism the frozen specification exists to eliminate. So the table is
generated here, once, from ``pilot_proxy.fxfft.MASTER_TWIDDLE_Q15``, and
emitted as source literals with the master's hash stamped into the file.

Because ``W_n[k] = W_M[k * M / n]`` exactly (see fxfft.py), one master serves
every supported transform length: a length-n transform indexes the master with
stride ``M / n``, which folds into the existing shift form -- stage ``s`` of a
length-n radix-2 DIT reads ``master[t << (log2(M) - s)]``. That is the same
expression the frozen 256-point code already uses with ``log2(256)``; only the
constant changes. No new rounding decisions are introduced at any size.

Usage:
    python3 tools/emit_fxfft_tables.py [--check] [-o cuda/fxfft_master_twiddle.h]

``--check`` regenerates in memory and diffs against the file on disk, exiting
non-zero if they differ. Wire it into CI so the committed header cannot drift
from the Python master.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pilot_proxy.fxfft import (  # noqa: E402
    MASTER_N,
    MASTER_TWIDDLE_Q15,
    MASTER_TWIDDLE_SHA256,
    master_twiddle_sha256,
)

DEFAULT_OUT = REPO_ROOT / "cuda" / "fxfft_master_twiddle.h"


def render() -> str:
    assert master_twiddle_sha256() == MASTER_TWIDDLE_SHA256, "master hash mismatch"
    half = MASTER_N // 2
    log2_master = MASTER_N.bit_length() - 1

    lines = [
        "/* GENERATED FILE -- do not edit by hand.",
        " *",
        " * Emitted by tools/emit_fxfft_tables.py from",
        " * pilot_proxy.fxfft.MASTER_TWIDDLE_Q15. Regenerate with:",
        " *",
        " *   python3 tools/emit_fxfft_tables.py",
        " *",
        " * and verify with --check (CI gate).",
        " *",
        " * Q15 twiddles W[k] = e^{-2 pi i k / FX_MASTER_N} for k < FX_MASTER_N/2:",
        " *   C[k] = nearest(32768 cos(2 pi k / FX_MASTER_N))",
        " *   S[k] = nearest(-32768 sin(2 pi k / FX_MASTER_N))",
        " *",
        " * A length-n radix-2 DIT (n a power of two dividing FX_MASTER_N) reads",
        " * stage s as master[t << (FX_MASTER_LOG2 - s)]. Entries for length n are",
        " * an exact decimation of this table -- identical integers, so adding a",
        " * transform length introduces no new rounding decisions.",
        " *",
        " * C[0] = +32768 exceeds int16; entries are stored in int32.",
        " *",
        f" * master sha256: {MASTER_TWIDDLE_SHA256}",
        " */",
        "#ifndef FXFFT_MASTER_TWIDDLE_H",
        "#define FXFFT_MASTER_TWIDDLE_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define FX_MASTER_N {MASTER_N}",
        f"#define FX_MASTER_LOG2 {log2_master}",
        f'#define FX_MASTER_SHA256 "{MASTER_TWIDDLE_SHA256}"',
        "",
        f"static const int32_t FX_TW_MASTER[{half}][2] = {{",
    ]
    per_line = 4
    for start in range(0, half, per_line):
        chunk = MASTER_TWIDDLE_Q15[start : start + per_line]
        body = ", ".join("{%d, %d}" % (c, s) for c, s in chunk)
        lines.append(f"    {body},")
    lines[-1] = lines[-1].rstrip(",")
    lines += ["};", "", "#endif /* FXFFT_MASTER_TWIDDLE_H */", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify the file on disk matches the generator")
    args = ap.parse_args()

    text = render()
    if args.check:
        if not args.output.is_file():
            print(f"FAIL: {args.output} does not exist", file=sys.stderr)
            return 1
        if args.output.read_text() != text:
            print(f"FAIL: {args.output} differs from the generator; "
                  f"run python3 tools/emit_fxfft_tables.py", file=sys.stderr)
            return 1
        print(f"OK: {args.output.name} matches the Python master "
              f"({MASTER_TWIDDLE_SHA256[:12]})")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output} ({MASTER_N // 2} entries, "
          f"master {MASTER_TWIDDLE_SHA256[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
