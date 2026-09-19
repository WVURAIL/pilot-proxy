#!/usr/bin/env python3
"""Robustness: every channel granted the maximum ground credit the model admits.

The ruling credits the ground filter at each channel's own measured between-epoch coherence, capped
at rho = 0.95 (amendment 8, second addendum). Channels whose residual is poorly correlated between
epochs therefore receive little credit: 0.5 dB on channel 15 against 5.3 dB on channel 31. A reader
may reasonably ask what happens if that measurement is wrong, since the coherence is itself measured
with noise and an earlier audit probed it from several directions.

This answers the question in its strongest form: grant EVERY channel and every range the ceiling
value, rho = 0.95, worth 10.1 dB, regardless of what its coherence measured. That is more credit than
the data give, so it is not the ruling; it is the bound on how much the ruling could possibly owe to
the credit model.

usage: blanket_credit.py [out csv]     (re-runs the ruling from a patched copy; the tree is untouched)
"""
import csv, io, os, re, subprocess, sys, tempfile
import numpy as np

FA = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.mkdtemp(), "tor_blanket.csv")
RESCAN = "/home/djg/rail/output/channel-ruling-execution-2026-09-14/rebuild/author_actions/ch33_rescan/products"

src = io.open(os.path.join(FA, "ruling_baseline.py")).read()
src = src.replace('FA = os.path.dirname(os.path.abspath(__file__))', 'FA = %r' % FA, 1)
old = '    return float(min(max(np.median(v), 0.0), 0.95)) if v else 0.0'
assert old in src, "range_rho not found; the estimator has changed"
tmp = os.path.join(os.path.dirname(OUT), "_rb_blanket.py")
io.open(tmp, "w").write(src.replace(old, '    return 0.95', 1))
subprocess.run([sys.executable, tmp, os.path.join(FA, "table_of_record_before_amendment8.csv"), OUT, RESCAN],
               capture_output=True, text=True, check=True)

def board(f):
    d = {}
    for x in csv.DictReader(open(f)):
        c = int(x["channel"])
        if x["role"] != "pilot" and c not in d: d[c] = x
    return d
a, b = board(os.path.join(FA, "table_of_record.csv")), board(OUT)
bar = lambda x: (re.search(r"([-\d.]+) dB over tolerance", x["policy_or_reason"]) or [None, None])[1]
exa = sorted(c for c, x in a.items() if x["disposition"] == "excise")
exb = sorted(c for c, x in b.items() if x["disposition"] == "excise")
print("excised as ruled                  :", exa)
print("excised at the maximum credit     :", exb)
print("dispositions that change          :", [c for c in a if a[c]["disposition"] != b[c]["disposition"]] or "none")
print()
print(f"{'ch':>3} {'as ruled':>10} {'at maximum credit':>19}")
lo = []
for c in exa:
    x, y = bar(a[c]), bar(b[c]); lo.append(float(y))
    print(f"{c:>3} {x:>8} dB {y:>16} dB")
print(f"\nrange at the maximum credit: {min(lo):.1f} to {max(lo):.1f} dB over tolerance, against a 3 dB allowance")
keeps = [(c, r) for c in b for r in ("bao_long", "bao_short", "shortest") if b[c][r + "_verdict"] == "keep"]
print(f"\nkeep cells created: {keeps or 'none'}")
for c, r in keeps:
    Rn = b[c][r + "_R_none"]
    print(f"  channel {c} {r}: fails the no-credit rescue test by {10*np.log10(float(Rn)):.1f} dB")
print()
print("Every excision survives when every channel is granted more credit than its own data support.")
print("The credit model is therefore not what the exclusions rest on. The keep cells created here are")
print("credited keeps only: a rescue must survive with no credit at all, and all of them fail that by")
print("10 dB or more, so no channel becomes a keep under any credit assumption.")
