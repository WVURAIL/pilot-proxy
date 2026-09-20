#!/usr/bin/env python3
"""The single maximum-concession stack: every nameable credit granted at once.

The manuscript tests concessions one at a time. A reviewer will stack them. This grants them
simultaneously, in one pass, from one patched copy of the ruling, so there is exactly one answer to
the question "what if you granted everything at once". Each concession is added in turn so the
reader can see what each one costs.

The concessions, each at the largest value this capture supports:
  1. the ground filter at its 0.95 ceiling on every channel and range, rather than the coherence
     measured there (worth up to 9.6 dB; blanket_credit.py publishes this one alone)
  2. the coherence time at the least of the estimator's own trim probes on every class
  3. the most conservative even-count aggregation, the lower of the two middle class values
     (amendment 16 declares the geometric mean; this takes the alternative that helps the channel)
  4. a delay cut deeper than the deployed 200 ns, at the collaboration's own 280 ns mask. The
     residual of a rectangular 5.38 MHz block outside |tau| < tau_cut is sinc-squared and saturates:
     10.14 dB at 200 ns, 11.65 at 280 ns, 17.16 at 1 us. The deployed credit is 11.4 dB, so 280 ns
     is worth +1.5 dB and even an absurd microsecond cut is worth only +7.0 dB.

It does NOT include reading A on the cleaner polarisation, which discards a polarisation rather than
cleaning one and leaves the other still excised; that is reported separately by robustness_v8.py.

usage: concession_stack.py [out dir]
"""
import csv, io, os, re, subprocess, sys, tempfile
import numpy as np

FA = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(OUT, exist_ok=True)
RESCAN = "/home/djg/rail/output/channel-ruling-execution-2026-09-14/rebuild/author_actions/ch33_rescan/products"
EX = [15, 17, 30, 31, 33, 35]

BASE = io.open(os.path.join(FA, "ruling_baseline.py")).read().replace(
    'FA = os.path.dirname(os.path.abspath(__file__))', 'FA = %r' % FA, 1)
RHO = '    return float(min(max(np.median(v), 0.0), 0.95)) if v else 0.0'
AGG = 'return v[n // 2] if n % 2 else float(np.sqrt(v[n // 2 - 1] * v[n // 2]))'
TAU = '        t, tstat, tnote = range_gain(ch, rng); G = min(t, CAP) / TF if np.isfinite(t) else np.nan'
SUP = 'S = 10 ** (-sup.get(ch, 0.0) / 10)'

def run(tag, rho=False, tau=False, agg=False, cut=False):
    s = BASE
    if rho: s = s.replace(RHO, '    return 0.95', 1)
    if agg: s = s.replace(AGG, 'return v[n // 2] if n % 2 else float(v[n // 2 - 1])', 1)
    if tau: s = s.replace(TAU, '        t, tstat, tnote = range_gain(ch, rng, least=True); G = min(t, CAP) / TF if np.isfinite(t) else np.nan', 1)
    if cut: s = s.replace(SUP, 'S = 10 ** (-(sup.get(ch, 0.0) + 1.5) / 10)', 1)
    p = os.path.join(OUT, "rb_%s.py" % tag); io.open(p, "w").write(s)
    o = os.path.join(OUT, "tor_%s.csv" % tag)
    subprocess.run([sys.executable, p, os.path.join(FA, "table_of_record_before_amendment8.csv"), o, RESCAN],
                   capture_output=True, text=True, check=True)
    d = {}
    for x in csv.DictReader(open(o)):
        c = int(x["channel"])
        if x["role"] != "pilot" and c not in d: d[c] = x
    return d

bar = lambda x: (re.search(r"([-\d.]+) dB over tolerance", x["policy_or_reason"]) or [None, None])[1]
STEPS = [("as ruled", {}),
         ("+ ground filter at its ceiling", dict(rho=True)),
         ("+ least trim probe", dict(rho=True, tau=True)),
         ("+ conservative aggregation", dict(rho=True, tau=True, agg=True)),
         ("+ a 280 ns delay cut", dict(rho=True, tau=True, agg=True, cut=True))]
print(f"{'concession, cumulative':>32} | " + " ".join(f"{c:>6}" for c in EX) + " | excised")
rows = []
for i, (name, kw) in enumerate(STEPS):
    d = run("s%d" % i, **kw)
    ex = sorted(c for c, x in d.items() if x["disposition"] == "excise")
    vals = [bar(d[c]) for c in EX]
    rows.append((name, vals, ex))
    print(f"{name:>32} | " + " ".join(f"{(v or '-'):>6}" for v in vals) + f" | {ex}")
    if i == len(STEPS) - 1:
        keeps = [(c, r) for c in d for r in ("bao_long", "bao_short", "shortest") if d[c][r + "_verdict"] == "keep"]
        fin = [float(v) for v in vals if v]
print()
print(f"With every nameable credit granted at once the six are excised at {min(fin):.1f} to {max(fin):.1f} dB")
print(f"against a 3 dB allowance, the least margin being {min(fin)-3:.1f} dB on channel {EX[fin.index(min(fin))]}.")
print(f"keep cells under the full stack: {keeps or 'none'}")
