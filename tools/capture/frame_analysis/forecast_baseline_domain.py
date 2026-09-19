#!/usr/bin/env python3
"""Evidence for amendment 10: the baseline domain the forecast of record prices.

Reads the four response banks behind the two tolerance worlds, reports the
baseline density file each was built on, and evaluates that density at every
baseline class the ruling reads. Prints PASS/FAIL against the claim made in
audit finding R3 and amendment 9 item 1, that the forecast prices no baseline
below 20 m.

usage: forecast_baseline_domain.py [banks_dir] [radiofisher_dir]
"""
import json, sys, glob, os
import numpy as np

BANKS = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/djg/rail/results/canfar_reanalysis_2026-09-09/archive/response-banks"
RFDIR = sys.argv[2] if len(sys.argv) > 2 else "/home/djg/rail/RadioFisher"
C = 299792458.0
CLASSES = [("(0,1)", 0.3), ("(0,8)", 2.4), ("(0,32)", 9.8), ("(0,64)", 19.5),
           ("(0,128)", 39.0), ("(0,255)", 78.0), ("(1,0)", 22.0), ("(2,0)", 44.0), ("(3,0)", 66.0)]
RULED = {"(0,32)", "(0,64)", "(0,128)", "(0,255)", "(1,0)", "(2,0)", "(3,0)"}

print("== banks behind the two tolerance worlds ==")
nx_files, ok = set(), True
for p in sorted(glob.glob(os.path.join(BANKS, "*.npz"))):
    m = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
    s = m["provenance"]["experiment"]["settings"]
    nx_files.add((s["n(x)"], m["provenance"]["experiment"]["baseline_sha256"]))
    print(f"  {os.path.basename(p)}")
    print(f"    label      : {m['experiment']}")
    print(f"    config     : {m['config']}   overrides: {m['expt_overrides']}")
    print(f"    n(x)       : {s['n(x)']}")
    print(f"    Dmin/Dmax  : {s['Dmin']} / {s['Dmax']} m")
print()
print("The 'label' field is a hardcoded literal in rfisher/fisherbank.py and is")
print("written for every config. The settings block is what was built.")
print()

if len(nx_files) != 1:
    print("FAIL: banks disagree on the density file:", nx_files); sys.exit(1)
nxname, nxsha = nx_files.pop()
path = os.path.join(RFDIR, nxname)
import hashlib
got = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(f"== density {nxname} ==")
print(f"  sha256 recorded in bank : {nxsha}")
print(f"  sha256 on disk          : {got}   {'match' if got == nxsha else 'MISMATCH'}")
ok &= got == nxsha

a = np.loadtxt(path)
x, nx = a[:, 0], a[:, 1]
nz = nx > 0
print(f"  support                 : d = {x[nz].min()*C/1e6:.3f} to {x[nz].max()*C/1e6:.1f} m")
tot = np.trapezoid(nx * x, x)
below = (x * C / 1e6) < 20.0
frac = np.trapezoid((nx * x)[below], x[below]) / tot
print(f"  weight below 20 m       : {100*frac:.1f}% of the integral, {(nx[below]>0).sum()} nonzero rows")
print()
print("RadioFisher's interferometer_response uses the n(x) table directly when")
print("one is supplied (baofisher.py, 'if \"n(x)\" in list(expt.keys())'); Dmin and")
print("Dmax gate only the uniform-density fallback, which this configuration does")
print("not take. The priced domain is the support of the table.")
print()
print("== density at each ruling baseline class ==")
print("  class      d [m]        x        n(x)      priced   rules")
for name, d in CLASSES:
    xi = d * 1e6 / C
    v = float(np.interp(xi, x, nx, left=0.0, right=0.0))
    priced = v > 0
    ok &= priced
    print(f"  {name:9s} {d:6.1f}  {xi:.6f}  {v:10.4g}     {'yes' if priced else 'NO ':3s}     {'yes' if name in RULED else 'reported'}")
print()
print("R3 / amendment 9 item 1 claimed: the forecast prices no baseline below 20 m.")
print("VERDICT:", "REFUTED - every class the ruling reads is priced." if ok else "not refuted.")
sys.exit(0 if ok else 1)
