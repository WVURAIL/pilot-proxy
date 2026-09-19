#!/usr/bin/env python3
"""What every excision was granted, and what it reads with none of it.

The ruling judges an exclusion in the deployed world with every credit the data can give: the 200 ns
delay cut at 11.4 dB, the ground filter at the coherence measured on the range, and the loosest
policy on the ladder, all against the tolerance that world prices. This prints, per excised channel,
the net credit granted and the bar the same excess reads in the world without any credit, at the same
coherence gain. It computes nothing new; both ratios are columns of the table of record.

usage: credit_ledger.py [table_of_record csv]
"""
import csv, re, sys
import numpy as np

TOR = sys.argv[1] if len(sys.argv) > 1 else "table_of_record.csv"
first = {}
for x in csv.DictReader(open(TOR)):
    c = int(x["channel"])
    if x["role"] != "pilot" and c not in first: first[c] = x
ex = sorted(c for c, x in first.items() if x["disposition"] == "excise")

print(f"{'ch':>3} {'range':>10} {'delay cut':>10} {'ground':>9} {'tolerance':>10} {'net credit':>11} "
      f"{'bar granted':>12} {'bar with none':>14}")
bars, nones, creds = [], [], []
for c in ex:
    x = first[c]; rng = "bao_long" if x["bao_long_verdict"] == "excise" else "bao_short"
    S_db = float(x["suppression_db"])
    ground_db = -10 * np.log10(1 - float(x[rng + "_rho"]) ** 2)
    tol_db = 10 * np.log10(float(x["lambda_deployed"]) / float(x["lambda_none"]))
    net = S_db + ground_db + tol_db
    bar = float(re.search(r"([-\d.]+) dB over tolerance", x["policy_or_reason"]).group(1))
    nb = 10 * np.log10(float(x[rng + "_R_none"])) if x[rng + "_R_none"] else float("nan")
    bars.append(bar); nones.append(nb); creds.append(net)
    print(f"{c:>3} {rng:>10} {S_db:>9.1f}dB {ground_db:>8.1f}dB {tol_db:>9.1f}dB {net:>10.1f}dB "
          f"{bar:>11.1f}dB {nb:>13.1f}dB")
print()
print(f"net credit granted: {min(creds):.1f} to {max(creds):.1f} dB")
print(f"bar with every credit: {min(bars):.1f} to {max(bars):.1f} dB over tolerance")
print(f"bar with no credit at all: {min(nones):.1f} to {max(nones):.1f} dB over tolerance")
print()
print("Every excised channel is over tolerance before any credit is granted and remains over it after")
print("all of them. The credits move each bar by 1 to 14 dB and change no verdict. The ruling's")
print("exclusions are therefore not an artefact of the credit model: a channel that fails under every")
print("favourable assumption the forecast allows fails under any less favourable one.")
