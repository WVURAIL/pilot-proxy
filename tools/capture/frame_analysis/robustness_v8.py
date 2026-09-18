#!/usr/bin/env python3
"""Robustness of the baseline-resolved excisions (amendment 8): per excised channel the bar on the range that excises
it with both credits (the ruling), the same bar without the ground-filter credit, the bar if the archive's booked shares
were granted instead of the measured coherence, and the bar on the cleaner polarisation of the range's classes.
Writes robustness_credit.csv in the column layout the dissertation generator reads."""
import csv, glob, json, os, re
import numpy as np
FA = os.path.dirname(os.path.abspath(__file__))
LEDGER = "/home/djg/rail/results/archive_v5_2026-09-08_corrected/ledger/channels"
RANGES = {"bao_short": [(0, 32), (0, 64)], "bao_long": [(0, 128), (0, 255), (1, 0), (2, 0), (3, 0)]}
SCIENCE = ["20260916162300", "20260917040230", "20260917090230", "20260917140230"]
tor = {}
for r in csv.DictReader(open(os.path.join(FA, "table_of_record.csv"))):
    ch = int(r["channel"])
    if r["role"] != "pilot" and ch not in tor: tor[ch] = r
shares = {}
for f in glob.glob(os.path.join(LEDGER, "ch*_fid*.json")):
    d = json.load(open(f)); c = d["sections"]["chain"]; shares[int(os.path.basename(f)[2:4])] = (c.get("intraday_share"), c.get("fast_share"), c.get("ground_filter_db"))
ex = {}
for r in csv.DictReader(open(os.path.join(FA, "class_excess_epochs.csv"))):
    if r["excess"] and r["epoch"] in SCIENCE: ex.setdefault((int(r["channel"]), (int(r["ew"]), int(r["ns"])), int(r["pol"])), []).append(float(r["excess"]))
fl = {}
for r in csv.DictReader(open(os.path.join(FA, "class_floor_bins.csv"))):
    if r["epoch"] in SCIENCE: fl.setdefault(((int(r["ew"]), int(r["ns"])), int(r["pol"])), []).append(float(r["excess"]))
floor = {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in fl.items()}
rows = []
print("ch  range      bar(ruling)  no-ground-credit  archive-shares  cleaner-pol")
for ch in sorted(tor):
    r = tor[ch]
    if r["disposition"] != "excise": continue
    rng = "bao_long" if r["bao_long_verdict"] == "excise" else "bao_short"
    bar = float(r[f"{rng}_R_deployed"]) if r[f"{rng}_R_deployed"] else np.nan
    m = re.search(r"subtraction bar ([-\d.]+) dB", r["policy_or_reason"]); bar_db = float(m.group(1)) if m else 10 * np.log10(bar)
    credit = float(r[f"{rng}_ground_credit_db"] or 0.0); G = float(r[f"{rng}_G"]); rho = float(r[f"{rng}_rho"])
    S = 10 ** (-float(r["suppression_db"]) / 10); lam = float(r["lambda_deployed"])
    isd, fsd, gdb = shares.get(ch, (None, None, None))
    R_ruling = 10 ** (bar_db / 10); R_noground = R_ruling * 10 ** (credit / 10)
    R_archive = (R_noground / G) * (isd * G + fsd) if isd is not None else R_noground
    # cleaner polarisation: per class the smaller-polarisation excess net of its floor, median over the range's measured classes
    cl = []
    for c in RANGES[rng]:
        vals = []
        for p in (0, 1):
            v = ex.get((ch, c, p)); f = floor.get((c, p))
            if v and f: vals.append(float(np.median(v)) - f[0])
        if vals: cl.append(min(vals))
    A_c = max(float(np.median(cl)), 3e-5) if cl else np.nan
    R_other = A_c * S * (1 - rho ** 2) * G / lam if np.isfinite(A_c) else np.nan
    rows.append(dict(channel=ch, bar_db=f"{bar_db:.1f}", ground_filter_db_archive=("" if gdb is None else f"{gdb:.1f}"), margin_credit_db=f"{10*np.log10(R_archive):.1f}",
                     rho_ew=f"{rho:.2f}", ground_filter_db_ew=f"{credit:.1f}", margin_credit_ew_db=f"{10*np.log10(R_noground):.1f}",
                     margin_otherpol_db=f"{10*np.log10(R_other):.1f}" if np.isfinite(R_other) and R_other > 0 else "", margin_both_ew_db=f"{bar_db:.1f}", basis=f"{rng}: {'long' if rng == 'bao_long' else 'short'} BAO baselines, both credits (amendment 8)"))
    print(f"{ch:2d}  {rng:9s}  {bar_db:6.1f}       {10*np.log10(R_noground):6.1f}          {10*np.log10(R_archive):6.1f}       {10*np.log10(R_other) if np.isfinite(R_other) and R_other > 0 else float('nan'):6.1f}")
with open(os.path.join(FA, "robustness_credit.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote robustness_credit.csv (amendment 8 basis)")
