#!/usr/bin/env python3
"""Robustness of the excisions to the credits the ruling refused (written 2026-09-18 after the table of record).

For each excised channel: (1) the ground-filter credit granted at the archive's own booked variance shares, applied in
RFIsher's consistent form R_credit = A S / lambda * (intraday_share * G + fast_share) with G the measured gain of the
table (the shares from the archive ledger, ch<NN>_fid<fid>.json sections.chain); (2) the other polarisation: R at the
measured gain from the smaller polarisation's reading, and from the estimator's noise floor where that reading is a
clipped zero; (3) the margin in dB in each case. Output: robustness_credit.csv and a printed table."""
import csv, glob, json, os, sys
import numpy as np
FA = os.path.dirname(os.path.abspath(__file__))
LEDGER = "/home/djg/rail/results/archive_v5_2026-09-08_corrected/ledger/channels"
tor = {}
for r in csv.DictReader(open(os.path.join(FA, "table_of_record.csv"))):
    ch = int(r["channel"])
    if r["role"] != "pilot" and ch not in tor: tor[ch] = r
shares = {}
for f in glob.glob(os.path.join(LEDGER, "ch*_fid*.json")):
    d = json.load(open(f)); ch = int(os.path.basename(f)[2:4]); c = d["sections"]["chain"]
    shares[ch] = (c.get("intraday_share"), c.get("fast_share"), c.get("ground_filter_db"))
# noise floor of the level estimator: median per-epoch sigma of the in-band level over the science dumps (cadence_tau, amendment 5)
rows = []
print("ch  disp    bar_dB  ground_db(archive)  R_credit  margin_dB | A_pol0    A_pol1   R_otherpol_Gm  margin_dB | both: R  margin_dB")
for ch in sorted(tor):
    r = tor[ch]
    if r["disposition"] != "excise": continue
    A = float(r["A_keep_all_ns1_bound"]); S = 10 ** (-float(r["suppression_db"]) / 10); lam = float(r["lambda_deployed"])
    G = float(r["G_measured"]); RG1 = float(r["R_deployed_G1"]); RGm = float(r["R_deployed_Gmeasured"])
    # the ruling's bar is the least over policies; use the row's keep_all ratio for the credit comparison (same basis as the shares)
    if "lowest-epoch" in r["policy_or_reason"]:
        A_used = float(r["A_lowest_epoch"]); RG1 = A_used * S / lam; RGm = RG1 * G
    isd, fsd, gdb = shares.get(ch, (None, None, None))
    if isd is None: R_credit = RGm; note = "no share (archive refused); no credit"
    else: R_credit = RG1 * (isd * G + fsd); note = ""
    a0 = float(r["A_keep_all_ns1_pol0"] or "nan"); a1 = float(r["A_keep_all_ns1_pol1"] or "nan")
    other = min(a0, a1) if np.isfinite(a0) and np.isfinite(a1) else np.nan
    floor = 3e-5   # per-epoch noise sigma of the level on a 33-frame dump (cadence_tau, sigma_d), used where the reading is a clipped zero
    A_other = max(other, floor) if np.isfinite(other) else np.nan
    R_other = A_other * S / lam * G if np.isfinite(A_other) else np.nan
    # both concessions at once: the cleaner polarisation and the archive's ground-filter credit
    R_both = (A_other * S / lam * (isd * G + fsd)) if (isd is not None and np.isfinite(A_other)) else R_other
    rows.append(dict(channel=ch, bar_db=f"{10*np.log10(RGm):.1f}", R_both=f"{R_both:.3g}", margin_both_db=f"{10*np.log10(R_both):.1f}" if R_both > 0 else "", ground_filter_db_archive=("" if gdb is None else f"{gdb:.1f}"), intraday_share=("" if isd is None else f"{isd:.4f}"), fast_share=("" if fsd is None else f"{fsd:.4f}"),
                     R_credit=f"{R_credit:.3g}", margin_credit_db=f"{10*np.log10(R_credit):.1f}", A_pol0=f"{a0:.3e}", A_pol1=f"{a1:.3e}", A_other_used=f"{A_other:.3e}", R_otherpol_Gm=f"{R_other:.3g}", margin_otherpol_db=f"{10*np.log10(R_other):.1f}" if R_other > 0 else "", note=note))
    print(f"{ch:2d}  excise  {10*np.log10(RGm):5.1f}   {('' if gdb is None else f'{gdb:5.1f}'):>6s}            {R_credit:8.3g}  {10*np.log10(R_credit):6.1f}  | {a0:.2e}  {a1:.2e}  {R_other:9.3g}  {10*np.log10(R_other):6.1f}  | both {R_both:8.3g} {10*np.log10(R_both):6.1f}  {note}")
with open(os.path.join(FA, "robustness_credit.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote robustness_credit.csv")
