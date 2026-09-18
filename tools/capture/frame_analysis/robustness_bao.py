#!/usr/bin/env python3
"""Robustness of the excisions on the BAO basis (amendment 7) to the concessions the ruling refuses: the archive's
ground-filter credit at its booked shares (RFIsher's consistent form, R (intraday_share G + fast_share)), the cleaner
polarisation (each east-west class read on its smaller polarisation, median over the classes and the four dumps), and
both at once. Writes robustness_credit.csv (BAO basis; the 0.3 m-basis file is kept as robustness_credit_ns1.csv)."""
import csv, glob, json, os, re, collections
import numpy as np
FA = os.path.dirname(os.path.abspath(__file__))
LEDGER = "/home/djg/rail/results/archive_v5_2026-09-08_corrected/ledger/channels"
tor = {}
for r in csv.DictReader(open(os.path.join(FA, "table_of_record.csv"))):
    ch = int(r["channel"])
    if r["role"] != "pilot" and ch not in tor: tor[ch] = r
shares = {}
for f in glob.glob(os.path.join(LEDGER, "ch*_fid*.json")):
    d = json.load(open(f)); c = d["sections"]["chain"]; shares[int(os.path.basename(f)[2:4])] = (c.get("intraday_share"), c.get("fast_share"), c.get("ground_filter_db"))
files = sorted(glob.glob(os.path.join(FA, "frame_residual_2026091[67]*.csv")))[:4]
files = [f for f in glob.glob(os.path.join(FA, "frame_residual_*.csv")) if any(e in f for e in ("20260916162300", "20260917040230", "20260917090230", "20260917140230"))]
vals = collections.defaultdict(list)
for f in files:
    for r in csv.DictReader(open(f)):
        if r["frames"] == "all" and (r["ew"], r["ns"]) in (("1", "0"), ("2", "0"), ("3", "0")) and r["excess_inband_median"]:
            vals[(int(r["channel"]), r["ew"], r["pol"])].append(float(r["excess_inband_median"]))
def med(v): return float(np.median(v)) if v else float("nan")
# the ground-filter credit the east-west measurement supports: the sidereal-mean subtraction removes the between-epoch
# coherent fraction of the power, rho^2, so the surviving fraction is 1 - rho^2 (rho = median ew1_x in-band coherence
# over the six dump pairs, clipped to [0, 0.95]); credit_ew_db = -10 log10(1 - rho^2)
rho = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(FA, "lag_coherence_allclasses_D.csv"))):
    if r["cls"] == "ew1_x" and r["bins"] == "inband": rho[int(r["channel"])].append(float(r["rho"]))
rows = []
print("ch  bar_bao  ground_db(archive) margin | measured EW credit | cleaner pol | both")
for ch in sorted(tor):
    r = tor[ch]
    if r["disposition"] != "excise": continue
    S = 10 ** (-float(r["suppression_db"]) / 10); lam = float(r["lambda_deployed"]); G = float(r["G_bao"])
    m = re.search(r"subtraction bar ([-\d.]+) dB, ([-\d.]+) dB even at G=1", r["policy_or_reason"])
    bar = float(m.group(1)); RG1 = 10 ** (float(m.group(2)) / 10); RGm = 10 ** (bar / 10)
    isd, fsd, gdb = shares.get(ch, (None, None, None))
    R_credit = RGm if isd is None else RG1 * (isd * G + fsd)
    cleaner = med([min(med(vals.get((ch, e, "0"), [])), med(vals.get((ch, e, "1"), []))) for e in ("1", "2", "3")])
    floor = 3e-5
    A_c = max(cleaner, floor) if np.isfinite(cleaner) else float("nan")
    R_other = A_c * S / lam * G
    R_both = R_other if isd is None else A_c * S / lam * (isd * G + fsd)
    rh = min(max(med(rho.get(ch, [])), 0.0), 0.95) if rho.get(ch) else 0.0
    credit_ew = -10 * np.log10(1 - rh ** 2)
    R_credit_ew = RGm * (1 - rh ** 2); R_both_ew = R_other * (1 - rh ** 2)
    rows.append(dict(channel=ch, bar_db=f"{bar:.1f}", ground_filter_db_archive=("" if gdb is None else f"{gdb:.1f}"), intraday_share=("" if isd is None else f"{isd:.4f}"), fast_share=("" if fsd is None else f"{fsd:.4f}"),
                     R_credit=f"{R_credit:.3g}", margin_credit_db=f"{10*np.log10(R_credit):.1f}", rho_ew=f"{rh:.2f}", ground_filter_db_ew=f"{credit_ew:.1f}", margin_credit_ew_db=f"{10*np.log10(R_credit_ew):.1f}",
                     A_cleaner_pol=f"{A_c:.3e}", R_otherpol_Gm=f"{R_other:.3g}", margin_otherpol_db=f"{10*np.log10(R_other):.1f}", R_both=f"{R_both:.3g}", margin_both_db=f"{10*np.log10(R_both):.1f}", margin_both_ew_db=f"{10*np.log10(R_both_ew):.1f}", basis="BAO (east-west classes, amendment 7)"))
    print(f"{ch:2d}  {bar:6.1f}  {('' if gdb is None else f'{gdb:5.1f}'):>6s}   {10*np.log10(R_credit):6.1f} | rho {rh:4.2f} credit_ew {credit_ew:4.1f} margin {10*np.log10(R_credit_ew):6.1f} | {A_c:.2e}  {10*np.log10(R_other):6.1f} | both(archive) {10*np.log10(R_both):6.1f}  both(ew) {10*np.log10(R_both_ew):6.1f}")
with open(os.path.join(FA, "robustness_credit.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote robustness_credit.csv (BAO basis)")
