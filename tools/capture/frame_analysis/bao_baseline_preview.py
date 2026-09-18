#!/usr/bin/env python3
"""Preview (2026-09-18, not a ruling): the excision test read on the east-west classes (22, 44, 66 m), which carry the
BAO scales, instead of the 0.3 m baseline. A_ew = median over the three classes of the per-class in-band excess (larger
polarisation, median over the four dumps); G_ew = from the class-level coherence time (cadence_tau_ew{1,2,3}.csv):
the median of the measured classes' tau_c, or, where every class is refused, the 0.3 m result (the persistence upper
bound) and where a class is bound its lower end. Bars in dB over tolerance with the deployed cut at G_ew and at G = 1."""
import csv, math, collections, re
tor = {}
for r in csv.DictReader(open("table_of_record.csv")):
    ch = int(r["channel"])
    if r["role"] != "pilot" and ch not in tor: tor[ch] = r
files = ["frame_residual_20260916162300.csv", "frame_residual_20260917040230.csv", "frame_residual_20260917090230.csv", "frame_residual_20260917140230.csv"]
vals = collections.defaultdict(list)
for f in files:
    for r in csv.DictReader(open(f)):
        if r["frames"] == "all" and r["ew"] != "-1" and r["excess_inband_median"]:
            vals[(int(r["channel"]), int(r["ew"]), int(r["ns"]), r["pol"])].append(float(r["excess_inband_median"]))
def med(v): v = sorted(v); return v[len(v) // 2] if v else float("nan")
tau = collections.defaultdict(dict)
for c in (1, 2, 3):
    for r in csv.DictReader(open(f"cadence_tau_ew{c}.csv")):
        if r["bins"] == "inband": tau[int(r["channel"])][c] = (r["status"], float(r["tau_c_s"]) if r["tau_c_s"] else float("nan"))
TF = 16384 * 2.56e-6; CAP = 86164.0905
print("ch  A_ns1     A_ew      tau_ew per class (s)                       tau_used  G_used   bar_ns1  bar_ew_at_G  bar_ew_G1  note")
rows = []
for ch in sorted(tor):
    r = tor[ch]
    if r["disposition"] != "excise": continue
    per = [max(med(vals.get((ch, c, 0, "0"), [])), med(vals.get((ch, c, 0, "1"), []))) for c in (1, 2, 3)]
    A_ew = med(per); A = float(r["A_keep_all_ns1_bound"])
    S = 10 ** (-float(r["suppression_db"]) / 10); lam = float(r["lambda_deployed"])
    measured = [t for s, t in tau[ch].values() if s == "measured"]
    bounds = [t for s, t in tau[ch].values() if s == "bound"]
    if measured: tau_used = med(measured); note = "ew measured"
    elif bounds: tau_used = bounds[0]; note = "ew bound"
    else: tau_used = float(r["tau_c_s"]) if r["tau_c_s"] else CAP; note = "ew refused: 0.3 m tau used"
    G = min(tau_used, CAP) / TF
    R_ew = A_ew * S / lam * G; R_ew1 = A_ew * S / lam
    bar_ns1 = float(re.search(r"subtraction bar ([-\d.]+) dB", r["policy_or_reason"]).group(1))
    desc = " ".join(f"{c}:{s[:4]}{('%.0f' % t) if t == t else ''}" for c, (s, t) in sorted(tau[ch].items()))
    print(f"{ch:2d}  {A:.2e}  {A_ew:.2e}  {desc:40s}  {tau_used:8.0f}  {G:8.0f}  {bar_ns1:6.1f}  {10*math.log10(R_ew):9.1f}  {10*math.log10(R_ew1):8.1f}  {note}")
    rows.append(dict(channel=ch, A_ns1=A, A_ew=f"{A_ew:.3e}", tau_ew_classes=desc, tau_used_s=f"{tau_used:.0f}", G_used=f"{G:.0f}", bar_ns1_db=bar_ns1, bar_ew_db=f"{10*math.log10(R_ew):.1f}", bar_ew_G1_db=f"{10*math.log10(R_ew1):.1f}", note=note))
with open("bao_baseline_preview.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
