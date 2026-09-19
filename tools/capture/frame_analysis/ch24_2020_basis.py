#!/usr/bin/env python3
"""Channel 24 on the 2020 full-array basis, and the 2020 cohort as a positive control.

Channel 24 is the one DTV channel with no products in the 2026 matched capture: no live node
records its band (freq_id 676 to 691), so the ruling of record leaves it undetermined for want of
data rather than for want of a measurement. This reads the 2020 CANFAR per-pilot baseband cohort,
which does cover it, and states what that measurement supports.

It is deliberately NOT merged into the 2026 table of record. The 2020 cohort has no channel 37
bin, carries one coarse bin per file (the pilot bin, not the in-band shelf the ruling reads), and
has 8 frames per epoch against 33. Its verdict is reported on its own basis.

usage: ch24_2020_basis.py [cohort_dir] [table_of_record csv]
"""
import json, os, sys
import numpy as np

COHORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_2026-09-19")
TOR = sys.argv[2] if len(sys.argv) > 2 else "/home/djg/rail/dissertation/figure_src/data/record/table_of_record.csv"
S_DB, RHO_CEIL, TF, CAP, ALLOW = 11.4, 0.95, 16384 * 2.56e-6, 86164.0905, 3.0
TAU_ARCHIVE = 7777.0      # channel 24's own archive-measured correlation time
TAU_FLOOR = 15.0          # amendment 5: no channel's tau_c lies below the campaign's shortest lag

exc = {int(k): np.array(v) for k, v in json.load(open(os.path.join(COHORT, "cohort_exc_long.json"))).items()}
pv = json.load(open(os.path.join(COHORT, "pilot_vs_inband_D1.json")))

print("== the 2020 cohort as a positive control ==")
print("Channel 35's transmitter was off until 2021-10, so its pilot bin is a transmitter-off null")
print("in the same frames. Every channel is read through the 2026 estimator on the same long BAO")
print("classes (39 and 78 m north-south, 22 to 66 m east-west).\n")
c35 = exc[35]; fm, fs = float(np.mean(c35)), float(np.std(c35, ddof=1))
gate = fm + 3 * fs
print(f"control ch35: excess mean {fm:.3e}, scatter {fs:.3e}, three-scatter gate {gate:.3e}\n")
print(f"{'ch':>3} {'A_2020':>10} {'sigma over ch35':>16}  status in 2020")
det = []
for ch in sorted(exc):
    a = float(np.median(exc[ch])); s = (a - fm) / fs
    det.append((ch, a, s))
    print(f"{ch:>3} {a:>10.3e} {s:>16.1f}  {'DETECTED' if a > gate else 'at the null'}")
on = [c for c, a, s in det if a > gate]
print(f"\ndetected in 2020: {on}")
print("Channels 26, 27 and 32 are detected here and read at the control floor in the 2026 capture,")
print("which is what a transmitter that has since been switched off looks like. Channel 35, off in")
print("2020, reads below its own null. The estimator therefore detects transmitters that were on")
print("and does not detect one that was off, on the baselines the ruling rules on.\n")

print("== channel 24 on the 2020 basis ==")
rat = []
for ch, v in pv.items():
    if len(v) != 4: continue
    pil, inb = max(v[0], v[1]), max(v[2], v[3])
    if pil > 0 and inb > 0: rat.append(10 * np.log10(inb / pil))
med_corr, worst_corr = float(np.median(rat)), float(min(rat))
print(f"pilot-bin to in-band correction, measured on the {len(rat)} channels the 2026 capture records:")
print(f"  median {med_corr:+.2f} dB, worst case {worst_corr:+.2f} dB. It is never measured on channel 24")
print("  itself, whose band is absent from every 2026 dump; that is the chief caveat on this row.\n")

lam = None
import csv
for x in csv.DictReader(open(TOR)):
    if int(x["channel"]) == 24 and x["role"] != "pilot" and x["lambda_deployed"]:
        lam = float(x["lambda_deployed"]); break
A_pilot = float(np.median(exc[24]))
S, credit = 10 ** (-S_DB / 10), 1 - RHO_CEIL ** 2
print(f"A on the pilot bin {A_pilot:.3e}; lambda_deployed {lam:.4g}; delay cut {S_DB} dB;")
print(f"ground credit at the 0.95 ceiling {-10*np.log10(credit):.1f} dB (the most generous the rule allows).\n")
print(f"{'correction':>16} {'A':>11} {'tau_c':>9} {'G':>9} {'bar':>9}")
rows = []
for name, corr in (("median", med_corr), ("worst case", worst_corr)):
    A = A_pilot * 10 ** (corr / 10)
    for tau in (TAU_ARCHIVE, 1200.0, TAU_FLOOR):
        G = min(tau, CAP) / TF
        bar = 10 * np.log10(A * S * credit * G / lam)
        rows.append((name, corr, tau, G, bar))
        print(f"{name:>16} {A:>11.3e} {tau:>9.1f} {G:>9.0f} {bar:>+8.2f} dB")
    thresh = (10 ** (ALLOW / 10)) * lam / (A * S * credit) * TF
    print(f"{name:>16} crosses the {ALLOW:.0f} dB allowance at tau_c = {thresh:.1f} s\n")
worst_at_floor = [b for n, c, t, g, b in rows if n == "worst case" and t == TAU_FLOOR][0]
worst_at_arch = [b for n, c, t, g, b in rows if n == "worst case" and t == TAU_ARCHIVE][0]
print("VERDICT (2020 basis, stated conditionally):")
print(f"  Channel 24 is over tolerance by {worst_at_arch:+.1f} dB at its own archive-measured correlation")
print(f"  time of {TAU_ARCHIVE:.0f} s, and by {worst_at_floor:+.1f} dB even at {TAU_FLOOR:.0f} s, the shortest coherence time")
print("  amendment 5 states any channel can have. It is over tolerance at every coherence time the")
print("  estimator can return, on the worst-case pilot-to-in-band correction, with every credit.")
print("  The condition is that the transmitter is still on. The archive ledger records channel 24")
print("  transmitting continuously from 2018-12 to 2026-04 with no declared off epoch, zero off")
print("  frames and 9,470 frames flagged transmitter-on; its bin ceased to be recorded on")
print("  2026-04-16 because no live node covers it, not because it went quiet.")
