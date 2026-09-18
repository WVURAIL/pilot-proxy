#!/usr/bin/env python3
"""Table of record: one row per freq_id 477..844, built from the dumps.

usage: table_of_record.py <out csv> <dump label>=<frame_residual csv>:<chime-run dir> [more dumps ...] [gain=<cadence_tau csv>]

Per channel and policy (cal_q0.1, cal_q0.5, cal_q0.9, keep_all): the dumps that policy keeps (dump median Q at or
below the archive's eta_q for that channel, see ladder_place.py), the physical A = in-band median excess on the
shortest same-polarisation north-south baseline over those dumps (all frames of each kept dump), G_dump (median),
and R = A * S_world * G / lambda_world in the world without credit and in the deployed 200 ns world (S_world the
table's suppression on the DTV shelf). Disposition follows the predeclaration: keep with the loosest passing policy,
excise when no policy passes with the deployed credit and G = 1 (bar = dB over tolerance), pilot for the detector bin.
"""
import csv, json, os, sys
import numpy as np
NFFT = 16384; W = 0.390625
PILOT = {14:844,15:829,16:813,17:798,18:783,19:767,20:752,21:736,22:721,23:706,24:690,25:675,26:660,27:644,28:629,29:614,30:598,31:583,32:568,33:552,34:537,35:521,36:506,37:491}
ARCH = "/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot"
POLICIES = ["cal_q0.1", "cal_q0.5", "cal_q0.9", "keep_all"]; QS = {"cal_q0.1": 0.1, "cal_q0.5": 0.5, "cal_q0.9": 0.9}
def channel_of(fid):
    f = 800.0 - fid * W; return 14 + int((f - 470.0) // 6.0)
# tolerances per world
lam_none = {}; 
for r in csv.DictReader(open("/home/djg/rail/output/forecast-convergence-2026-09-09/exact-time-reference/channels.csv")):
    if r["channel"].isdigit(): lam_none[int(r["channel"])] = float(r["conditional_reference_amplitude"])
lam_dep = {}; sup_dep = {}
for r in csv.DictReader(open("/home/djg/rail/results/canfar_reanalysis_2026-09-09/coarse-worlds/coarse-world-sensitivity.csv")):
    if r["world"] == "deployed" and r["science_group"] == "dilation" and r["zeta"] == "1.0" and r["tolerance_basis"] == "primary" and r["base_tolerance_zeta1"]:
        lam_dep[int(r["channel"])] = float(r["base_tolerance_zeta1"]); sup_dep[int(r["channel"])] = float(r["suppression_db"])
# the archive's gain bound per channel (board_final.csv: G_nocredit = min(tau bound, cap) / T_frame)
G_arch = {}; tau_arch = {}
for r in csv.DictReader(open("/home/djg/rail/output/channel-ruling-execution-2026-09-14/rebuild/ruling/final/board_final.csv")):
    G_arch[int(r["channel"])] = float(r["G_nocredit"]); tau_arch[int(r["channel"])] = r["tau_quality"] + " " + r["tau_booked_s"]
# archive ladder thresholds
eta = {}
for ch, fid in PILOT.items():
    p = f"{ARCH}/{fid}.npz"
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    mu0 = 2.0 * float(z["target_norm_sq"][0]) / float(z["reference_norm_sum_sq"][0])
    Q = z["coarse_power_ratio"][:, 0].astype(float) / mu0; Q = Q[v & np.isfinite(Q)]
    eta[ch] = {k: float(np.quantile(Q, q, method="higher")) for k, q in QS.items()}
# dumps
dumps = {}
gain = {}   # amendment 3: measured coherence time per channel from cadence_tau.py (pilot-bin row), keyed by channel
for arg in sys.argv[2:]:
    label, rest = arg.split("=", 1)
    if label == "gain":
        for r in csv.DictReader(open(rest)):
            if r["bins"] == "pilot" and r["status"] in ("measured", "bound", "constant"):
                gain[int(r["channel"])] = dict(status=r["status"], tau=float(r["tau_c_s"]), tau_hi=float(r["tau_c_high_s"]), G=float(r["G"]))
        continue
    fr_csv, run_dir = rest.split(":", 1)
    allrows = [r for r in csv.DictReader(open(fr_csv)) if r["frames"] == "all" and r["ew"] != "-1"]
    rows = [r for r in allrows if r["ew"] == "0" and r["ns"] == "1" and r["pol"] == "0"]
    ex = {int(r["channel"]): (float(r["excess_inband_median"] or "nan"), float(r["excess_pilot_bin"] or "nan"), float(r["G_dump_pilot"] or "nan"), float(r["phi_fast_pilot"] or "nan")) for r in rows}
    # typical residual over all predeclared baseline classes and both polarisations (median)
    typ = {}
    for r in allrows:
        v = float(r["excess_inband_median"] or "nan")
        if np.isfinite(v): typ.setdefault(int(r["channel"]), []).append(v)
    typ = {c: float(np.median(v)) for c, v in typ.items()}
    d = np.load(os.path.join(run_dir, "chime_detector_outputs.npz"), allow_pickle=True)
    Qmed = {}
    for j, c in enumerate(d["physical_channel"]):
        v = d["valid"][:, j].astype(bool); Q = d["coarse_power_ratio"][:, j][v] / float(d["null_power_ratio"][j]); Qmed[int(c)] = float(np.median(Q))
    dumps[label] = dict(ex=ex, Qmed=Qmed, typ=typ)
chans = sorted(set(c for dmp in dumps.values() for c in dmp["ex"]))
verdicts = {}
for ch in chans:
    kept = {}
    for pol in POLICIES:
        if pol == "keep_all": kept[pol] = list(dumps)
        elif ch in eta: kept[pol] = [lbl for lbl, dmp in dumps.items() if ch in dmp["Qmed"] and dmp["Qmed"][ch] <= eta[ch][pol]]
        else: kept[pol] = []   # no live pilot bin: the mask cannot be evaluated
    res = {}
    for pol in POLICIES:
        vals = [dumps[l]["ex"][ch][0] for l in kept[pol] if ch in dumps[l]["ex"] and np.isfinite(dumps[l]["ex"][ch][0])]
        Gs = [dumps[l]["ex"][ch][2] for l in kept[pol] if ch in dumps[l]["ex"] and np.isfinite(dumps[l]["ex"][ch][2])]
        A = float(np.median(vals)) if vals else np.nan; G = float(np.median(Gs)) if Gs else np.nan
        tv = [dumps[l]["typ"][ch] for l in kept[pol] if ch in dumps[l]["typ"]]; A_typ = float(np.median(tv)) if tv else np.nan
        ln, ld, sd = lam_none.get(ch, np.nan), lam_dep.get(ch, np.nan), sup_dep.get(ch, 0.0)
        R_none_G1 = A / ln if ln else np.nan; R_dep_G1 = A * 10 ** (-sd / 10) / ld if ld else np.nan
        Ga = G_arch.get(ch, np.nan)
        Gm = gain[ch]["G"] if ch in gain else np.nan
        res[pol] = dict(kept=kept[pol], n=len(vals), A=A, A_typ=A_typ, G=G, R_none_G1=R_none_G1, R_dep_G1=R_dep_G1,
                        R_none_Gd=R_none_G1 * G if np.isfinite(G) else np.nan, R_dep_Gd=R_dep_G1 * G if np.isfinite(G) else np.nan,
                        R_dep_Ga=R_dep_G1 * Ga if np.isfinite(Ga) else np.nan,
                        R_none_Gm=R_none_G1 * Gm if np.isfinite(Gm) else np.nan, R_dep_Gm=R_dep_G1 * Gm if np.isfinite(Gm) else np.nan)
    def loosest(world, key_dep="R_dep_Ga"):
        key = "R_none_Gd" if world == "none" else key_dep
        for pol in reversed(POLICIES):
            if res[pol]["n"] and np.isfinite(res[pol][key]) and res[pol][key] <= 1: return pol
        return None
    # a pilot bin must exist in the dumps themselves (a dead node leaves the archive product but no new frames)
    has_pilot = ch in eta and any(ch in d["ex"] and np.isfinite(d["ex"][ch][2]) for d in dumps.values())
    if not any(res[p]["n"] for p in POLICIES): disp = ("undetermined", "no residual measured")
    elif ch not in lam_none: disp = ("undetermined", "no tolerance on the board")
    else:
        keep_arch = loosest("deployed", "R_dep_Ga")            # passes at the archive's gain bound: the rule's keep
        keep_meas = loosest("deployed", "R_dep_Gd")            # passes only at the measured within-dump gain
        keep_none = loosest("none")
        best_dep_G1 = min((res[p]["R_dep_G1"] for p in POLICIES if res[p]["n"] and np.isfinite(res[p]["R_dep_G1"])), default=np.nan)
        keep_meas_G = loosest("deployed", "R_dep_Gm")          # amendment 3: passes at the measured coherence time
        keep_none_G = loosest("none", "R_none_Gm") if ch in gain else None
        best_dep_Gm = min((res[p]["R_dep_Gm"] for p in POLICIES if res[p]["n"] and np.isfinite(res[p]["R_dep_Gm"])), default=np.nan)
        gtxt = (f"tau_c {'= ' if gain[ch]['status'] == 'measured' else '>= '}{gain[ch]['tau']:.0f} s"
                + (f" (to {gain[ch]['tau_hi']:.0f} s)" if gain[ch]['status'] == 'bound' else "")
                + (" (level constant within the day; cap)" if gain[ch]['status'] == 'constant' else "")
                + f", G {gain[ch]['G']:.0f}") if ch in gain else ""
        if keep_arch: disp = ("keep", f"{keep_arch} at the archive gain bound ({tau_arch.get(ch,'')}; {'also without credit' if keep_none else 'needs the deployed cut'})")
        elif ch in gain and has_pilot and gain[ch]["status"] in ("measured", "constant") and keep_meas_G:
            disp = ("keep", f"{keep_meas_G} at the measured coherence time ({gtxt}; {'also without credit' if keep_none_G else 'needs the deployed cut'})")
        elif ch in gain and has_pilot and np.isfinite(best_dep_Gm) and best_dep_Gm > 1:
            disp = ("excise", f"{10*np.log10(best_dep_Gm):.1f} dB over tolerance at the measured coherence time ({gtxt}) with the deployed cut under every policy; subtraction bar {10*np.log10(best_dep_Gm):.1f} dB, {10*np.log10(best_dep_G1):.1f} dB even at G=1")
        elif np.isfinite(best_dep_G1) and best_dep_G1 > 1 and has_pilot: disp = ("excise", f"{10*np.log10(best_dep_G1):.1f} dB over tolerance at G=1 with the deployed cut; subtraction bar {10*np.log10(best_dep_G1):.1f} dB")
        elif not has_pilot: disp = ("undetermined", "pilot bin has no live node; residual measured, mask not evaluable")
        elif keep_meas: disp = ("undetermined", f"passes at the measured gain (G {res[keep_meas]['G']:.1f}, lags to 1.3 s) but not at the archive bound ({tau_arch.get(ch,'')}): tau_c between 1.4 s and the bound decides")
        else: disp = ("undetermined", "over tolerance at the measured gain; no policy passes")
    verdicts[ch] = (disp, res)
with open(sys.argv[1], "w") as fh:
    w = csv.writer(fh); w.writerow(["freq_id", "channel", "role", "disposition", "policy_or_reason", "A_keep_all_ns1_bound", "A_keep_all_class_median", "G_dump", "lambda_none", "lambda_deployed", "suppression_db", "R_none_G1", "R_deployed_G1", "R_none_Gdump", "R_deployed_Gdump", "G_archive", "R_deployed_Garchive", "tau_c_s", "tau_c_status", "G_measured", "R_none_Gmeasured", "R_deployed_Gmeasured", "kept_dumps_cal_q0.5", "kept_dumps_cal_q0.9", "n_dumps"])
    for fid in range(477, 845):
        ch = channel_of(fid); role = "pilot" if PILOT.get(ch) == fid else "data"
        if ch not in verdicts: w.writerow([fid, ch, role, "undetermined", "no products", "", "", "", lam_none.get(ch, ""), lam_dep.get(ch, ""), sup_dep.get(ch, ""), "", "", "", "", "", "", "", "", "", "", "", "", "", 0]); continue
        (disp, reason), res = verdicts[ch]; ka = res["keep_all"]
        if role == "pilot": disp, reason = "pilot", "detector bin, recorded on every channel"
        w.writerow([fid, ch, role, disp, reason, f"{ka['A']:.3e}" if np.isfinite(ka['A']) else "", f"{ka['A_typ']:.3e}" if np.isfinite(ka['A_typ']) else "", f"{ka['G']:.2f}" if np.isfinite(ka['G']) else "", lam_none.get(ch, ""), lam_dep.get(ch, ""), sup_dep.get(ch, ""),
                    f"{ka['R_none_G1']:.3f}" if np.isfinite(ka['R_none_G1']) else "", f"{ka['R_dep_G1']:.3f}" if np.isfinite(ka['R_dep_G1']) else "", f"{ka['R_none_Gd']:.3f}" if np.isfinite(ka['R_none_Gd']) else "", f"{ka['R_dep_Gd']:.3f}" if np.isfinite(ka['R_dep_Gd']) else "", f"{G_arch.get(ch, float('nan')):.0f}", f"{ka['R_dep_Ga']:.3f}" if np.isfinite(ka['R_dep_Ga']) else "",
                    f"{gain[ch]['tau']:.0f}" if ch in gain else "", gain[ch]['status'] if ch in gain else "", f"{gain[ch]['G']:.0f}" if ch in gain else "", f"{ka['R_none_Gm']:.3f}" if np.isfinite(ka['R_none_Gm']) else "", f"{ka['R_dep_Gm']:.3f}" if np.isfinite(ka['R_dep_Gm']) else "",
                    " ".join(res["cal_q0.5"]["kept"]), " ".join(res["cal_q0.9"]["kept"]), len(dumps)])
print(f"wrote {sys.argv[1]} with {845-477} rows from {len(dumps)} dump(s): {', '.join(dumps)}; measured gain on {len(gain)} channel(s)")
print("ch  disposition   policy / reason                                                       A_ns1     A_typ     Gd    R_dep(G1)  R_dep(Gd)  R_dep(Garch)")
for ch in chans:
    (disp, reason), res = verdicts[ch]; ka = res["keep_all"]
    print(f"{ch:2d}  {disp:12s}  {reason[:70]:70s}  {ka['A']:.2e}  {ka['A_typ']:.2e}  {ka['G']:4.1f}  {ka['R_dep_G1']:8.2f}  {ka['R_dep_Gd']:8.2f}  {ka['R_dep_Ga']:9.1f}")
