#!/usr/bin/env python3
"""Table of record: one row per freq_id 477..844, built from the dumps.

usage: table_of_record.py <out csv> <dump label>=<frame_residual csv>:<chime-run dir> [more dumps ...] [gain=<cadence_tau csv>] [lowbound=<frame_residual csv>,...] [bao=<gain_bao csv>]

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
lowb = {}   # amendment 6: channel -> pol -> [(in-band excess, epoch)]
lowb_bao = {}   # amendment 7: channel -> [(east-west median excess, epoch)] over epochs
bao_gain = {}   # amendment 7: channel -> BAO-basis coherence time
EW_CLASSES = ((1, 0), (2, 0), (3, 0))
gain = {}   # amendments 3 and 5: coherence time per channel from cadence_tau.py (in-band row), keyed by channel; refused takes the cap
for arg in sys.argv[2:]:
    label, rest = arg.split("=", 1)
    if label == "lowbound":
        # amendment 6: per-channel, per-polarisation minimum over epochs of the in-band median excess (all frames, (0,1) class)
        for fr in rest.split(","):
            percls = {}
            for r in csv.DictReader(open(fr)):
                if r["frames"] == "all" and r["ew"] == "0" and r["ns"] == "1" and r["pol"] in ("0", "1") and r["excess_inband_median"]:
                    v = float(r["excess_inband_median"])
                    if np.isfinite(v): lowb.setdefault(int(r["channel"]), {}).setdefault(r["pol"], []).append((v, os.path.basename(fr)[15:29]))
                if r["frames"] == "all" and (int(r["ew"]), int(r["ns"])) in EW_CLASSES and r["pol"] in ("0", "1") and r["excess_inband_median"]:
                    v = float(r["excess_inband_median"])
                    if np.isfinite(v): percls.setdefault(int(r["channel"]), {}).setdefault((int(r["ew"]), int(r["ns"])), []).append(v)
            for ch, d in percls.items():
                cls_vals = [max(vv) for vv in d.values()]                       # larger polarisation per class
                if cls_vals: lowb_bao.setdefault(ch, []).append((float(np.median(cls_vals)), os.path.basename(fr)[15:29]))
        continue
    if label == "bao":
        # amendment 7: BAO-basis coherence time per channel (gain_bao.csv built from the east-west classes)
        for r in csv.DictReader(open(rest)):
            if r["bins"] == "inband" and r["tau_c_s"]:
                bao_gain[int(r["channel"])] = dict(status=r["status"], tau=float(r["tau_c_s"]), G=float(r["G"]), basis=r["basis"])
        continue
    if label == "gain":
        for r in csv.DictReader(open(rest)):
            if r["bins"] == "inband" and (r["status"] in ("measured", "bound", "constant") or r["status"].startswith("refused")):   # in-band level, the same quantity as A (amendment 5)
                gain[int(r["channel"])] = dict(status=r["status"], tau=float(r["tau_c_s"]), tau_hi=float(r["tau_c_high_s"]), G=float(r["G"]))
        continue
    fr_csv, run_dir = rest.split(":", 1)
    allrows = [r for r in csv.DictReader(open(fr_csv)) if r["frames"] == "all" and r["ew"] != "-1"]
    # amendment 4: A is the larger of the two same-polarisation readings on the (0, 1) class; G and phi from that polarisation
    ex = {}; expol = {}; exbao = {}
    percls = {}
    for r in allrows:
        if (int(r["ew"]), int(r["ns"])) in EW_CLASSES and r["pol"] in ("0", "1") and r["excess_inband_median"]:
            v = float(r["excess_inband_median"])
            if np.isfinite(v): percls.setdefault(int(r["channel"]), {}).setdefault((int(r["ew"]), int(r["ns"])), []).append(v)
    for ch, d in percls.items():
        exbao[ch] = float(np.median([max(vv) for vv in d.values()]))        # median over the three classes of the larger-polarisation reading
    for r in [r for r in allrows if r["ew"] == "0" and r["ns"] == "1" and r["pol"] in ("0", "1")]:
        ch = int(r["channel"]); vals = (float(r["excess_inband_median"] or "nan"), float(r["excess_pilot_bin"] or "nan"), float(r["G_dump_pilot"] or "nan"), float(r["phi_fast_pilot"] or "nan"))
        expol.setdefault(ch, {})[r["pol"]] = vals
    for ch, pv in expol.items():
        cand = [v for v in pv.values() if np.isfinite(v[0])]
        if cand: ex[ch] = max(cand, key=lambda v: v[0])
        else: ex[ch] = next(iter(pv.values()))
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
    dumps[label] = dict(ex=ex, Qmed=Qmed, typ=typ, expol=expol, exbao=exbao)
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
        Apol = {}
        for pp in ("0", "1"):
            vv = [dumps[l]["expol"][ch][pp][0] for l in kept[pol] if ch in dumps[l]["expol"] and pp in dumps[l]["expol"][ch] and np.isfinite(dumps[l]["expol"][ch][pp][0])]
            Apol[pp] = float(np.median(vv)) if vv else np.nan
        bv = [dumps[l]["exbao"][ch] for l in kept[pol] if ch in dumps[l]["exbao"] and np.isfinite(dumps[l]["exbao"][ch])]
        A_bao = float(np.median(bv)) if bv else np.nan
        Gb = bao_gain[ch]["G"] if ch in bao_gain else np.nan
        R_bao_none_G1 = A_bao / ln if ln else np.nan; R_bao_dep_G1 = A_bao * 10 ** (-sd / 10) / ld if ld else np.nan
        res[pol] = dict(kept=kept[pol], n=len(vals), A=A, A_typ=A_typ, G=G, A_pol0=Apol["0"], A_pol1=Apol["1"], R_none_G1=R_none_G1, R_dep_G1=R_dep_G1,
                        A_bao=A_bao, R_bao_none_G1=R_bao_none_G1, R_bao_dep_G1=R_bao_dep_G1,
                        R_bao_none_Gb=R_bao_none_G1 * Gb if np.isfinite(Gb) else np.nan, R_bao_dep_Gb=R_bao_dep_G1 * Gb if np.isfinite(Gb) else np.nan,
                        R_none_Gd=R_none_G1 * G if np.isfinite(G) else np.nan, R_dep_Gd=R_dep_G1 * G if np.isfinite(G) else np.nan,
                        R_dep_Ga=R_dep_G1 * Ga if np.isfinite(Ga) else np.nan,
                        R_none_Gm=R_none_G1 * Gm if np.isfinite(Gm) else np.nan, R_dep_Gm=R_dep_G1 * Gm if np.isfinite(Gm) else np.nan)
    # (low and low_bao are attached to res below, before any disposition is made)
    # amendment 6: the lowest-epoch bound on the polarisation that sets A
    ka_ = res["keep_all"]; polstar = "1" if (np.isfinite(ka_["A_pol1"]) and (not np.isfinite(ka_["A_pol0"]) or ka_["A_pol1"] >= ka_["A_pol0"])) else "0"
    lowvals = lowb.get(ch, {}).get(polstar, [])
    A_low, ep_low = (min(lowvals) if lowvals else (np.nan, ""))
    ln_, ld_, sd_ = lam_none.get(ch, np.nan), lam_dep.get(ch, np.nan), sup_dep.get(ch, 0.0)
    Gm_ = gain[ch]["G"] if ch in gain else np.nan
    R_low_dep_G1 = A_low * 10 ** (-sd_ / 10) / ld_ if (np.isfinite(A_low) and ld_) else np.nan
    R_low_dep_Gm = R_low_dep_G1 * Gm_ if np.isfinite(R_low_dep_G1) and np.isfinite(Gm_) else np.nan
    low = dict(A=A_low, epoch=ep_low, pol=polstar, R_G1=R_low_dep_G1, R_Gm=R_low_dep_Gm)
    lb = lowb_bao.get(ch, [])
    A_low_b, ep_low_b = (min(lb) if lb else (np.nan, ""))
    Gb_ = bao_gain[ch]["G"] if ch in bao_gain else np.nan
    R_lowb_G1 = A_low_b * 10 ** (-sd_ / 10) / ld_ if (np.isfinite(A_low_b) and ld_) else np.nan
    low_bao = dict(A=A_low_b, epoch=ep_low_b, R_G1=R_lowb_G1, R_Gb=R_lowb_G1 * Gb_ if np.isfinite(R_lowb_G1) and np.isfinite(Gb_) else np.nan)
    res["low"] = low; res["low_bao"] = low_bao
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
                + (f" ({gain[ch]['status']}; cap)" if gain[ch]['status'].startswith('refused') else "")
                + f", G {gain[ch]['G']:.0f}") if ch in gain else ""
        # amendment 7: the BAO basis decides where a BAO-basis gain exists
        ALLOW_DB = 3.0
        if ch in bao_gain:
            bg = bao_gain[ch]; btxt = (f"tau_bao {'= ' if bg['status'] == 'measured' else '>= '}{bg['tau']:.0f} s ({bg['status']}), G_bao {bg['G']:.0f}")
            keep_bao = loosest("deployed", "R_bao_dep_Gb"); keep_bao_none = loosest("none", "R_bao_none_Gb")
            best_bao = min((res[p]["R_bao_dep_Gb"] for p in POLICIES if res[p]["n"] and np.isfinite(res[p]["R_bao_dep_Gb"])), default=np.nan)
            best_bao_G1 = min((res[p]["R_bao_dep_G1"] for p in POLICIES if res[p]["n"] and np.isfinite(res[p]["R_bao_dep_G1"])), default=np.nan)
            bar_b = 10 * np.log10(best_bao) if np.isfinite(best_bao) and best_bao > 0 else np.nan
            worst = f"worst case on the 0.3 m baseline {10*np.log10(best_dep_Gm):.1f} dB over at G {gain[ch]['G']:.0f}" if (ch in gain and np.isfinite(best_dep_Gm) and best_dep_Gm > 0) else ""
            flips = False
            if bg["status"].startswith("fallback") and has_pilot and np.isfinite(best_bao_G1):
                # the east-west coherence is unmeasurable: does the verdict differ between the fallback gain and the persistence bound (the 3600 s class)?
                Gpers = 3600.0 / 0.04194304
                a = best_bao_G1 * bg["G"]; b = best_bao_G1 * Gpers
                flips = (10 * np.log10(a) > ALLOW_DB) != (10 * np.log10(b) > ALLOW_DB) or (a <= 1) != (b <= 1)
            if has_pilot and flips:
                disp = ("undetermined", f"east-west coherence unmeasurable (refused in every class): {10*np.log10(best_bao_G1*bg['G']):.1f} dB at the fallback gain {bg['G']:.0f}, {10*np.log10(best_bao_G1*3600/0.04194304):.1f} dB at the persistence bound; {worst}")
            elif has_pilot and keep_bao:
                disp = ("keep", f"{keep_bao} on the BAO baselines (22 to 66 m; {btxt}; {10*np.log10(res[keep_bao]['R_bao_dep_Gb']):.1f} dB under tolerance with the deployed cut; {'also without credit' if keep_bao_none else 'needs the deployed cut'}); {worst}")
            elif has_pilot and np.isfinite(bar_b) and bar_b > ALLOW_DB:
                disp = ("excise", f"{bar_b:.1f} dB over tolerance on the BAO baselines (22 to 66 m) at the measured coherence time ({btxt}) with the deployed cut under every policy; subtraction bar {bar_b:.1f} dB, {10*np.log10(best_bao_G1):.1f} dB even at G=1; {worst}")
            elif has_pilot and np.isfinite(bar_b):
                disp = ("undetermined", f"marginal: {bar_b:.1f} dB over tolerance on the BAO baselines ({btxt}), within the 3 dB allowance; {worst}")
            elif not has_pilot and np.isfinite(low_bao["R_Gb"]) and low_bao["R_Gb"] > 10 ** (ALLOW_DB / 10):
                disp = ("excise", f"no pilot bin in the dumps; the lowest-epoch residual on the BAO baselines ({low_bao['A']:.2e}, {low_bao['epoch']}) is {10*np.log10(low_bao['R_Gb']):.1f} dB over tolerance at the measured coherence time ({btxt}) with the deployed cut, so no policy can pass (amendments 6 and 7); subtraction bar {10*np.log10(low_bao['R_Gb']):.1f} dB, {10*np.log10(low_bao['R_G1']):.1f} dB even at G=1; {worst}")
            elif not has_pilot and np.isfinite(low_bao["R_Gb"]):
                disp = ("undetermined", f"pilot bin has no live node; the lowest-epoch residual on the BAO baselines is within tolerance or the 3 dB allowance ({10*np.log10(low_bao['R_Gb']):.1f} dB, {btxt}); mask not evaluable")
            else:
                disp = ("undetermined", "no BAO-basis residual")
            res["bao"] = dict(bar=bar_b, G=bg["G"], tau=bg["tau"], status=bg["status"], best_G1=best_bao_G1)
            verdicts[ch] = (disp, res); continue
        if keep_arch: disp = ("keep", f"{keep_arch} at the archive gain bound ({tau_arch.get(ch,'')}; {'also without credit' if keep_none else 'needs the deployed cut'})")
        elif ch in gain and has_pilot and (gain[ch]["status"] in ("measured", "constant") or gain[ch]["status"].startswith("refused")) and keep_meas_G:
            disp = ("keep", f"{keep_meas_G} at the measured coherence time ({gtxt}; {'also without credit' if keep_none_G else 'needs the deployed cut'})")
        elif ch in gain and has_pilot and np.isfinite(best_dep_Gm) and best_dep_Gm > 1:
            disp = ("excise", f"{10*np.log10(best_dep_Gm):.1f} dB over tolerance at the measured coherence time ({gtxt}) with the deployed cut under every policy; subtraction bar {10*np.log10(best_dep_Gm):.1f} dB, {10*np.log10(best_dep_G1):.1f} dB even at G=1")
        elif np.isfinite(best_dep_G1) and best_dep_G1 > 1 and has_pilot: disp = ("excise", f"{10*np.log10(best_dep_G1):.1f} dB over tolerance at G=1 with the deployed cut; subtraction bar {10*np.log10(best_dep_G1):.1f} dB")
        elif not has_pilot and np.isfinite(low["R_Gm"]) and low["R_Gm"] > 1:
            disp = ("excise", f"no pilot bin in the dumps; the lowest-epoch residual ({low['A']:.2e}, {low['epoch']}, pol {low['pol']}) is {10*np.log10(low['R_Gm']):.1f} dB over tolerance at the measured coherence time ({gtxt}) with the deployed cut, so no policy can pass (amendment 6); subtraction bar {10*np.log10(low['R_Gm']):.1f} dB, {10*np.log10(low['R_G1']):.1f} dB even at G=1")
        elif not has_pilot and np.isfinite(low["R_Gm"]): disp = ("undetermined", f"pilot bin has no live node; the lowest-epoch residual passes at the measured coherence time ({gtxt}); mask not evaluable")
        elif not has_pilot: disp = ("undetermined", "pilot bin has no live node; residual measured, mask not evaluable")
        elif keep_meas: disp = ("undetermined", f"passes at the measured gain (G {res[keep_meas]['G']:.1f}, lags to 1.3 s) but not at the archive bound ({tau_arch.get(ch,'')}): tau_c between 1.4 s and the bound decides")
        else: disp = ("undetermined", "over tolerance at the measured gain; no policy passes")
    res["low"] = low; res["low_bao"] = low_bao
    verdicts[ch] = (disp, res)
with open(sys.argv[1], "w") as fh:
    w = csv.writer(fh); w.writerow(["freq_id", "channel", "role", "disposition", "policy_or_reason", "A_keep_all_ns1_bound", "A_keep_all_ns1_pol0", "A_keep_all_ns1_pol1", "A_keep_all_class_median", "G_dump", "lambda_none", "lambda_deployed", "suppression_db", "R_none_G1", "R_deployed_G1", "R_none_Gdump", "R_deployed_Gdump", "G_archive", "R_deployed_Garchive", "tau_c_s", "tau_c_status", "G_measured", "R_none_Gmeasured", "R_deployed_Gmeasured", "A_lowest_epoch", "lowest_epoch", "R_deployed_lowest_Gmeasured", "A_bao_keep_all", "tau_bao_s", "tau_bao_status", "G_bao", "R_none_Gbao", "R_deployed_Gbao", "R_deployed_bao_G1", "A_bao_lowest_epoch", "kept_dumps_cal_q0.5", "kept_dumps_cal_q0.9", "n_dumps"])
    for fid in range(477, 845):
        ch = channel_of(fid); role = "pilot" if PILOT.get(ch) == fid else "data"
        if ch not in verdicts: w.writerow([fid, ch, role, "undetermined", "no products", "", "", "", "", "", lam_none.get(ch, ""), lam_dep.get(ch, ""), sup_dep.get(ch, ""), "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0]); continue
        (disp, reason), res = verdicts[ch]; ka = res["keep_all"]
        if role == "pilot": disp, reason = "pilot", "detector bin, recorded on every channel"
        w.writerow([fid, ch, role, disp, reason, f"{ka['A']:.3e}" if np.isfinite(ka['A']) else "", f"{ka['A_pol0']:.3e}" if np.isfinite(ka['A_pol0']) else "", f"{ka['A_pol1']:.3e}" if np.isfinite(ka['A_pol1']) else "", f"{ka['A_typ']:.3e}" if np.isfinite(ka['A_typ']) else "", f"{ka['G']:.2f}" if np.isfinite(ka['G']) else "", lam_none.get(ch, ""), lam_dep.get(ch, ""), sup_dep.get(ch, ""),
                    f"{ka['R_none_G1']:.3f}" if np.isfinite(ka['R_none_G1']) else "", f"{ka['R_dep_G1']:.3f}" if np.isfinite(ka['R_dep_G1']) else "", f"{ka['R_none_Gd']:.3f}" if np.isfinite(ka['R_none_Gd']) else "", f"{ka['R_dep_Gd']:.3f}" if np.isfinite(ka['R_dep_Gd']) else "", f"{G_arch.get(ch, float('nan')):.0f}", f"{ka['R_dep_Ga']:.3f}" if np.isfinite(ka['R_dep_Ga']) else "",
                    f"{gain[ch]['tau']:.0f}" if ch in gain else "", gain[ch]['status'] if ch in gain else "", f"{gain[ch]['G']:.0f}" if ch in gain else "", f"{ka['R_none_Gm']:.3f}" if np.isfinite(ka['R_none_Gm']) else "", f"{ka['R_dep_Gm']:.3f}" if np.isfinite(ka['R_dep_Gm']) else "",
                    f"{res['low']['A']:.3e}" if np.isfinite(res['low']['A']) else "", res['low']['epoch'], f"{res['low']['R_Gm']:.3f}" if np.isfinite(res['low']['R_Gm']) else "",
                    f"{ka['A_bao']:.3e}" if np.isfinite(ka['A_bao']) else "", f"{bao_gain[ch]['tau']:.0f}" if ch in bao_gain else "", bao_gain[ch]['status'] if ch in bao_gain else "", f"{bao_gain[ch]['G']:.0f}" if ch in bao_gain else "",
                    f"{ka['R_bao_none_Gb']:.3f}" if np.isfinite(ka['R_bao_none_Gb']) else "", f"{ka['R_bao_dep_Gb']:.3f}" if np.isfinite(ka['R_bao_dep_Gb']) else "", f"{ka['R_bao_dep_G1']:.4f}" if np.isfinite(ka['R_bao_dep_G1']) else "", f"{res['low_bao']['A']:.3e}" if np.isfinite(res['low_bao']['A']) else "",
                    " ".join(res["cal_q0.5"]["kept"]), " ".join(res["cal_q0.9"]["kept"]), len(dumps)])
print(f"wrote {sys.argv[1]} with {845-477} rows from {len(dumps)} dump(s): {', '.join(dumps)}; measured gain on {len(gain)} channel(s)")
print("ch  disposition   policy / reason                                                       A_ns1     A_typ     Gd    R_dep(G1)  R_dep(Gd)  R_dep(Garch)")
for ch in chans:
    (disp, reason), res = verdicts[ch]; ka = res["keep_all"]
    print(f"{ch:2d}  {disp:12s}  {reason[:70]:70s}  {ka['A']:.2e}  {ka['A_typ']:.2e}  {ka['G']:4.1f}  {ka['R_dep_G1']:8.2f}  {ka['R_dep_Gd']:8.2f}  {ka['R_dep_Ga']:9.1f}")
