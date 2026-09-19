#!/usr/bin/env python3
"""The baseline-resolved ruling (amendment 8 and its addenda): the table of record per baseline range, with the
control floor, the measured credits and the per-range coherence times.

usage: ruling_baseline.py <in table_of_record csv> <out csv>

Inputs (this directory): class_excess_epochs.csv, class_floor_bins.csv, cadence_tau.csv (class (0,1)),
cadence_tau_ns{32,64,128,255}.csv, cadence_tau_ew{1,2,3}.csv, lag_coherence_tenclasses_D.csv, the kernel run
directories ../kernel_<event>_k230 (policy kept sets) and the archive per-pilot products (eta thresholds)."""
import csv, json, os, sys, re
import numpy as np
FA = os.path.dirname(os.path.abspath(__file__))
PILOT = {14:844,15:829,16:813,17:798,18:783,19:767,20:752,21:736,22:721,23:706,24:690,25:675,26:660,27:644,28:629,29:614,30:598,31:583,32:568,33:552,34:537,35:521,36:506,37:491}
ARCH = "/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot"
POLICIES = ["cal_q0.1", "cal_q0.5", "cal_q0.9", "keep_all"]; QS = {"cal_q0.1": 0.1, "cal_q0.5": 0.5, "cal_q0.9": 0.9}
SCIENCE = {"20260916162300": "pilot", "20260917040230": "D1", "20260917090230": "D2", "20260917140230": "D3"}
TF = 16384 * 2.56e-6; CAP = 86164.0905; ALLOW_DB = 3.0; NSIG = 3.0
RANGES = {"shortest": [(0, 1)], "bao_short": [(0, 32), (0, 64)], "bao_long": [(0, 128), (0, 255), (1, 0), (2, 0), (3, 0)]}
RANGE_TEXT = {"shortest": "the 0.3 m baseline", "bao_short": "the short BAO baselines (9.8 and 19.5 m north-south)", "bao_long": "the long BAO baselines (39 and 78 m north-south, 22 to 66 m east-west)"}
TAU_FILES = {(0, 1): "cadence_tau.csv", (0, 32): "cadence_tau_ns32.csv", (0, 64): "cadence_tau_ns64.csv", (0, 128): "cadence_tau_ns128.csv", (0, 255): "cadence_tau_ns255.csv", (1, 0): "cadence_tau_ew1.csv", (2, 0): "cadence_tau_ew2.csv", (3, 0): "cadence_tau_ew3.csv"}
CLS_NAME = {(0, 1): "ns1", (0, 8): "ns8", (0, 32): "ns32", (0, 64): "ns64", (0, 128): "ns128", (0, 255): "ns255", (1, 0): "ew1", (1, 32): "ew1ns32", (2, 0): "ew2", (3, 0): "ew3"}
inp, outp = sys.argv[1], sys.argv[2]
CH33_RESCAN = sys.argv[3] if len(sys.argv) > 3 else None   # products dir of the measured-pilot rescan: replaces channel 33's ladder thresholds and dump statistics
rows_in = list(csv.DictReader(open(inp)))
lam_none = {int(r["channel"]): float(r["lambda_none"]) for r in rows_in if r["lambda_none"]}
lam_dep = {int(r["channel"]): float(r["lambda_deployed"]) for r in rows_in if r["lambda_deployed"]}
sup = {int(r["channel"]): float(r["suppression_db"]) for r in rows_in if r["suppression_db"]}
# excess per channel, epoch, class, pol
ex = {}
for r in csv.DictReader(open(os.path.join(FA, "class_excess_epochs.csv"))):
    if r["excess"]: ex[(int(r["channel"]), r["epoch"], (int(r["ew"]), int(r["ns"])), int(r["pol"]))] = float(r["excess"])
# floor per class, pol over the four science dumps (channel 37 bins)
fl = {}
for r in csv.DictReader(open(os.path.join(FA, "class_floor_bins.csv"))):
    if r["epoch"] in SCIENCE: fl.setdefault(((int(r["ew"]), int(r["ns"])), int(r["pol"])), []).append(float(r["excess"]))
floor = {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in fl.items()}
# amendment 11: the polarisation that sets A on each class over the four science dumps. Amendment 5 item 1 reads the
# cadence level "on the polarisation that sets the table's A"; after amendment 8 made the ruling per class, that is a
# per-class choice, and the class files of record were produced on a per-class argmax of the raw level over all
# fourteen epochs instead, which is a different quantity. G prices the persistence of the residual A measures, so the
# two must be the same polarisation.
POL_OF_CLASS = {}
for ch in PILOT:
    for c in CLS_NAME:
        best = None
        for p in (0, 1):
            vals = [ex[(ch, e, c, p)] for e in SCIENCE if (ch, e, c, p) in ex]
            if not vals or (c, p) not in floor: continue
            an = float(np.median(vals)) - floor[(c, p)][0]
            if best is None or an > best[1]: best = (p, an)
        if best is not None: POL_OF_CLASS[(ch, c)] = best[0]

# coherence times per class, read on the polarisation that sets A there (amendment 11)
tau = {}
tau_least = {}
tau_pol = {}
for c, f in TAU_FILES.items():
    byp = {}
    for p in (0, 1):
        fp = os.path.join(FA, f[:-4] + f"_pol{p}.csv")
        if not os.path.exists(fp): continue
        for r in csv.DictReader(open(fp)):
            if r["bins"] == "inband": byp[(int(r["channel"]), p)] = r
    for r in csv.DictReader(open(os.path.join(FA, f))):
        if r["bins"] != "inband": continue
        ch = int(r["channel"])
        p = POL_OF_CLASS.get((ch, c))
        row = byp.get((ch, p), r) if p is not None else r
        tau[(ch, c)] = (row["status"], float(row["tau_c_s"]) if row["tau_c_s"] else np.nan)
        # amendment 10 item 7: the least of the channel's own trim probes on this class
        vals = [float(m) for m in re.findall(r"(?:measured|bound|constant)\s+([\d.]+)", row.get("probes", "") or "")]
        tau_least[(ch, c)] = min(vals) if vals else np.nan
        tau_pol[(ch, c)] = row["pol"]
# phasor coherence per class (median over the six pairs, in-band)
rho_v = {}
for r in csv.DictReader(open(os.path.join(FA, "lag_coherence_tenclasses_D.csv"))):
    if r["bins"] == "inband" and r["cls"].endswith("_x"): rho_v.setdefault((int(r["channel"]), r["cls"][:-2]), []).append(float(r["rho"]))
rho = {k: float(np.median(v)) for k, v in rho_v.items()}
# policies: eta from the archive, kept dumps from the kernel runs
eta = {}
for ch, fid in PILOT.items():
    p = f"{ARCH}/{fid}.npz"
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    mu0 = 2.0 * float(z["target_norm_sq"][0]) / float(z["reference_norm_sum_sq"][0])
    Q = z["coarse_power_ratio"][:, 0].astype(float) / mu0; Q = Q[v & np.isfinite(Q)]
    eta[ch] = {k: float(np.quantile(Q, q, method="higher")) for k, q in QS.items()}
Qmed = {}
for ev in SCIENCE:
    d = np.load(os.path.join(FA, "..", f"kernel_{ev}_k230", "chime_detector_outputs.npz"), allow_pickle=True)
    for j, c in enumerate(d["physical_channel"]):
        v = d["valid"][:, j].astype(bool); Q = d["coarse_power_ratio"][:, j][v] / float(d["null_power_ratio"][j]); Qmed[(int(c), ev)] = float(np.median(Q))
if CH33_RESCAN:
    z = np.load(os.path.join(CH33_RESCAN, "chime_detector_outputs.npz"), allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    Q = z["coarse_power_ratio"][:, 0].astype(float) / float(z["null_power_ratio"][0]); Q = Q[v & np.isfinite(Q)]
    eta[33] = {k: float(np.quantile(Q, q, method="higher")) for k, q in QS.items()}
    for ev in SCIENCE:
        d = np.load(os.path.join(FA, "..", f"kernel_{ev}_k230_ch33measured", "chime_detector_outputs.npz"), allow_pickle=True)
        for j, c in enumerate(d["physical_channel"]):
            if int(c) == 33:
                vv = d["valid"][:, j].astype(bool); Qd = d["coarse_power_ratio"][:, j][vv] / float(d["null_power_ratio"][j]); Qmed[(33, ev)] = float(np.median(Qd))
    print("channel 33 under the measured-pilot bank: eta", {k: round(x, 2) for k, x in eta[33].items()}, "dump Q medians", {ev: round(Qmed[(33, ev)], 2) for ev in SCIENCE})
def kept(ch, pol):
    if pol == "keep_all": return list(SCIENCE)
    if ch not in eta: return []
    return [ev for ev in SCIENCE if (ch, ev) in Qmed and Qmed[(ch, ev)] <= eta[ch][pol]]
def gmed(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else float(np.sqrt(v[n // 2 - 1] * v[n // 2]))
def db(x): return 10 * np.log10(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan

def class_reading(ch, epochs, c):
    """(A_net, measured, floor_mean, floor_sd, pol) on the larger-polarisation reading over the given epochs."""
    best = None
    for p in (0, 1):
        vals = [ex[(ch, e, c, p)] for e in epochs if (ch, e, c, p) in ex]
        if not vals or (c, p) not in floor: continue
        A = float(np.median(vals)); fm, fs = floor[(c, p)]
        cand = dict(A_net=A - fm, measured=A > fm + NSIG * fs, fm=fm, fs=fs, pol=p, A=A)
        if best is None or cand["A_net"] > best["A_net"]: best = cand
    return best

def _tau_of(ch, c, least):
    """The class's coherence time, or under least= the smallest of its trim probes."""
    if least:
        v = tau_least.get((ch, c), np.nan)
        if np.isfinite(v): return v
    return tau[(ch, c)][1]

def rescue_rho(bar_db, rh):
    """The coherence a ground filter must reach to bring bar_db inside the allowance."""
    if not np.isfinite(bar_db) or bar_db <= ALLOW_DB: return np.nan
    rem = (1.0 - rh ** 2) * 10 ** ((ALLOW_DB - bar_db) / 10.0)
    return float(np.sqrt(1.0 - rem)) if 0.0 < rem < 1.0 else np.nan

def gain_classes(ch, rng):
    """Amendment 11 item 2: a range's gain is read only on classes whose excess is measured above the
    control floor. G prices the persistence of the residual A measures, and a class at the floor has no
    measured residual to persist."""
    out = [c for c in RANGES[rng] if (lambda r: bool(r) and r["measured"])(class_reading(ch, list(SCIENCE), c))]
    return out or RANGES[rng]

def range_gain(ch, rng, least=False):
    CLASSES = gain_classes(ch, rng)
    meas = [_tau_of(ch, c, least) for c in CLASSES if (ch, c) in tau and tau[(ch, c)][0] == "measured" and np.isfinite(tau[(ch, c)][1])]
    meas = [v for v in meas if np.isfinite(v)]
    bnd = [_tau_of(ch, c, least) for c in CLASSES if (ch, c) in tau and tau[(ch, c)][0] == "bound" and np.isfinite(tau[(ch, c)][1])]
    bnd = [v for v in bnd if np.isfinite(v)]
    rh = range_rho(ch, rng)
    if meas: return gmed(meas), "measured", ""
    if bnd: return bnd[0], "bound", ""
    if rng == "bao_long" and (ch, (0, 64)) in tau and tau[(ch, (0, 64))][0] == "measured" and np.isfinite(tau[(ch, (0, 64))][1]):
        return _tau_of(ch, (0, 64), least), "borrowed", f"no measured class on the long range; the channel's own measured coherence time on the 19.5 m class is used (amendment 9 item 3)"
    return np.nan, "unmeasured", "no measured class on the range and no borrowed measurement (amendment 9 item 3)"
def range_rho(ch, rng):
    v = [rho[(ch, CLS_NAME[c])] for c in RANGES[rng] if (ch, CLS_NAME[c]) in rho]
    return float(min(max(np.median(v), 0.0), 0.95)) if v else 0.0

results = {}
has_pilot = {ch: (ch in eta and any((ch, ev) in Qmed for ev in SCIENCE)) for ch in PILOT}
for ch in sorted({r for r in PILOT}):
    if ch not in lam_dep: continue
    S = 10 ** (-sup.get(ch, 0.0) / 10); ld = lam_dep[ch]; ln = lam_none.get(ch, np.nan)
    per = {}
    for rng, classes in RANGES.items():
        t, tstat, tnote = range_gain(ch, rng); G = min(t, CAP) / TF if np.isfinite(t) else np.nan
        tL, _, _ = range_gain(ch, rng, least=True); GL = min(tL, CAP) / TF if np.isfinite(tL) else np.nan
        probe_db = db(GL / G) if (np.isfinite(G) and np.isfinite(GL) and G > 0) else 0.0
        rh = range_rho(ch, rng); credit = 1 - rh ** 2
        pols = {}
        for pol in POLICIES:
            eps = kept(ch, pol) if has_pilot[ch] else (list(SCIENCE) if pol == "keep_all" else [])
            if not eps: continue
            readings = [class_reading(ch, eps, c) for c in classes]; readings = [x for x in readings if x]
            measured = [x for x in readings if x["measured"]]
            if not readings: continue
            A_net = float(np.median([x["A_net"] for x in measured])) if measured else np.nan
            floor_level = float(np.median([x["fm"] + NSIG * x["fs"] for x in readings]))
            pols[pol] = dict(A_net=A_net, measured=bool(measured), n_measured=len(measured), floor_level=floor_level,
                             R_dep=(A_net * S * credit * G / ld) if (measured and np.isfinite(G)) else np.nan,
                             R_none=(A_net * G / ln) if (measured and np.isfinite(G) and np.isfinite(ln)) else np.nan,
                             R_dep_G1=(A_net * S * credit / ld) if measured else np.nan,
                             R_floor=(floor_level * S * credit * G / ld) if np.isfinite(G) else np.nan,
                             R_dep_Gdump=(A_net * S * credit / ld) if measured else np.nan, kept=eps)
        # lowest-epoch bound for channels without a pilot bin (all fourteen epochs, keep-all)
        low = None
        if not has_pilot[ch]:
            eps14 = sorted({e for (c_, e, cc, p) in ex if c_ == ch})
            vals = []
            at_floor_epoch = None
            for e in eps14:
                rd_all = [class_reading(ch, [e], c) for c in classes]; rd = [x for x in rd_all if x and x["measured"]]
                if rd: vals.append((float(np.median([x["A_net"] for x in rd])), e))
                elif any(rd_all): at_floor_epoch = e                        # amendment 9 item 2: an epoch at the floor is the minimum
            if at_floor_epoch is not None: low = dict(A=np.nan, epoch=at_floor_epoch, R_dep=np.nan, R_dep_G1=np.nan, at_floor=True)
            elif vals:
                a, e = min(vals); low = dict(A=a, epoch=e, R_dep=(a * S * credit * G / ld) if np.isfinite(G) else np.nan, R_dep_G1=a * S * credit / ld, at_floor=False)
        per[rng] = dict(tau=t, tstat=tstat, tnote=tnote, G=G, tau_least=tL, G_least=GL, probe_db=probe_db, rho=rh, credit_db=-db(credit) if credit > 0 else np.nan, pols=pols, low=low)
        # verdict on the range
        cand = [(p, d) for p, d in pols.items() if d["measured"] and np.isfinite(d["R_dep"])]
        anymeas = any(d["measured"] for d in pols.values())
        if not pols: v = ("undetermined", "no reading")
        elif not anymeas:
            fr = pols["keep_all"]["R_floor"] if "keep_all" in pols else np.nan
            v = ("at floor", f"not measured above the control floor (the floor itself is {db(fr):.1f} dB over tolerance at G {G:.0f})" if np.isfinite(fr) else "not measured above the control floor (gain unmeasured)")
        elif tstat == "unmeasured":
            best_G1 = min(d["R_dep_G1"] for p, d in pols.items() if d["measured"])
            v = ("unpriced", f"measured above the floor but the gain is unmeasured: {db(best_G1):.1f} dB at G = 1, {db(best_G1 * 3600 / TF):.1f} dB at the persistence bound; {tnote}")
        else:
            if not has_pilot[ch] and low is not None and low.get("at_floor"):
                v = ("at floor", f"the lowest epoch ({low['epoch']}) is not measured above the control floor on any class of the range, so the bound is at the floor and cannot excise (amendment 9)")
            elif not has_pilot[ch] and low is not None and np.isfinite(low["R_dep"]):
                bar = db(low["R_dep"]); pbest = "lowest-epoch bound"
                if bar > ALLOW_DB and bar + probe_db > ALLOW_DB: v = ("excise", f"{bar:.1f} dB over tolerance on the lowest epoch ({low['epoch']}) with both credits at G {G:.0f} ({tstat}), {bar + probe_db:.1f} dB at the least trim probe; no policy can pass; {db(low['R_dep_G1']):.1f} dB at G = 1; rescued only by a ground filter reaching coherence {rescue_rho(bar, rh):.3f} against the measured {rh:.2f}")
                elif bar > ALLOW_DB: v = ("marginal", f"{bar:.1f} dB over tolerance on the lowest epoch ({low['epoch']}) but {bar + probe_db:.1f} dB at the least trim probe, inside the {ALLOW_DB:.0f} dB allowance (amendment 10 item 7)")
                else: v = ("undetermined", f"lowest-epoch bound {bar:.1f} dB, within the allowance; mask not evaluable")
            elif cand:
                pbest, dbest = min(cand, key=lambda x: x[1]["R_dep"]); bar = db(dbest["R_dep"])
                loosest = next((p for p in reversed(POLICIES) if p in pols and pols[p]["measured"] and np.isfinite(pols[p]["R_dep"]) and pols[p]["R_dep"] <= 1), None)
                ka_meas = "keep_all" in pols and pols["keep_all"]["measured"]
                if loosest: v = ("keep", f"{loosest} passes ({db(pols[loosest]['R_dep']):.1f} dB) with both credits at G {G:.0f} ({tstat})")
                elif not ka_meas:
                    fr = pols["keep_all"]["R_floor"] if "keep_all" in pols else np.nan
                    v = ("at floor", f"keep-all is not measured above the control floor (the floor itself is {db(fr):.1f} dB over tolerance at G {G:.0f}); the calibrated policies read {bar:.1f} dB over on their kept dumps")
                elif bar > ALLOW_DB and bar + probe_db > ALLOW_DB: v = ("excise", f"{bar:.1f} dB over tolerance under every policy (least: {pbest}) with both credits at G {G:.0f} ({tstat}), {bar + probe_db:.1f} dB at the least trim probe; {db(dbest['R_dep_G1']):.1f} dB at G = 1; rescued only by a ground filter reaching coherence {rescue_rho(bar, rh):.3f} against the measured {rh:.2f}")
                elif bar > ALLOW_DB: v = ("marginal", f"{bar:.1f} dB over tolerance ({pbest}) at G {G:.0f} ({tstat}) but {bar + probe_db:.1f} dB at the least trim probe, inside the {ALLOW_DB:.0f} dB allowance (amendment 10 item 7)")
                else: v = ("marginal", f"{bar:.1f} dB over tolerance ({pbest}), within the {ALLOW_DB:.0f} dB allowance, at G {G:.0f} ({tstat})")
            else: v = ("undetermined", "measured on no evaluable policy")
        per[rng]["verdict"] = v
    results[ch] = per
# summary verdict per channel
summary = {}
for ch, per in results.items():
    L, Sh, W = per["bao_long"]["verdict"], per["bao_short"]["verdict"], per["shortest"]["verdict"]
    def bartxt(v): m = re.search(r"([-\d.]+) dB over tolerance", v[1]); return m.group(1) if m else ""
    if L[0] == "excise":
        disp, reason = "excise", f"excised on {RANGE_TEXT['bao_long']}: {L[1]}; on the short BAO baselines: {Sh[0]} ({Sh[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]}); subtraction bar {bartxt(L)} dB, {re.search(r'([-\d.]+) dB at G = 1', L[1]).group(1) if re.search(r'([-\d.]+) dB at G = 1', L[1]) else ''} dB even at G=1"
    elif Sh[0] == "excise":
        disp, reason = "excise", f"excised on {RANGE_TEXT['bao_short']}: {Sh[1]}; on the long BAO baselines: {L[0]} ({L[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]}); subtraction bar {bartxt(Sh)} dB, {re.search(r'([-\d.]+) dB at G = 1', Sh[1]).group(1) if re.search(r'([-\d.]+) dB at G = 1', Sh[1]) else ''} dB even at G=1"
    elif L[0] == "keep" and Sh[0] in ("keep", "at floor"):
        disp, reason = "keep", f"keeps on the BAO baselines: long, {L[1]}; short, {Sh[0]} ({Sh[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]})"
    elif L[0] == "unpriced" or Sh[0] == "unpriced":
        disp, reason = "undetermined", f"measured above the floor but unpriced: long BAO baselines, {L[0]} ({L[1]}); short BAO baselines, {Sh[0]} ({Sh[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]})"
    elif L[0] == "marginal" or Sh[0] == "marginal":
        disp, reason = "undetermined", f"marginal: long BAO baselines, {L[0]} ({L[1]}); short BAO baselines, {Sh[0]} ({Sh[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]})"
    else:
        disp, reason = "undetermined", f"at the control floor on the BAO baselines: long, {L[1]}; short, {Sh[0]} ({Sh[1]}); worst case on the 0.3 m baseline: {W[0]} ({W[1]})"
    summary[ch] = (disp, reason)
# write: keep the input columns, replace disposition and reason for data rows, append range columns
newcols = []
for rng in RANGES: newcols += [f"{rng}_A_net", f"{rng}_measured", f"{rng}_floor_level", f"{rng}_tau_s", f"{rng}_tau_status", f"{rng}_tau_least_s", f"{rng}_probe_db", f"{rng}_rescue_rho", f"{rng}_G", f"{rng}_rho", f"{rng}_ground_credit_db", f"{rng}_R_deployed", f"{rng}_R_none", f"{rng}_verdict"]
fields = list(rows_in[0].keys()) + newcols
with open(outp, "w") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    for r in rows_in:
        ch = int(r["channel"]); o = dict(r)
        if r["role"] != "pilot" and ch in summary and r["policy_or_reason"] != "no products":
            o["disposition"], o["policy_or_reason"] = summary[ch]
            if has_pilot.get(ch):
                o["kept_dumps_cal_q0.5"] = " ".join(SCIENCE[e] for e in kept(ch, "cal_q0.5")); o["kept_dumps_cal_q0.9"] = " ".join(SCIENCE[e] for e in kept(ch, "cal_q0.9"))
            if ch == 37: o["disposition"], o["policy_or_reason"] = "undetermined", "reference channel (no DTV allocation): the control whose reading defines the floor; no tolerance on the board"
        for rng in RANGES:
            p = results.get(ch, {}).get(rng)
            if not p: continue
            ka = p["pols"].get("keep_all", {})
            o[f"{rng}_A_net"] = f"{ka.get('A_net', np.nan):.3e}" if ka and np.isfinite(ka.get("A_net", np.nan)) else ""
            o[f"{rng}_measured"] = str(ka.get("measured", "")) if ka else ""
            o[f"{rng}_floor_level"] = f"{ka.get('floor_level', np.nan):.3e}" if ka and np.isfinite(ka.get("floor_level", np.nan)) else ""
            o[f"{rng}_tau_s"] = f"{p['tau']:.0f}" if np.isfinite(p["tau"]) else ""; o[f"{rng}_tau_status"] = p["tstat"]
            o[f"{rng}_tau_least_s"] = f"{p['tau_least']:.0f}" if np.isfinite(p["tau_least"]) else ""; o[f"{rng}_probe_db"] = f"{p['probe_db']:+.1f}"
            o[f"{rng}_G"] = f"{p['G']:.0f}" if np.isfinite(p["G"]) else ""; o[f"{rng}_rho"] = f"{p['rho']:.2f}"; o[f"{rng}_ground_credit_db"] = f"{p['credit_db']:.1f}" if np.isfinite(p["credit_db"]) else ""
            o[f"{rng}_R_deployed"] = f"{ka.get('R_dep', np.nan):.4g}" if ka and np.isfinite(ka.get("R_dep", np.nan)) else ""
            o[f"{rng}_R_none"] = f"{ka.get('R_none', np.nan):.4g}" if ka and np.isfinite(ka.get("R_none", np.nan)) else ""
            o[f"{rng}_verdict"] = p["verdict"][0]
            mb = re.search(r"([-\d.]+) dB over tolerance", p["verdict"][1])
            rr = rescue_rho(float(mb.group(1)), p["rho"]) if mb else np.nan
            o[f"{rng}_rescue_rho"] = f"{rr:.3f}" if np.isfinite(rr) else ""
        w.writerow(o)
print("ch  summary        | shortest              | short BAO              | long BAO")
for ch in sorted(results):
    per = results[ch]
    print(f"{ch:2d}  {summary[ch][0]:13s} | " + " | ".join(f"{per[r]['verdict'][0]:10s} {db(per[r]['pols'].get('keep_all', {}).get('R_dep', np.nan)):+6.1f}dB G{per[r]['G']:.0f} r{per[r]['rho']:.2f}" if np.isfinite(per[r]['G']) else f"{per[r]['verdict'][0]:10s}   n/a" for r in RANGES))
