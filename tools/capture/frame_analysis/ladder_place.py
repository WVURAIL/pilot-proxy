#!/usr/bin/env python3
"""Place a dump on each channel's archive policy ladder.

The archive keeps a frame under policy cal_q when Q = F / mu0 <= eta_q, with eta_q the q-quantile of Q over the
calibration frames (coarse_retention_diagnostic_v1.threshold, method 'higher'). Here eta_q is taken over all valid
archive frames of the channel (per-pilot products of record), and each dump's 33 frames are placed against it.
usage: ladder_place.py <chime-run dir> [<chime-run dir> ...]"""
import glob, os, sys, json
import numpy as np
ARCH = "/home/djg/rail/products/chime_pilots_rebuild_20260829/products/_per_pilot"
PILOT = {14:844,15:829,16:813,17:798,18:783,19:767,20:752,21:736,22:721,23:706,24:690,25:675,26:660,27:644,28:629,29:614,30:598,31:583,32:568,33:552,34:537,35:521,36:506}
QS = (0.1, 0.5, 0.9)
eta = {}
for ch, fid in PILOT.items():
    p = f"{ARCH}/{fid}.npz"
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    mu0 = 2.0 * float(z["target_norm_sq"][0]) / float(z["reference_norm_sum_sq"][0])   # null_power_ratio of the bank
    Q = z["coarse_power_ratio"][:, 0].astype(float) / mu0; Q = Q[v & np.isfinite(Q)]
    eta[ch] = {q: float(np.quantile(Q, q, method="higher")) for q in QS}; eta[ch]["n"] = int(Q.size); eta[ch]["median"] = float(np.median(Q))
print("archive eta (Q = F/mu0) per channel: ch  n_frames  eta0.1  eta0.5  eta0.9")
for ch in sorted(eta): print(f"  {ch:2d}  {eta[ch]['n']:6d}   {eta[ch][0.1]:.4f}  {eta[ch][0.5]:.4f}  {eta[ch][0.9]:.4f}")
for run in sys.argv[1:]:
    d = np.load(os.path.join(run, "chime_detector_outputs.npz"), allow_pickle=True)
    ch_ = d["physical_channel"]; F = d["coarse_power_ratio"]; mu0 = d["null_power_ratio"]; v = d["valid"].astype(bool)
    print(f"\n{os.path.basename(run)}: per channel, median dump Q, and the fraction of dump frames kept under each policy")
    print("  ch   Q_med    archive Q_med   kept@0.1  kept@0.5  kept@0.9   verdict")
    for j in np.argsort(ch_):
        c = int(ch_[j]); 
        if c not in eta: continue
        Q = F[:, j][v[:, j]] / float(mu0[j])
        frac = {q: float(np.mean(Q <= eta[c][q])) for q in QS}
        verdict = "kept by cal_q0.1" if frac[0.1] >= 0.5 else "kept by cal_q0.5" if frac[0.5] >= 0.5 else "kept by cal_q0.9" if frac[0.9] >= 0.5 else "rejected by every calibrated policy"
        print(f"  {c:2d}   {np.median(Q):6.3f}     {eta[c]['median']:6.3f}       {frac[0.1]:.2f}      {frac[0.5]:.2f}      {frac[0.9]:.2f}    {verdict}")
