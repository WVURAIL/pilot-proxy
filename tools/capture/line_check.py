#!/usr/bin/env python3
"""Where is the pilot line, per pilot file, under both spectral senses.
usage: line_check.py <products dir>"""
import glob, json, os, sys
import numpy as np
NFFT = 16384; W = 0.390625; DK = W * 1e3 / NFFT   # kHz per fine bin
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
    z = np.load(f); m = json.loads(str(z["meta"]))
    if not m["is_pilot"]: continue
    psd = z["block_psd"].sum(axis=1).mean(axis=0)          # mean over frames, sum over blocks -> [16384]
    med = np.median(psd); kp = m["pilot_fine_bin_naive"]
    def best(c, half=600):
        idx = (c + np.arange(-half, half + 1)) % NFFT; j = idx[np.argmax(psd[idx])]
        d = ((j - c + NFFT // 2) % NFFT) - NFFT // 2
        return j, round(float(psd[j] / med), 1), round(d * DK, 2)
    jn, rn, dn = best(kp % NFFT)          # naive sense: bin +k <-> sky above centre
    ji, ri, di = best((-kp) % NFFT)       # inverted sense: bin +k <-> sky below centre
    top = np.argsort(psd)[-3:][::-1]
    def sky_inv(b): return m["freq_mhz"] - (((b + NFFT // 2) % NFFT) - NFFT // 2) * DK / 1e3
    tops = ", ".join(f"{sky_inv(b):.4f} MHz x{psd[b]/med:.0f}" for b in top)
    print(f"ch{m['channel']:02d} fid {m['freq_id']} pilot nominal {m['freq_mhz'] + kp*DK/1e3:.6f} MHz | naive: x{rn} at {dn:+.2f} kHz | inverted: x{ri} at {di:+.2f} kHz (sky {sky_inv(ji):.4f}) | top lines (inverted sky): {tops}")
