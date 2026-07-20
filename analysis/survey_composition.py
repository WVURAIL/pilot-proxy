#!/usr/bin/env python3
"""Survey composition and class-stratified secular robustness.

Joins the archive inventory (event -> obs_date, dataset classes, per-fid
n_frames) to the per-frame products, and answers the survey-bias question:
do the secular rate transitions survive restriction to a single uniform
trigger class (classified.FRB)?

Inputs (env-overridable):
  PP_INVENTORY  datatrawl inventory.jsonl        (default ~/data/chime-pilots/inventory.jsonl)
  PP_EVENTKEYS  event_presence_keys.csv.gz       (default: the copy committed
                in data/provenance/survey_stratum_20260718/)
  PP_PERFRAME   per-frame dump (shared _paths default)

Unit->event assignment: units carry only capture times; events carry only
obs_date (day). Within each (channel, day+/-1) group, units are matched to
events by frame count (unit frames == floor(inventory n_frames), the rule
measured at 97.4 per cent on singleton days). Outcomes are tallied as
  exact   day+framecount unique class
  classok day+framecount multiple events but one class
  daypure day fallback, single-class day
  mixed   irreducibly ambiguous class (excluded from strata)
  orphan  no inventory day within +/-1 (excluded)

Outputs (in PP_OUT):
  survey_composition_by_channel.csv   sampled vs archive class mix per channel
  survey_quarterly_exposure.csv       units and valid frames per quarter
  survey_frb_stratum_rates.csv        quarterly hi-rates, all vs FRB stratum
  survey_quarterly_rates_all23.csv    quarterly hi-rates, EVERY calibrated
                                      channel (all + FRB strata; the
                                      completeness scan behind the Sec. 6.3
                                      "one further transition" statement and
                                      the Sec. 8.1 tier flatness claims;
                                      refused channels 24/30 excluded -- no
                                      calibrated zero point to rate against)
  fig_secular_frb_stratum.(png|pdf)   episodic channels, all vs FRB stratum
"""
import collections
import csv
import datetime
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from pilot_proxy.plot_style import setup_matplotlib

plt = setup_matplotlib()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"
OUT = _paths.OUT
INV = Path(os.environ.get("PP_INVENTORY",
                          str(Path.home() / "data/chime-pilots/inventory.jsonl")))
KEYS = Path(os.environ.get("PP_EVENTKEYS",
                           str(_paths.REPO / "data/provenance"
                               / "survey_stratum_20260718"
                               / "event_presence_keys.csv.gz")))
EPISODIC = (17, 32, 33, 35)
C_ALL, C_FRB = "0.35", "#0072B2"
TRANS = {33: 2020.2, 32: 2023.3, 35: 2021.7, 17: None}    # measured falls/rises


def classify(ds):
    ds = ds or []
    joined = " ".join(ds)
    if "classified.FRB" in ds:
        return "classified.FRB"
    if any("SGR" in d for d in ds):
        return "SGR"
    if "B0531+21.commissioning" in joined:
        return "crab.commissioning"
    if "pulsar" in joined.lower() or "PULSAR" in joined:
        return "pulsar"
    if "scheduled" in joined:
        return "scheduled"
    return "other"


CLASSES = ["classified.FRB", "crab.commissioning", "pulsar", "SGR",
           "scheduled", "other"]

# ---- inventory ------------------------------------------------------------
ev_date, ev_cls = {}, {}
nf = {}
ev_fids = collections.defaultdict(set)
for line in open(INV):
    r = json.loads(line)
    e = r["event"]
    ev_fids[e].add(r["freq_id"])
    nf[(e, r["freq_id"])] = float(r["n_frames"])
    if e not in ev_date:
        ev_date[e] = r["obs_date"]
        ev_cls[e] = classify(r.get("datasets"))

# ---- product presence -----------------------------------------------------
with gzip.open(KEYS, "rt") as fh:
    rd = csv.reader(fh)
    ORDER = [int(x) for x in next(rd)[1:]]
    next(rd)
    prod = [(ev.replace("baseband_", "").replace(".h5", ""), int(bm))
            for ev, bm in rd]

z = np.load(_paths.PERFRAME)
zs = np.load(_paths.SPECTRA)
fid2ch = {int(zs[k][2]): int(k[2:].split("_")[0])
          for k in zs.files if k.endswith("_meta")}
study = {int(r["atsc_channel"]): r for r in
         csv.DictReader(open(OUT / "empirical_zero_points.csv"))}

# ---- unit -> event class assignment --------------------------------------
outcome = collections.Counter()
unit_class = {}          # ch -> list of class-or-None per unit
comp_sampled = {ch: collections.Counter() for ch in fid2ch.values()}
comp_archive = {ch: collections.Counter() for ch in fid2ch.values()}
for e, fids in ev_fids.items():
    for fid in fids:
        if fid in fid2ch:
            comp_archive[fid2ch[fid]][ev_cls[e]] += 1

for i, fid in enumerate(ORDER):
    ch = fid2ch[fid]
    evs = [n for n, bm in prod if (bm >> i) & 1]
    for e in evs:
        comp_sampled[ch][ev_cls[e]] += 1
    by_day = collections.defaultdict(list)
    for e in evs:
        by_day[ev_date[e]].append(e)
    fui = z[f"ch{ch}_frame_unit_index"]
    t0 = z[f"ch{ch}_unit_time0_ctime"]
    ufr = np.bincount(fui, minlength=len(t0))
    used = set()
    ucls = []
    for u in range(len(t0)):
        d0 = datetime.datetime.utcfromtimestamp(t0[u]).date()
        cands_days = [str(d0), str(d0 - datetime.timedelta(1)),
                      str(d0 + datetime.timedelta(1))]
        chosen = None
        for dstr in cands_days:
            grp = [e for e in by_day.get(dstr, []) if e not in used]
            if not grp:
                continue
            match = [e for e in grp if int(ufr[u]) == int(nf[(e, fid)])]
            pool = match if match else grp
            classes = {ev_cls[e] for e in pool}
            if match and len(match) == 1:
                outcome["exact"] += 1
            elif match and len(classes) == 1:
                outcome["classok"] += 1
            elif not match and len(classes) == 1:
                outcome["daypure"] += 1
            else:
                outcome["mixed"] += 1
                used.add(pool[0])
                chosen = (pool[0], None)
                break
            used.add(pool[0])
            chosen = (pool[0], classes.pop())
            break
        if chosen is None:
            outcome["orphan"] += 1
            ucls.append(None)
        else:
            ucls.append(chosen[1])
    unit_class[ch] = ucls

tot_u = sum(outcome.values())
print("unit->class assignment outcomes "
      f"({tot_u} units): " + ", ".join(f"{k}={v} ({100*v/tot_u:.1f}%)"
                                       for k, v in outcome.most_common()))

with open(OUT / "survey_assignment_quality.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "n_units", "n_assigned",
                "n_ambiguous_or_orphan", "assigned_fraction"])
    for i, fid in enumerate(ORDER):
        ch = fid2ch[fid]
        cls_list = unit_class[ch]
        n = len(cls_list)
        n_ok = sum(1 for c in cls_list if c is not None)
        w.writerow([ch, fid, n, n_ok, n - n_ok,
                    f"{n_ok / max(n, 1):.4f}"])

# ---- composition table ----------------------------------------------------
with open(OUT / "survey_composition_by_channel.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["atsc_channel", "freq_id", "population", "n_events"]
               + CLASSES)
    for i, fid in enumerate(ORDER):
        ch = fid2ch[fid]
        for pop, cc in (("sampled", comp_sampled[ch]),
                        ("archive", comp_archive[ch])):
            n = sum(cc.values())
            w.writerow([ch, fid, pop, n]
                       + [f"{cc.get(c, 0) / max(n, 1):.4f}" for c in CLASSES])

# ---- quarterly exposure + stratified rates -------------------------------
def quarter(ts):
    d = datetime.datetime.utcfromtimestamp(ts)
    return d.year + ((d.month - 1) // 3) / 4.0

rate_rows = []
fig, axes = plt.subplots(len(EPISODIC), 1, figsize=(7.2, 8.6), sharex=True)
qx_rows = []
for ax, ch in zip(axes, EPISODIC):
    s = study[ch]
    mu0 = float(s["mu0_analytic"])
    mu_hat = float(s["mu0_empirical"])
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    ok = valid & np.isfinite(f)
    hi = ok & (f > mu_hat + 12e-3 * mu0)
    fui = z[f"ch{ch}_frame_unit_index"]
    t0 = z[f"ch{ch}_unit_time0_ctime"]
    uq = np.array([quarter(t) for t in t0])
    ucls = unit_class[ch]
    frb_units = np.array([c == "classified.FRB" for c in ucls])
    frame_q = uq[fui]
    frame_frb = frb_units[fui]
    for stratum, sel_frames in (("all", np.ones_like(ok)),
                                ("frb", frame_frb)):
        qs = sorted(set(uq))
        xs, ys, ns = [], [], []
        for q in qs:
            m = ok & sel_frames & (frame_q == q)
            n = int(m.sum())
            if n >= 40:                      # rate needs a floor of frames
                xs.append(q)
                ys.append(hi[m].mean())
                ns.append(n)
                rate_rows.append({"atsc_channel": ch, "stratum": stratum,
                                  "quarter": f"{q:.2f}", "n_valid_frames": n,
                                  "hi_rate": f"{hi[m].mean():.4f}"})
        lw = 0.0 if ch == 17 else 1.1     # ch17: endpoints only, no
        # connecting line across its coverage hole (quarters failing the
        # exposure floor)
        style = dict(color=C_ALL, lw=lw, ls="-", marker="o", ms=3.0) \
            if stratum == "all" else \
            dict(color=C_FRB, lw=lw, ls="-", marker="s", ms=3.0, alpha=0.9)
        ax.plot(xs, ys, label=("all sampled events" if stratum == "all"
                               else "classified.FRB stratum"), **style)
    # quarterly exposure bookkeeping (all units)
    for q in sorted(set(uq)):
        m_u = uq == q
        qx_rows.append({"atsc_channel": ch, "quarter": f"{q:.2f}",
                        "n_units": int(m_u.sum()),
                        "n_frb_units": int((m_u & frb_units).sum())})
    if TRANS.get(ch):
        ax.axvline(TRANS[ch], color="0.75", ls=":", lw=0.9)
    ax.set_ylabel(f"ch{ch} hi-rate", fontsize=8.5)
    ax.grid(color="0.93", lw=0.4)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8, frameon=False, ncol=2)
axes[-1].set_xlabel("year (quarterly bins; quarters with $\\geq$40 valid "
                    "frames in the stratum)")
fig.suptitle("Secular detection-rate transitions: all sampled events vs the "
             "classified.FRB stratum\n(hi-rate $= P[F > \\hat\\mu_0 + "
             "12\\times10^{-3}\\mu_0]$ per quarter)", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "fig_secular_frb_stratum.png", dpi=220,
            bbox_inches="tight")
fig.savefig(OUT / "fig_secular_frb_stratum.pdf", bbox_inches="tight")

with open(OUT / "survey_quarterly_exposure.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(qx_rows[0].keys()))
    w.writeheader()
    w.writerows(qx_rows)
with open(OUT / "survey_frb_stratum_rates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rate_rows[0].keys()))
    w.writeheader()
    w.writerows(rate_rows)

# ---- completeness scan: quarterly hi-rates for EVERY calibrated channel ---
# The episodic set of Fig. 7 is defined by the full-period high-tail
# fraction (>3 per cent), which dilutes any loud epoch that is short
# relative to a channel's total exposure. This scan re-derives quarterly
# rates (all events + classified.FRB stratum, same 40-frame floor) for all
# calibrated channels so that epoch structure below the tail criterion is
# archived rather than asserted. Refused channels (zero_point_trusted=0)
# are excluded: with no calibrated null, a rate against mu_hat is not
# meaningful there.
all_rows = []
for ch in sorted(unit_class):
    s = study[ch]
    if s.get("zero_point_trusted") != "1":
        continue
    mu0 = float(s["mu0_analytic"])
    mu_hat = float(s["mu0_empirical"])
    pt = z[f"ch{ch}_p_target_u64"].astype(np.float64)
    pr = z[f"ch{ch}_p_ref_sum_u64"].astype(np.float64)
    valid = z[f"ch{ch}_valid"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * pt / pr
    ok = valid & np.isfinite(f)
    hi = ok & (f > mu_hat + 12e-3 * mu0)
    fui = z[f"ch{ch}_frame_unit_index"]
    t0 = z[f"ch{ch}_unit_time0_ctime"]
    uq = np.array([quarter(t) for t in t0])
    frb_units = np.array([c == "classified.FRB" for c in unit_class[ch]])
    frame_q = uq[fui]
    frame_frb = frb_units[fui]
    for q in sorted(set(uq)):
        m_all = ok & (frame_q == q)
        if int(m_all.sum()) < 40:
            continue
        m_frb = m_all & frame_frb
        all_rows.append({
            "atsc_channel": ch, "quarter": f"{q:.2f}",
            "n_valid_frames": int(m_all.sum()),
            "hi_rate_all": f"{hi[m_all].mean():.4f}",
            "n_frb_frames": int(m_frb.sum()),
            "hi_rate_frb": (f"{hi[m_frb].mean():.4f}"
                            if int(m_frb.sum()) >= 40 else "")})
with open(OUT / "survey_quarterly_rates_all23.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)

print("wrote survey_composition_by_channel.csv, survey_quarterly_exposure.csv,")
print("      survey_frb_stratum_rates.csv, survey_quarterly_rates_all23.csv,")
print("      fig_secular_frb_stratum")
