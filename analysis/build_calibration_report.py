#!/usr/bin/env python3
"""Render the calibration report, with every figure embedded.

    python3 analysis/build_calibration_report.py [--out FILE]
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from string import Template

import _calibration_paths as P
from ppcal.state import CARRIER_DOMINATED_LEVEL_DB

OUT = str(P.OUT)
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")


def rows(name, required=True):
    """Table rows, or an empty list for an optional table not yet built."""
    path = os.path.join(TAB, name)
    if not os.path.exists(path):
        if required:
            raise SystemExit(
                f"{path} is missing; run make_calibration_data.py first")
        print(f"note: {name} not found, its section will be omitted")
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def img(name, alt, caption, cls=""):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return ('<figure class="fig %s">\n'
            '<img src="data:image/png;base64,%s" alt="%s" loading="lazy">\n'
            '<figcaption>%s</figcaption>\n</figure>\n'
            % (cls, b64, html.escape(alt), caption))


def table(headers, body_rows, cls="", align=None):
    align = align or []
    th = "".join('<th%s>%s</th>'
                 % (' class="num"' if i in align else "", h)
                 for i, h in enumerate(headers))
    trs = []
    for r in body_rows:
        tds = "".join('<td%s>%s</td>'
                      % (' class="num"' if i in align else "", c)
                      for i, c in enumerate(r))
        trs.append("<tr>%s</tr>" % tds)
    return ('<div class="scroll"><table class="%s"><thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table></div>' % (cls, th, "".join(trs)))


def fmt(v, spec="%.3f", dash="&mdash;"):
    try:
        if v in ("", None):
            return dash
        return spec % float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(OUT, "calibration_report.html"))
    args = ap.parse_args(argv)

    doc = json.load(open(os.path.join(OUT, "calibration.json"), encoding="utf-8"))
    t = doc["totals"]
    cal = rows("calibration.csv")
    inv = {r["ch"]: r for r in rows("channel_inventory.csv")}
    eras = rows("eras.csv")
    drift = rows("carrier_drift.csv", required=False)
    tx = rows("transmitter_census.csv")

    multi = sorted({r["ch"] for r in eras if int(r["n_eras"]) > 1}, key=int)
    carrier_cut = float(
        doc.get("report_policy", {}).get(
            "carrier_dominated_level_db", CARRIER_DOMINATED_LEVEL_DB))
    quiet = [r for r in cal if float(r["mu_shift_db"]) < carrier_cut]
    shifts = sorted(abs(float(r["mu_shift_db"])) for r in quiet)
    sigmas = sorted(float(r["sigma_over_mu"]) for r in quiet)

    stats = [
        ("%d" % t["channels"], "channels calibrated", "all 23 freq_ids, "
         "Dec 2018 &ndash; Jul 2026"),
        ("{:,}".format(t["frames"]), "frames", "%.2f h of integration across "
         "{:,} acquisition units".format(t["units"]) % t["integration_hours"]),
        ("%.1f%% &rarr; %.1f%%" % (100 * t["kept_median_occ_provisional"],
                                   100 * t["kept_median_occ_working"]),
         "masked, before &rarr; after", "median over the 17 kept channels, "
         "F&nbsp;&gt;&nbsp;1 against F&nbsp;&gt;&nbsp;&eta;&mu;"),
        ("%d / %d" % (t["kept"], t["excised"]), "keep / excise",
         "%d of %d agree with the published policy"
         % (t["agree_with_published"], t["channels"])),
    ]
    strip = "".join(
        '<div class="stat"><div class="stat-v">%s</div>'
        '<div class="stat-l">%s</div><div class="stat-n">%s</div></div>'
        % s for s in stats)

    LOCKED = {"35": "sign-on Nov 2021", "19": "sign-off Dec 2024",
              "26": "sign-off Apr 2023", "20": "step down Sep 2022",
              "27": "sign-off in the 2021-22 gap",
              "32": "sign-off in the 2021-22 gap"}
    era_rows = []
    for ch in multi:
        segs = [r for r in eras if r["ch"] == ch]
        era_rows.append([
            "ch&nbsp;%s" % ch,
            " &rarr; ".join("%s <span class='dim'>(%+.2f dB)</span>"
                            % (s["span"], float(s["level_median_db"]))
                            for s in segs),
            "<code>%s</code>" % segs[-1]["month_start"],
            (LOCKED.get(ch) or "<em class='new'>not in the locked table</em>")])

    cal_rows = []
    for r in sorted(cal, key=lambda r: int(r["ch"])):
        v = r["verdict"]
        threshold_status = r.get("threshold_status", "legacy_report")
        pill = ('<span class="pill %s">%s</span>'
                % ("excise" if v == "excise" else "keep", v))
        cal_rows.append([
            "ch&nbsp;%s" % r["ch"],
            "<code>%s</code>" % inv[r["ch"]]["freq_id"],
            fmt(r["mu0_provisional"], "%.5f"),
            "<strong>%s</strong>" % fmt(r["mu"], "%.4f"),
            fmt(r["mu_shift_db"], "%+.3f"),
            fmt(r["sigma_over_mu"], "%.1e"),
            fmt(r["eta_channel"], "%.3f"),
            fmt(float(r["occupancy_provisional"]) * 100, "%.1f"),
            fmt(float(r["occ_at_eta_channel"]) * 100, "%.1f"),
            fmt(r["mask_band_suppression_db"], "%.2f"),
            html.escape(threshold_status.replace("_", " ")),
            pill])

    sec = [r for r in tx if r["kind"] == "secondary"]
    sec_rows = [["ch&nbsp;%s" % r["ch"],
                 fmt(r["offset_from_pilot_khz"], "%+.2f"),
                 fmt(r["db_rel_median"], "%.1f")] for r in sec]

    coh = [r for r in drift if r["coherent_drift"] == "True"]
    stable = sorted((r for r in drift if r["coherent_drift"] != "True"
                     and float(r["scatter_hz"]) < 6),
                    key=lambda r: abs(float(r["slope_hz_per_year"])))
    drift_rows = [["ch&nbsp;%s" % r["ch"], r["era"], r["n_months"],
                   fmt(r["slope_hz_per_year"], "%+.1f"),
                   fmt(r["scatter_hz"], "%.1f"), fmt(r["r2"], "%.2f")]
                  for r in coh]
    stable_rows = [["ch&nbsp;%s" % r["ch"], r["era"], r["n_months"],
                    fmt(r["slope_hz_per_year"], "%+.1f"),
                    fmt(r["scatter_hz"], "%.1f")] for r in stable]

    body = Template(TEMPLATE).substitute(
        strip=strip,
        n_channels=t["channels"],
        n_frames="{:,}".format(t["frames"]),
        n_units="{:,}".format(t["units"]),
        hours="%.2f" % t["integration_hours"],
        span_first="2018-12", span_last="2026-07",
        n_keep=t["kept"], n_exc=t["excised"], n_agree=t["agree_with_published"],
        eta_min="%.2f" % t["eta_min"], eta_max="%.2f" % t["eta_max"],
        eta_med="%.2f" % t["eta_median"],
        prov_med="%.1f" % (100 * t["kept_median_occ_provisional"]),
        work_med="%.1f" % (100 * t["kept_median_occ_working"]),
        shift_min="%.4f" % shifts[0], shift_med="%.3f" % shifts[len(shifts)//2],
        shift_max="%.2f" % shifts[-1],
        sig_med="%.1e" % sigmas[len(sigmas)//2],
        n_multi=len(multi),
        era_table=table(["channel", "eras recovered (median level)",
                         "boundary", "locked table"], era_rows),
        cal_table=table(["ch", "freq_id", "&mu;<sub>0</sub> prov.",
                         "&mu; calibrated", "shift dB", "&sigma;/&mu;",
                         "&eta;", "% masked F&gt;1", "% masked F&gt;&eta;&mu;",
                         "mask gain dB", "threshold source", "report label"],
                        align=[2, 3, 4, 5, 6, 7, 8, 9]),
        sec_table=table(["channel", "offset from pilot (kHz)",
                         "dB rel. median"], sec_rows, align=[1, 2]),
        drift_table=table(["channel", "era", "months", "Hz / yr",
                           "scatter (Hz)", "r&sup2;"], drift_rows,
                          align=[2, 3, 4, 5]),
        stable_table=table(["channel", "era", "months", "Hz / yr",
                            "scatter (Hz)"], stable_rows, align=[2, 3, 4]),
        fig_timeline=img("fig01_occupancy_timeline.png",
                         "Monthly occupancy heat map for all 23 channels",
                         "Every channel&rsquo;s monthly median pilot level "
                         "across the survey. Black rules are recovered era "
                         "boundaries; the outlined box is the latest era, the "
                         "only one used for characterisation."),
        fig_mu=img("fig02_mu_calibration.png", "Calibrated mu per channel",
                   "Left: how far each channel&rsquo;s calibrated &mu; sits "
                   "above the provisional constant. Right: the null width of "
                   "the 17 kept channels, all within a few parts per "
                   "thousand."),
        fig_hist=img("fig03_histograms.png",
                     "Latest-era F distributions, 23 panels",
                     "The statistic in every channel&rsquo;s latest era, with "
                     "all three rungs of the ladder marked. The bimodal "
                     "panels are the six channels with a live carrier."),
        fig_ladder_sum=img("fig11_ladder_summary.png",
                           "Masked fraction at each rung of the ladder",
                           "What calibrating &mu; buys. Hatched bars are the "
                           "excised channels, where &mu; is the carrier "
                           "itself and the report-rule bar is not a "
                           "report point."),
        fig_eta=img("fig13_eta_per_channel.png",
                    "Per-channel eta and the redshift-bin tolerances",
                    "Each channel&rsquo;s own &eta;, and the two tolerance "
                    "tiers behind it. The growth-rate tier varies by only "
                    "1.3&times; across the whole band, so it is not what "
                    "makes &eta; differ between channels."),
        fig_residual_bracket=img("fig12_residual_tolerance_bracket.png",
                        "Residual against tolerance at both bracket ends",
                        "The coherence bracket spans four to six orders of "
                        "magnitude. The red end is the adopted basis: every "
                        "residual on it is an upper bound, so a channel "
                        "outside tolerance there is uncertified, not "
                        "disqualified."),
        fig_bracket=img("fig15_bracket_stability.png",
                        "Coherence-bracket stability of eta",
                        "Each channel's threshold at both ends of the coherence bracket. Where the two coincide the threshold is identified by the data; where they do not, it is identified only by the choice of basis."),
        fig_mask=img("fig14_mask_effect.png",
                     "Latest-era fine spectrum before and after masking",
                     "What the threshold actually removes. On the kept "
                     "channels the two traces separate; on the excised ones "
                     "they lie on top of each other, which is the excision "
                     "argument shown spectrally rather than argued."),
        fig_thr=img("fig06_threshold_ladder.png",
                    "Masked fraction against eta",
                    "Masked fraction against threshold for every channel, "
                    "with each channel&rsquo;s report point marked."),
        fig_era=img("fig07_era_effect.png",
                    "Era-blind against latest-era masked fraction",
                    "Restricting to the latest era moves only the channels "
                    "that had a transition &mdash; and moves them in the "
                    "direction the physics demands."),
        fig_disp=img("fig08_dispositions.png", "Historical report labels",
                     "The two pieces of evidence behind every report label: how "
                     "much masking the report threshold costs, and whether a "
                     "null exists in that era at all."),
        fig_wide=img("fig04_wide_spectra.png",
                     "Channel-wide spectra, 23 panels",
                     "The full 390.6 kHz CHIME channel for each allocation, "
                     "with the target coarse bin, the two guard references, "
                     "the synthesized pilot position and the channel centre "
                     "marked."),
        fig_zoom=img("fig05_zoom_spectra.png",
                     "Zoom on each pilot, 23 panels",
                     "&plusmn;15 kHz about each synthesized pilot. The guard "
                     "references sit &plusmn;6.1 kHz from the target bin; "
                     "every dominant carrier lands inside the target bin "
                     "except ch33&rsquo;s."),
        fig_census=img("fig09_carrier_census.png", "Carrier census",
                       "Every feature within &plusmn;30 kHz of each pilot, "
                       "classified. The line at each channel&rsquo;s centre "
                       "is instrumental and appears in all 23."),
        fig_tracks=img("fig10_carrier_tracks.png",
                       "Carrier frequency tracked month by month",
                       "Carrier position through the survey for the 18 "
                       "channels that hold a trackable line."),
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body)
    print("wrote %s (%.2f MB)"
          % (args.out, os.path.getsize(args.out) / 1048576))
    return 0


TEMPLATE = r"""<title>Pilot Detector Calibration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#fbfaf7; --raise:#ffffff; --ink:#16160f; --ink-2:#4a4940;
  --dim:#7d7b70; --rule:#ddd8ca; --rule-2:#eceadf;
  --accent:#1f5fb0; --accent-soft:#e6eefa;
  --keep:#0d7f57; --keep-soft:#e2f2eb;
  --excise:#bb3328; --excise-soft:#fae9e7;
  --warn:#a3690a;
  --measure:68ch;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#12120e; --raise:#1b1b16; --ink:#eeece3; --ink-2:#b6b4a8;
    --dim:#8b897d; --rule:#33322a; --rule-2:#26251f;
    --accent:#79aef2; --accent-soft:#16283f;
    --keep:#4fc496; --keep-soft:#13291f;
    --excise:#f0857a; --excise-soft:#301713;
    --warn:#e0a640;
  }
}
:root[data-theme="dark"]{
  --paper:#12120e; --raise:#1b1b16; --ink:#eeece3; --ink-2:#b6b4a8;
  --dim:#8b897d; --rule:#33322a; --rule-2:#26251f;
  --accent:#79aef2; --accent-soft:#16283f;
  --keep:#4fc496; --keep-soft:#13291f;
  --excise:#f0857a; --excise-soft:#301713;
  --warn:#e0a640;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;
  font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:0 clamp(18px,4vw,44px) 96px}
.measure{max-width:var(--measure)}
h1,h2,h3{font-family:Spectral,Georgia,serif; text-wrap:balance; margin:0}
h1{font-size:clamp(2.1rem,4.6vw,3.15rem); font-weight:700; line-height:1.08;
   letter-spacing:-0.015em}
h2{font-size:clamp(1.42rem,2.5vw,1.78rem); font-weight:600; line-height:1.2}
h3{font-size:1.1rem; font-weight:600; margin-top:2em}
p{margin:0 0 1.05em}
a{color:var(--accent)}
code,.num,kbd{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:0.88em; background:var(--rule-2); padding:0.1em 0.36em;
  border-radius:3px}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:0.72rem;
  letter-spacing:0.16em; text-transform:uppercase; color:var(--dim);
  margin:0 0 0.9em}
.dim{color:var(--dim)}
em.new{color:var(--warn); font-style:normal; font-weight:500}

header.top{border-bottom:1px solid var(--rule); padding:clamp(46px,7vw,86px) 0 0}
.lede{font-family:Spectral,Georgia,serif; font-size:1.2rem; line-height:1.55;
  color:var(--ink-2); max-width:62ch; margin:1.15em 0 0}
.meta{display:flex; flex-wrap:wrap; gap:1.4em; margin:2em 0 0;
  font-family:"IBM Plex Mono",monospace; font-size:0.76rem; color:var(--dim)}
.strip{display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
  border:1px solid var(--rule); margin:2.6em 0 0}
.stat{background:var(--raise); padding:1.05em 1.15em 1.15em}
.stat-v{font-family:Spectral,Georgia,serif; font-size:1.72rem; font-weight:600;
  line-height:1.1; font-variant-numeric:tabular-nums}
.stat-l{font-size:0.83rem; font-weight:500; margin-top:0.28em}
.stat-n{font-size:0.76rem; color:var(--dim); margin-top:0.32em;
  line-height:1.42}

section{padding-top:clamp(48px,6vw,78px)}
.sec-head{display:flex; gap:0.85em; align-items:baseline; margin-bottom:1.1em}
.sec-n{font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
  color:var(--accent); font-weight:500; padding-top:0.22em}

.ladder{list-style:none; padding:0; margin:1.5em 0; display:grid; gap:1px;
  background:var(--rule); border:1px solid var(--rule)}
.ladder li{background:var(--raise); padding:1em 1.2em; display:grid;
  grid-template-columns:auto 1fr; gap:0 1.1em; align-items:baseline}
.rung{font-family:"IBM Plex Mono",monospace; font-weight:500;
  font-size:1.02rem; color:var(--accent); white-space:nowrap}
.ladder .what{font-weight:600}
.ladder .why{grid-column:2; color:var(--ink-2); font-size:0.93rem;
  margin-top:0.2em}

.fig{margin:2.2em 0; padding:0}
.fig img{width:100%; height:auto; display:block; border:1px solid var(--rule);
  border-radius:2px; background:#fcfcfb}
.fig figcaption{font-size:0.85rem; color:var(--ink-2); margin-top:0.75em;
  max-width:78ch; line-height:1.5}

.scroll{overflow-x:auto; margin:1.5em 0; border:1px solid var(--rule)}
table{border-collapse:collapse; width:100%; font-size:0.855rem;
  background:var(--raise)}
th,td{text-align:left; padding:0.52em 0.8em;
  border-bottom:1px solid var(--rule-2); white-space:nowrap}
th{font-size:0.72rem; letter-spacing:0.05em; text-transform:uppercase;
  color:var(--dim); font-weight:600; background:var(--paper);
  position:sticky; top:0}
td.num,th.num{text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
.pill{display:inline-block; font-size:0.72rem; font-weight:600;
  padding:0.16em 0.6em; border-radius:99px; letter-spacing:0.02em}
.pill.keep{background:var(--keep-soft); color:var(--keep)}
.pill.excise{background:var(--excise-soft); color:var(--excise)}

.note{border-left:2px solid var(--accent); background:var(--accent-soft);
  padding:1em 1.2em; margin:1.6em 0; font-size:0.94rem}
.note.warn{border-left-color:var(--warn);
  background:color-mix(in srgb, var(--warn) 9%, var(--paper))}
.note p:last-child{margin-bottom:0}
.note .h{font-weight:600; display:block; margin-bottom:0.3em}

.findings{display:grid; gap:1px; background:var(--rule);
  border:1px solid var(--rule); margin:1.8em 0;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}
.finding{background:var(--raise); padding:1.15em 1.25em}
.finding h3{margin:0 0 0.4em; font-size:1rem}
.finding p{font-size:0.9rem; color:var(--ink-2); margin:0}

footer{border-top:1px solid var(--rule); margin-top:clamp(56px,7vw,90px);
  padding-top:1.6em; font-size:0.82rem; color:var(--dim)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">CHIME &middot; pilot-proxy detector &middot; calibration</p>
  <h1>Pilot Detector Calibration</h1>
  <p class="lede">The completed trawl, read back as a calibration: where each
  channel&rsquo;s null actually sits, where the transmitters are relative to
  the detector&rsquo;s guards, what a supplied science tolerance implies, and
  how the historical report classifies each allocation.</p>
  <div class="meta">
    <span>${n_channels} channels</span>
    <span>${n_frames} frames</span>
    <span>${hours} h integration</span>
    <span>${span_first} &ndash; ${span_last}</span>
    <span>products: _per_pilot complete-23</span>
  </div>
  <div class="strip">${strip}</div>
  <div class="note warn">
    <span class="h">Report only.</span>
    <p>These keep/excise labels reproduce the calibration report. They are not
    operational threshold exports. Missing supplied thresholds use the
    historical fallback and are labeled as such in the table.</p>
  </div>
</header>

<section>
  <div class="sec-head"><span class="sec-n">01</span>
  <h2>The threshold ladder, and which rung the archive was collected on</h2></div>
  <div class="measure">
  <p>The statistic is <code>F = 2&middot;P<sub>target</sub> /
  (P<sub>ref,lo</sub> + P<sub>ref,hi</sub>)</code>, summed over all 2048 input
  streams before the ratio is taken. Its null mean is fixed analytically by
  the weight norms, and that constant &mdash; the <code>mu0</code> stored in
  every product &mdash; sits within 1.2% of unity on all 23 channels. The
  archive&rsquo;s reject mask is that rule, so in practice the survey was
  collected under <strong>F&nbsp;&gt;&nbsp;1</strong>: a provisional setting,
  the strictest available, chosen so a threshold could be exercised while the
  archive was still being built.</p>
  <p>It is not a calibrated report point, and the cost of treating it as one
  is the single largest number in this report. Across the 17 channels worth keeping,
  it masks a median of <strong>${prov_med}%</strong> of frames.</p>
  </div>

  <ol class="ladder">
    <li><span class="rung">F &gt; 1</span><span class="what">provisional,
      as collected</span>
      <span class="why">The analytic weight-norm constant. Not calibrated
      against anything the sky did; masks ${prov_med}% of frames at the
      median.</span></li>
    <li><span class="rung">F &gt; &mu;</span><span class="what">calibrated on
      the collected data</span>
      <span class="why">&mu; measured per channel, per era, from the frames
      the survey actually holds. This is the real null centre.</span></li>
    <li><span class="rung">F &gt; &eta;&mu;</span><span class="what">the
      report rule</span>
      <span class="why">&eta; supplied from the science residual
      tolerance when available: ${eta_min} to ${eta_max} across the band, median
      ${eta_med}. Masks ${work_med}% at the median.</span></li>
  </ol>
  ${fig_ladder_sum}
</section>

<section>
  <div class="sec-head"><span class="sec-n">02</span>
  <h2>Activity eras</h2></div>
  <div class="measure">
  <p>A channel&rsquo;s occupancy is piecewise stationary: a transmitter signs
  on, signs off, or steps power, and between those events the level is flat
  apart from propagation scatter. Characterising a channel across a
  transition mixes two physically different populations.</p>
  <p><strong>The latest era is used because it is the forecast.</strong> The
  question these numbers have to answer is what the band will do on the
  telescope next, and the only evidence bearing on that is the configuration
  the transmitters are in now. A channel&rsquo;s 2019 behaviour is history,
  not prediction &mdash; which is also why ch35 is allowed to get
  <em>worse</em> under era resolution rather than better: it signed on, so
  its latest era is the loud one, and that is the honest forward answer.</p>
  <p>The timelines below are not decoration on that choice; they are what
  licenses it. An era is only a meaningful unit if the transitions are real
  and datable, so every split has to be visible in the occupancy history
  before it is trusted to partition anything.</p>
  <p>The segmentation is deliberately conservative &mdash; it looks for
  transmitter transitions, not seasonal propagation. Monthly medians of the
  per-unit level are split by recursive binary segmentation; a split is
  accepted only if both sides hold at least six observed months and nine
  months of wall-clock, the median step is at least 2&nbsp;dB, and a
  rank-based Mann&ndash;Whitney |z| exceeds 4.</p>
  </div>

  <div class="note">
    <span class="h">It reproduces the hand-maintained table exactly.</span>
    <p>All six transitions carried in
    <code>analysis/make_policy_data.py</code> come back at the correct
    calendar month, with no false positives anywhere in the other 17
    channels &mdash; and one transition the locked table did not carry.</p>
  </div>

  ${era_table}
  ${fig_timeline}

  <div class="measure">
  <p>Restricting to the latest era moves exactly the channels that had a
  transition, and moves them in the direction the physics demands: the five
  sign-off and step-down channels get 2.0&ndash;6.0&times; cheaper, while
  ch17, whose latest era is the louder one, gets slightly dearer. Every
  single-era channel is unchanged, which is the control.</p>
  </div>
  ${fig_era}
</section>

<section>
  <div class="sec-head"><span class="sec-n">03</span>
  <h2>Calibrating &mu;</h2></div>
  <div class="measure">
  <p>F has a very narrow null core with a heavy transient tail on the right,
  so no moment-based estimator survives contact with it. The centre is taken
  as the half-sample mode &mdash; the densest region of the sample. That
  choice does double duty: where a null exists the mode lands on it, and
  where the channel is fully occupied the mode lands on the carrier lobe
  instead, which is itself the evidence for the report label. The scale comes
  from frames at or below the centre, which a carrier can only ever add to
  from above, using the released
  <code>rfisher.residual.NULL_SCALE_PROBES</code> convention.</p>
  <p>On the 17 channels with a recoverable null, the calibrated &mu; agrees
  with the analytic constant to between <strong>${shift_min} dB</strong> and
  <strong>${shift_max} dB</strong> (median ${shift_med} dB), and the null
  itself is extraordinarily tight &mdash; &sigma;/&mu; of ${sig_med} at the
  median. That tightness is exactly why F&nbsp;&gt;&nbsp;1 masks almost
  everything: with a null a few parts per thousand wide, a shift of a
  hundredth of a decibel puts most of the distribution on the wrong side of
  the line.</p>
  </div>
  ${fig_mu}
  ${fig_hist}
</section>

<section>
  <div class="sec-head"><span class="sec-n">04</span>
  <h2>&eta; from the contamination-residual tolerance</h2></div>
  <div class="measure">
  <p>&mu; says where the null is. It does not say how far above the null the
  threshold belongs &mdash; that is a science question, and it is answered
  the way <code>scripts/optimal_thresholds.py</code> answers it: minimise the
  survey-time cost <code>(1+r)/(1&minus;f)</code> over the family
  F&nbsp;&gt;&nbsp;&eta;&mu;, subject to the Fisher-bias tolerance, with the
  measured 10&nbsp;dB fine-stage credit applied to the bound. Masking more
  lowers the residual <em>r</em> and raises the discarded fraction <em>f</em>,
  so the optimum is interior wherever the residual is large enough to
  matter.</p>
  <p>Every channel gets its own &eta;, and they differ:
  <strong>${eta_min} to ${eta_max}</strong>, median ${eta_med}. What makes
  them differ is mostly each channel&rsquo;s own occupancy and coherence, not
  its redshift bin &mdash; the growth-rate tolerance varies by only
  1.3&times; from ch14 to ch36.</p>
  </div>
  ${fig_eta}

  <div class="note warn">
    <span class="h">These thresholds are identified by the adopted basis, not
    by measurement.</span>
    <p>A per-channel &eta; is only well defined if it does not depend on which
    end of the coherence bracket you evaluate it at. On this archive it
    mostly does: the ratio between the thermal-end and cap-end optimum has a
    median of <strong>2.08&times;</strong> and reaches
    <strong>9.92&times;</strong> on ch14. The bracket collapses to exactly
    1.00&times; on ch24, ch31 and ch35 &mdash; the three channels with a
    measured coherence time &mdash; which is the cleanest statement available
    of what measuring &tau; actually buys.</p>
    <p>So the &eta; values above are identified <em>because the bounded basis
    was adopted</em>, not because the data pins them. That is a legitimate
    position and it is the one taken here, but it is a decision rather than a
    measurement, and it is worth reading the ordering below with that in
    mind: <strong>&eta; is pinned on five channels and all five are
    excised</strong>. On the seventeen kept channels &mdash; the ones where
    the threshold actually governs anything &mdash; it is not.</p>
  </div>
  ${fig_bracket}

  <div class="note">
    <span class="h">Adopted basis: the bound, not the measurement.</span>
    <p>The residual depends on how long contamination stays coherent, and only
    three channels carry a measured &tau; (ch24 62 min, ch35 46 min, ch31
    34 min). The other twenty are evaluated at the sidereal-day cap, which is
    the physical ceiling &mdash; anything longer-lived has already been removed
    as <em>m</em> = 0. Since &tau;<sub>cap</sub> &ge; &tau;<sub>true</sub> and
    the coherent amplification grows with &tau;, every residual quoted here is
    an <strong>upper bound</strong> on the true one, not an estimate of it.</p>
    <p>Two things follow, and they are the reason this is a usable position
    rather than a gap. A residual above tolerance means that tolerance is
    <strong>not certified</strong> on present evidence &mdash; not that it is
    exceeded. A measured &tau; can only move the residual down, so it can only
    enlarge the feasible set. And no report label rests on the bound at all:
    every excised channel fails on carrier dominance, which is a &tau;-free
    measurement of the latest era. The bound governs what the kept channels
    cost, not which channels are kept.</p>
    <p class="dim">Deferred, not dismissed. Measuring &tau; per channel stays
    the single highest-value follow-up in this chain, and the acquisition is
    already specified. Until it runs these bounds stand, and every number
    downstream of them inherits the same one-sided qualification.</p>
  </div>
  ${fig_residual_bracket}
</section>

<section>
  <div class="sec-head"><span class="sec-n">05</span>
  <h2>What survives</h2></div>
  <div class="measure">
  <p>The historical report labels a channel excised when the densest
  population in its latest era is
  the carrier rather than a null &mdash; there is then nothing to threshold
  against &mdash; or when reaching the report point costs more than half
  the band-time. Six channels fail on the first test and none on the second
  alone. The separation is clean rather than marginal: every kept channel
  sits below 1&nbsp;dB of null shift and every excised one above
  5.3&nbsp;dB, with nothing in between.</p>
  <p>All <strong>${n_agree} of ${n_channels}</strong> report labels match
  the published complete-23 policy. This agreement does not promote them to
  operational thresholds.</p>
  </div>
  ${fig_disp}
  ${cal_table}
  ${fig_thr}
</section>

<section>
  <div class="sec-head"><span class="sec-n">06</span>
  <h2>What the threshold removes</h2></div>
  <div class="measure">
  <p>The ladder says how many frames a threshold masks. It does not say what
  that buys spectrally, and those are different questions: masking a large
  fraction of frames is worthless if the frames removed carried no more power
  than the ones kept. The pair below is the direct answer &mdash; the
  latest-era spectrum averaged over all frames, against the same average over
  only the frames that survive F &gt; &eta;&mu;.</p>
  <p>Both traces are <em>means</em>, not medians. Mean power is what
  integrates into a map, so it is the statistic that says what masking
  actually takes out; the medians used for the spectrograms answer a
  different question, about persistence, and the two part company on a bursty
  channel.</p>
  <p>Across the kept channels the threshold removes 0.13 to 3.07 dB of
  band-integrated power and up to 13.7 dB at the carrier itself. Across the
  excised ones it removes essentially nothing &mdash; 0.02 to 0.07 dB on
  ch24, ch30, ch31 and ch35 &mdash; because the carrier is present in
  substantially every frame, so the surviving set is the same population as
  the full one. That is the excision case made spectrally instead of
  argumentatively: there is no subset of these eras worth keeping.</p>
  </div>
  ${fig_mask}

  <div class="note warn">
    <span class="h">This view is era-resolved; the channel-wide one cannot
    be.</span>
    <p>The fine statistic is stored per frame, so it can be restricted to the
    latest era and split by any threshold after the fact. The channel-wide
    spectra cannot: they are accumulated over the whole archive and the
    per-frame version is not retained. On the sixteen single-era channels
    that distinction is empty &mdash; archive and latest era are the same
    population, so the stored before/after pair already is the latest-era
    pair. On the seven with a transition (ch17, 19, 20, 26, 27, 32, 35) it
    blends eras and no post-hoc split is possible. Recovering it needs a
    per-era accumulator at trawl time: cheap to add, impossible to reconstruct
    afterwards. <code>wide_pair_is_latest_era</code> in the tables flags which
    channels are which.</p>
  </div>
</section>

<section>
  <div class="sec-head"><span class="sec-n">07</span>
  <h2>Where the transmitters are</h2></div>
  <div class="measure">
  <p>The channel-wide spectrum spans the full 390.6 kHz CHIME channel at
  23.8&nbsp;Hz resolution, so it shows every carrier in the allocation and
  where each sits relative to the detector&rsquo;s geometry: the target
  coarse bin, and the two &plusmn;2-bin guard references at
  &plusmn;6.1&nbsp;kHz that form the statistic&rsquo;s denominator.</p>
  </div>
  ${fig_zoom}

  <div class="note warn">
    <span class="h">The line at every channel&rsquo;s centre is
    instrumental.</span>
    <p>A strong feature sits at the exact channel centre in all 23 channels,
    including ones with no transmitter near it &mdash; the detector&rsquo;s
    own contract names that bin a forbidden tone. Before the released health
    repair it stands at a median of 19.6&nbsp;dB above the channel median;
    the repair, which exactly subtracts frames whose every decoded sample is
    (&minus;8,&nbsp;&minus;8), takes it down to 9.9&nbsp;dB. It is not a
    carrier and should not be counted as one. Two channels have their pilot
    close enough to the centre for both to fall inside a &plusmn;15 kHz
    census window: <strong>ch28</strong> (pilot +12.57 kHz, so the centre
    line lands at &minus;12.57 kHz from the pilot) and <strong>ch14</strong>
    (pilot &minus;3.06 kHz).</p>
  </div>

  <p class="measure">Six channels carry a genuine secondary carrier standing
  at least 3&nbsp;dB above the channel median, none of them inside a guard
  reference:</p>
  ${sec_table}
  ${fig_census}
  ${fig_wide}
</section>

<section>
  <div class="sec-head"><span class="sec-n">08</span>
  <h2>Time-resolved: what the fine band shows</h2></div>
  <div class="measure">
  <p>The channel-wide spectrum is integrated over the whole archive and
  cannot be split by era. The fine statistic can: it is stored per frame, 256
  bins of 11.92&nbsp;Hz spanning &plusmn;1.53&nbsp;kHz about the target
  coarse bin, which makes it the only time-resolved spectral product in the
  archive. Binned by month it gives a spectrogram, and tracking its peak
  gives each transmitter&rsquo;s frequency history.</p>
  <p>Five channels show a drift the straight line genuinely explains. The
  clearest is <strong>ch17</strong>, whose carrier walks down
  97&nbsp;Hz&nbsp;per&nbsp;year over 46 months of its latest era with only
  28&nbsp;Hz of scatter about the fit.</p>
  </div>
  ${drift_table}
  <p class="measure">Against which, four carriers are effectively frequency
  references &mdash; stable to a few hertz over seven years:</p>
  ${stable_table}
  ${fig_tracks}
</section>

<section>
  <div class="sec-head"><span class="sec-n">09</span>
  <h2>Where this is not finished</h2></div>
  <div class="findings">
    <div class="finding">
      <h3>&tau; measured on 3 of 23 &mdash; deferred by decision</h3>
      <p>The acquisition that would measure the other twenty is specified but
      not scheduled, so the chain is carried on the bounded basis and every
      residual here is an upper limit. That is a deliberate position, not an
      omission: it is one-sided, so nothing kept can be wrong. What it costs
      is the ability to quote a residual instead of a bound, and it stays open
      until the campaign runs.</p>
    </div>
    <div class="finding">
      <h3>The floor basis changes feasibility</h3>
      <p>This run measures each floor on the quietest era the channel
      actually has. The released selector offers two other bases, and where
      they disagree about feasibility that disagreement is the finding. On
      ch31, ch32, ch33 and ch35 the published selector reports a feasible
      &eta; where the quiet-era floor does not.</p>
    </div>
    <div class="finding">
      <h3>Resolved lines are not automatically emitters</h3>
      <p>ch22 resolves six lines in its fine band, but they sit at matched
      offsets either side of one carrier (roughly &plusmn;96 and
      &plusmn;200&nbsp;Hz) &mdash; sideband structure, not six transmitters.
      Read the geometry before counting.</p>
    </div>
    <div class="finding">
      <h3>ch17&rsquo;s era split was not in the locked table</h3>
      <p>The segmentation finds a boundary at 2022-10 between a +12.6 dB and
      a +16.4 dB era. It does not change ch17&rsquo;s report label, but any
      era-blind characterisation of that channel &mdash; including its
      measured carrier offset &mdash; averages two transmitter
      configurations.</p>
    </div>
    <div class="finding">
      <h3>The sub-null tail is broadband, not the guards</h3>
      <p>0.24% of frames sit well below &mu;, which F can only produce if the
      references outweigh the target bin. Their median total power is
      3.35&times; the null bulk&rsquo;s and they are spread across many
      acquisitions, so they are loud frames rather than contaminated
      references &mdash; the incumbent flagger&rsquo;s business, not this
      detector&rsquo;s.</p>
    </div>
    <div class="finding">
      <h3>No era-resolved wide spectrum exists</h3>
      <p>Per-frame spectra are retained only in the &plusmn;1.53 kHz fine
      band. For the seven channels with a transition, the channel-wide
      spectrum blends eras. For the other sixteen it is already the latest
      era, so the question is confined.</p>
    </div>
  </div>
</section>

<footer>
  <p>Built from the completed <code>_per_pilot</code> complete-23 products
  (23 npz, ${n_frames} frames, ${n_units} units). Geometry, frame health and
  CFAR conventions are taken from WVURAIL/pilot-proxy; residuals, coherence
  times and tolerances from WVURAIL/RFIsher and its completed
  forecast run.</p>
</footer>
</div>
"""

if __name__ == "__main__":
    raise SystemExit(main())
