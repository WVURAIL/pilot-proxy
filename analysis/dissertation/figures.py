#!/usr/bin/env python3
"""Dissertation figures owned by pilot-proxy.

These render the survey-derived dissertation figures -- the archive-averaged
pilot spectra (both bands), the worked fine-spectrum examples, the
epoch-operating-point diagram, and the channel evidence-status matrix -- in
the dissertation's exact style (Latin Modern through LaTeX, the WVU semantic
palette, pinned PDF metadata). The dissertation bundle vendors the resulting
PDFs; this module and the tables under ``data/`` are their editable source.

Each table in ``data/`` is a frozen snapshot of a generator this repository
owns (see data/README.md for the exact regeneration commands). Rendering
requires latex + dvipng + kpsewhich, like the dissertation build itself.

    python3 figures.py --out out/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import style

DATA = Path(__file__).resolve().parent / "data"


def read_csv(name: str):
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canvas(height, title):
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, height))
    ax.set_axis_off()
    ax.text(.02, .96, rf"\textbf{{{title}}}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.0)
    return fig, ax


def fig_census_psd(out: Path) -> Path:
    rows=read_csv("census_psd.csv")
    # Fixed GridSpec geometry avoids allowing the explanatory key to distort
    # the scientific panel widths.  The first nine channels fill a 3x3 grid;
    # channel 36 and the key share the fourth row.
    fig=plt.figure(figsize=(style.TEXT_WIDTH,7.05))
    gs=fig.add_gridspec(
        4,3,left=.075,right=.985,bottom=.065,top=.945,
        wspace=.30,hspace=.34,
    )
    axes=[]
    for idx in range(9):
        axes.append(fig.add_subplot(gs[idx//3,idx%3]))
    axes.append(fig.add_subplot(gs[3,0]))
    for i,ch in enumerate(range(27,37)):
        sub=[r for r in rows if int(r["channel"])==ch]
        x=np.asarray([float(r["offset_khz"]) for r in sub]); y=np.asarray([float(r["db_rel_median"]) for r in sub])
        order=np.argsort(x); x=x[order]; y=y[order]; ax=axes[i]
        ax.axvspan(-1.526,1.526,facecolor=style.LIGHT_BLUE,edgecolor="none")
        for ref in (-6.1036,6.1036): ax.axvline(ref,color=style.CONDITIONAL,ls=(0,(3,2)),lw=.85)
        ax.plot(x,y,color=style.INK,lw=.78); ax.axvline(0,color=style.MUTED,lw=.5)
        ax.text(.04,.91,rf"\textbf{{ch {ch}}}",transform=ax.transAxes,ha="left",va="top",fontsize=7.7)
        ax.set_xlim(-15,15); ax.grid(True,color=style.GRID,lw=.42)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=6.8)
        if i%3==0: ax.set_ylabel(r"dB rel. median",fontsize=7.3)
        if i>=6: ax.set_xlabel(r"offset [kHz]",fontsize=7.3)
    key=fig.add_subplot(gs[3,1:]); key.set_axis_off()
    key.text(.02,.90,r"\textbf{Detector geometry and provenance}",transform=key.transAxes,fontsize=8.0,va="top")
    key.add_patch(Rectangle((.03,.62),.10,.12,transform=key.transAxes,facecolor=style.LIGHT_BLUE,edgecolor="none"))
    key.text(.17,.68,r"fine span $\pm1.526$ kHz",transform=key.transAxes,va="center",fontsize=7.1)
    key.plot([.03,.13],[.47,.47],transform=key.transAxes,color=style.CONDITIONAL,ls=(0,(3,2)),lw=1.2)
    key.text(.17,.47,r"references at $\pm6.10$ kHz",transform=key.transAxes,va="center",fontsize=7.1)
    key.plot([.03,.13],[.30,.30],transform=key.transAxes,color=style.INK,lw=1)
    key.text(.17,.30,r"archive-averaged spectrum",transform=key.transAxes,va="center",fontsize=7.1)
    key.text(.02,.07,r"Direct archive-result export from the August 2026 survey products.",
             transform=key.transAxes,va="bottom",fontsize=6.5,color=style.MUTED)
    fig.suptitle(r"Archive-averaged spectrum around each pilot: main lobe, companions, and references",fontsize=9.7,y=.988)
    return style.save(fig,out/"fig_census_psd.pdf",title="Archive-averaged spectra around DTV pilots")


def fig_census_psd_lower(out: Path) -> Path:
    rows=read_csv("census_psd.csv")
    # Same fixed-GridSpec design as fig_census_psd: the first nine lower-band
    # channels fill a 3x3 grid; channels 25 and 26 and a compact key share the
    # fourth row.
    fig=plt.figure(figsize=(style.TEXT_WIDTH,7.05))
    gs=fig.add_gridspec(
        4,3,left=.075,right=.985,bottom=.065,top=.945,
        wspace=.30,hspace=.34,
    )
    axes=[]
    for idx in range(9):
        axes.append(fig.add_subplot(gs[idx//3,idx%3]))
    axes.append(fig.add_subplot(gs[3,0]))
    axes.append(fig.add_subplot(gs[3,1]))
    for i,ch in enumerate(range(16,27)):
        sub=[r for r in rows if int(r["channel"])==ch]
        x=np.asarray([float(r["offset_khz"]) for r in sub]); y=np.asarray([float(r["db_rel_median"]) for r in sub])
        order=np.argsort(x); x=x[order]; y=y[order]; ax=axes[i]
        ax.axvspan(-1.526,1.526,facecolor=style.LIGHT_BLUE,edgecolor="none")
        for ref in (-6.1036,6.1036): ax.axvline(ref,color=style.CONDITIONAL,ls=(0,(3,2)),lw=.85)
        ax.plot(x,y,color=style.INK,lw=.78); ax.axvline(0,color=style.MUTED,lw=.5)
        ax.text(.04,.91,rf"\textbf{{ch {ch}}}",transform=ax.transAxes,ha="left",va="top",fontsize=7.7)
        ax.set_xlim(-15,15); ax.grid(True,color=style.GRID,lw=.42)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=6.8)
        if i%3==0: ax.set_ylabel(r"dB rel. median",fontsize=7.3)
        if i in (8,9,10): ax.set_xlabel(r"offset [kHz]",fontsize=7.3)
    key=fig.add_subplot(gs[3,2]); key.set_axis_off()
    key.text(.04,.92,r"\textbf{Key}",transform=key.transAxes,fontsize=7.6,va="top")
    key.add_patch(Rectangle((.06,.64),.14,.11,transform=key.transAxes,facecolor=style.LIGHT_BLUE,edgecolor="none"))
    key.text(.26,.695,r"fine span $\pm1.526$ kHz",transform=key.transAxes,va="center",fontsize=6.7)
    key.plot([.06,.20],[.50,.50],transform=key.transAxes,color=style.CONDITIONAL,ls=(0,(3,2)),lw=1.1)
    key.text(.26,.50,r"references at $\pm6.10$ kHz",transform=key.transAxes,va="center",fontsize=6.7)
    key.plot([.06,.20],[.35,.35],transform=key.transAxes,color=style.INK,lw=1)
    key.text(.26,.35,r"archive-averaged spectrum",transform=key.transAxes,va="center",fontsize=6.7)
    key.text(.04,.10,r"Direct archive-result export" + "\n" + r"(August 2026 survey products).",
             transform=key.transAxes,va="bottom",fontsize=6.3,color=style.MUTED)
    fig.suptitle(r"The lower band's spectral face: archive-averaged spectrum around each pilot, channels 16--26",fontsize=9.7,y=.988)
    return style.save(fig,out/"fig_census_psd_lower.pdf",title="Archive-averaged spectra around lower-band DTV pilots")


def fig_worked_example(out: Path) -> Path:
    rows=read_csv("worked_example_spectra.csv"); panels={}
    for key in ("a","b"):
        sub=[r for r in rows if r["panel"]==key]
        x=np.asarray([float(r["fine_bin"]) for r in sub]); y=np.asarray([float(r["T"]) for r in sub])
        o=np.argsort(x); panels[key]=(x[o],y[o])
    fig,axes=plt.subplots(2,1,figsize=(style.TEXT_WIDTH,5.0),sharex=True,constrained_layout=True)
    info={
        "a":dict(title=r"exemplar frame, 2025-07-31: $F/\mu_0=1.258$",median=1.009,peak=18.62,ratio=18.4,ylim=(.72,24)),
        "b":dict(title=r"weakest valid frame, 2025-05-16: $F/\mu_0=0.897$",median=.833,peak=2.59,ratio=3.1,ylim=(.43,3.35)),
    }
    for ax,key in zip(axes,("a","b")):
        x,y=panels[key]; d=info[key]
        ax.axvspan(59.5,64.5,color=style.LIGHT_BLUE); ax.plot(x,y,color=style.MEASURED,lw=.95)
        usable=(x<60)|(x>64); ax.scatter(x[usable][::4],y[usable][::4],s=7,color=style.PENDING,alpha=.75)
        ax.axhline(d["median"],color=style.INK,lw=.85); ax.axhline(1.5*d["median"],color=style.INK,lw=.85,ls=(0,(3,2)))
        ax.text(253,d["median"]*1.03,rf"bulk median ${d['median']:.3f}$",ha="right",va="bottom",fontsize=7.0)
        ax.text(253,1.5*d["median"]*1.03,r"$\eta T_{(\rho)}$ (illustrative $\eta=1.5$)",ha="right",va="bottom",fontsize=6.9,color=style.MUTED)
        ax.text(67,d["peak"]*.92,rf"window max ${d['peak']:.2f}$" + "\n" + rf"$({d['ratio']:.1f}\times$ bulk median)",fontsize=7.5,va="top")
        ax.text(56,d["ylim"][0]*1.45,r"designated window" + "\n" + r"$f_a\pm2$",ha="right",fontsize=7.0,color=style.MEASURED)
        ax.axvline(62,color=style.MEASURED,lw=.5,alpha=.6)
        ax.set_yscale("log"); ax.set_ylim(*d["ylim"]); ax.set_ylabel(r"$T[f]$")
        ax.set_title(rf"\textbf{{({key})}} {d['title']}",loc="left",fontsize=8.9,pad=3); style.clean_axes(ax)
    axes[-1].set_xlabel(r"Fine bin $f$"); axes[-1].set_xlim(0,255)
    return style.save(fig,out/"fig_worked_example.pdf",title="Worked fine-spectrum examples")


def fig_epoch_operating_points(out: Path) -> Path:
    rows=read_csv("epoch_operating_points.csv")
    by_key={(int(r["channel"]),int(r["epoch_group"])):r for r in rows}

    def body(row):
        survey=100*float(row["survey_mask_fraction"])
        first=rf"survey mask ${survey:.1f}\%$"
        if row["retained_frames"]:
            count=int(float(row["retained_frames"]))
            words={0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten"}
            count_text=words.get(count,str(count))
            second=rf"fine calibration retains {count_text} frames"
        elif row["fine_mask_fraction"] and row["residual_ratio"]:
            fine=100*float(row["fine_mask_fraction"])
            second=rf"fine point: $f={fine:.1f}\%$, $r={float(row['residual_ratio']):.4g}$"
        elif row["fine_mask_fraction"]:
            fine=100*float(row["fine_mask_fraction"])
            second=rf"fine point: $f={fine:.1f}\%$"
        else:
            second=r"fine point unavailable"
        return first+r"\\"+second

    fig,ax=canvas(3.12,"An archive-average operating point can describe no physical era")
    ax.plot([.12,.90],[.72,.72],transform=ax.transAxes,color=style.INK,lw=.9)
    ax.plot([.52,.52],[.23,.84],transform=ax.transAxes,color=style.PENDING,lw=.8,ls=(0,(3,2)))
    ax.text(.52,.85,r"2021 boundary",transform=ax.transAxes,ha="center",va="bottom",fontsize=7.3,color=style.PENDING)
    ax.text(.29,.75,r"\textbf{pre-2021}",transform=ax.transAxes,ha="center",fontsize=8)
    ax.text(.72,.75,r"\textbf{post-2021}",transform=ax.transAxes,ha="center",fontsize=8)
    positions={32:(.57,.49),35:(.32,.24)}
    for channel,(label_y,box_y) in positions.items():
        ax.text(.04,label_y,rf"\textbf{{ch {channel}}}",transform=ax.transAxes,fontsize=8,color=style.MEASURED)
        for group,x in ((0,.14),(1,.60)):
            row=by_key[(channel,group)]
            style.diagram_box(
                ax,(x,box_y),(.29,.15),title=row["epoch_label"],body=body(row),
                status=row["status"],fontsize=6.5,title_size=7.1,
            )
    ax.text(.5,.09,r"Deployment unit: epoch-valid anchor, null model, rank, multiplier, and verdict.",transform=ax.transAxes,ha="center",fontsize=7.3,color=style.MUTED)
    return style.save(fig,out/"fig_epoch_operating_points.pdf",title="Epoch-specific operating points")


def fig_channel_status_matrix(out: Path) -> Path:
    rows=read_csv("channel_status.csv")
    fig,ax=plt.subplots(figsize=(style.TEXT_WIDTH,3.20))
    fig.subplots_adjust(left=.205,right=.985,bottom=.17,top=.76)
    ax.set_xlim(13.4,36.8); ax.set_ylim(-.05,3.78)
    ax.set_xticks(range(14,37)); ax.set_xticklabels([str(i) for i in range(14,37)],fontsize=6.3)
    y={"unmeasured":3.0,"conditional_recovery":2.15,"measurement_bound":1.30,"excision_candidate":.45}
    ycentres=[y[k]+.27 for k in ("unmeasured","conditional_recovery","measurement_bound","excision_candidate")]
    ax.set_yticks(ycentres)
    ax.set_yticklabels([r"unmeasured",r"conditional recovery",r"measurement-bound",r"excision candidate"],fontsize=7.0)
    for tick,c in zip(ax.get_yticklabels(),(style.PENDING,style.CONDITIONAL,style.MODEL,style.FAILURE)):
        tick.set_color(c); tick.set_ha("right")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="x",length=0,pad=1.5); ax.tick_params(axis="y",length=0,pad=8)
    ax.set_xlabel(r"ATSC physical channel",labelpad=4)
    appearance={
        "unmeasured":(style.LIGHT_GRAY,style.PENDING),
        "conditional_recovery":(style.LIGHT_GREEN,style.CONDITIONAL),
        "measurement_bound":(style.LIGHT_ORANGE,style.MODEL),
        "excision_candidate":(style.LIGHT_RED,style.FAILURE),
    }
    def cell(ch,status):
        face,edge=appearance[status]; yy=y[status]
        patch=FancyBboxPatch((ch-.36,yy),.72,.54,boxstyle="round,pad=.01,rounding_size=.04",facecolor=face,edgecolor=edge,lw=.75)
        ax.add_patch(patch); ax.text(ch,yy+.27,str(ch),ha="center",va="center",fontsize=6.2,fontweight="bold")
    for row in rows:
        channel=int(row["channel"]); primary=row["status"]; secondary=row["secondary_status"]
        cell(channel,primary)
        if secondary:
            yy=y[primary]; face,_=appearance[secondary]
            ax.add_patch(Rectangle((channel,yy),.35,.54,facecolor=face,edgecolor="none"))
            ax.text(channel,yy+.27,str(channel),ha="center",va="center",fontsize=6.2,fontweight="bold")
    measured_count=sum(r["status"]!="unmeasured" for r in rows)
    count_label="Ten" if measured_count==10 else str(measured_count)
    split_count=sum(bool(r["secondary_status"]) for r in rows)
    split_note=r"; half-filled cells are epoch-split between transmitter eras" if split_count else ""
    fig.suptitle(r"Current evidence matrix: a status map, not a completed 23-channel verdict",y=.975,fontsize=9.6,fontweight="bold")
    fig.text(.5,.885,rf"{count_label} channels have archive products{split_note}; transfer and estimator validation remain global gates.",ha="center",va="center",fontsize=7.2,color=style.MUTED)
    return style.save(fig,out/"fig_channel_status_matrix.pdf",title="Current 23-channel evidence matrix")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args(argv)
    style.configure(require_tex=True)
    outputs = [
        fig_census_psd(args.out),
        fig_census_psd_lower(args.out),
        fig_worked_example(args.out),
        fig_epoch_operating_points(args.out),
        fig_channel_status_matrix(args.out),
    ]
    for path in outputs:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
