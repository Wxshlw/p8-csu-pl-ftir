"""Fig.3 —— 外推主图：代表性体系的预测-参考曲线（含校准区间与 0 % 锚点补救）。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
from unmix_ftir import pubstyle as P

P.apply_pub_style()
rcParams.update({
    "axes.titlesize": 9.0, "axes.titlepad": 3.5,
    "axes.labelsize": 8.8, "axes.labelpad": 2.6,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 6.7, "lines.linewidth": 1.5,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
})

PANELS = [
    dict(dataset="D1", protocol="high", base="CLS-2pt", name="Olive oil / sunflower oil",
         train=(0, 15), note="upward, 2× range", cal="calibration 0–15 %",
         xlim=(0, 32), ylim=(0, 32), vspan=True, aspect="equal", extra=None),
    dict(dataset="D7d", protocol="low", base="Ridge", name="Cinnamon / wood dust (low)",
         train=(6, 24), note="nearly constant: 4.73 ± 0.25", cal="calibration 6–24 % (off scale)",
         xlim=(0, 5.2), ylim=(0, 7.5), vspan=False, aspect="auto", extra=None),
    dict(dataset="D2", protocol="low", base="PLS", name="Coffee / malt flour, low end",
         train=(15, 50), note="0 % anchor restores", cal="",
         xlim=(0, 12), ylim=(0, 15), vspan=False, aspect="auto", extra="anchor"),
    dict(dataset="D3", protocol="high", base="PLS", name="Orange juice / apple juice",
         train=(0, 25), note="low contrast (CNR 4)", cal="calibration 0–25 %",
         xlim=(0, 52), ylim=(0, 52), vspan=True, aspect="equal", extra=None),
]


def level_stats(d):
    return d.groupby("y").pred.agg(["mean", "std"]).reset_index()


def main():
    df = pd.read_csv(os.path.join(OUT, "exp03_predictions.csv"))
    anc = pd.read_csv(os.path.join(OUT, "exp03c_predictions.csv"))
    # label the near-constant D7d response from the same predictions that are plotted
    d7d = df[(df.dataset == "D7d") & (df.protocol == "low") & (df.method == "CSU-PL")]
    PANELS[1]["note"] = (f"nearly constant: {d7d.pred.mean():.2f} "
                         f"± {d7d.pred.std(ddof=1):.2f}")
    fig, axes = plt.subplots(2, 2, figsize=(P.DOUBLE_W, 4.3))
    letters = "abcd"
    for ax, spec, letter in zip(axes.ravel(), PANELS, letters):
        sub = df[(df.dataset == spec["dataset"]) & (df.protocol == spec["protocol"])]
        ref = sub[sub.method == "CSU-PL"].drop_duplicates("group")
        if spec["vspan"]:
            ax.axvspan(*spec["train"], color="#EDEDED", zorder=0, lw=0)
        top = spec["ylim"][1]
        ax.plot([0, top], [0, top], ls=(0, (3, 2)), color=P.C_GRAY2, lw=1.0, zorder=1)
        series = [(sub[sub.method == "CSU-PL"], P.C_NAVY, "o", "-", "CSU-PL"),
                  (sub[sub.method == spec["base"]], P.C_VERM, "s", "-", spec["base"])]
        if spec["extra"] == "anchor":
            a = anc[(anc.dataset == spec["dataset"]) & (anc.protocol == spec["protocol"])
                    & (anc.variant == "plus_all_0") & (anc.method == "CSU-PL")]
            series.insert(1, (a, P.C_TEAL, "D", "--", "CSU-PL + 0 % anchor"))
        for d, col, mk, ls, lab in series:
            g = level_stats(d)
            ax.errorbar(g.y, g["mean"], yerr=g["std"].fillna(0), fmt=mk + ls, color=col,
                        ms=3.6, lw=1.5, capsize=2.0, elinewidth=.8, label=lab, zorder=4)
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*spec["ylim"])
        if spec["cal"]:
            ax.text(.04, .875, spec["cal"], transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.2, color=P.C_INK)
        ax.set_xlabel("reference adulterant (%)")
        ax.set_ylabel("predicted (%)")
        ax.set_title(f"({letter}) {spec['name']}")
        ax.text(.04, .96, spec["note"], transform=ax.transAxes, ha="left", va="top",
                fontsize=6.8, color=P.C_INK)
        if spec["extra"] == "anchor":
            ax.text(.04, .87, "RMSEP 4.0 → 1.7 %", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.8, color=P.C_TEAL)
        ax.set_aspect(spec["aspect"], adjustable="box")
        if spec["dataset"] == "D2" and spec["protocol"] == "low":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    fig.subplots_adjust(wspace=.30, hspace=.42)
    P.save(fig, os.path.join(FIG, "fig3_extrapolation.pdf"))
    P.save(fig, os.path.join(FIG, "fig3_extrapolation.png"))
    print("wrote figures/fig3_extrapolation.pdf/png")


if __name__ == "__main__":
    main()
