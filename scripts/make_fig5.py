"""Fig.5 —— 操作包络：稳健性、尺度约束、区间校准。"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
from unmix_ftir import pubstyle as P

P.apply_pub_style()
rcParams.update({
    "axes.titlesize": 8.6, "axes.titlepad": 3.2,
    "axes.labelsize": 8.2, "axes.labelpad": 2.4,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.4, "lines.linewidth": 1.4,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
})

FACTORS = [
    ("noise", [1, 5, 10, 20], [1, 5, 10, 20], "noise\n($\\times\\sigma_0$)", "(a) Noise"),
    ("scatter", [0.0, 0.03, 0.06, 0.12], [0, 3, 6, 12], "scatter CV\n(%)", "(b) Scattering"),
    ("baseline", [0.0, 0.05, 0.10, 0.20], [0, 5, 10, 20],
     "baseline\n(% level)", "(c) Baseline"),
    ("nonlinear", [0.0, 0.05, 0.10, 0.20, 0.40], [0, 5, 10, 20, 40],
     "nonlinearity\n(% slope loss)", "(d) Nonlinearity"),
    ("matrixvar", [0.0, 0.10, 0.25, 0.50], [0, 10, 25, 50],
     "matrix spread\n(% contrast)", "(e) Matrix spread"),
]
PROTO_MK = {"high": "o", "interp": "s", "low": "^"}


def audit_gate():
    au = json.load(io.open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8"))
    gate = {}
    for ds, r in au.items():
        gate[ds] = (r["endmember"]["cnr"] >= 10) and (abs(r["curvature"]["gamma"]) <= 1)
    return gate


def top_row(fig, gs, rob):
    from matplotlib.ticker import NullLocator
    axes = []
    for i, (fac, vals, ticks, xlab, title) in enumerate(FACTORS):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        sub = rob[rob.factor == fac]
        for meth, col, mk, ls, lab, fill in [
                ("CSU-PL", P.C_NAVY, "o", "-", "CSU-PL", True),
                ("additive-only", P.C_VERM, "s", "--", "fixed scale", False)]:
            g = sub[sub.method == meth].groupby("value").rmse.agg(["mean", "std"]).reset_index()
            g = g[g.value.isin(vals)].copy()
            pos = {v: i for i, v in enumerate(vals)}
            ax.plot(g.value.map(pos), g["mean"], mk + ls, color=col, ms=2.8, lw=1.3,
                    mfc=col if fill else "white", mec=col, mew=0.9, label=lab, zorder=4)
        ax.axhline(1.0, color="#9A9A9A", lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.07, 22)
        ax.set_xlim(-0.35, len(vals) - 0.65)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=6.6)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(xlab, fontsize=7.0, labelpad=2.0)
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.grid(False)
        ax.tick_params(which="minor", left=False)
        ax.set_yticks([0.1, 1, 10])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0.1", "1", "10"])
            ax.set_ylabel("RMSEP (%)")
            ax.legend(loc="lower right", handletextpad=.4, borderpad=.15,
                      labelspacing=.18, fontsize=6.0, borderaxespad=.15)
    return axes


def panel_f(ax, fx, gate):
    from matplotlib.ticker import NullLocator
    piv = fx.pivot_table(index=["dataset", "protocol"], columns="fix_scale",
                         values="rmse").reset_index()
    piv = piv.rename(columns={False: "free", True: "fixed"})
    for _, r in piv.iterrows():
        col = P.C_NAVY if gate[r.dataset] else P.C_VERM
        ax.scatter(r.fixed, r.free, s=17, marker=PROTO_MK[r.protocol],
                   facecolor=col, edgecolor="white", linewidth=0.4, zorder=4)
    lim = [2.0, 45]
    ax.plot(lim, lim, color="#777777", lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xticks([3, 10, 30]); ax.set_xticklabels(["3", "10", "30"])
    ax.set_yticks([3, 10, 30]); ax.set_yticklabels(["3", "10", "30"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("fixed-scale RMSEP (%)")
    ax.set_ylabel("free-scale RMSEP (%)")
    ax.annotate("D1 high\n23.3 → 5.0", (23.3, 4.99), xytext=(-4, -10),
                textcoords="offset points", fontsize=6.2, color=P.C_INK,
                ha="right", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777",
                                shrinkA=1, shrinkB=3))
    ax.text(0.03, 0.97, f"{int((piv.free < piv.fixed).sum())}/{len(piv)} below identity",
            transform=ax.transAxes, fontsize=6.2, color=P.C_INK, ha="left", va="top")
    ax.grid(False)
    P.panel_title(ax, "(f) Scale constraint", pad=3.2)


def panel_g(ax, summ, gate):
    for _, r in summ.iterrows():
        col = P.C_NAVY if gate[r.dataset] else P.C_VERM
        ax.scatter(r.cov95, r.cov95f, s=20, marker=PROTO_MK[r.protocol],
                   facecolor=col, edgecolor="white", linewidth=0.4, zorder=4)
    ax.plot([-0.03, 1.03], [-0.03, 1.03], color="#777777", lw=1.0,
            ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.95, color=P.C_GREN, lw=0.9, ls=(0, (2, 2)), zorder=0)
    ax.text(0.02, 0.905, "nominal 95 %", fontsize=6.0, color=P.C_TEAL,
            ha="left", va="top")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("raw 95 % coverage")
    ax.set_ylabel("calibrated 95 % coverage")
    ax.text(0.97, 0.05, f"{int((summ.cov95f >= 0.75).sum())}/{len(summ)} ≥ 0.75",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2,
            color=P.C_INK)
    ax.grid(False)
    P.panel_title(ax, "(g) Interval calibration", pad=3.2)


def main():
    rob = pd.read_csv(os.path.join(OUT, "exp06_robustness.csv"))
    fx = pd.read_csv(os.path.join(OUT, "exp10_fixscale.csv"))
    summ = pd.read_csv(os.path.join(OUT, "exp08_summary.csv"))
    gate = audit_gate()
    fig = plt.figure(figsize=(P.DOUBLE_W, 4.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.30], hspace=0.60)
    gt = outer[0].subgridspec(1, 5, wspace=0.20)
    gb = outer[1].subgridspec(1, 2, wspace=0.30)
    top_row(fig, gt, rob)
    axf = fig.add_subplot(gb[0, 0])
    axg = fig.add_subplot(gb[0, 1])
    panel_f(axf, fx, gate)
    panel_g(axg, summ, gate)
    P.save(fig, os.path.join(FIG, "fig5_robustness.pdf"))
    P.save(fig, os.path.join(FIG, "fig5_robustness.png"))
    print("wrote figures/fig5_robustness.pdf/png")


if __name__ == "__main__":
    main()
