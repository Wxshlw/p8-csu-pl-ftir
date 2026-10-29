"""Fig.2 —— 可辨识性审计：先验诊断 -> 外推结局。

(a) 体系定位图：端元 CNR vs 校准曲率 |γ|，阴影为可外推操作区
(b) 21 个条件的 RMSEP 比值（CSU / 最佳基线），按审计通过/未通过分组
(c) 曲率 |γ| 与校准斜率畸变
(d) D7 标签可分辨性置换检验（样品级置换，固定 GroupKFold）
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

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
    "legend.fontsize": 6.6, "lines.linewidth": 1.4,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
})

NAME = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4",
        "D7w": "D6", "D7p": "D5", "D8": "D7"}
FULL = {"D1": "olive/sunflower oil", "D2": "coffee/malt flour",
        "D3": "orange/apple juice", "D7d": "cinnamon/wood dust",
        "D7w": "cinnamon/walnut shell", "D7p": "cinnamon/peanut shell",
        "D8": "turmeric/wheat flour"}
CNR_GATE, GAM_GATE = 10.0, 1.0
PROTO_MK = {"high": "o", "interp": "s", "low": "^"}


def load():
    au = json.load(io.open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8"))
    vb = pd.read_csv(os.path.join(OUT, "exp03_vs_best.csv"))
    pm = json.load(io.open(os.path.join(OUT, "exp05b_d7_permutation.json"), encoding="utf-8"))
    tab = []
    for ds, r in au.items():
        tab.append(dict(dataset=ds, cnr=r["endmember"]["cnr"],
                        absg=abs(r["curvature"]["gamma"]),
                        leak=r["endmember"]["baseline_leak"],
                        snr=r["resolvability"]["adjacent_snr_med"]))
    tab = pd.DataFrame(tab)
    tab["gate"] = (tab.cnr >= CNR_GATE) & (tab.absg <= GAM_GATE)
    return au, vb, pm, tab


def panel_a(ax, tab):
    gate_x0, gate_x1 = CNR_GATE, 60.0
    gate_y0, gate_y1 = 0.055, GAM_GATE
    ax.add_patch(Rectangle((gate_x0, gate_y0), gate_x1 - gate_x0, gate_y1 - gate_y0,
                           facecolor=P.C_GREN, alpha=0.10, edgecolor=P.C_GREN,
                           lw=0.9, ls=(0, (4, 2)), zorder=0))
    ax.text(0.97, 0.60, "gate\npasses", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.4, color=P.C_TEAL, linespacing=1.15)
    ax.axvline(CNR_GATE, color=P.C_GREN, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.axhline(GAM_GATE, color=P.C_GREN, lw=0.8, ls=(0, (2, 2)), zorder=0)
    off = {"D2": (7, -2, "left"), "D8": (7, 2, "left"),
           "D1": (-2, 8, "center"), "D7d": (-7, 1, "right"),
           "D3": (-7, -3, "right"), "D7w": (7, 2, "left"),
           "D7p": (7, 4, "left")}
    for _, r in tab.iterrows():
        col = P.C_NAVY if r.gate else P.C_VERM
        ax.scatter(r.cnr, r.absg, s=34, facecolor=col,
                   edgecolor=col, linewidth=1.0, zorder=4)
        dx, dy, ha = off[r.dataset]
        ax.annotate(NAME[r.dataset], (r.cnr, r.absg), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=7.0, color=col if r.gate else P.C_REDD, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.6, 60)
    ax.set_ylim(0.055, 6.0)
    ax.set_xlabel("endmember CNR")
    ax.set_ylabel(r"calibration curvature $|\gamma|$")
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(["1", "3", "10", "30"])
    ax.set_yticks([0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.grid(False)
    P.panel_title(ax, "(a) Regime map")


def panel_b(ax, tab, vb):
    """21 个条件对 4 个预指定比较器的比值：点为中位数，横线为四条比值的跨度。

    显著性取预指定比较（exp25_prespecified.csv 的 p_holm_pre）：实心 = 至少一条
    比较在 Holm 校正后 p < 0.05。逐个条件的“最佳比较器”只在 SI 里作描述性上界。
    """
    pre = pd.read_csv(os.path.join(OUT, "exp25_prespecified.csv"))
    pre = pre[pre.p_holm_pre.notna()]
    old = {"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D7d", "D5": "D7p",
           "D6": "D7w", "D7": "D8"}
    order = []
    for gate_flag in (True, False):
        sub = tab[tab.gate == gate_flag].sort_values("cnr", ascending=False)
        order += list(sub.dataset)
    ypos = {ds: 6.0 - i for i, ds in enumerate(order)}
    ax.axvspan(0.95, 1.0, color="#E9E9E9", zorder=0, lw=0)
    for (ds, prot), g in pre.groupby(["dataset", "protocol"]):
        key = old[ds]
        yy = ypos[key]
        gate_flag = bool(tab.loc[tab.dataset == key, "gate"].iloc[0])
        col = P.C_NAVY if gate_flag else P.C_VERM
        r = np.sort(g.ratio.values)
        sig = bool((g.p_holm_pre < 0.05).any())
        ax.plot([r[0], r[-1]], [yy, yy], color=col, lw=0.9, alpha=0.55, zorder=3,
                solid_capstyle="butt")
        ax.scatter(np.median(r), yy, marker=PROTO_MK[prot], s=30,
                   facecolor=col if sig else "white", edgecolor=col,
                   linewidth=0.9, zorder=4)
    ax.axvline(1.0, color="#777777", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.axhline(3.5, color="#C8C8C8", lw=0.8, zorder=1)
    ax.text(4.55, 5.0, "pass", fontsize=6.6, color=P.C_NAVY, ha="right", va="center",
            fontweight="bold")
    ax.text(4.55, 1.5, "fail", fontsize=6.6, color=P.C_REDD, ha="right", va="center",
            fontweight="bold")
    ax.set_yticks([ypos[d] for d in order])
    ax.set_yticklabels([f"{NAME[d]}" for d in order])
    ax.set_ylim(-0.7, 6.7)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 5.2)
    ax.set_xticks([0.1, 0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.1", "0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel("RMSEP ratio (CSU / prespecified comparator)")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    P.panel_title(ax, "(b) Extrapolation outcome")


def panel_c(ax, au, vb, tab):
    tb = pd.read_csv(os.path.join(OUT, "exp03_table.csv"))
    tb = tb[tb.method == "CSU-PL"]
    rows = []
    for _, r in tb.iterrows():
        a = au[r.dataset]
        rows.append(dict(dataset=r.dataset, protocol=r.protocol,
                         absg=abs(a["curvature"]["gamma"]), slope=r.slope))
    d = pd.DataFrame(rows)
    order = tab.sort_values("absg").dataset.tolist()
    xpos = {ds: i for i, ds in enumerate(order)}
    ax.axhspan(0.8, 1.2, color="#EDEDED", zorder=0, lw=0)
    ax.axhline(1.0, color="#777777", lw=0.9, ls=(0, (3, 2)), zorder=1)
    for _, r in d.iterrows():
        gate = bool(tab.loc[tab.dataset == r.dataset, "gate"].iloc[0])
        col = P.C_NAVY if gate else P.C_VERM
        dx = {"high": -0.14, "interp": 0.0, "low": 0.14}[r.protocol]
        ax.scatter(xpos[r.dataset] + dx, r.slope, marker=PROTO_MK[r.protocol], s=28,
                   facecolor=col, edgecolor=col,
                   linewidth=0.9, zorder=4, clip_on=False)
    gattr = {r.dataset: r.absg for _, r in tab.iterrows()}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{NAME[ds]}\n{gattr[ds]:.2f}" for ds in order], fontsize=6.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-2.4, 3.0)
    ax.set_xlabel(r"dataset (ordered by $|\gamma|$)")
    ax.set_ylabel("calibration slope (pred./ref.)")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.text(0.03, 0.97, "Grey band: slope 0.8–1.2", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2, color=P.C_INK)
    ax.grid(False)
    P.panel_title(ax, "(c) Curvature effect")


def panel_d(ax, pm):
    labels = ["D7d", "D7p", "D7w"]
    for i, ds in enumerate(labels):
        r = pm[ds]
        y = 2 - i
        col = P.C_NAVY if r["p_value"] < 0.05 else P.C_VERM
        ax.plot([r["null_mean"], r["null_p95"]], [y, y], color=P.C_GRAY2, lw=3.2,
                solid_capstyle="butt", zorder=2)
        ax.plot([r["null_mean"]], [y], "|", color=P.C_GREY, ms=7, mew=0.9, zorder=3)
        ax.scatter(r["observed_r2"], y, s=34, facecolor=col, edgecolor=col,
                   linewidth=0.9, zorder=4)
        ax.text(1.02, y, f"p = {r['p_value']:.2f}", fontsize=6.6, va="center",
                ha="left", color=col)
    ax.axvline(0.0, color="#777777", lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([NAME[d] for d in labels])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-1.15, 1.42)
    ax.set_xlabel(r"cross-validated $R^2$ (PLS)")
    ax.text(0.02, 0.97, "Grey: permutation null (mean–95%)", transform=ax.transAxes,
            fontsize=6.2, color=P.C_INK, ha="left", va="top")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    P.panel_title(ax, "(d) Label resolvability")


def main():
    au, vb, pm, tab = load()
    fig, axes = plt.subplots(2, 2, figsize=(P.DOUBLE_W, 4.35))
    panel_a(axes[0, 0], tab)
    panel_b(axes[0, 1], tab, vb)
    panel_c(axes[1, 0], au, vb, tab)
    panel_d(axes[1, 1], pm)
    fig.subplots_adjust(wspace=0.34, hspace=0.46)
    P.save(fig, os.path.join(FIG, "fig2_audit.pdf"))
    P.save(fig, os.path.join(FIG, "fig2_audit.png"))
    print("wrote figures/fig2_audit.pdf/png")


if __name__ == "__main__":
    main()
