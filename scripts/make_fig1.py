"""Fig.1 —— 前向模型概念 + 合成外推验证 + profile 可辨识性。"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
from unmix_ftir import pubstyle as P

P.apply_pub_style()
# 三联图：单栏宽 ~1.7 in，字号按版面可读性下调（ACA 多面板惯例 8-9 pt）
rcParams.update({
    "axes.titlesize": 8.8, "axes.titlepad": 3.0,
    "axes.labelsize": 8.6, "axes.labelpad": 2.5,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 6.6, "lines.linewidth": 1.8,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
})


def main():
    d = json.load(io.open(os.path.join(OUT, "exp02_theory.json"), encoding="utf-8"))
    wvn = np.array(d["endmembers"]["wvn"])
    gm = np.array(d["endmembers"]["gm"])
    ga = np.array(d["endmembers"]["ga"])
    preds = d["predictions"]
    met = {m["scenario"]: m for m in d["metrics"]}
    prof = d["profile"]

    fig, axes = plt.subplots(1, 3, figsize=(P.DOUBLE_W, 2.15))
    # (a) spectra
    ax = axes[0]
    ax.plot(wvn, gm, color=P.C_NAVY, lw=2.0, label=r"matrix $g_m$")
    ax.plot(wvn, ga, color=P.C_VERM, lw=2.0, label=r"adulterant $g_a$")
    ax.plot(wvn, 0.75 * gm + 0.25 * ga, color=P.C_TEAL, lw=1.6, ls=(0, (5, 2)),
            label=r"25 % mixture $A_i$")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_xlim(400, 4000)
    ax.set_ylim(-0.02, 1.15)
    ax.legend(loc="upper right", fontsize=6.6, handlelength=1.4, borderpad=0.2,
              labelspacing=0.3)
    P.panel_title(ax, "(a) Forward model")

    # (b) synthetic extrapolation
    ax = axes[1]
    styles = {"exact": (P.C_NAVY, "o", "exact"),
              "additive": (P.C_TEAL, "s", r"$s\equiv1$"),
              "nonlinear": (P.C_VERM, "^", "nonlinear")}
    for sc, (c, mk, lab) in styles.items():
        arr = [(p["y"], p["pred"]) for p in preds if p["scenario"] == sc]
        y, p = np.array(arr).T
        ax.scatter(y, p, s=15, facecolor=c, edgecolor="white", linewidth=0.35,
                   marker=mk, alpha=0.9,
                   label=f"{lab} ({met[sc]['rmse']:.2f})")
    ax.plot([0, 32], [0, 32], color="#888888", lw=1.2, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(18, 31)
    ax.set_ylim(13, 34)
    ax.set_xlabel("Nominal concentration (%)")
    ax.set_ylabel("Predicted (%)")
    leg = ax.legend(loc="lower right", fontsize=6.3, handletextpad=0.3,
                    borderpad=0.2, labelspacing=0.26,
                    title="RMSEP (%)")
    leg.get_title().set_fontsize(6.1)
    P.panel_title(ax, "(b) Extrapolation")

    # (c) profile likelihood：随部分对比度衰减，内部极小值消失、估计被推向边界
    # 只在窗口内部确有小值时标点；以 40 % 为网格端点截断的 argmin 不是极小值
    ax = axes[2]
    c = np.array(prof["c"]) * 100
    keys = ["1", "0.5", "0.2", "0.05"]
    cols = [P.C_NAVY, P.C_BLUE, P.C_SKY, P.C_GREY]
    ymax = max(np.array(prof["1"]).max(), 1e-12)
    c_max = 30.0
    for k, col in zip(keys, cols):
        R = np.array(prof[k])
        dR = R - R.min()
        ax.plot(c, dR / ymax, color=col, lw=1.8, label=f"{float(k):.2f}")
        j = int(np.argmin(R))
        if c[0] + 0.05 * (c[-1] - c[0]) < c[j] < c_max:
            ax.plot(c[j], 0.0, "o", ms=3.2, color=P.C_INK, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.plot([], [], "o", ms=3.2, color=P.C_INK, markeredgecolor="white",
                    markeredgewidth=0.5, label="minimum")
    ax.axvline(25.0, color="#9A9A9A", lw=0.9, ls=(0, (2, 2)), zorder=0)
    ax.text(25.7, 0.44, "25 %", rotation=90, fontsize=6.2, color=P.C_INK,
            ha="left", va="center")
    ax.set_xlabel("Concentration (%)")
    ax.set_ylabel(r"$R(c)-R_{\min}$ (norm.)")
    ax.set_xlim(0, c_max)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(-0.04, 1.04)
    leg = ax.legend(loc="upper right", fontsize=6.2, handlelength=1.2,
                    labelspacing=0.22, borderpad=0.12, ncol=2,
                    columnspacing=0.8, title=r"contrast $\rho_d$")
    leg.get_title().set_fontsize(6.0)
    P.panel_title(ax, "(c) Identifiability")

    for a in axes:
        a.tick_params(labelsize=7.8)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(os.path.join(FIG, "fig1_forward.pdf"), pad_inches=0.03)
    fig.savefig(os.path.join(FIG, "fig1_forward.png"), pad_inches=0.03)
    plt.close(fig)
    print("saved fig1")


if __name__ == "__main__":
    main()
