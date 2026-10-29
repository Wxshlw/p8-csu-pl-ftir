"""Fig.4 —— 设计扫描：杠杆预测外推误差，条件数不预测；设计规模与 0 锚点。"""
from __future__ import annotations

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
    "axes.titlesize": 9.0, "axes.titlepad": 3.5,
    "axes.labelsize": 8.8, "axes.labelpad": 2.6,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 7.0, "lines.linewidth": 1.5,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4,
})


def binned(x, y, nbin=8):
    idx = np.argsort(x)
    edges = np.array_split(idx, nbin)
    xs = np.array([np.median(x[i]) for i in edges if len(i) > 2])
    ys = np.array([np.median(y[i]) for i in edges if len(i) > 2])
    lo = np.array([np.quantile(y[i], .25) for i in edges if len(i) > 2])
    hi = np.array([np.quantile(y[i], .75) for i in edges if len(i) > 2])
    return xs, ys, lo, hi


def main():
    df = pd.read_csv(os.path.join(OUT, "exp04_design_sweep.csv"))
    real = df[df.source == "real"].copy()
    z0 = real[real.has_zero == 0]
    z1 = real[real.has_zero == 1]
    cor = pd.read_csv(os.path.join(OUT, "exp04_correlations.csv")).set_index(["source", "metric"])
    b = pd.read_csv(os.path.join(OUT, "exp04b_leverage_check.csv"))
    bk = json.load(open(os.path.join(OUT, "exp04b_summary.json")))
    byk = pd.read_csv(os.path.join(OUT, "exp04_by_k.csv"))

    # ---- SI Fig. S1：条件数面板（移出正文）----
    fs, axs = plt.subplots(figsize=(3.05, 2.45))
    axs.scatter(z0.kappa, z0.rmse, s=7, c=P.C_NAVY, alpha=.45, lw=0,
                label="no 0 % anchor")
    axs.scatter(z1.kappa, z1.rmse, s=7, c=P.C_VERM, alpha=.45, lw=0,
                label="0 % anchor")
    xs, ys, lo, hi = binned(real.kappa.values, real.rmse.values)
    axs.plot(xs, ys, "-o", color=P.C_INK, ms=3.4, lw=1.5, zorder=5)
    axs.fill_between(xs, lo, hi, color=P.C_INK, alpha=.10, lw=0)
    axs.set_xscale("log")
    axs.set_xlabel("concentration condition number  $\\kappa(\\mathbf{C})$")
    axs.set_ylabel("extrapolation RMSEP (%)")
    rho = cor.loc[("real", "kappa")]
    axs.text(.04, .97, f"$\\rho$ = +{rho.rho:.2f}\n$p$ = {rho.p:.2f}",
             transform=axs.transAxes, ha="left", va="top", fontsize=7.4)
    leg = axs.legend(loc="upper left", bbox_to_anchor=(0.02, 0.80),
                     handletextpad=.5, borderpad=.25, labelspacing=.3,
                     frameon=True, facecolor="white", edgecolor="none",
                     framealpha=.92)
    leg.set_zorder(10)
    axs.set_ylim(3.1, 8.0)
    fs.tight_layout()
    P.save(fs, os.path.join(FIG, "figS1_kappa.pdf"))
    P.save(fs, os.path.join(FIG, "figS1_kappa.png"))
    plt.close(fs)
    print("wrote figures/figS1_kappa.pdf/png")

    fig = plt.figure(figsize=(P.DOUBLE_W, 4.2))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.30)
    axes = {(0, 0): fig.add_subplot(gs[0, :]),
            (1, 0): fig.add_subplot(gs[1, 0]),
            (1, 1): fig.add_subplot(gs[1, 1])}

    # ---- (a) leverage ----
    ax = axes[0, 0]
    ax.scatter(z0.leverage50, z0.rmse, s=7, c=P.C_NAVY, alpha=.45, lw=0,
               label="no 0 % anchor")
    ax.scatter(z1.leverage50, z1.rmse, s=7, c=P.C_VERM, alpha=.45, lw=0,
               label="0 % anchor")
    xs, ys, lo, hi = binned(real.leverage50.values, real.rmse.values)
    ax.plot(xs, ys, "-o", color=P.C_INK, ms=3.4, lw=1.5, zorder=5,
            label="binned median")
    ax.fill_between(xs, lo, hi, color=P.C_INK, alpha=.10, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("prediction leverage  $h(c^*)$")
    ax.set_ylabel("extrapolation RMSEP (%)")
    ax.set_title("(a) Design leverage")
    rho = cor.loc[("real", "leverage50")]
    m, e = f"{rho.p:.1e}".split("e")
    ax.text(.04, .96, f"$\\rho$ = +{rho.rho:.2f}\n$p$ = {m}$\\times10^{{{int(e)}}}$",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=7.4)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), handletextpad=.5,
                    borderpad=.25, labelspacing=.3,
                    frameon=True, facecolor="white", edgecolor="none", framealpha=.92)
    leg.set_zorder(10)
    ax.set_ylim(3.1, 8.0)

    # ---- (b) synthetic leverage scaling (log-log) ----
    ax = axes[1, 0]
    h = b.h50.values
    y = b.rmse50.values
    ok = np.isfinite(h) & np.isfinite(y) & (h > 0) & (y > 0)
    ax.scatter(h[ok], y[ok], s=9, c=P.C_NAVY, alpha=.55, lw=0)
    pfit = np.polyfit(np.log(h[ok]), np.log(y[ok]), 1)
    xx = np.geomspace(h[ok].min(), h[ok].max(), 80)
    ax.plot(xx, np.exp(np.polyval(pfit, np.log(xx))), color=P.C_VERM, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("leverage  $h(c^*)$")
    ax.set_ylabel("RMSEP at 50 % (%)")
    ax.set_title("(b) Leverage scaling")
    pred = np.exp(np.polyval(pfit, np.log(h[ok])))
    r2 = 1 - np.sum((np.log(y[ok]) - np.log(pred)) ** 2) / np.sum(
        (np.log(y[ok]) - np.log(y[ok]).mean()) ** 2)
    ax.text(.04, .96, f"RMSEP $\\propto h^{{{pfit[0]:.2f}}}$\n$R^2$ = {r2:.2f} (log–log)\n"
                      f"$n$ = {ok.sum()} designs", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.4)

    # ---- (d) k and zero anchor ----
    ax = axes[1, 1]
    for hz, col, lab in ((0, P.C_NAVY, "no 0 % anchor"), (1, P.C_VERM, "0 % anchor")):
        g = byk[(byk.has_zero == hz) & (byk.k <= 7)]
        ax.plot(g.k, g.rmse_med, "-o", color=col, ms=4, lw=1.6, label=lab)
        ax.fill_between(g.k, g.q25, g.q75, color=col, alpha=.13, lw=0)
    ax.set_xlabel("number of calibration levels  $k$")
    ax.set_ylabel("median RMSEP (%)")
    ax.set_title("(c) Design size")
    ax.set_xticks(range(2, 8))
    ax.legend(loc="upper right", handletextpad=.5, borderpad=.2, labelspacing=.3)
    ax.set_ylim(3.3, 5.0)

    P.save(fig, os.path.join(FIG, "fig4_design.pdf"))
    P.save(fig, os.path.join(FIG, "fig4_design.png"))
    print("wrote figures/fig4_design.pdf/png")


if __name__ == "__main__":
    main()
