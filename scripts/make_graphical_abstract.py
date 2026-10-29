"""Graphical abstract（13 × 5 cm，600 dpi）。

版式：一条从左到右的流程——参考集 → 事先审计 → 越界样品（通过/否决），
底部一句结论加一行误差比。图内不出现任何未定义的缩写或符号：CNR 写为
"contrast-to-noise"，γ 写为 "curvature"，阈值直接写在数量旁边；两行注释
（"contrast clears noise…"）为可读性让位给放大的判定行。

数字来源（正文 §4.2 / 摘要，七数据集 21 条件）：通过区条件下中位比值 0.92，
未通过区 1.59，均相对逐条件最优比较器。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import pubstyle as P

P.apply_pub_style()
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42      # 以 TrueType(Type 42) 嵌入，投稿系统可解析
rcParams["ps.fonttype"] = 42
FIG = os.path.join(ROOT, "figures")


def rbox(ax, x, y, w, h, face, edge, lw=0.8, ls="-", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                facecolor=face, edgecolor=edge, linewidth=lw,
                                linestyle=ls, zorder=2))


def arrow(ax, xy1, xy2, color=P.C_GREY, lw=1.0):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, zorder=4,
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(5.12, 1.97))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- left: reference set ----
    rbox(ax, 0.012, 0.19, 0.300, 0.760, "#EEF3F9", P.C_NAVY)
    ax.text(0.162, 0.885, "Reference set", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=P.C_NAVY)
    ax.text(0.162, 0.816, "spans the reporting range", ha="center", va="center",
            fontsize=6.8, color=P.C_INK)

    sax = fig.add_axes([0.035, 0.245, 0.255, 0.420])
    x = np.linspace(0, 1, 300)
    peaks = [(0.22, 0.10, 0.9), (0.40, 0.06, 0.6), (0.58, 0.08, 0.75),
             (0.76, 0.05, 0.5), (0.88, 0.06, 0.35)]

    def spec(shift=0.0, scale=1.0):
        y = np.zeros_like(x)
        for mu, sd, a in peaks:
            y += a * np.exp(-0.5 * ((x - mu - shift) / sd) ** 2)
        return scale * y

    gm = spec(0.0, 1.0)
    ga = spec(0.035, 0.7)
    mix = 0.55 * gm + 0.45 * ga
    sax.plot(x, gm, color=P.C_NAVY, lw=1.1, label="matrix")
    sax.plot(x, ga, color=P.C_VERM, lw=1.1, label="adulterant")
    sax.plot(x, mix, color=P.C_GREY, lw=0.9, ls="--", label="mixture")
    sax.set_xticks([]); sax.set_yticks([])
    sax.set_ylim(0, 1.55)
    sax.set_facecolor("none")
    for s in sax.spines.values():
        s.set_visible(False)
    sax.legend(fontsize=6.2, loc="upper right", handlelength=1.2,
               labelspacing=0.22, borderaxespad=0.1)

    # ---- middle: audit ----
    ax.text(0.495, 0.885, "Audit", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=P.C_NAVY)
    ax.text(0.495, 0.816, "two thresholds, fixed in advance", ha="center",
            va="center", fontsize=6.8, color=P.C_INK)
    rbox(ax, 0.345, 0.420, 0.300, 0.330, "#F7F9FC", P.C_BLUE, lw=1.0)
    ax.text(0.495, 0.660, "contrast-to-noise ≥ 10", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=P.C_NAVY)
    ax.text(0.495, 0.520, "|curvature| ≤ 1", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=P.C_NAVY)
    ax.text(0.495, 0.300, "both criteria must hold,\ndecided before measurement",
            ha="center", va="center", fontsize=6.4, color=P.C_INK,
            linespacing=1.35)

    # ---- right: outcomes ----
    ax.text(0.827, 0.885, "Out-of-range sample", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=P.C_NAVY)
    ax.text(0.827, 0.816, "outside the calibrated range", ha="center",
            va="center", fontsize=6.8, color=P.C_INK)

    rbox(ax, 0.678, 0.545, 0.298, 0.215, "#E7F0EA", P.C_GREEN, lw=1.0)
    ax.text(0.827, 0.703, "pass", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=P.C_GREEN)
    ax.text(0.827, 0.608, "report estimate ± interval", ha="center",
            va="center", fontsize=7.2, color=P.C_INK)

    rbox(ax, 0.678, 0.255, 0.298, 0.215, "#F8EDE7", P.C_VERM, lw=1.0)
    ax.text(0.827, 0.413, "fail", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=P.C_VERM)
    ax.text(0.827, 0.318, "withhold the value", ha="center", va="center",
            fontsize=7.2, color=P.C_INK)

    # arrows
    arrow(ax, (0.315, 0.600), (0.342, 0.618))
    arrow(ax, (0.647, 0.640), (0.675, 0.652))
    arrow(ax, (0.647, 0.505), (0.675, 0.375))

    # footer
    ax.text(0.5, 0.112, "Audit the data before trusting the extrapolation",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=P.C_NAVY)
    ax.text(0.5, 0.042,
            "Median error relative to the best comparator: "
            "0.92 where the audit passes, 1.59 where it fails",
            ha="center", va="center", fontsize=6.4, color=P.C_INK)

    fig.savefig(os.path.join(FIG, "graphical_abstract.png"), dpi=600,
                bbox_inches="tight", pad_inches=0.01, facecolor="white")
    fig.savefig(os.path.join(FIG, "graphical_abstract.pdf"),
                bbox_inches="tight", pad_inches=0.01, facecolor="white")
    print("saved", os.path.join(FIG, "graphical_abstract.png"))


if __name__ == "__main__":
    main()
