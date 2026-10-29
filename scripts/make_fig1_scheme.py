# -*- coding: utf-8 -*-
"""Fig. 1 —— 研究设计流程图（5 个阶段，图中嵌真实数据小图）。

版式：单向流程 + 阶段色带。每个方框顶部是一条实色标题带（论文色板：
深蓝 / 中蓝 / 青 / 蓝灰 / 赤陶 / 中性深灰），白字粗体；框内放该阶段的真实
数据小图（谱库、划分标尺、审计平面、剖面曲线、判定）。不分 (a)(b)(c)，
图内不引用其他图号。句子首字母大写；除数据线外文字一律黑色。

字号/线宽与 Fig. 2-6 同级：标题 9.2 pt、说明 7.0、刻度 6.8-7.2、主数据线
1.4-1.8 pt、框线 0.9、箭头 1.4。Canvas = 15.5 cm（6.10 in），1:1 排版。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import pubstyle as P

P.apply_pub_style()
rcParams.update({
    "font.size": 7.0,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
    "xtick.color": "#1A1A1A",
    "ytick.color": "#1A1A1A",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
})

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

W, H = 6.10, 4.08

# 阶段色带（实色），标题白字
BANDS = {
    "data":   P.C_NAVY,
    "split":  P.C_BLUE,
    "audit":  P.C_TEAL,
    "est":    P.C_PURP,
    "decide": P.C_VERM,
    "eval":   "#4A5560",
}
# 谱库按食品基质着色：橄榄油 / 咖啡 / 橙汁 / 肉桂（三档浓度）/ 姜黄
TRACE_COL = {"D1": P.C_NAVY, "D2": P.C_BLUE, "D3": P.C_TEAL,
             "D7w": "#8FA3CC", "D7p": "#6B83B8", "D7d": "#40598F",
             "D8": P.C_VERM}
# 图内数据集编号：正文按 D1–D7 连续编号，分析键 D7d/D7p/D7w/D8 依次显示为 D4–D7
DS_LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4",
            "D7p": "D5", "D7w": "D6", "D8": "D7"}
EDGE = "#C9D2DB"
TEAL_FACE = "#EAF2F1"
WARM_FACE = "#FBF0E9"
INK = "#1A1A1A"


def box(ax, x, y, w, h, face="white", edge=EDGE, lw=0.9, radius=0.05, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z))


def band(ax, x, y, w, h, key):
    """标题色带：顶部圆角、底边平直。"""
    col = BANDS[key]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.05",
        linewidth=0, facecolor=col, zorder=2))
    ax.add_patch(Rectangle((x, y), w, 0.07, linewidth=0, facecolor=col,
                           zorder=2))


def arrow(ax, x0, y0, x1, y1, color="#4A5560", lw=1.4, ms=9.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, shrinkA=0.0, shrinkB=0.0, zorder=4))


def txt(ax, x, y, s, size=7.0, color=INK, ha="left", va="baseline",
        weight="normal", z=6, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=z, **kw)


# ---------------------------------------------------------------------------
def d1_example():
    """真实 D1：high-end 协议、30 % 测试谱（按绝对误差取中位样本）。"""
    from unmix_ftir import data, design, forward as F, unmix

    cfg = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0,
                            resolution=np.nan, n_iter=40, atr_correct=False)
    sg = F.SG(window=15, poly=3, deriv=1)

    d = data.load("D1")
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    br = data.brands("D1", groups)
    Xv = sg.transform(X)

    tr_lv, te_lv = design.protocol_split("D1", "high")
    tr = np.isin(y, tr_lv)
    te = np.isin(y, te_lv)
    mdl = unmix.fit(Xv[tr], y[tr], wvn, cfg, brand=br[tr])

    idx = np.where(te & (y == 30.0))[0]
    pred_all = mdl.predict(Xv[idx], brand=br[idx], n_grid=1001)
    c_all = pred_all["c_hat"] * 100.0
    k = int(np.argsort(np.abs(c_all - 30.0))[len(idx) // 2])
    i = int(idx[k])

    pred = mdl.predict(Xv[i:i + 1], brand=br[i:i + 1], n_grid=1001)
    grid, curve = pred["grid"], pred["curve"][0]
    c_hat = float(pred["c_hat"][0]) * 100.0
    j = int(np.argmin(curve))
    s2 = float(mdl.sigma_hat) ** 2

    thr = curve[j] + 3.841 * s2
    left, right = j, j
    while left > 0 and curve[left - 1] <= thr:
        left -= 1
    while right < len(grid) - 1 and curve[right + 1] <= thr:
        right += 1
    lo, hi = float(grid[left]) * 100.0, float(grid[right]) * 100.0

    return dict(grid=grid * 100.0, r=(curve - curve[j]) / s2,
                c_hat=c_hat, lo=lo, hi=hi, y=float(y[i]))


def dataset_traces():
    """七套数据的平滑平均谱（真实数据，仅用于谱库缩略图）。"""
    from unmix_ftir import data, forward as F

    sg = F.SG(window=31, poly=2, deriv=0)
    out = {}
    for k in data.ALL:
        d = data.load(k)
        m = sg.transform(d["X"].mean(axis=0, keepdims=True))[0]
        m = m - np.percentile(m, 2)
        s = np.percentile(m, 99.5)
        out[k] = (d["wvn"], m / s if s > 0 else m, d["X"].shape[0])
    return out


# ---------------------------------------------------------------------------
def visual_spectra(axv, traces):
    keys = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
    step = 0.30
    axv.set_xlim(4560, 400)
    axv.set_ylim(-0.40, 6 * step + 0.28)
    off = 0.0
    for k in keys:
        w, m, n = traces[k]
        axv.plot(w, off + 0.17 * m, color=TRACE_COL[k], lw=1.4, zorder=3)
        axv.text(4500, off + 0.015, DS_LABEL[k], fontsize=6.6, color=INK,
                 ha="left", va="bottom", zorder=4)
        off += step
    axv.set_xticks([])
    axv.set_yticks([])
    for s in axv.spines.values():
        s.set_visible(False)
    axv.text(0.995, 0.0, "Mean spectra, stacked", fontsize=6.6,
             color=INK, ha="right", va="bottom", transform=axv.transAxes)


def visual_splits(axv):
    from unmix_ftir import design
    axv.set_xlim(0, 1)
    axv.set_ylim(0, 1)
    axv.axis("off")
    x0, x1 = 0.29, 0.93
    levels = np.arange(0, 31, 3)

    # 图例用绘制的实心/空心点，不用字符
    axv.plot([0.33], [0.95], marker="o", ms=4.2, mfc=P.C_NAVY, mec="white",
             mew=0.5, ls="none", transform=axv.transAxes, clip_on=False)
    axv.text(0.39, 0.95, "calibration", fontsize=6.4, color=INK,
             ha="left", va="center", transform=axv.transAxes)
    axv.plot([0.68], [0.95], marker="o", ms=4.2, mfc="white", mec=P.C_VERM,
             mew=1.2, ls="none", transform=axv.transAxes, clip_on=False)
    axv.text(0.74, 0.95, "held out", fontsize=6.4, color=INK,
             ha="left", va="center", transform=axv.transAxes)

    rows = [("Interp.", "interp", 0.66), ("Low-end", "low", 0.44),
            ("High-end", "high", 0.22)]
    for name, key, yy in rows:
        tr_lv, te_lv = design.protocol_split("D1", key)
        axv.plot([x0 - 0.02, x1], [yy, yy], color="#CFCFCF", lw=1.0, zorder=1)
        axv.text(0.0, yy, name, fontsize=6.6, color=INK,
                 ha="left", va="center", transform=axv.transData)
        for lv in levels:
            xx = x0 + lv / 30.0 * (x1 - x0)
            if lv in tr_lv:
                axv.plot([xx], [yy], marker="o", ms=3.8, mfc=P.C_NAVY,
                         mec="white", mew=0.5, zorder=4,
                         transform=axv.transData)
            else:
                axv.plot([xx], [yy], marker="o", ms=3.8, mfc="white",
                         mec=P.C_VERM, mew=1.2, zorder=4,
                         transform=axv.transData)
    axv.text(x0 - 0.02, 0.03, "0", fontsize=6.4, color=INK, ha="center",
             transform=axv.transData)
    axv.text(x1, 0.03, "30 %", fontsize=6.4, color=INK, ha="right",
             transform=axv.transData)


def visual_audit(axv, tab):
    axv.set_xscale("log")
    axv.set_yscale("log")
    axv.set_xlim(0.55, 60)
    axv.set_ylim(0.05, 7)
    axv.add_patch(Rectangle((10, 0.05), 50, 0.95, facecolor=P.C_TEAL,
                            alpha=0.14, edgecolor="none", zorder=0))
    axv.axvline(10, color=P.C_TEAL, lw=1.0, ls=(0, (3, 2)), zorder=1)
    axv.axhline(1.0, color=P.C_TEAL, lw=1.0, ls=(0, (3, 2)), zorder=1)

    lab = {"D7p": (1.02, 0.078, "left", "center"),
           "D7w": (2.95, 3.9, "left", "center"),
           "D8": (20.0, 3.3, "right", "bottom"),
           "D3": (5.3, 0.30, "left", "center"),
           "D7d": (17.3, 0.60, "left", "center"),
           "D1": (21.0, 0.081, "left", "bottom"),
           "D2": (34.5, 0.20, "center", "bottom")}
    for _, r in tab.iterrows():
        k = r["Dataset"]
        col = P.C_NAVY if r["Gate"] == "pass" else P.C_VERM
        axv.plot([r["CNR"]], [abs(r["Gamma"])], marker="o", ms=4.6, mfc=col,
                 mec="white", mew=0.8, zorder=5)
        lx, ly, ha, va = lab[k]
        axv.text(lx, ly, DS_LABEL[k], fontsize=6.4, color=col, ha=ha, va=va, zorder=6)

    axv.set_xticks([1, 10])
    axv.set_xticklabels(["1", "10"])
    axv.set_yticks([0.1, 1])
    axv.set_yticklabels(["0.1", "1"])
    axv.tick_params(labelsize=7.0, length=2.4, width=0.8, pad=1.2,
                    which="major", colors=INK)
    axv.tick_params(which="minor", length=1.2, width=0.6, colors=INK)
    axv.set_xlabel("CNR", fontsize=7.6, labelpad=1.2, color=INK)
    axv.set_ylabel(r"$|\gamma|$", fontsize=7.6, labelpad=1.6, color=INK)
    axv.grid(False)
    for s in ("top", "right"):
        axv.spines[s].set_visible(False)


def visual_profile(axv, ex):
    rr = np.clip(ex["r"], 1e-2, None)
    axv.axvspan(0, 15, color=P.C_NAVY, alpha=0.06, lw=0)
    axv.axvspan(ex["lo"], ex["hi"], ymin=0.06, ymax=0.82,
                color=P.C_ORAN, alpha=0.55, lw=0)
    axv.plot(ex["grid"], np.log10(rr), color=P.C_VERM, lw=1.8, zorder=3)
    axv.axhline(np.log10(3.841), color="#6E6E6E", lw=0.9, ls=(0, (2.4, 2)))
    axv.axvline(15.0, color=P.C_NAVY, lw=0.9, ls=(0, (3, 2.2)))
    axv.axvline(ex["y"], color="#6E6E6E", lw=0.9, ls=(0, (1.2, 1.6)))
    axv.set_xlim(0, 36.5)
    axv.set_ylim(-2.4, 4.4)
    axv.set_xticks([0, 15, 30])
    axv.set_yticks([-2, 0, 2, 4])
    axv.set_yticklabels([r"$10^{-2}$", "1", r"$10^{2}$", r"$10^{4}$"])
    axv.tick_params(labelsize=6.8, length=2.4, width=0.8, pad=1.2,
                    colors=INK)
    axv.set_xlabel("c (%)", fontsize=7.6, labelpad=1.2, color=INK)
    axv.set_ylabel(r"$\Delta R/\hat\sigma^{2}$", fontsize=7.6, labelpad=1.6,
                   color=INK)
    axv.grid(False)
    for s in ("top", "right"):
        axv.spines[s].set_visible(False)
    axv.text(6.6, 1.85, "Calibration\n0–15 %", fontsize=6.4,
             color=INK, ha="center", va="center", linespacing=1.15)
    axv.text(31.6, 0.66, "95%", fontsize=6.4, color=INK,
             ha="left", va="bottom")
    axv.text(30.6, 3.50, "30.0 %", fontsize=6.4, color=INK,
             ha="left", va="center")
    axv.text(21.2, -1.35, rf"$\hat c$ =" + "\n" + f"{ex['c_hat']:.1f} %",
             fontsize=6.8, color=P.C_VERM, ha="center", va="center",
             fontweight="bold", linespacing=1.15)


def visual_decision(axv):
    axv.set_xlim(0, 1)
    axv.set_ylim(0, 1)
    axv.axis("off")
    axv.add_patch(FancyBboxPatch((0.0, 0.56), 1.0, 0.42,
                                 boxstyle="round,pad=0.004,rounding_size=0.03",
                                 facecolor=TEAL_FACE, edgecolor=P.C_GREN,
                                 lw=0.9, zorder=2))
    axv.plot([0.048, 0.072, 0.112], [0.878, 0.850, 0.918], color=P.C_GREN,
             lw=1.8, solid_capstyle="round", zorder=5)
    axv.text(0.145, 0.840, "Report", fontsize=7.8, color=P.C_GREN,
             fontweight="bold", va="baseline")
    axv.text(0.145, 0.735, "Audit passed; interval", fontsize=6.6,
             color=INK, va="baseline")
    axv.text(0.145, 0.645, "resolves the range", fontsize=6.6,
             color=INK, va="baseline")

    axv.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.42,
                                 boxstyle="round,pad=0.004,rounding_size=0.03",
                                 facecolor=WARM_FACE, edgecolor=P.C_VERM,
                                 lw=0.9, zorder=2))
    axv.plot([0.048, 0.112], [0.297, 0.377], color=P.C_VERM, lw=1.8,
             solid_capstyle="round", zorder=5)
    axv.plot([0.112, 0.048], [0.297, 0.377], color=P.C_VERM, lw=1.8,
             solid_capstyle="round", zorder=5)
    axv.text(0.145, 0.300, "Withhold", fontsize=7.8, color=P.C_VERM,
             fontweight="bold", va="baseline")
    axv.text(0.145, 0.195, "Audit failed, or interval", fontsize=6.6,
             color=INK, va="baseline")
    axv.text(0.145, 0.105, "wider than the range", fontsize=6.6,
             color=INK, va="baseline")


# ---------------------------------------------------------------------------
def main():
    ex = d1_example()
    traces = dataset_traces()
    tab = pd.read_csv(os.path.join(ROOT, "tables", "table1_datasets.csv"))

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    mg, cw, ch, gap = 0.10, 1.806, 1.82, 0.24
    x1 = mg
    x2 = mg + cw + gap
    x3 = mg + 2 * (cw + gap)
    ytop = mg + ch + gap
    ybot = mg

    cells = [
        (x1, ytop, "data", "Data", "7 systems, 3,470 spectra"),
        (x2, ytop, "split", "Sample-level splits",
         "Three held-out protocols"),
        (x3, ytop, "audit", "A priori audit",
         "CNR ≥ 10, |γ| ≤ 1, no labels"),
        (x3, ybot, "est", "Profile-likelihood estimate",
         "Fixed endmembers, profile CI"),
        (x2, ybot, "decide", "Report or withhold",
         "Report only when identified"),
    ]
    for x, y, key, title, cap in cells:
        box(ax, x, y, cw, ch)
        band(ax, x, y + ch - 0.40, cw, 0.40, key)
        box(ax, x, y, cw, ch, face="none", lw=0.9, z=3)
        txt(ax, x + 0.12, y + ch - 0.265, title, size=9.2, color="white",
            weight="bold")
        txt(ax, x + 0.12, y + ch - 0.60, cap, size=7.0, color=INK)

    # 流程箭头
    ymid_t, ymid_b = ytop + 0.88, ybot + 0.88
    arrow(ax, x1 + cw + 0.03, ymid_t, x2 - 0.03, ymid_t)
    arrow(ax, x2 + cw + 0.03, ymid_t, x3 - 0.03, ymid_t)
    arrow(ax, x3 + cw / 2, ytop - 0.02, x3 + cw / 2, ybot + ch + 0.02)
    arrow(ax, x3 - 0.03, ymid_b, x2 + cw + 0.03, ymid_b)

    # 各阶段图形
    av = fig.add_axes([(x1 + 0.08) / W, (ytop + 0.10) / H,
                       (cw - 0.16) / W, (ch - 0.74) / H])
    visual_spectra(av, traces)

    av = fig.add_axes([(x2 + 0.06) / W, (ytop + 0.14) / H,
                       (cw - 0.12) / W, (ch - 0.78) / H])
    visual_splits(av)

    av = fig.add_axes([(x3 + 0.44) / W, (ytop + 0.31) / H,
                       (cw - 0.56) / W, (ch - 0.93) / H])
    visual_audit(av, tab)

    av = fig.add_axes([(x3 + 0.46) / W, (ybot + 0.30) / H,
                       (cw - 0.58) / W, (ch - 0.99) / H])
    visual_profile(av, ex)

    av = fig.add_axes([(x2 + 0.07) / W, (ybot + 0.11) / H,
                       (cw - 0.14) / W, (ch - 0.76) / H])
    visual_decision(av)

    # 评价网格（不是流程的一步，深灰带以示区别）：三个计数 + 两行方法
    box(ax, x1, ybot, cw, ch, face="#FAFBFC")
    band(ax, x1, ybot + ch - 0.40, cw, 0.40, "eval")
    box(ax, x1, ybot, cw, ch, face="none", lw=0.9, z=3)
    txt(ax, x1 + 0.12, ybot + ch - 0.265, "Evaluation", size=9.2,
        color="white", weight="bold")
    txt(ax, x1 + 0.12, ybot + ch - 0.60, "How the workflow is tested",
        size=7.0, color=INK)

    stats = [(0.12, "21", "conditions", P.C_NAVY),
             (0.68, "8", "comparators", P.C_TEAL),
             (1.22, "640", "designs", P.C_VERM)]
    for dx, num, lab, col in stats:
        txt(ax, x1 + dx, ybot + 0.95, num, size=13.0, color=col,
            weight="bold")
        txt(ax, x1 + dx + 0.01, ybot + 0.82, lab, size=6.4, color=INK)
    ax.plot([x1 + 0.12, x1 + cw - 0.12], [ybot + 0.70, ybot + 0.70],
            color="#D5D9DE", lw=0.8, zorder=3)
    txt(ax, x1 + 0.12, ybot + 0.54, "Dataset-level permutation and bootstrap",
        size=6.4, color=INK)
    txt(ax, x1 + 0.12, ybot + 0.40, "Holm correction across comparators",
        size=6.4, color=INK)

    fig.savefig(os.path.join(FIG, "fig1_scheme.png"), dpi=600,
                facecolor="white")
    fig.savefig(os.path.join(FIG, "fig1_scheme.pdf"), facecolor="white")
    plt.close(fig)
    print("saved fig1_scheme  c_hat=%.2f  [%.2f, %.2f]  y=%.1f"
          % (ex["c_hat"], ex["lo"], ex["hi"], ex["y"]))


if __name__ == "__main__":
    main()
