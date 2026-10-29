"""出版级绘图样式（Elsevier 投稿版式，对标 Analytica Chimica Acta 画风）。

设计原则（2026-10-23 重制， ACA 绝对字号标准）：
  * Arial 无衬线（Elsevier 图主流）；**所有字号按"最终排版绝对 pt"设定**
    —— 各图脚本的 figsize 宽度 == LaTeX 排版宽度（scale=1.0），
    因此脚本内 fontsize 即读者看到的实际 pt 数，不再靠大画布缩放
  * ACA 实测参考：轴刻度 ~9-10pt、轴标 ~10pt、图例 ~8.5-9pt、
    面板标题 ~10.5pt 粗体、主数据线 ~2pt、误差棒 caps ~4pt
  * 轴线仅 left/bottom，1.1pt 深灰，刻度朝外
  * 网格仅 y 向，浅灰细实线，置底
  * Okabe-Ito 深色变体（色盲友好，印刷可辨）
  * savefig 600 dpi（Elsevier 线图要求 >=500 dpi）
  * 白底、无装饰、图例无框
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from matplotlib import font_manager

# ---- 色板（Okabe-Ito 深色变体，印刷友好）----
# 统一色系：蓝系渐变（主） + 邻近青（次） + 单一暖强调（赤陶） + 中性灰
C_NAVY = "#1B3A6B"    # 深蓝（主色锚点）
C_BLUE = "#2E6DA4"    # 中蓝
C_SKY = "#7FB2D9"     # 浅蓝
C_TEAL = "#2A7F86"    # 邻近青（第二系列）
C_GREN = "#4C9A8F"    # 软青绿（正向/通过标记）
C_GREEN = C_GREN      # 兼容别名
C_VERM = "#C4622D"    # 暖强调（赤陶，唯一暖色，用于对比/高亮）
C_ORAN = "#DE9A68"    # 浅赤陶
C_REDD = "#A34A2E"    # 深赤陶（负向标记）
C_PURP = "#5B7DB1"    # 蓝灰（池化/FC 等）
C_GREY = "#6E6E6E"
C_GRAY2 = "#B0B0B0"
C_INK = "#222222"
C_BLACK = "#000000"

PALETTE = [C_NAVY, C_BLUE, C_TEAL, C_SKY, C_VERM, C_PURP, C_GREY]
PALETTE_SOFT = ["#A9C2DE", "#E8B394", "#93C4C9", "#F0CDB4", "#8FA8CE", "#CFE3F0", "#C8C8C8"]

# 画布宽度（in）：正文排版宽 \textwidth≈5.64in；各图按 LaTeX 中的
# includegraphics 宽度设计（0.85/0.9/0.95/0.98 × 5.64 ≈ 4.79/5.08/5.36/5.53），
# 再留 ~3% 余量给 bbox=tight，保证 scale>=1.0、有效字号不低于设计值。
SINGLE_W = 5.08   # 0.90\textwidth
DOUBLE_W = 5.53   # 0.98\textwidth


def _register_arial():
    """注册 Arial（失败安全：Helvetica -> DejaVu Sans）。"""
    import glob
    for pat in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/Arial.ttf",
                "C:/Windows/Fonts/helvetica.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        for f in sorted(glob.glob(pat)):
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Helvetica", "Liberation Sans"):
        if cand in have:
            return [cand]
    return ["DejaVu Sans"]


def apply_pub_style():
    """应用出版级 rcParams（幂等）。字号/线宽均为 ACA 绝对标准。"""
    fams = _register_arial()
    if "DejaVu Sans" not in fams:
        fams = fams + ["DejaVu Sans"]   # 缺字形（∈/ℝ 等）逐字兜底
    rcParams.update({
        # 字体：正文 Arial；mathtext 同源（custom），缺字形自动落到 DejaVu
        "font.family": "sans-serif",
        "font.sans-serif": fams,
        "mathtext.fontset": "custom",
        "mathtext.rm": fams[0],
        "mathtext.it": fams[0] + ":italic",
        "mathtext.bf": fams[0] + ":bold",
        "mathtext.default": "it",
        # ---- ACA 绝对字号（figsize 宽度=排版宽，scale=1.0 时即读者实际 pt）----
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 5.0,
        "axes.labelsize": 10.0,
        "axes.labelpad": 3.5,
        # 轴线 / 刻度
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        # 网格
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#D6D6D6",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        # 图例
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.loc": "best",
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.4,
        # 线 / 标记（ACA：主线 ~2pt，标记 4.5-6pt 带白/深描边）
        "lines.linewidth": 2.0,
        # 统一默认配色循环（否则未显式着色的线会走 matplotlib tab10 彩虹色）
        "axes.prop_cycle": cycler(color=[C_NAVY, C_BLUE, C_TEAL, C_VERM,
                                         C_SKY, C_PURP, C_GREY]),
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.7,
        # 误差棒
        "errorbar.capsize": 4.0,
        # 输出
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, path, pad=0.03):
    """保存并关闭；pad 单位 in。"""
    fig.savefig(path, pad_inches=pad)
    plt.close(fig)
    return path


def panel_title(ax, txt, loc="left", pad=4.0):
    """面板标题：左上/居中，加粗小字（默认继承 axes.titlesize=10.5）。"""
    if loc == "left":
        ax.set_title(txt, loc="left", pad=pad)
    else:
        ax.set_title(txt, loc="center", pad=pad)


def fit_text(ax, txt, loc="upper right", fontsize=8.5, color=None, mono=False, **kw):
    """坐标框内统计注记（R^2/p 值等）。"""
    import matplotlib.transforms as mtrans
    x, y = (0.97, 0.96) if loc == "upper right" else (0.03, 0.04)
    ha, va = ("right", "top") if loc == "upper right" else ("left", "bottom")
    if loc == "upper left":
        x, y = (0.03, 0.96); ha, va = ("left", "top")
    return ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=fontsize, color=color or C_INK,
                   family="monospace" if mono else None,
                   linespacing=1.35, **kw)


def identity_line(ax, ls=(0, (4, 3)), color="#888888", lw=1.4, label="identity"):
    """45 度恒等参考线（按数据范围）。"""
    lo, hi = ax.get_xlim()
    m = min(lo, hi); M = max(lo, hi)
    ln = ax.plot([m, M], [m, M], ls=ls, color=color, lw=lw, label=label,
                 zorder=1)
    return ln


def band_arrow(ax, x0, x1, y, text, color=None, fs=8.0, lw=1.4):
    """频带标注：水平双头箭头 + 文字（DCT 三带图用）。"""
    import numpy as np
    color = color or C_VERM
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw))
    ax.text(0.5 * (x0 + x1), y, text, ha="center", va="bottom",
            fontsize=fs, color=color)
