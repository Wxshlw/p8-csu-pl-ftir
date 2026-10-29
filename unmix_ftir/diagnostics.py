"""unmix_ftir.diagnostics —— 前向模型是否成立的"训练集自检"。

这些量**只用训练数据 + 已知浓度**，不需要测试标签，因此可以作为
"要不要相信外推"的预警指标：

1. curvature γ     —— 水平均值沿浓度方向的响应是否线性（前向模型的核心假设）
2. endmember CNR   —— 端元对比 Q(g_a-g_m) 相对噪声的强度
3. baseline leak   —— 端元对比中有多大比例落在基线子空间（不可辨识部分）
4. brand coherence —— 多基质数据中各品牌的方向是否一致（D3 的失效诊断）
5. label resolvability —— 相邻浓度水平的谱差异是否高于重复测量噪声（D7 审计）
"""
from __future__ import annotations

import numpy as np

from . import forward as F


def level_means(y, X):
    y = np.asarray(y, float)
    levels = np.unique(np.round(y, 8))
    means = np.array([X[np.abs(y - v) < 1e-8].mean(0) for v in levels])
    sds = np.array([X[np.abs(y - v) < 1e-8].std(0, ddof=1) for v in levels])
    return levels, means, sds


def curvature_diagnostic(y, X):
    """水平均值对浓度的线性度。γ = 二次项在训练区间末端的相对偏离。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    levels, means, sds = level_means(y, X)
    if len(levels) < 3:
        return dict(gamma=np.nan, quad_gain=np.nan, p_quad=np.nan,
                    slope_first=np.nan, slope_last=np.nan, n_levels=len(levels))
    c = (levels - levels.mean()) / max(float(np.ptp(levels)), 1e-12)
    M = means - means.mean(0)
    d = (c[:, None] * M).sum(0) / max(float(c @ c), 1e-12)
    t = M @ d / max(float(d @ d), 1e-12)
    # 线性 vs 二次
    A1 = np.column_stack([np.ones_like(c), c])
    A2 = np.column_stack([A1, c ** 2])
    r1 = t - A1 @ np.linalg.lstsq(A1, t, rcond=None)[0]
    r2 = t - A2 @ np.linalg.lstsq(A2, t, rcond=None)[0]
    ss1, ss2 = float(r1 @ r1), float(r2 @ r2)
    n = len(c)
    # 二次项 F 检验（1 个额外参数）
    if ss2 <= 1e-18:
        p = 0.0
    else:
        Fv = max((ss1 - ss2) / (ss2 / max(n - 3, 1)), 0.0)
        from scipy.stats import f as fdist
        p = float(1 - fdist.cdf(Fv, 1, max(n - 3, 1)))
    b1 = float(np.polyfit(c, t, 1)[0])
    q = float(np.polyfit(c, t, 2)[0])
    gamma = float(q / b1) if abs(b1) > 1e-12 else np.nan
    # 相邻水平增量（按单位浓度归一）的首末比
    dt = np.diff(t) / np.diff(levels)
    return dict(gamma=gamma, quad_gain=float(1 - ss2 / max(ss1, 1e-18)), p_quad=p,
                slope_first=float(dt[0]), slope_last=float(dt[-1]),
                slope_ratio=float(dt[-1] / dt[0]) if abs(dt[0]) > 1e-12 else np.nan,
                n_levels=int(len(levels)), contrast=float(np.linalg.norm(d)))


def endmember_audit(mdl, X, y):
    """端元物理性：对比强度 / 噪声、落在基线子空间的比例、端元相似度。"""
    X = np.asarray(X, float)
    psi = mdl.psi
    Qb = F.projector_out(psi) if psi.size else np.eye(X.shape[1])
    gm = mdl.g[:-1].mean(0)          # 多品牌时取基质端元均值
    d = mdl.g[-1] - gm
    Qd = F.project_out(d[None, :], Qb)[0]
    Qgm = F.project_out(gm[None, :], Qb)[0]
    # 关键量：Qd 中与 Qgm 正交的分量（profile 信息量 ∝ ||Qd_perp||²）
    qgm2 = float(Qgm @ Qgm)
    Qd_perp = Qd - (float(Qd @ Qgm) / qgm2) * Qgm if qgm2 > 1e-18 else Qd.copy()
    # 噪声：一阶差分的高频残差（稳健）
    dx = np.diff(X, axis=1)
    sigma = 1.4826 * np.median(np.abs(dx - np.median(dx, axis=1, keepdims=True))) / np.sqrt(2)
    N = X.shape[1]
    return dict(contrast_norm=float(np.linalg.norm(d)),
                qd_norm=float(np.linalg.norm(Qd)),
                qd_perp_norm=float(np.linalg.norm(Qd_perp)),
                cos_qd_qgm=float(Qd @ Qgm / (np.linalg.norm(Qd) * np.linalg.norm(Qgm) + 1e-18)),
                baseline_leak=float(1 - (Qd @ Qd) / max(d @ d, 1e-18)),
                cnr=float(np.linalg.norm(Qd) / (sigma * np.sqrt(N) + 1e-18)),
                cnr_partial=float(np.linalg.norm(Qd_perp) / (sigma * np.sqrt(N) + 1e-18)),
                sigma=float(sigma),
                cos_gm_ga=float(gm @ mdl.g[-1] /
                                (np.linalg.norm(gm) * np.linalg.norm(mdl.g[-1]) + 1e-18)),
                nonneg_frac=float((mdl.g >= 0).mean()))


def extrapolation_bias_forecast(y, X, c_star):
    """只用训练数据预测外推偏差：二次 vs 线性响应在 c* 处的差（浓度单位）。

    返回负值 = 线性外推会低估；正值 = 高估。假设谱响应对 c 的二次项在训练区间
    可估，并在 c* 处延续（这是"预警"而非校正）。
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    levels, means, _ = level_means(y, X)
    if len(levels) < 4:
        return np.nan
    c = levels / 100.0
    M = means - means.mean(0)
    cc = c - c.mean()
    d = (cc[:, None] * M).sum(0) / max(float(cc @ cc), 1e-18)
    t = M @ d / max(float(d @ d), 1e-18)          # 单位：浓度
    A1 = np.column_stack([np.ones_like(cc), cc])
    b1 = np.linalg.lstsq(A1, t, rcond=None)[0]
    A2 = np.column_stack([A1, cc ** 2])
    b2 = np.linalg.lstsq(A2, t, rcond=None)[0]
    x = np.array([1.0, c_star / 100.0 - c.mean(), (c_star / 100.0 - c.mean()) ** 2])
    return float((x @ b2 - x[:2] @ b1) * 100.0)


def brand_consistency(y, X, brand):
    """逐品牌方向一致性：谱响应是否与品牌无关（D3 失效诊断）。"""
    y = np.asarray(y, float)
    brand = np.asarray(brand, dtype=object)
    dirs, r2s, slopes = {}, {}, {}
    for b in sorted(set(brand)):
        m = brand == b
        levels, means, _ = level_means(y[m], X[m])
        if len(levels) < 3:
            continue
        c = (levels - levels.mean()) / max(float(np.ptp(levels)), 1e-12)
        M = means - means.mean(0)
        d = (c[:, None] * M).sum(0) / max(float(c @ c), 1e-12)
        t = M @ d
        A = np.column_stack([np.ones_like(c), c])
        th = np.linalg.lstsq(A, t, rcond=None)[0]
        r = t - A @ th
        r2s[b] = float(1 - (r @ r) / max(float(((t - t.mean()) ** 2).sum()), 1e-18))
        slopes[b] = float(th[1])
        dirs[b] = d / (np.linalg.norm(d) + 1e-18)
    keys = sorted(dirs)
    cos = {f"{a}|{b}": float(dirs[a] @ dirs[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    return dict(r2=r2s, slope=slopes,
                cos_min=float(min(cos.values())) if cos else np.nan,
                cos_max=float(max(cos.values())) if cos else np.nan,
                cos=cos)


def label_resolvability(y, X, groups=None, max_levels=15):
    """相邻浓度水平的谱差异 vs 重复测量噪声。

    先对每个通道做单因素方差分析取 F 的最大值（log 尺度），再报告
    "相邻水平均值差 / 水平内标准差" 的中位数与最小值。
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    levels, means, sds = level_means(y, X)
    if len(levels) < 2:
        return dict(adjacent_snr_min=np.nan, adjacent_snr_med=np.nan, f_max=np.nan)
    snr = []
    for i in range(len(levels) - 1):
        diff = np.linalg.norm(means[i + 1] - means[i])
        noise = float(np.sqrt(np.mean(sds[i] ** 2 + sds[i + 1] ** 2)))
        snr.append(diff / (noise + 1e-18))
    # 单因素 F（逐通道，取 95 分位）
    gm = X.mean(0)
    ssb = np.zeros(X.shape[1])
    ssw = np.zeros(X.shape[1])
    for v in levels:
        m = np.abs(y - v) < 1e-8
        n_l = m.sum()
        if n_l < 2:
            continue
        dev = X[m] - X[m].mean(0)
        ssw += (dev ** 2).sum(0)
        ssb += n_l * (X[m].mean(0) - gm) ** 2
    dof_b, dof_w = len(levels) - 1, len(y) - len(levels)
    Fv = (ssb / max(dof_b, 1)) / (ssw / max(dof_w, 1) + 1e-18)
    return dict(adjacent_snr_min=float(np.min(snr)),
                adjacent_snr_med=float(np.median(snr)),
                f_max=float(np.max(Fv)), f_p95=float(np.percentile(Fv, 95)),
                n_levels=int(len(levels)))
