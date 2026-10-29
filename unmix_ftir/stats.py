"""unmix_ftir.stats —— 指标、聚类 bootstrap、配对比较与 Holm 校正。

统计口径（方案 §4.6 纪律）
--------------------------
* 划分：**固定的物理样品级划分**。sklearn GroupKFold 的组分配对样本顺序置换不变，
  所以禁止用 "N 次随机 repeat" 制造重复；本文所有不确定性用
  **聚类 bootstrap（重采样单位 = 物理样品）** 报告。
* 配对比较：同一测试谱上的 RMSEP 差，按物理样品整簇重采样
  （同一样品的所有扫描一起进出），得到差值的 95 % CI。
* 多重比较：同一数据集 / 协议内，所有基线 vs 本文方法构成一个 family，
  用 Holm step-down 校正。
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, slope=np.nan, slope_inv=np.nan,
                    r2=np.nan, rpd=np.nan, neg_rate=np.nan, max_abs=np.nan)
    e = yhat - y
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = float(np.sum(e ** 2))
    slope, slope_inv = np.nan, np.nan
    if np.std(yhat) > 1e-12 and np.std(y) > 1e-12:        # 单浓度测试集无斜率
        slope = float(np.polyfit(y, yhat, 1)[0])          # 预测 ~ 实测（b1；理想 = 1）
        slope_inv = float(np.polyfit(yhat, y, 1)[0])      # 实测 ~ 预测
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(n=int(len(y)), rmse=rmse, mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)), slope=slope, slope_inv=slope_inv,
                r2=float(1 - ssr / sst) if sst > 0 else np.nan,
                rpd=float(sd / rmse) if rmse > 0 else np.nan,
                neg_rate=float(np.mean(yhat < 0)),
                max_abs=float(np.max(np.abs(e))))


def by_level(y, yhat, tol=1e-6):
    """按浓度水平分层的指标。"""
    y = np.asarray(y, float)
    out = {}
    for v in sorted(set(np.round(y / max(tol, 1e-9)) * max(tol, 1e-9))):
        m = np.abs(y - v) < 1e-6
        out[f"{v:g}"] = metrics(y[m], np.asarray(yhat)[m])
    return out


# ---------------------------------------------------------------------------
# 聚类 bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap(y, yhat, groups, n_boot=2000, seed=0, metric="rmse", alpha=0.05):
    """按物理样品整簇重采样的 bootstrap CI（百分位法）。"""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[k] for k in pick])
        vals[b] = _one_metric(metric, y[idx], yhat[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=_one_metric(metric, y, yhat), lo=float(lo), hi=float(hi),
                n_boot=int(n_boot), n_clusters=int(len(uniq)), metric=metric)


def _one_metric(metric, y, yhat):
    if metric == "rmse":
        return float(np.sqrt(np.mean((yhat - y) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(yhat - y)))
    if metric == "bias":
        return float(np.mean(yhat - y))
    raise ValueError(metric)


def paired_diff(y, yhat_a, yhat_b, groups, n_boot=2000, seed=0, alpha=0.05,
                metric="rmse", relative=True):
    """a 相对 b 的改进：默认返回 (RMSE_b - RMSE_a)/RMSE_b（正 = a 更好）。

    正值的 95 % CI 下界 > 0 即 "a 在 5 % 水平上显著优于 b"（未做多重校正前）。
    """
    y = np.asarray(y, float)
    ya, yb = np.asarray(yhat_a, float), np.asarray(yhat_b, float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inv == k)[0] for k in range(len(uniq))]

    def stat(idx):
        ra = _one_metric(metric, y[idx], ya[idx])
        rb = _one_metric(metric, y[idx], yb[idx])
        if relative:
            return (rb - ra) / rb if rb > 0 else np.nan
        return rb - ra

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[k] for k in pick])
        vals[b] = stat(idx)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = stat(np.arange(len(y)))
    # 双侧 bootstrap p：差值分布跨 0 的比例（方向性，仅用于 Holm 排序）
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return dict(point=float(point), lo=float(lo), hi=float(hi), p=float(min(max(p, 0.0), 1.0)),
                n_boot=int(len(vals)), n_clusters=int(len(uniq)), metric=metric)


def holm(pvals):
    """Holm step-down；返回校正后 p（与输入同序）。"""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def sign_stability(values):
    """≥5 seeds 的符号稳定性：返回 (median, frac_same_sign, min, max)。"""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(median=np.nan, frac_positive=np.nan, min=np.nan, max=np.nan, n=0)
    med = float(np.median(v))
    frac = float(np.mean(np.sign(v) == np.sign(med))) if med != 0 else 1.0
    return dict(median=med, frac_positive=float(np.mean(v > 0)), min=float(v.min()),
                max=float(v.max()), n=int(len(v)), frac_same_sign=frac)
