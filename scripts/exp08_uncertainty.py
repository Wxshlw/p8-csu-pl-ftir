"""exp08 —— profile-likelihood 不确定度的校准：噪声区间 vs 模型失配区间。

问题：ΔR(c) ≤ qσ² 给出的是**统计噪声**区间。ATR-FTIR 通道多、噪声极小，
外推误差却由模型失配（基体变异、非线性、基线）主导 → 原始 profile 区间严重
欠覆盖（实测 coverage 远低于名义值）。

做法：
1. 原始 profile 区间（σ² = σ̂²，σ̂ 为训练残差 MAD）；
2. 训练集内 5 折 **样品级** GroupKFold：每折用其余折拟合、对被留出折算 profile
   区间 → 得到 CV-RMSE 与 CV 中位区间半宽，其比值 f = CV-RMSE/SE_CV 即模型失配
   的膨胀因子；
3. 用 σ_eff² = f²σ̂² 重算测试区间，重新评估 coverage。

若校准后 coverage 接近名义值，说明"profile 形状 + 经验膨胀"可以作为外推区间；
若仍欠覆盖，则模型失配是系统性的（偏差），只能作为失败信号而非区间。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
Q68, Q95 = 1.0, 3.841
NG = 4001


def profile_interval(grid, curve, j, sigma2, q):
    thr = curve[j] + q * sigma2
    n = len(grid)
    left = j
    while left > 0 and curve[left - 1] <= thr:
        left -= 1
    right = j
    while right < n - 1 and curve[right + 1] <= thr:
        right += 1
    lo, hi = grid[left], grid[right]
    if left > 0:
        y0, y1 = curve[left - 1], curve[left]
        if y1 != y0:
            lo = grid[left - 1] + (thr - y0) * (grid[left] - grid[left - 1]) / (y1 - y0)
    if right < n - 1:
        y0, y1 = curve[right], curve[right + 1]
        if y1 != y0:
            hi = grid[right] + (thr - y0) * (grid[right + 1] - grid[right]) / (y1 - y0)
    return float(max(0.0, lo)), float(min(1.0, hi))


def predict_intervals(mdl, X, brand):
    out = mdl.predict(X, brand=brand, n_grid=NG, lo=0.0, hi=1.0)
    grid, curve, c_hat = out["grid"], out["curve"], out["c_hat"]
    s2 = float(mdl.sigma_hat) ** 2
    j = np.argmin(curve, axis=1)
    lo68, hi68, lo95, hi95 = [], [], [], []
    for i in range(len(c_hat)):
        a, b = profile_interval(grid, curve[i], j[i], s2, Q68)
        c, d = profile_interval(grid, curve[i], j[i], s2, Q95)
        lo68.append(a); hi68.append(b); lo95.append(c); hi95.append(d)
    return (np.asarray(c_hat), np.asarray(lo68), np.asarray(hi68),
            np.asarray(lo95), np.asarray(hi95))


def cv_inflation(Xv, y, wvn, br, groups):
    """训练集内 5 折样品级 CV，返回膨胀因子 f 与 CV-RMSE。"""
    gk = GroupKFold(n_splits=5)
    c_hat, se, c_true = [], [], []
    for tr_i, te_i in gk.split(Xv, y, groups=groups):
        if len(np.unique(br[tr_i])) < len(np.unique(br)):
            continue                      # 某折缺品牌则跳过（不平衡设计）
        m = unmix.fit(Xv[tr_i], y[tr_i], wvn, CFG, brand=br[tr_i])
        c, lo, hi, _, _ = predict_intervals(m, Xv[te_i], br[te_i])
        c_hat.append(c); se.append(0.5 * (hi - lo)); c_true.append(y[te_i] / 100.0)
    if not c_hat:
        return 1.0, np.nan
    c_hat = np.concatenate(c_hat); se = np.concatenate(se); c_true = np.concatenate(c_true)
    rmse = float(np.sqrt(np.mean((c_hat - c_true) ** 2)))
    med_se = float(np.median(se))
    f = max(1.0, rmse / med_se) if med_se > 0 else 1.0
    return f, rmse


def main():
    rows, summ = [], []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            if te.sum() == 0 or tr.sum() == 0:
                continue
            mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
            c, lo68, hi68, lo95, hi95 = predict_intervals(mdl, Xv[te], br[te])
            f, cv_rmse = cv_inflation(Xv[tr], y[tr], wvn, br[tr], groups[tr])
            ct = y[te] / 100.0
            # 膨胀区间（保持 c_hat，半宽 × f）
            mid = 0.5 * (lo68 + hi68)
            w68 = 0.5 * (hi68 - lo68) * f
            w95 = 0.5 * (hi95 - lo95) * f
            lo68f, hi68f = np.maximum(0, mid - w68), np.minimum(1, mid + w68)
            mid95 = 0.5 * (lo95 + hi95)
            lo95f, hi95f = np.maximum(0, mid95 - w95), np.minimum(1, mid95 + w95)
            df = pd.DataFrame(dict(
                dataset=nm, protocol=prot, group=[str(g) for g in groups[te]],
                y=y[te], c_hat=c, c_true=ct, f=f, cv_rmse=cv_rmse,
                lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95,
                hit68=((lo68 <= ct) & (ct <= hi68)).astype(int),
                hit95=((lo95 <= ct) & (ct <= hi95)).astype(int),
                hit68f=((lo68f <= ct) & (ct <= hi68f)).astype(int),
                hit95f=((lo95f <= ct) & (ct <= hi95f)).astype(int),
                width95=hi95 - lo95, width95f=hi95f - lo95f))
            rows.append(df)
            rmse = float(np.sqrt(np.mean((c - ct) ** 2)) * 100)
            summ.append(dict(dataset=nm, protocol=prot, n=len(df), f=f, cv_rmse=cv_rmse * 100,
                             rmse=rmse, cov68=df.hit68.mean(), cov95=df.hit95.mean(),
                             cov68f=df.hit68f.mean(), cov95f=df.hit95f.mean(),
                             width95=df.width95.median(), width95f=df.width95f.median()))
            print(f"{nm:3s} {prot:6s} f={f:5.2f} cvRMSE={cv_rmse*100:5.2f} "
                  f"testRMSE={rmse:5.2f} cov95 {df.hit95.mean():.2f}->{df.hit95f.mean():.2f} "
                  f"width95 {df.width95.median()*100:.2f}->{df.width95f.median()*100:.2f}",
                  flush=True)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "exp08_uncertainty.csv"),
                                              index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, "exp08_summary.csv"), index=False)
    print("\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
