"""exp18 —— LOQ / 可报告范围：校准噪声口径 vs 失配区间口径（P1-7）。

两个口径
--------
1) 校准曲线口径（噪声限制）：profile 似然在 1σ 处的半宽即 SE(ĉ)。按二元校准
   曲线的常规写法给出 LOD = 3.3 SE、LOQ = 10 SE。它只包含通道噪声，不含基体
   失配，因此是"模型完全正确时"的下限。
2) 区间宽度口径（失配限制）：把同一 SE 用训练集 5 折样品级 CV 的膨胀因子 f
   放大得到 1σ_eff = f SE，要求它在 LOQ 处不超过浓度的给定比例（常规 RSD
   20 % 用于筛查、10 % 用于定量）。满足该条件的最低浓度即该口径下的 LOQ。
   两个口径只差一个 f：「10σ/S」描述的是拟合区间内的噪声，不是外推区的
   可报告下限。

特征曲线由拟合端元的合成谱给出（s=1、无基线），并对 D1–D3 的五个品牌取中位；
真实校准谱上的 SE 作为同浓度处的一致性检查一并输出。

输出
----
out/exp18_loq.csv        每数据集一行
out/exp18_loq_curve.csv  特征曲线（c, SE, U95）
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

from unmix_ftir import data, forward as F, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
AUDIT = {"D1": "pass", "D2": "pass", "D3": "fail", "D7w": "fail",
         "D7p": "fail", "D7d": "pass", "D8": "fail"}
NG = 2001
Q68, Q95 = 1.0, 3.841
REL_TOL = 0.20          # 可报告范围的相对扩展不确定度上限
Z95 = 1.959964


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


def se_of(mdl, X, brand):
    """逐谱 1σ profile 半宽（单位：浓度分数）。"""
    out = mdl.predict(X, brand=brand, n_grid=NG, lo=0.0, hi=1.0)
    grid, curve = out["grid"], out["curve"]
    j = np.argmin(curve, axis=1)
    s2 = float(mdl.sigma_hat) ** 2
    se = np.empty(len(curve))
    for i in range(len(curve)):
        a, b = profile_interval(grid, curve[i], j[i], s2, Q68)
        se[i] = 0.5 * (b - a)
    return se, out["c_hat"]


def cv_inflation(Xv, y, wvn, br, groups):
    gk = GroupKFold(n_splits=5)
    se_all, err_all = [], []
    for tr_i, te_i in gk.split(Xv, y, groups=groups):
        if len(np.unique(br[tr_i])) < len(np.unique(br)):
            continue
        m = unmix.fit(Xv[tr_i], y[tr_i], wvn, CFG, brand=br[tr_i])
        se, c_hat = se_of(m, Xv[te_i], br[te_i])
        se_all.append(se)
        err_all.append(c_hat - y[te_i] / 100.0)
    if not se_all:
        return 1.0, np.nan
    se = np.concatenate(se_all)
    err = np.concatenate(err_all)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    med = float(np.median(se))
    return (max(1.0, rmse / med) if med > 0 else 1.0), rmse


def main():
    rows, curves = [], []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        Xv = SG.transform(X)
        mdl = unmix.fit(Xv, y, wvn, CFG, brand=br)
        brands = sorted(set(br.tolist()))
        f, cv_rmse = cv_inflation(Xv, y, wvn, br, groups)

        # 特征曲线：合成谱（s=1、无基线），品牌间取中位
        cgrid = np.arange(0.005, 0.951, 0.005)
        Xs, bs = [], []
        bl = list(mdl.brand_levels or [])
        for c in cgrid:
            for b in brands:
                k = bl.index(b) if b in bl else 0
                Xs.append((1 - c) * mdl.g[k] + c * mdl.g[-1])
                bs.append(b)
        Xs = np.asarray(Xs)
        bs = np.asarray(bs, dtype=object) if mdl.brand_levels else None
        se_syn, _ = se_of(mdl, Xs, bs)
        se_syn = se_syn.reshape(len(cgrid), len(brands))
        se_med = np.median(se_syn, axis=1)

        u95 = Z95 * f * se_med
        u1 = f * se_med          # CV 膨胀后的 1σ，与 10σ/S 同一口径

        def solve(frac):
            okk = u1 <= frac * cgrid
            return float(cgrid[np.argmax(okk)] * 100) if okk.any() else np.nan

        loq20, loq10 = solve(0.20), solve(0.10)

        # 校准噪声口径：非零校准水平处的 SE
        levels = np.unique(y)
        nz = levels[levels > 0]
        se_at_lv = [float(np.interp(v / 100.0, cgrid, se_med)) for v in nz]
        se_cal = float(np.median(se_at_lv)) * 100

        # 真实校准谱的一致性检查
        trn = np.isin(y, nz)
        se_real, _ = se_of(mdl, Xv[trn], br[trn])
        se_real_med = float(np.median(se_real)) * 100

        rows.append(dict(dataset=nm, audit=AUDIT[nm], n_levels=int(len(levels)),
                         c_max=float(levels.max()), sigma=float(mdl.sigma_hat),
                         se_cal_pp=se_cal, se_real_pp=se_real_med,
                         synth_over_real=se_cal / max(se_real_med, 1e-12),
                         lod_noise_pp=3.3 * se_cal, loq_noise_pp=10.0 * se_cal,
                         f=f, cv_rmse_pp=cv_rmse * 100,
                         u95_at_cmax_pp=float(np.interp(levels.max() / 100.0,
                                                        cgrid, u95)) * 100,
                         loq20_pp=loq20, loq10_pp=loq10,
                         loq20_in_range=bool(np.isfinite(loq20) and loq20 <= levels.max()),
                         loq10_in_range=bool(np.isfinite(loq10) and loq10 <= levels.max())))
        for c, s, u in zip(cgrid, se_med, u95):
            curves.append(dict(dataset=nm, c_pct=c * 100, se_pp=s * 100, u95_pp=u * 100))
        print(f"{nm:4s} {AUDIT[nm]:4s} sigma={mdl.sigma_hat:.4f} SEcal={se_cal:5.2f}pp "
              f"SEreal={se_real_med:5.2f}pp LOD={3.3*se_cal:5.2f} LOQn={10*se_cal:6.2f} "
              f"f={f:5.2f} LOQ20={loq20:6.2f} LOQ10={loq10:6.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp18_loq.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "exp18_loq_curve.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
