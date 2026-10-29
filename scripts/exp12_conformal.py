"""exp12 —— 与 conformal prediction 的正面对比。

目的：当前 profile 区间的失效（原始 95% coverage 中位数 7%，CV 膨胀后 83%）
很容易被审稿人归因于"区间方法没选对"。本实验用 split conformal / CV+ 这一
分布无关区间作为最强对照，检验"外推时任何无标签区间都会失效"这一命题。

做法（每个数据集 × 协议）：
1. 校准集内 5 折样品级 GroupKFold，收集 out-of-fold 预测 c_hat_cv 与真值；
2. 取 |残差| 的有限样本 95% 分位数 q（split conformal）；
3. 测试集区间 = c_hat_test ± q（截断到 [0,1]）；
4. 与原始 profile 区间、CV-RMSE 高斯区间（1.96·RMSE_CV）比较 coverage 与宽度。

输出 out/exp12_conformal.csv, out/exp12_summary.csv
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

from unmix_ftir import data, design, forward as F, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
ALPHA = 0.05


def conformal_q(abs_res, alpha=ALPHA):
    """有限样本 split-conformal 分位数。"""
    n = len(abs_res)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(abs_res))
    return float(np.sort(abs_res)[k - 1])


def main():
    rows = []
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
            c_hat = mdl.predict_c(Xv[te], brand=br[te])
            ct = y[te] / 100.0

            # 校准集内 out-of-fold 残差
            res = []
            gk = GroupKFold(n_splits=5)
            for a, b in gk.split(Xv[tr], y[tr], groups=groups[tr]):
                if len(np.unique(br[tr][a])) < len(np.unique(br[tr])):
                    continue
                m = unmix.fit(Xv[tr][a], y[tr][a], wvn, CFG, brand=br[tr][a])
                p = m.predict_c(Xv[tr][b], brand=br[tr][b])
                res.append(p - y[tr][b] / 100.0)
            if not res:
                continue
            res = np.concatenate(res)
            abs_r = np.abs(res)
            q = conformal_q(abs_r)
            rmse_cv = float(np.sqrt(np.mean(res ** 2)))

            lo = np.clip(c_hat - q, 0, 1)
            hi = np.clip(c_hat + q, 0, 1)
            lo_g = np.clip(c_hat - 1.96 * rmse_cv, 0, 1)
            hi_g = np.clip(c_hat + 1.96 * rmse_cv, 0, 1)

            rows.append(dict(
                dataset=nm, protocol=prot, n_cal=len(abs_r), n_test=len(ct),
                q95=q, rmse_cv=rmse_cv,
                cov_conf=float(np.mean((lo <= ct) & (ct <= hi))),
                width_conf=float(np.median(hi - lo)),
                cov_gauss=float(np.mean((lo_g <= ct) & (ct <= hi_g))),
                width_gauss=float(np.median(hi_g - lo_g)),
                rmse=float(np.sqrt(np.mean((c_hat - ct) ** 2))),
                bias=float(np.mean(c_hat - ct))))
            print(f"{nm:3s} {prot:6s} q95={q*100:6.2f}pp covC={rows[-1]['cov_conf']:.2f} "
                  f"covG={rows[-1]['cov_gauss']:.2f} wC={rows[-1]['width_conf']*100:6.2f} "
                  f"wG={rows[-1]['width_gauss']*100:6.2f} rmse={rows[-1]['rmse']*100:5.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp12_conformal.csv"), index=False)
    if df.empty:
        return
    s = df.groupby("dataset").agg(
        cov_conf=("cov_conf", "median"), cov_gauss=("cov_gauss", "median"),
        width_conf=("width_conf", "median"), width_gauss=("width_gauss", "median"),
        rmse=("rmse", "median")).reset_index()
    s.to_csv(os.path.join(OUT, "exp12_summary.csv"), index=False)
    print("\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
