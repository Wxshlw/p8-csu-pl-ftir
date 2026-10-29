"""exp04c —— 设计扫描的外部复现：D1 / D3 / D8。

exp04 只在 D2 上枚举了 247 个真实浓度设计。为使 C3（杠杆/κ 诊断）不是
单数据集结论，这里在 D1、D3、D8 上重复同一套"训练水平组合 -> 外推测试段"
的扫描，使用与 exp04 完全相同的 CSU 配置和导数预处理。

设计空间
--------
D1: pool 0..18 (step 3), test 21/24/27/30  -> 120 designs
D3: pool 0..35 (step 5), test 40/45/50     -> 247 designs
D8: pool 0/1/2/3/5,      test 7/9          -> 26 designs

输出
----
out/exp04c_design_other.csv   每个设计一行（含 κ、杠杆、RMSEP、斜率、偏置）
out/exp04c_summary.json       各数据集内 Spearman 相关与分层统计
"""
from __future__ import annotations

import io
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)

SPECS = {
    "D1": dict(pool=[0, 3, 6, 9, 12, 15, 18], test=[21, 24, 27, 30]),
    "D3": dict(pool=[0, 5, 10, 15, 20, 25, 30, 35], test=[40, 45, 50]),
    "D8": dict(pool=[0, 1, 2, 3, 5], test=[7, 9]),
}


def run_dataset(nm, pool, test):
    d = data.load(nm)
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    br = data.brands(nm, groups)
    Xv = SG.transform(X)
    te = np.isin(y, test)
    rows = []
    designs = []
    for k in range(2, len(pool) + 1):
        designs += list(itertools.combinations(pool, k))
    for di, combo in enumerate(designs):
        tr_lv = np.array(combo, float)
        tr = np.isin(y, tr_lv)
        if tr.sum() < 40:
            continue
        t0 = time.time()
        mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
        p = mdl.predict_c(Xv[te], brand=br[te]) * 100.0
        met = stats.metrics(y[te], p)
        rep = design.condition_report(y[tr], y[te])
        rows.append(dict(
            dataset=nm, index=di, k=len(combo),
            levels="|".join(f"{v:g}" for v in combo),
            has_zero=int(combo[0] == 0), cmin=float(min(combo)), cmax=float(max(combo)),
            span=float(max(combo) - min(combo)), n_train=int(tr.sum()),
            kappa=rep["kappa"], kappa_intercept=rep["kappa_intercept"],
            leverage_test=rep["leverage"][f"{max(test):g}"],
            max_leverage=rep["max_leverage"],
            rmse=met["rmse"], slope=met["slope"], bias=met["bias"], r2=met["r2"],
            train_rmse=float(np.sqrt(np.mean((mdl.predict_c(Xv[tr], brand=br[tr]) * 100 - y[tr]) ** 2))),
            bias_forecast=diagnostics.extrapolation_bias_forecast(y[tr], Xv[tr], float(max(test))),
            seconds=time.time() - t0))
        if di % 10 == 0:
            print(f"  {nm} {di}/{len(designs)} k={len(combo)} rmse={met['rmse']:.2f} "
                  f"kappa={rep['kappa']:.2f} h={rep['max_leverage']:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return rows


def main():
    rows = []
    for nm, sp in SPECS.items():
        print(f"\n=== {nm}: {len(sp['pool'])} pool levels, test {sp['test']} ===", flush=True)
        rows += run_dataset(nm, sp["pool"], sp["test"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp04c_design_other.csv"), index=False)

    summ = {}
    for nm, g in df.groupby("dataset"):
        rec = dict(n_designs=int(len(g)),
                   rmse_median=float(g.rmse.median()),
                   kappa_median=float(g.kappa.median()),
                   leverage_median=float(g.leverage_test.median()))
        for x in ["kappa", "leverage_test", "max_leverage", "span", "cmin", "has_zero", "n_train"]:
            r, p = sps.spearmanr(g[x], g.rmse)
            rec[f"rho_rmse_{x}"] = float(r)
            rec[f"p_rmse_{x}"] = float(p)
        # 分层：同一 k 内，杠杆最高的 1/4 设计 vs 其余
        q = g.leverage_test.quantile(0.75)
        hi = g[g.leverage_test >= q]
        lo = g[g.leverage_test < q]
        rec["rmse_top_leverage"] = float(hi.rmse.median())
        rec["rmse_rest_leverage"] = float(lo.rmse.median())
        rec["ratio_top_leverage"] = float(hi.rmse.median() / lo.rmse.median())
        summ[nm] = rec
    with io.open(os.path.join(OUT, "exp04c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print("\n" + json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    print("\n" + df.groupby("dataset")[["rmse", "kappa", "leverage_test"]].median().to_string(), flush=True)


if __name__ == "__main__":
    main()
