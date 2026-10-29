"""exp09 —— D7 外部验证（cinnamon, 2523 通道, 不同采集网格）。

三种掺杂物：D7w（核桃）、D7p（花生，标签不可分辨 -> 阴性对照）、D7d（木屑）。
每个样品只有 1 条谱（无重复、无品牌），因此是"单基体"模型，正好检验
品牌无关场景下 CSU-PL 的表现。基线取 PLS / Ridge / CLS-2pt / MCR-ALS。

输出与 exp03 相同的长表，并把 D7 行并入 out/exp03_predictions.csv（表 2、exp11 与
作图脚本都从那张长表读取）。这一步以前是手工做的，重跑 exp09 会让分析链停留在
过期谱上，因此现在由脚本执行：先删掉已有的 D7 行，再追加本次结果。
运行顺序：exp03 -> exp03b -> exp09 -> analyze03 -> make_tables -> exp11。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import baselines, data, design, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D7w", "D7p", "D7d"]
PROTOCOLS = ["low", "high", "interp"]
METHODS = ["PLS", "Ridge", "CLS-2pt", "MCR-ALS"]


def main():
    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            key = f"{nm}|{prot}"
            print(f"\n=== {key}: train {min(tr_lv):g}-{max(tr_lv):g} n={tr.sum()}, "
                  f"test {sorted(te_lv)} n={te.sum()} ===", flush=True)
            t0 = time.time()
            mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=None)
            p = mdl.predict_c(Xv[te]) * 100
            m = stats.metrics(y[te], p)
            for i, idx in enumerate(np.where(te)[0]):
                rows.append(dict(dataset=nm, protocol=prot, method="CSU-PL", seed=0,
                                 group=str(groups[idx]), y=float(y[idx]), pred=float(p[i])))
            print(f"  CSU-PL rmse={m['rmse']:6.2f} slope={m['slope']:5.2f} "
                  f"bias={m['bias']:+6.2f} ({time.time()-t0:.0f}s)", flush=True)
            for meth in METHODS:
                t0 = time.time()
                q = baselines.fit_predict(meth, Xv[tr], y[tr], groups[tr], Xv[te], wvn, seed=0)
                met = stats.metrics(y[te], q)
                for i, idx in enumerate(np.where(te)[0]):
                    rows.append(dict(dataset=nm, protocol=prot, method=meth, seed=0,
                                     group=str(groups[idx]), y=float(y[idx]), pred=float(q[i])))
                print(f"  {meth:8s} rmse={met['rmse']:6.2f} slope={met['slope']:5.2f} "
                      f"bias={met['bias']:+6.2f} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp09_d7_predictions.csv"), index=False)
    summ = (df.groupby(["dataset", "protocol", "method"])
            .apply(lambda g: pd.Series(stats.metrics(g.y.values, g.pred.values)))
            .reset_index())
    summ.to_csv(os.path.join(OUT, "exp09_d7_summary.csv"), index=False)

    # 把 D7 行并入 exp03 的长表：表 2 / exp11 / 作图都只读那张表
    main_pred = os.path.join(OUT, "exp03_predictions.csv")
    if os.path.exists(main_pred):
        base = pd.read_csv(main_pred)
        kept = base[~base["dataset"].astype(str).str.startswith("D7")]
        merged = pd.concat([kept, df], ignore_index=True).sort_values(
            ["dataset", "protocol", "method", "seed", "group", "y"], kind="stable")
        merged.to_csv(main_pred, index=False)
        print(f"spliced {len(df)} D7 rows into {main_pred} "
              f"({len(kept)} rows kept, {len(merged)} total)")
    else:
        print(f"note: {main_pred} absent, D7 rows left in exp09_d7_predictions.csv only")

    print("\n" + summ[["dataset", "protocol", "method", "rmse", "slope", "bias", "r2"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
