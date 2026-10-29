"""analyze04c —— 设计诊断的外部复现汇总（D1/D2/D3/D8）。

把 exp04（D2，247 个设计）和 exp04c（D1 120、D3 247、D8 26）合并，回答两个问题：

1. 杠杆/κ 与 RMSEP 的秩相关是否跨数据集稳定？——不稳定。
2. 杠杆能否作为"病态设计"的阈值筛查？——h>1 的设计在四个数据集上误差
   中位数均更高（1.14–1.65 倍），而 κ 对同一批设计没有筛查能力
   （AUC 0.00–0.75，且 D1/D3 反向）。

输出
----
tables/table8_design_replication.csv
out/exp04c_tail.json
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tables")
OUT = os.path.join(ROOT, "out")


def main():
    rows = []
    d2 = pd.read_csv(os.path.join(OUT, "exp04_design_sweep.csv"))
    d2 = d2[d2.source == "real"].copy()
    d2["dataset"] = "D2"
    d2 = d2.rename(columns={"leverage50": "leverage_test"})
    cols = ["dataset", "k", "levels", "has_zero", "n_train", "kappa", "leverage_test", "rmse"]
    d = pd.concat([pd.read_csv(os.path.join(OUT, "exp04c_design_other.csv"))[cols],
                   d2[cols]], ignore_index=True)

    tail = {}
    for nm, g in d.groupby("dataset"):
        hi = g[g.leverage_test > 1.0]
        lo = g[g.leverage_test <= 1.0]
        y = (g.rmse > 2.0 * g.rmse.median()).astype(int)
        auc = {}
        for x in ["leverage_test", "kappa"]:
            auc[x] = (float(stats.mannwhitneyu(g[x][y == 1], g[x][y == 0]).statistic /
                            (y.sum() * (len(y) - y.sum()))) if y.sum() else np.nan)
        rho_l = stats.spearmanr(g.leverage_test, g.rmse)
        rho_k = stats.spearmanr(g.kappa, g.rmse)
        rec = dict(dataset=nm, n_designs=len(g), n_high_leverage=len(hi),
                   median_rmse_high=float(hi.rmse.median()) if len(hi) else np.nan,
                   median_rmse_rest=float(lo.rmse.median()),
                   ratio_high_rest=float(hi.rmse.median() / lo.rmse.median()) if len(hi) else np.nan,
                   max_rmse=float(g.rmse.max()),
                   n_twofold= int(y.sum()),
                   auc_leverage=auc["leverage_test"], auc_kappa=auc["kappa"],
                   spearman_leverage=float(rho_l.statistic), p_leverage=float(rho_l.pvalue),
                   spearman_kappa=float(rho_k.statistic), p_kappa=float(rho_k.pvalue))
        tail[nm] = rec
        rows.append(rec)

    t = pd.DataFrame(rows).sort_values("dataset")
    t.to_csv(os.path.join(TAB, "table8_design_replication.csv"), index=False)
    with io.open(os.path.join(OUT, "exp04c_tail.json"), "w", encoding="utf-8") as f:
        json.dump(tail, f, ensure_ascii=False, indent=1)
    print(t.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
