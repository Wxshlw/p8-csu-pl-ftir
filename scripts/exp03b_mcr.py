"""exp03b —— 用修正后的转导式 MCR-ALS 重算 exp03 的 MCR 行并重写汇总。

旧版 MCR 的校准（无截距 + 单纯形闭合）会退化，数值不可用；本脚本只替换
MCR-ALS 方法行，其余方法的结果保持不变。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")

from unmix_ftir import baselines, data, design, forward as F, stats

DATASETS = ["D1", "D2", "D3", "D8"]
PROTOCOLS = ["low", "high", "interp"]
SG = F.SG(window=15, poly=3, deriv=1)


def main():
    df = pd.read_csv(os.path.join(OUT, "exp03_predictions.csv"))
    df = df[df.method != "MCR-ALS"].copy()
    new = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        br = data.brands(nm, groups)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            if te.sum() == 0 or tr.sum() == 0:
                continue
            q = baselines.fit_predict("MCR-ALS", Xv[tr], y[tr], br[tr], Xv[te],
                                      wvn, seed=0)
            for i, idx in enumerate(np.where(te)[0]):
                new.append(dict(dataset=nm, protocol=prot, method="MCR-ALS", seed=0,
                                group=str(groups[idx]), y=float(y[idx]),
                                pred=float(q[i])))
            m = stats.metrics(y[te], q)
            print(f"{nm} {prot:6s} MCR rmse={m['rmse']:6.2f} slope={m['slope']:5.2f} "
                  f"bias={m['bias']:+6.2f}", flush=True)
    df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
    df.to_csv(os.path.join(OUT, "exp03_predictions.csv"), index=False)

    rows = []
    for (nm, prot, meth), g in df.groupby(["dataset", "protocol", "method"]):
        per_seed = []
        for sd, gs in g.groupby("seed"):
            gs = gs.sort_values("group", kind="stable")
            per_seed.append(stats.metrics(gs["y"].values, gs["pred"].values))
        arr = pd.DataFrame(per_seed)
        rows.append(dict(dataset=nm, protocol=prot, method=meth, n_seeds=len(arr),
                         rmse=arr["rmse"].mean(), rmse_sd=arr["rmse"].std(ddof=0),
                         slope=arr["slope"].mean(), slope_sd=arr["slope"].std(ddof=0),
                         bias=arr["bias"].mean(), r2=arr["r2"].mean(),
                         mae=arr["mae"].mean(), neg_rate=arr["neg_rate"].mean(),
                         rpd=arr["rpd"].mean()))
    summ = pd.DataFrame(rows).sort_values(["dataset", "protocol", "rmse"])
    summ.to_csv(os.path.join(OUT, "exp03_summary.csv"), index=False)


if __name__ == "__main__":
    main()
