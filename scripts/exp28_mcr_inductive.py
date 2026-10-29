"""exp28 —— 归纳式 MCR-ALS 对照（回应"转导式比较不公平"）。

exp03 的 MCR-ALS 把测试谱并入分解（转导、X_extra=Xte），对比较器最有利。
这里给同一算法的归纳版本：只在校准谱上分解，测试谱通过固定因子投影
（NMF.transform）。其余设置、划分、聚类口径与 exp03/exp25 完全一致。

输出
  out/exp28_mcr_inductive.csv  每条件：归纳式 MCR-ALS 的 RMSEP、CSU-PL/比较器比值、配对 bootstrap p
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import baselines, data, design, forward as F, stats

OUT = os.path.join(ROOT, "out")
PRED = os.path.join(OUT, "exp03_predictions.csv")
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
PROTOCOLS = ["low", "interp", "high"]
SG = F.SG(window=15, poly=3, deriv=1)


def csu_preds():
    d = pd.read_csv(PRED)
    d = d[d.method == "CSU-PL"].copy()
    d["ds"] = d.dataset.map(LABEL)
    d["spec"] = d.groupby(["dataset", "protocol", "method", "seed"]).cumcount()
    return d.groupby(["ds", "protocol", "group", "y", "spec"], as_index=False).agg(
        pred=("pred", "mean"))


def main():
    csu = csu_preds()
    rows = []
    for nm in ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]:
        ds = LABEL[nm]
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            if te.sum() == 0 or tr.sum() == 0:
                continue
            t0 = time.time()
            q_tra = baselines.fit_predict("MCR-ALS", Xv[tr], y[tr], groups[tr], Xv[te], wvn, seed=0)
            m = baselines.fit_mcr(Xv[tr], y[tr], groups[tr], wvn, seed=0, X_extra=None)
            q_ind = baselines.pred_mcr(m, Xv[te])
            yte = y[te]
            grp = np.asarray([str(g) for g in groups[te]])
            mr_tra = stats.metrics(yte, q_tra)["rmse"]
            mr_ind = stats.metrics(yte, q_ind)["rmse"]
            base = csu[(csu.ds == ds) & (csu.protocol == prot)]
            key = pd.DataFrame(dict(group=grp, y=yte, spec=np.arange(len(yte))))
            # 与 exp03 的对齐方式一致：按 group 顺序逐条比对
            order = np.argsort(grp, kind="stable")
            b = base.sort_values(["group"], kind="stable")
            if len(b) == len(order):
                yb = b["y"].to_numpy(float)
                pb = b["pred"].to_numpy(float)
                pa = q_ind[order]
                cl = grp[order]
                mr_csu = stats.metrics(yb, pb)["rmse"]
                dd = stats.paired_diff(yb, pb, np.asarray(pa, float), cl,
                                       n_boot=1000, metric="rmse", relative=True)
                rows.append(dict(dataset=ds, protocol=prot,
                                 rmse_csu=mr_csu, rmse_mcr_transduct=mr_tra,
                                 rmse_mcr_ind=mr_ind,
                                 ratio_transduct=mr_csu / mr_tra,
                                 ratio_inductive=mr_csu / mr_ind,
                                 p_inductive=dd["p"], n=len(yb), n_clusters=len(set(cl))))
            print(f"{ds} {prot:6s} transduct {mr_tra:6.2f}  inductive {mr_ind:6.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp28_mcr_inductive.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
