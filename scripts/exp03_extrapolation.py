"""exp03 —— 外推主实验：P-out-low / P-out-high / P-interp × 数据集 × 基线族。

输出
----
out/exp03_predictions.csv   每方法/每 seed 的逐谱预测（含样品组）
out/exp03_summary.csv       指标汇总（跨 seed 均值）
out/exp03_splits.json       每个划分的设计诊断（κ、杠杆、曲率）
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import baselines, data, design, diagnostics, forward as F, stats, unmix

DATASETS = ["D1", "D2", "D3", "D8"]
PROTOCOLS = ["low", "high", "interp"]
SEEDS = {"MLP": [0, 1, 2, 3, 4], "RF": [0, 1, 2], "HGB": [0, 1, 2], "MCR-ALS": [0]}
METHODS = ["CSU-PL"] + baselines.ALL

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)


def seed_list(method):
    return SEEDS.get(method, [0])


def run():
    records = []
    splits = {}
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
            key = f"{nm}|{prot}"
            rep = design.condition_report(y[tr], y[te])
            rep.update(diagnostics.curvature_diagnostic(y[tr], Xv[tr]))
            # 训练集内部的品牌一致性（只对有品牌的 KAN 数据有意义）
            if nm in ("D1", "D2", "D3"):
                bc = diagnostics.brand_consistency(y[tr], Xv[tr], br[tr])
                rep["brand_r2"] = bc["r2"]
                rep["brand_cos_min"] = bc["cos_min"]
            splits[key] = rep
            print(f"\n=== {nm} {prot}: train {min(tr_lv):g}-{max(tr_lv):g} n={tr.sum()}, "
                  f"test {sorted(set(te_lv))} n={te.sum()} | kappa={rep['kappa']:.2f} "
                  f"gamma={rep['gamma']:.2f} ===", flush=True)

            # --- CSU-PL ---
            t0 = time.time()
            mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
            pred = mdl.predict_c(Xv[te], brand=br[te]) * 100.0
            for i, idx in enumerate(np.where(te)[0]):
                records.append(dict(dataset=nm, protocol=prot, method="CSU-PL", seed=0,
                                    group=str(groups[idx]), y=float(y[idx]), pred=float(pred[i])))
            m = stats.metrics(y[te], pred)
            print(f"  {'CSU-PL':9s} rmse={m['rmse']:6.2f} slope={m['slope']:5.2f} "
                  f"bias={m['bias']:+6.2f} ({time.time()-t0:.0f}s)", flush=True)
            mdl_full = mdl

            for meth in baselines.ALL:
                for sd in seed_list(meth):
                    t0 = time.time()
                    try:
                        q = baselines.fit_predict(meth, Xv[tr], y[tr], groups[tr], Xv[te], wvn, seed=sd)
                    except Exception as e:  # 记录而不是中断
                        print(f"  {meth} seed={sd} FAIL {type(e).__name__}: {e}", flush=True)
                        continue
                    for i, idx in enumerate(np.where(te)[0]):
                        records.append(dict(dataset=nm, protocol=prot, method=meth, seed=sd,
                                            group=str(groups[idx]), y=float(y[idx]),
                                            pred=float(q[i])))
                    m = stats.metrics(y[te], q)
                    print(f"  {meth:9s} s={sd} rmse={m['rmse']:6.2f} slope={m['slope']:5.2f} "
                          f"bias={m['bias']:+6.2f} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT, "exp03_predictions.csv"), index=False)

    # 汇总：先按 seed 计算，再跨 seed 汇总
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
    with io.open(os.path.join(OUT, "exp03_splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=1, default=float)
    print("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    run()
