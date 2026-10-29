"""analyze03 —— 汇总 exp03：指标、聚类 bootstrap CI、配对差与 Holm。

注意：D1/D2/D3 的每个物理样品（group）含多条重复扫描，**指标必须按谱计算**，
bootstrap 的重采样单位才是 group。旧版按 group 聚合会把重复谱压成一条，错误。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
from unmix_ftir import stats


def main():
    df = pd.read_csv(os.path.join(OUT, "exp03_predictions.csv"))
    # 每条方法/seed 的测试谱顺序一致；spec 为谱内序号
    df["spec"] = df.groupby(["dataset", "protocol", "method", "seed"]).cumcount()
    # 跨 seed 平均同一条测试谱的预测
    pred = (df.groupby(["dataset", "protocol", "method", "group", "y", "spec"],
                       as_index=False)
              .agg(pred=("pred", "mean"), n_seeds=("seed", "nunique")))
    rows = []
    for (nm, prot), g in pred.groupby(["dataset", "protocol"]):
        base = g[g.method == "CSU-PL"].set_index(["group", "spec"])
        recs = {}
        pvals, keys = [], []
        for meth in sorted(g.method.unique()):
            gm = g[g.method == meth].set_index(["group", "spec"])
            common = base.index.intersection(gm.index)
            y = gm.loc[common, "y"].values
            p = gm.loc[common, "pred"].values
            clusters = np.asarray(common.get_level_values("group"))
            m = stats.metrics(y, p)
            ci = stats.cluster_bootstrap(y, p, clusters, n_boot=1000, metric="rmse")
            rec = dict(dataset=nm, protocol=prot, method=meth, n=len(y),
                       n_clusters=len(set(clusters)),
                       rmse=m["rmse"], rmse_lo=ci["lo"], rmse_hi=ci["hi"],
                       slope=m["slope"], bias=m["bias"], r2=m["r2"], rpd=m["rpd"],
                       n_seeds=int(gm["n_seeds"].iloc[0]))
            if meth != "CSU-PL":
                d = stats.paired_diff(y, base.loc[common, "pred"].values, p, clusters,
                                      n_boot=1000, metric="rmse", relative=True)
                rec.update(diff_rel=d["point"], diff_lo=d["lo"], diff_hi=d["hi"],
                           p_raw=d["p"])
                pvals.append(d["p"]); keys.append(meth)
            recs[meth] = rec
        if pvals:
            for meth, pa in zip(keys, stats.holm(pvals)):
                recs[meth]["p_holm"] = float(pa)
        # 跨 seed 符号稳定性（CSU 相对每个基线）
        sub = df[(df.dataset == nm) & (df.protocol == prot)]
        b = sub[sub.method == "CSU-PL"].set_index(["group", "spec"])["pred"]
        for meth in sorted(sub.method.unique()):
            if meth == "CSU-PL":
                continue
            diffs = []
            for sd in sorted(sub[sub.method == meth].seed.unique()):
                q = (sub[(sub.method == meth) & (sub.seed == sd)]
                     .set_index(["group", "spec"])["pred"])
                idx = b.index.intersection(q.index)
                yy = sub[sub.method == "CSU-PL"].set_index(["group", "spec"]).loc[idx, "y"].values
                ra = np.sqrt(np.mean((b.loc[idx].values - yy) ** 2))
                rb = np.sqrt(np.mean((q.loc[idx].values - yy) ** 2))
                diffs.append((rb - ra) / rb if rb > 0 else np.nan)
            ss = stats.sign_stability(diffs)
            recs[meth]["sign_frac"] = ss["frac_positive"]
            recs[meth]["diff_seed_median"] = ss["median"]
        rows += list(recs.values())

    out = pd.DataFrame(rows).sort_values(["dataset", "protocol", "rmse"])
    out.to_csv(os.path.join(OUT, "exp03_table.csv"), index=False)

    # CSU vs 该条件下最优基线：表 2、图 2 与数据集层推断都读这张表
    vs = []
    for (nm, prot), g in out.groupby(["dataset", "protocol"]):
        csu = g[g.method == "CSU-PL"]
        base = g[g.method != "CSU-PL"]
        if csu.empty or base.empty:
            continue
        c, b = csu.iloc[0], base.loc[base.rmse.idxmin()]
        vs.append(dict(dataset=nm, protocol=prot, n=int(c["n"]), csu=float(c.rmse),
                       best=b.method, best_rmse=float(b.rmse),
                       rel=float(b.diff_rel) * 100.0, p_holm=float(b.p_holm),
                       sign=float(b.sign_frac), csu_lo=float(c.rmse_lo),
                       csu_hi=float(c.rmse_hi)))
    pd.DataFrame(vs).to_csv(os.path.join(OUT, "exp03_vs_best.csv"), index=False)

    # exp03_summary.csv：按 seed 计算指标再跨 seed 汇总（列口径同 exp03_extrapolation.py）
    srows = []
    for (nm, prot, meth), g in df.groupby(["dataset", "protocol", "method"]):
        per_seed = []
        for sd, gs in g.groupby("seed"):
            gs = gs.sort_values("group", kind="stable")
            per_seed.append(stats.metrics(gs["y"].values, gs["pred"].values))
        arr = pd.DataFrame(per_seed)
        srows.append(dict(dataset=nm, protocol=prot, method=meth, n_seeds=len(arr),
                          rmse=arr["rmse"].mean(), rmse_sd=arr["rmse"].std(ddof=0),
                          slope=arr["slope"].mean(), slope_sd=arr["slope"].std(ddof=0),
                          bias=arr["bias"].mean(), r2=arr["r2"].mean(),
                          mae=arr["mae"].mean(), neg_rate=arr["neg_rate"].mean(),
                          rpd=arr["rpd"].mean()))
    summ = pd.DataFrame(srows).sort_values(["dataset", "protocol", "rmse"])
    summ.to_csv(os.path.join(OUT, "exp03_summary.csv"), index=False)
    print(f"wrote exp03_summary.csv ({len(summ)} rows from {len(df)} predictions)")

    print(f"wrote exp03_table.csv ({len(out)} rows) and "
          f"exp03_vs_best.csv ({len(vs)} conditions)")

    cols = ["dataset", "protocol", "method", "n", "n_clusters", "rmse", "rmse_lo",
            "rmse_hi", "slope", "bias", "diff_rel", "diff_lo", "diff_hi",
            "p_holm", "sign_frac"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
