"""exp20 —— 容量匹配对照（P1-5）：CSU-PL 的优势是否只是"模型更简单"？

背景
----
主表里的 PLS 由训练集内 GroupKFold 选成分数（常选到 8–20 个），远多于 CSU-PL
的有效自由度（浓度 1 + 逐谱尺度 1 + 常数基线 1，谱方向 2）。若把 PLS 压到同
等容量后它仍然在门内落后，才说明差别来自约束形式而不是参数量。

做法
----
门内数据集（D1、D2、D7d）× 3 协议：
  * PLS 固定成分数 k = 2,3,4,5,6（不做 CV 选择）；
  * Ridge 用**有效自由度** trace(H) = 3,4,5 反解 alpha（标准化后 SVD），
    dof(alpha) = sum s_i^2/(s_i^2+alpha)；
  * 参照：现有 PLS-CV（exp03/exp09 出货值）与 CSU-PL（同一长表）。
评价：每条件 RMSEP 与 比值 CSU/对照；门内 9 个条件的中位比值。描述性统计，
不跨条件做显著性检验（条件间相关，推断口径见正文 §4.7）。

输出
----
out/exp20_capacity.csv          每 条件 x 对照 一行
out/exp20_capacity_summary.csv  每 对照 的中位比值 / 胜场
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import baselines, data, design, forward as F, stats

SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D7d"]
PROTOCOLS = ["low", "high", "interp"]
K_LIST = [2, 3, 4, 5, 6]
DOF_LIST = [3, 4, 5]


def ridge_dof_alpha(s, dof_target):
    """在 [1e-8, 1e10] 上二分 log(alpha)，使 sum s^2/(s^2+alpha) = dof_target。"""
    s2 = s ** 2

    def dof(a):
        return float(np.sum(s2 / (s2 + a)))

    lo, hi = -8.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dof(10 ** mid) > dof_target:
            lo = mid
        else:
            hi = mid
    return float(10 ** (0.5 * (lo + hi)))


def fit_ridge_dof(Xtr, ytr, Xte, dof_target):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd
    Zt = (Xte - mu) / sd
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    alpha = ridge_dof_alpha(s, dof_target)
    coef = Vt.T @ ((s / (s ** 2 + alpha)) * (U.T @ (ytr - ytr.mean())))
    achieved = float(np.sum(s ** 2 / (s ** 2 + alpha)))
    return Zt @ coef + ytr.mean(), alpha, achieved


def load_existing():
    """从 exp03 长表取 CSU-PL 与 PLS-CV 的逐谱预测（D7 已在 exp09 中并入）。"""
    p = os.path.join(OUT, "exp03_predictions.csv")
    df = pd.read_csv(p)
    df = df[df.method.isin(["CSU-PL", "PLS"]) & df.dataset.isin(DATASETS)
            & df.protocol.isin(PROTOCOLS)].copy()
    return df


def rmse_by_condition(df, method, seed=None):
    g = df[df.method == method]
    if seed is not None:
        g = g[g.seed == seed]
    out = {}
    for (nm, prot), gg in g.groupby(["dataset", "protocol"]):
        out[(nm, prot)] = stats.metrics(gg.y.values, gg.pred.values)
    return out


def main():
    ref = load_existing()
    csu = rmse_by_condition(ref, "CSU-PL")
    pls_cv = rmse_by_condition(ref, "PLS", seed=0)

    rows = []
    for (nm, prot), m in csu.items():
        rows.append(dict(dataset=nm, protocol=prot, method="CSU-PL", param=np.nan,
                         dof_effective=np.nan, n=m["n"], rmse=m["rmse"],
                         slope=m["slope"], bias=m["bias"]))
    for (nm, prot), m in pls_cv.items():
        rows.append(dict(dataset=nm, protocol=prot, method="PLS-CV", param=np.nan,
                         dof_effective=np.nan, n=m["n"], rmse=m["rmse"],
                         slope=m["slope"], bias=m["bias"]))

    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            Xtr, ytr, Xte, yte = Xv[tr], y[tr], Xv[te], y[te]
            for k in K_LIST:
                mdl = PLSRegression(n_components=k, scale=False)
                mdl.fit(Xtr, ytr)
                p = mdl.predict(Xte).ravel()
                m = stats.metrics(yte, p)
                rows.append(dict(dataset=nm, protocol=prot, method=f"PLS-{k}", param=k,
                                 dof_effective=float(k), n=m["n"], rmse=m["rmse"],
                                 slope=m["slope"], bias=m["bias"]))
            for dof in DOF_LIST:
                p, alpha, achieved = fit_ridge_dof(Xtr, ytr, Xte, dof)
                m = stats.metrics(yte, p)
                rows.append(dict(dataset=nm, protocol=prot, method=f"Ridge-dof{dof}",
                                 param=dof, dof_effective=achieved, n=m["n"],
                                 rmse=m["rmse"], slope=m["slope"], bias=m["bias"]))
            print(f"{nm} {prot}: done", flush=True)

    df = pd.DataFrame(rows)
    base = (df[df.method == "CSU-PL"][["dataset", "protocol", "rmse"]]
            .rename(columns={"rmse": "rmse_csu"}))
    df = df.merge(base, on=["dataset", "protocol"], how="left")
    df["ratio"] = df.rmse / df.rmse_csu
    df["csu_better"] = df.ratio > 1.0
    df.to_csv(os.path.join(OUT, "exp20_capacity.csv"), index=False)

    summ = []
    for meth, g in df[df.method != "CSU-PL"].groupby("method"):
        per_ds = g.groupby("dataset").ratio.median()
        summ.append(dict(method=meth, n_conditions=len(g),
                         median_ratio=float(g.ratio.median()),
                         n_csu_better=int((g.ratio > 1).sum()),
                         worst_ratio=float(g.ratio.max()),
                         best_ratio=float(g.ratio.min()),
                         median_D1=float(per_ds.get("D1", np.nan)),
                         median_D2=float(per_ds.get("D2", np.nan)),
                         median_D7d=float(per_ds.get("D7d", np.nan))))
    s = pd.DataFrame(summ)
    order = (["PLS-CV"] + [f"PLS-{k}" for k in K_LIST] +
             [f"Ridge-dof{d}" for d in DOF_LIST])
    s["o"] = s.method.map({m: i for i, m in enumerate(order)})
    s = s.sort_values("o").drop(columns="o")
    s.to_csv(os.path.join(OUT, "exp20_capacity_summary.csv"), index=False)
    print("\n" + s.round(3).to_string(index=False))

    # 参照锚点：CV 选择的 PLS 成分数与 CV ridge 的有效自由度
    anchors = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            pls = baselines.fit_pls(Xv[tr], y[tr], groups[tr])
            rid = baselines.fit_ridge(Xv[tr], y[tr], groups[tr])
            alpha = float(rid.named_steps["ridge"].alpha)
            Z = rid.named_steps["sc"].transform(Xv[tr])
            sv = np.linalg.svd(Z, compute_uv=False)
            dof = float(np.sum(sv ** 2 / (sv ** 2 + alpha)))
            anchors.append(dict(dataset=nm, protocol=prot,
                                pls_components=int(pls.n_components),
                                ridge_alpha=alpha, ridge_dof=dof))
            print(f"anchor {nm} {prot}: PLS k={pls.n_components} ridge alpha={alpha:.3g} "
                  f"dof={dof:.1f}", flush=True)
    pd.DataFrame(anchors).to_csv(os.path.join(OUT, "exp20_capacity_dof.csv"), index=False)


if __name__ == "__main__":
    main()
