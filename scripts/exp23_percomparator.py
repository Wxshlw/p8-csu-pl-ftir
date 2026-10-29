"""exp23 —— 审计量对每一个比较器的外推误差，逐方法计算（SI Table S22）。

正文的核心统计量是 CNR 与"每条件最佳比较器"误差的数据集层秩相关（Table S5）。
本脚本把同一统计量对**每一个**比较器分别计算，用来回答"审计是否只对本文估计器
成立"。口径与 Table 2/Table 5 完全一致：

* 条件层 RMSEP = 各 seed 的逐谱 RMSE 的均值（Table 2 的 RMSEP）；
* 数据集层误差 = 三个协议的中位数（Table 5 的 dataset 行）；
* 相关 = 数据集层 Spearman，p 为精确置换（7! 或 4!，取决于该方法的覆盖范围）；
* 另外记录留一数据集后的系数范围，以及 |gamma| 对 |1-slope| 的相关。

输出 out/exp23_percomparator.csv，同时把每条件最佳比较器的一行作为 oracle 参照。
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
TAB = os.path.join(ROOT, "tables")

METHODS = ["CSU-PL", "CLS-2pt", "MCR-ALS", "PLS", "Ridge", "SVR", "RF", "HGB", "MLP"]
PROT = {"high-end": "high", "interp.": "interp", "low-end": "low"}


def exact_perm(x, y) -> tuple[float, float]:
    """数据集层 Spearman 与精确双侧置换 p（全部排列）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    cnt = 0
    for p in itertools.permutations(range(n)):
        if abs(stats.spearmanr(x[list(p)], y).statistic) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / math.factorial(n)


def condition_rmse(pred: pd.DataFrame) -> pd.DataFrame:
    """每方法/数据集/协议：各 seed 的 RMSE 的均值；同时给出斜率与偏差。"""
    rows = []
    for (meth, ds, pr), s in pred.groupby(["method", "dataset", "protocol"]):
        r, sl, bi = [], [], []
        for _, ss in s.groupby("seed"):
            e = ss.pred.to_numpy() - ss.y.to_numpy()
            r.append(np.sqrt(np.mean(e ** 2)))
            sl.append(np.polyfit(ss.y.to_numpy(), ss.pred.to_numpy(), 1)[0])
            bi.append(np.mean(e))
        rows.append(dict(method=meth, dataset=ds, protocol=pr,
                         rmse=float(np.mean(r)), slope=float(np.mean(sl)),
                         bias=float(np.mean(bi))))
    return pd.DataFrame(rows)


def main() -> None:
    pred = pd.read_csv(os.path.join(OUT, "exp03_predictions.csv"))
    t1 = pd.read_csv(os.path.join(TAB, "table1_datasets.csv")).rename(columns={"Dataset": "dataset"})
    t2 = pd.read_csv(os.path.join(TAB, "table2_main.csv"))
    audit = t1.set_index("dataset")[["CNR", "Gamma"]].astype(float)

    cond = condition_rmse(pred)

    # 对账 1：CSU-PL 的 21 个条件必须与 Table 2 的 RMSEP 一致
    t2["protocol"] = t2.Protocol.map(PROT)
    chk = cond[cond.method == "CSU-PL"].merge(
        t2[["Dataset", "protocol", "RMSEP", "RMSEP_best"]],
        left_on=["dataset", "protocol"], right_on=["Dataset", "protocol"])
    assert float(np.abs(chk.rmse - chk.RMSEP).max()) < 1e-9, "CSU-PL RMSEP 与 Table 2 不一致"
    # 对账 2：每条件最佳比较器必须与 Table 2 的 RMSEP_best 一致
    best = cond[cond.method != "CSU-PL"].groupby(["dataset", "protocol"]).rmse.min()
    chk["best"] = [best[(a, b)] for a, b in zip(chk.dataset, chk.protocol)]
    assert float(np.abs(chk.best - chk.RMSEP_best).max()) < 1e-3, "最佳比较器与 Table 2 不一致"

    # 每条件最佳比较器作为 oracle 参照，聚合口径与 Table 5 相同
    win = cond[cond.method != "CSU-PL"].sort_values("rmse").groupby(
        ["dataset", "protocol"]).head(1)
    oracle = win.groupby("dataset").apply(
        lambda g: pd.Series(dict(rmse=g.rmse.median(), slope=g.slope.median(),
                                 bias=float(np.abs(g.bias).median()))),
        include_groups=False).reset_index()

    rows = []
    for meth in METHODS + ["Best comparator"]:
        if meth == "Best comparator":
            sub_ds = oracle.copy()
            sub_ds["method"] = meth
        else:
            sub_ds = cond[cond.method == meth].groupby("dataset").agg(
                rmse=("rmse", "median"), slope=("slope", "median"),
                bias=("bias", lambda v: float(np.abs(v).median()))).reset_index()
        cov = sorted(sub_ds.dataset)
        a = audit.loc[cov]
        y = sub_ds.set_index("dataset").reindex(cov)
        rho_cnr, p_cnr = exact_perm(a.CNR.values, y.rmse.values)
        rho_g, p_g = exact_perm(a.Gamma.abs().values, (1 - y.slope).abs().values)
        rho_b, p_b = exact_perm(a.CNR.values, y.bias.abs().values)
        loo = []
        for k in cov:
            keep = [i for i, c in enumerate(cov) if c != k]
            loo.append(float(stats.spearmanr(a.CNR.values[keep], y.rmse.values[keep]).statistic))
        cc = cond[cond.method == meth].merge(a.reset_index(), on="dataset") if meth != "Best comparator" else None
        rows.append(dict(
            method=meth, n_datasets=len(cov), datasets=", ".join(cov),
            rho_cnr=rho_cnr, p_cnr=p_cnr,
            loo_min=min(loo), loo_max=max(loo),
            loo_worst_dataset=cov[int(np.argmin(loo))],
            rho_gamma_slope=rho_g, p_gamma_slope=p_g,
            rho_cnr_absbias=rho_b, p_absbias=p_b,
            rho_cnr_cond=(float(stats.spearmanr(cc.CNR, cc.rmse).statistic)
                          if cc is not None else np.nan),
            median_rmsep=float(y.rmse.median())))
        print(f"{meth:15s} n={len(cov)} rho(CNR)={rho_cnr:+.3f} p={p_cnr:.4f} "
              f"LOO [{min(loo):+.2f},{max(loo):+.2f}] "
              f"rho(|g|,slope)={rho_g:+.3f} p={p_g:.4f}")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "exp23_percomparator.csv"), index=False)

    # 断言：全部方法同号（正文与 S22 的说明依赖这一点）
    assert (R.rho_cnr < 0).all(), "存在非负的 CNR 相关"
    assert (R.rho_gamma_slope > 0).all(), "存在非正的 gamma–斜率相关"
    # 断言：参照行复现 Table 5 的 -0.786 / 0.048
    o = R[R.method == "Best comparator"].iloc[0]
    assert abs(o.rho_cnr + 0.7857142857142859) < 1e-6 and abs(o.p_cnr - 0.04801587301587302) < 1e-6
    print(f"\nwrote {os.path.join(OUT, 'exp23_percomparator.csv')} ({len(R)} rows)")
    print(f"oracle 行复现 Table 5：rho={o.rho_cnr:+.3f}, p={o.p_cnr:.4f}")


if __name__ == "__main__":
    main()
