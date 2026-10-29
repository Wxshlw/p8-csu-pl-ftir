"""exp16 —— 扫描级 vs 样品级划分泄漏对照（P0-2）。

背景
----
公开 FTIR 掺假定量文献里最常见的划分是**按谱随机划分**。当同一个物理样品
有多次扫描（技术重复）时，测试谱在训练集中存在"同一样品的兄弟谱"，
指标被系统性抬高。本文的协议是按**浓度水平 / 物理样品**切分，因此不存在这种泄漏；
但审稿人一定会问"你的绝对误差是不是被这种泄漏保护出来的"。

对照设计（只在有技术重复的数据集上做，D7/D8 每样品只有 1 条谱，无泄漏可言）
  D1 / D2 / D3：55 个物理样品 × 11 个水平，每样品 5–36 条扫描。
  * scan   ：按水平分层、对**谱**随机取 20 % 作测试（同一水平、同一样品的兄弟谱留在训练集）
  * sample ：按水平分层、对**物理样品**随机取 20 % 作测试（同一样品的所有扫描整簇进出）
  两种模式的浓度覆盖完全相同（都在 0–max 全范围内做内插），差异只有一个：
  测试谱是否在训练集中有同一样品的技术重复 → 差值就是泄漏带来的乐观偏差。

纪律
----
* 超参数选择始终用 GroupKFold（组 = 物理样品），即使在 scan 模式下也一样；
  这样比较只隔离"划分"这一个因素，不混入交叉验证口径的变化。
* 划分随机种子 3 个（0/1/2），模型随机种子沿用 exp03（MLP 3 个）。
* 所有数字实跑，不引用未计算的近似值。
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import baselines, data, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3"]
MODES = ["scan", "sample"]
METHODS = ["CSU-PL", "PLS", "Ridge", "SVR", "MLP", "CLS-2pt"]
SEEDS = {"MLP": [0, 1, 2]}
SPLIT_SEEDS = [0, 1, 2]
FRAC_TEST = 0.2


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def split(y, groups, seed, mode, frac=FRAC_TEST):
    """按水平分层的 20 % 测试划分。mode='scan' 以谱为单位，'sample' 以物理样品为单位。"""
    y = np.asarray(y, float)
    groups = np.asarray(groups, dtype=object)
    rng = np.random.default_rng(seed)
    te = np.zeros(len(y), dtype=bool)
    for lv in np.unique(y):
        m = np.where(np.abs(y - lv) < 1e-6)[0]
        if len(m) < 5:
            continue
        n_te = max(1, int(round(frac * len(m))))
        if mode == "scan":
            pick = rng.permutation(m)[:n_te]
        elif mode == "sample":
            # 整簇进出：随机顺序累加物理样品，直到达到目标测试规模（不裁剪，裁剪会把
            # 同一样品的扫描拆到两侧，本身就成了泄漏）
            gs = rng.permutation(np.unique(groups[m]))
            acc, got = [], 0
            for g in gs:
                if got >= n_te:
                    break
                idx = m[groups[m] == g]
                acc.extend(idx)
                got += len(idx)
            if not acc:
                acc = [m[groups[m] == gs[0]][0]]
            pick = np.array(acc, dtype=int)
        else:
            raise KeyError(mode)
        te[pick] = True
    return ~te, te


def leakage_rate(groups, tr, te):
    """测试谱中，其物理样品同时出现在训练集里的比例。"""
    g = np.asarray(groups, dtype=object)
    seen = set(g[tr])
    return float(np.mean([x in seen for x in g[te]]))


# ---------------------------------------------------------------------------
# 跑
# ---------------------------------------------------------------------------
def run_block(nm, mode, sd):
    d = data.load(nm)
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    br = data.brands(nm, groups)
    Xv = SG.transform(X)
    inv = data.inventory(d)
    rows = []
    tr, te = split(y, groups, sd, mode)
    lr = leakage_rate(groups, tr, te)
    meta_row = dict(dataset=nm, mode=mode, split_seed=sd,
                    n_train=int(tr.sum()), n_test=int(te.sum()), leak_rate=lr,
                    n_groups_train=int(len(set(groups[tr]))),
                    n_groups_test=int(len(set(groups[te]))),
                    scans_per_group=inv["scans_per_group"]["mean"])
    print(f"\n=== {nm} {mode} seed={sd}: train={tr.sum()} test={te.sum()} "
          f"leak={lr:.2f} ===", flush=True)
    for meth in METHODS:
        for ms in SEEDS.get(meth, [0]):
            t0 = time.time()
            if meth == "CSU-PL":
                mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
                q = mdl.predict_c(Xv[te], brand=br[te]) * 100
            else:
                q = baselines.fit_predict(meth, Xv[tr], y[tr], groups[tr],
                                          Xv[te], wvn, seed=ms)
            m = stats.metrics(y[te], q)
            for i, idx in enumerate(np.where(te)[0]):
                rows.append(dict(dataset=nm, mode=mode, split_seed=sd,
                                 method=meth, seed=ms, leak=lr,
                                 group=str(groups[idx]),
                                 y=float(y[idx]), pred=float(q[i])))
            print(f"  {nm} {mode} s{sd} {meth:8s} s={ms} rmse={m['rmse']:6.2f} "
                  f"slope={m['slope']:5.2f} bias={m['bias']:+6.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return rows, meta_row


def run(n_jobs=6):
    jobs = [(nm, mode, sd) for nm in DATASETS for mode in MODES for sd in SPLIT_SEEDS]
    if n_jobs == 1:
        res = [run_block(*j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_block)(*j) for j in jobs)
    rows = [r for rr, _ in res for r in rr]
    meta = [m for _, m in res]
    pred = pd.DataFrame(rows)
    pred.to_csv(os.path.join(OUT, "exp16_predictions.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "exp16_splitmeta.csv"), index=False)
    return pred, pd.DataFrame(meta)


def aggregate(pred):
    """先按 (模型种子) 算指标，再对种子与划分种子取均值。"""
    out = []
    for (nm, mode, meth), g in pred.groupby(["dataset", "mode", "method"]):
        per = []
        for (sd, ms), gs in g.groupby(["split_seed", "seed"]):
            gs = gs.sort_values("group", kind="stable")
            per.append(stats.metrics(gs["y"].values, gs["pred"].values))
        per = pd.DataFrame(per)
        out.append(dict(dataset=nm, mode=mode, method=meth, n_splits=len(per),
                        rmse=float(per["rmse"].mean()),
                        rmse_sd=float(per["rmse"].std(ddof=0)),
                        slope=float(per["slope"].mean()),
                        bias=float(per["bias"].mean()),
                        r2=float(per["r2"].mean())))
    return pd.DataFrame(out)


def markdown(agg, meta):
    lines = ["# 扫描级 vs 样品级划分泄漏对照（exp16，全部实跑值）", ""]
    lines += ["## 划分口径", "",
              "| 数据集 | 模式 | 训练谱 | 测试谱 | 泄漏率(测试谱在训练集有同样品兄弟谱) |",
              "|---|---|---|---|---|"]
    mm = meta.groupby(["dataset", "mode"]).agg(n_train=("n_train", "mean"),
                                               n_test=("n_test", "mean"),
                                               leak=("leak_rate", "mean"),
                                               gtr=("n_groups_train", "mean"),
                                               gte=("n_groups_test", "mean")).reset_index()
    for r in mm.itertuples():
        lines.append(f"| {r.dataset} | {r.mode} | {r.n_train:.0f} | {r.n_test:.0f} | "
                     f"{r.leak:.2f} |")
    lines += ["", "## 内插精度：同一浓度覆盖下，只看划分单位", "",
              "| 数据集 | 方法 | RMSE scan | RMSE sample | 乐观偏差 (sample-scan) | 比值 sample/scan |",
              "|---|---|---|---|---|---|"]
    for nm in DATASETS:
        for meth in METHODS:
            a = agg[(agg.dataset == nm) & (agg.method == meth)].set_index("mode")
            if "scan" not in a.index or "sample" not in a.index:
                continue
            s, p = a.loc["scan", "rmse"], a.loc["sample", "rmse"]
            lines.append(f"| {nm} | {meth} | {s:.2f} | {p:.2f} | {p - s:+.2f} | {p / s:.2f} |")
    lines += ["", "## 跨数据集汇总（3 数据集 × 6 方法）", "",
              "| 方法 | 中位乐观偏差 (pp) | 为正的比例 | 中位比值 |", "|---|---|---|---|"]
    piv = agg.pivot_table(index=["dataset", "method"], columns="mode", values="rmse")
    piv = piv.dropna()
    piv["gap"] = piv["sample"] - piv["scan"]
    piv["ratio"] = piv["sample"] / piv["scan"]
    for meth in METHODS:
        t = piv[piv.index.get_level_values("method") == meth]
        if len(t) == 0:
            continue
        lines.append(f"| {meth} | {t.gap.median():+.2f} | {float((t.gap > 0).mean()):.2f} | "
                     f"{t.ratio.median():.2f} |")
    lines += ["", f"整体：{int((piv.gap > 0).sum())}/{len(piv)} 个 (数据集,方法) 组合上 "
              f"scan 划分的 RMSE 更低；中位乐观偏差 {piv.gap.median():+.2f} pp，"
              f"最大 {piv.gap.max():+.2f} pp。", ""]

    lines += ["", "## 结论是否依赖划分单位（同数据集内的相对表现）", "",
              "| 数据集 | 模式 | CSU/最优基线 | CSU 排名（6 法） | 最优基线 |",
              "|---|---|---|---|---|"]
    for nm in DATASETS:
        for mode in MODES:
            t = agg[(agg.dataset == nm) & (agg["mode"] == mode)]
            base = t[t.method != "CSU-PL"].sort_values("rmse")
            csu = float(t[t.method == "CSU-PL"].rmse.iloc[0])
            order = list(t.sort_values("rmse").method)
            lines.append(f"| {nm} | {mode} | {csu / float(base.rmse.iloc[0]):.2f} | "
                         f"{order.index('CSU-PL') + 1}/{len(order)} | {base.method.iloc[0]} |")
    lines += ["", "注：内插区（浓度全覆盖）里 CSU-PL 并不占优；本文的主张只在**留出浓度水平**"
              "的外推区成立（正文 Table 2）。两种划分单位下这一相对格局方向一致。", ""]

    # 与本文真实外推口径的对照
    s3 = pd.read_csv(os.path.join(OUT, "exp03_summary.csv"))
    s3 = s3[s3.dataset.isin(DATASETS)]
    ext = (s3.groupby(["dataset", "method"])
           .agg(ext_rmse=("rmse", "median"), n_cond=("protocol", "nunique")).reset_index())
    j = piv.reset_index()[["dataset", "method", "sample"]].merge(ext, on=["dataset", "method"])
    lines += ["## 与按浓度水平留出的外推口径对比", "",
              "| 数据集 | 方法 | 内插(样品级) | 外推(21 条件中位) | 外推代价 |",
              "|---|---|---|---|---|"]
    for r in j.itertuples():
        base = getattr(r, "sample")
        lines.append(f"| {r.dataset} | {r.method} | {base:.2f} | {r.ext_rmse:.2f} | "
                     f"{r.ext_rmse - base:+.2f} |")
    io.open(os.path.join(OUT, "exp16_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    agg.to_csv(os.path.join(OUT, "exp16_summary.csv"), index=False)
    piv.reset_index().to_csv(os.path.join(OUT, "exp16_pivot.csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"                  # 块内 BLAS 限 2 线程，避免与并行块抢核
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    pred, meta = run(n_jobs=nj)
    markdown(aggregate(pred), meta)
