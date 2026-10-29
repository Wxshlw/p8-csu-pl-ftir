"""exp11 —— 统计推断的单位修正：条件层描述 vs 数据集层推断。

论文 v1 把 21 个条件当作独立样本报告 Spearman p 值，但 21 个条件来自 7 个
数据集，CNR/γ 在每个数据集内是常数，属于典型的伪重复。本脚本给出三层证据：

1. 条件层 Spearman（复现论文数字，仅作描述，附 naive p）
2. 数据集层 Spearman + 7! 精确置换 p + 按数据集重采样 bootstrap CI
3. 审计门限的敏感性网格；以及固定基线（而非测试集 oracle）的敏感性

另计算：
  * 偏 Spearman：CNR–RMSEP 控制 |γ|；|γ|–|1−slope| 控制 CNR（精确置换）
  * 门限分类与“CSU 是否不劣于最佳基线”的 7/7 符号检验
  * 每个数据集的误差比中位数、条件胜负数

输出
----
out/exp11_inference.json
tables/table5_inference.csv        数据集层相关 + 条件层描述
tables/table6_fixed_baseline.csv   固定基线敏感性
tables/table7_gate_sensitivity.csv 门限网格
"""
from __future__ import annotations

import io
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
TAB = os.path.join(ROOT, "tables")
os.makedirs(OUT, exist_ok=True)

BOOT = 5000
RNG = np.random.default_rng(20261023)


def cnr_eff_map():
    """CNR_eff = ||d_perp||/(sigma sqrt(N)) from the fitted audit (Section 2.3)."""
    with io.open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8") as f:
        aud = json.load(f)
    return {k: float(v["endmember"]["cnr_partial"]) for k, v in aud.items()}


def load():
    t1 = pd.read_csv(os.path.join(TAB, "table1_datasets.csv")).rename(columns={"Dataset": "dataset"})
    t2 = pd.read_csv(os.path.join(TAB, "table2_main.csv")).rename(columns={
        "Dataset": "dataset", "Protocol": "protocol", "RMSEP": "rmse", "Slope": "slope",
        "Bias": "bias", "RMSEP_best": "best_rmse", "BestBaseline": "best_method",
        "Ratio": "ratio", "p_Holm": "p_holm"})
    t2["protocol"] = (t2.protocol.str.replace("-end", "", regex=False)
                      .str.replace("interp.", "interp", regex=False))
    m = t2.merge(t1[["dataset", "CNR", "Gamma", "Gate"]], on="dataset")
    ce = cnr_eff_map()
    assert set(ce) == set(t1.dataset), "CNR_eff audit keys differ from Table 1 datasets"
    m["CNR_eff"] = m.dataset.map(ce)
    m["gabs"] = m.Gamma.abs()
    m["sdist"] = (1 - m.slope).abs()
    m["abs_bias"] = m.bias.abs()
    m["pass"] = m.Gate.str.lower().eq("pass")
    # p_Holm 在表 2 中是字符串（含 "<0.001"）；仅为计数解析为数值，不用于报告
    m["p_holm_num"] = pd.to_numeric(m.p_holm.astype(str).str.replace("<", "", regex=False),
                                    errors="coerce")
    return t1, t2, m


def dataset_table(m):
    agg = dict(CNR=("CNR", "first"), CNR_eff=("CNR_eff", "first"), gabs=("gabs", "first"),
               rmse=("rmse", "median"),
               best=("best_rmse", "median"), ratio=("ratio", "median"),
               sdist=("sdist", "median"), pass_=("pass", "first"),
               n=("rmse", "size"), slope=("slope", "median"))
    if "best_sdist" in m.columns:                      # 每条件最佳基线的 |1−slope|
        agg["best_sdist"] = ("best_sdist", "median")
    g = m.groupby("dataset").agg(**agg).reset_index()
    return g.sort_values("CNR")


def exact_perm(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    cnt = 0
    for p in itertools.permutations(range(n)):
        if abs(stats.spearmanr(x[list(p)], y).statistic) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / math.factorial(n)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(np.asarray(v, float)) for v in (x, y, z))

    def resid(a, b):
        A = np.column_stack([np.ones_like(b), b])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


def exact_partial(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    obs = partial_spearman(x, y, z)
    cnt = 0
    for p in itertools.permutations(range(len(x))):
        if abs(partial_spearman(x[list(p)], y, z)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / math.factorial(len(x))


def boot_rho(g, x, y):
    vals = []
    n = len(g)
    xv = g[x].values; yv = g[y].values
    for _ in range(BOOT):
        idx = RNG.integers(0, n, n)
        if len(np.unique(xv[idx])) < 3 or len(np.unique(yv[idx])) < 3:
            continue
        vals.append(stats.spearmanr(xv[idx], yv[idx]).statistic)
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def boot_median_ratio(m, group):
    """按数据集重采样，再在重采样条件的并集上取中位数。"""
    sub = m[m["pass"] == group]
    ds = sorted(sub.dataset.unique())
    parts = [sub[sub.dataset == d].ratio.values for d in ds]
    vals = []
    for _ in range(BOOT):
        pick = RNG.integers(0, len(ds), len(ds))
        arr = np.concatenate([parts[i] for i in pick])
        vals.append(np.median(arr))
    vals = np.asarray(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gate_grid(g):
    rows = []
    for cnr_t in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        for gam_t in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
            ps = g[(g.CNR >= cnr_t) & (g.gabs <= gam_t)]
            fl = g[~((g.CNR >= cnr_t) & (g.gabs <= gam_t))]
            if len(ps) == 0 or len(fl) == 0:
                continue
            rows.append(dict(cnr_threshold=cnr_t, gamma_threshold=gam_t,
                             n_pass=len(ps), n_fail=len(fl),
                             pass_median=float(ps.ratio.median()),
                             fail_median=float(fl.ratio.median()),
                             separation=float(fl.ratio.median() - ps.ratio.median()),
                             ordered=int(ps.ratio.max() < fl.ratio.min()),
                             pass_datasets="|".join(sorted(ps.dataset)),
                             fail_datasets="|".join(sorted(fl.dataset))))
    return pd.DataFrame(rows)


def fixed_baseline(m, exp3):
    rows = []
    inside = m["pass"]
    # 顺序与正文 §3.4 的八个比较器枚举一致；SVR/RF/HGB/MLP 只在 D1–D3 与 D8 上跑，
    # 故它们在 inside 的 n 比 PLS 等四个方法小（D7d 缺失），outside 同样少三个 D7 条件。
    for meth in ["PLS", "Ridge", "SVR", "RF", "HGB", "MLP", "CLS-2pt", "MCR-ALS"]:
        e = exp3[exp3.method == meth].copy()
        e["ratio"] = 1.0 - e.diff_rel          # diff_rel = (baseline - CSU)/baseline
        e = e.merge(m[["dataset", "protocol", "pass"]], on=["dataset", "protocol"], how="left")
        for grp, gsub in [("inside", e[e["pass"]]), ("outside", e[~e["pass"]])]:
            sig_better = int(((gsub.p_holm < 0.05) & (gsub.diff_rel > 0)).sum())
            sig_worse = int(((gsub.p_holm < 0.05) & (gsub.diff_rel < 0)).sum())
            rows.append(dict(method=meth, region=grp, n=len(gsub),
                             median_ratio=float(gsub.ratio.median()),
                             dataset_balanced_median=float(
                                 gsub.groupby("dataset").ratio.median().median()),
                             n_csu_better=int((gsub.ratio < 1).sum()),
                             n_sig_better=sig_better, n_sig_worse=sig_worse))
    # oracle 行直接来自表 2
    for grp, gsub in [("inside", m[inside]), ("outside", m[~inside])]:
        rows.append(dict(method="Best comparator (oracle)", region=grp, n=len(gsub),
                         median_ratio=float(gsub.ratio.median()),
                         dataset_balanced_median=float(gsub.groupby("dataset").ratio.median().median()),
                         n_csu_better=int((gsub.ratio < 1).sum()),
                         n_sig_better=int(((gsub.p_holm_num < 0.05) & (gsub.ratio < 1)).sum()),
                         n_sig_worse=int(((gsub.p_holm_num < 0.05) & (gsub.ratio > 1)).sum())))
    return pd.DataFrame(rows)


def main():
    t1, t2, m = load()
    exp3 = pd.read_csv(os.path.join(OUT, "exp03_table.csv"))

    # 每条件"最佳基线"的斜率畸变：Table 2 只给出其 RMSEP，斜率要从 exp03_table 同一
    # (dataset, protocol, method) 行取，正文 "|γ| vs |1−slope| for the best baseline" 由此复算。
    b = exp3[exp3.method != "CSU-PL"].copy()
    b["best_sdist"] = (1.0 - b.slope).abs()
    b = b[["dataset", "protocol", "method", "best_sdist"]].rename(columns={"method": "best_method"})
    assert not b.duplicated(["dataset", "protocol", "best_method"]).any(), \
        "exp03_table repeats a (dataset, protocol, method) row; the baseline slope lookup would be ambiguous"
    m = m.merge(b, on=["dataset", "protocol", "best_method"], how="left")
    assert len(m) == 21, f"condition table grew to {len(m)} rows through the baseline merge"
    miss = int(m.best_sdist.isna().sum())
    assert miss == 0, f"{miss} conditions have no baseline slope row in exp03_table"

    g = dataset_table(m)
    print(g.to_string(index=False), flush=True)

    rows = []

    # --- 条件层（论文 v1 的口径，作为描述保留）---
    cond_pairs = [("CNR", "rmse", "CNR vs CSU RMSEP"),
                  ("CNR", "best_rmse", "CNR vs best-comparator RMSEP"),
                  ("CNR", "abs_bias", "CNR vs |bias|"),
                  ("gabs", "sdist", "|gamma| vs |1-slope| (CSU)"),
                  ("gabs", "rmse", "|gamma| vs CSU RMSEP"),
                  ("CNR", "ratio", "CNR vs CSU/best ratio")]
    for x, y, lab in cond_pairs:
        r, p = stats.spearmanr(m[x], m[y])
        rows.append(dict(relation=lab, x=x, y=y, unit="condition", n=len(m),
                         statistic=float(r), p_exact=np.nan, p_naive=float(p),
                         ci_lo=np.nan, ci_hi=np.nan,
                         note="descriptive; 21 conditions cluster in 7 datasets"))

    # --- 数据集层 ---
    ds_pairs = [("CNR", "rmse", "CNR vs CSU RMSEP"),
                ("CNR", "best", "CNR vs best-comparator RMSEP"),
                ("CNR_eff", "best", "CNR_eff vs best-comparator RMSEP"),
                ("CNR_eff", "rmse", "CNR_eff vs CSU RMSEP"),
                ("gabs", "sdist", "|gamma| vs |1-slope| (CSU)"),
                ("gabs", "best_sdist", "|gamma| vs |1-slope| (best comparator)"),
                ("gabs", "rmse", "|gamma| vs CSU RMSEP"),
                ("CNR", "ratio", "CNR vs CSU/best ratio")]
    for x, y, lab in ds_pairs:
        rho, p = exact_perm(g[x].values, g[y].values)
        ci, nb = boot_rho(g, x, y)
        rows.append(dict(relation=lab, x=x, y=y, unit="dataset", n=len(g),
                         statistic=float(rho), p_exact=float(p), p_naive=np.nan,
                         ci_lo=ci[0], ci_hi=ci[1],
                         note=f"exact 7! permutation; cluster bootstrap {nb} reps"))

    # --- 偏相关（数据集层）---
    for x, y, z, lab in [("CNR", "rmse", "gabs", "CNR vs CSU RMSEP | |gamma|"),
                         ("gabs", "sdist", "CNR", "|gamma| vs |1-slope| | CNR")]:
        rho, p = exact_partial(g[x].values, g[y].values, g[z].values)
        rows.append(dict(relation=lab, x=x, y=y, unit="dataset", n=len(g),
                         statistic=float(rho), p_exact=float(p), p_naive=np.nan,
                         ci_lo=np.nan, ci_hi=np.nan,
                         note=f"partial Spearman controlling {z}; exact 7! permutation"))

    infer = pd.DataFrame(rows)
    infer.to_csv(os.path.join(TAB, "table5_inference.csv"), index=False)
    print("\n" + infer.to_string(index=False), flush=True)

    # --- 门限分类与符号检验 ---
    ps = g[g.pass_]; fl = g[~g.pass_]
    n_concord = int((ps.ratio < 1).sum() + (fl.ratio > 1).sum())
    sign_p = float(stats.binomtest(n_concord, len(g), 0.5).pvalue)
    mw = stats.mannwhitneyu(ps.ratio, fl.ratio, alternative="two-sided")
    rng = {
        "n_datasets": len(g), "n_pass": len(ps), "n_fail": len(fl),
        "pass_datasets": ps.dataset.tolist(), "fail_datasets": fl.dataset.tolist(),
        "pass_median_ratios": ps.set_index("dataset").ratio.round(4).to_dict(),
        "fail_median_ratios": fl.set_index("dataset").ratio.round(4).to_dict(),
        "pass_ratio_median": float(ps.ratio.median()),
        "fail_ratio_median": float(fl.ratio.median()),
        "pass_ratio_ci": boot_median_ratio(m, True),
        "fail_ratio_ci": boot_median_ratio(m, False),
        "condition_ratio_median_inside": float(m[m["pass"]].ratio.median()),
        "condition_ratio_median_outside": float(m[~m["pass"]].ratio.median()),
        "n_concordant": n_concord, "sign_test_p": sign_p,
        "mannwhitney_p": float(mw.pvalue),
        "n_sig_worse_outside": int(((m[~m["pass"]].p_holm_num < 0.05) & (m[~m["pass"]].ratio > 1)).sum()),
        "n_sig_better_outside": int(((m[~m["pass"]].p_holm_num < 0.05) & (m[~m["pass"]].ratio < 1)).sum()),
        "n_sig_comparisons_inside": int((m[m["pass"]].p_holm_num < 0.05).sum()),
        "oracle_counts": m.best_method.value_counts().to_dict(),
        "cnr_eff_min": float(g.CNR_eff.min()), "cnr_eff_max": float(g.CNR_eff.max()),
        "cnr_eff_same_cnr_classification":
            int((g.CNR.ge(10).values == g.CNR_eff.ge(10).values).sum()),
    }
    print("\n" + json.dumps(rng, ensure_ascii=False, indent=1), flush=True)

    grid = gate_grid(g)
    grid.to_csv(os.path.join(TAB, "table7_gate_sensitivity.csv"), index=False)
    print("\ngate grid (n_pass 2-5):", flush=True)
    print(grid[(grid.n_pass >= 2) & (grid.n_pass <= 5)].to_string(index=False), flush=True)

    fb = fixed_baseline(m, exp3)
    fb.to_csv(os.path.join(TAB, "table6_fixed_baseline.csv"), index=False)
    print("\n" + fb.to_string(index=False), flush=True)

    # 门限平台摘要：在 2<=n_pass<=5 的格点里，分类是否与 (10,1) 相同、结论是否同向
    chosen = grid[(grid.cnr_threshold == 10) & (grid.gamma_threshold == 1.0)].iloc[0]
    ok = grid[(grid.n_pass >= 2) & (grid.n_pass <= 5)].copy()
    same = (ok.n_pass == chosen.n_pass) & (ok.pass_datasets == chosen.pass_datasets)
    plateau = {
        "n_grid_points": int(len(ok)),
        "n_same_classification": int(same.sum()),
        "n_pass_lt1_fail_gt1": int(((ok.pass_median < 1) & (ok.fail_median > 1)).sum()),
        "chosen_n_pass": int(chosen.n_pass),
        "chosen_pass_datasets": chosen.pass_datasets,
    }
    print("\n" + json.dumps(plateau, ensure_ascii=False, indent=1), flush=True)

    allout = dict(inference=infer.to_dict("records"), gate=rng, plateau=plateau,
                  dataset_table=g.to_dict("records"),
                  fixed_baseline=fb.to_dict("records"))
    with io.open(os.path.join(OUT, "exp11_inference.json"), "w", encoding="utf-8") as f:
        json.dump(allout, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
