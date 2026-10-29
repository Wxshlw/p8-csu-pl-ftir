"""exp04 —— ★ 实验设计扫描：浓度水平数 / 位置 / 是否含 0 % → κ(C) → 外推误差。

设计 A（真实）：在 D2 咖啡数据上枚举所有训练水平组合（0–35 %，8 个水平），
固定外推测试段 40/45/50 %，逐个组合拟合 CSU-PL。

设计 B（半合成）：以 D2 全数据拟合的端元为"真值"，按同样的浓度设计模拟
（精确前向模型 + 噪声 + 散射 + 基线），验证 κ / 杠杆与误差的关系，并给出
端元恢复误差。

输出：out/exp04_design_sweep.csv, out/exp04_summary.json
"""
from __future__ import annotations

import io
import itertools
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

from unmix_ftir import data, design, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
POOL = [0, 5, 10, 15, 20, 25, 30, 35]
TEST = [40, 45, 50]


def run_real(Xv, y, wvn, groups, brand, ref_g):
    te = np.isin(y, TEST)
    rows = []
    designs = []
    for k in range(2, len(POOL) + 1):
        for combo in itertools.combinations(POOL, k):
            designs.append(combo)
    for di, combo in enumerate(designs):
        tr_lv = np.array(combo, float)
        tr = np.isin(y, tr_lv)
        if tr.sum() < 30:
            continue
        t0 = time.time()
        mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=brand[tr])
        p = mdl.predict_c(Xv[te], brand=brand[te]) * 100.0
        met = stats.metrics(y[te], p)
        rep = design.condition_report(y[tr], y[te])
        gm = mdl.g[:-1].mean(0)
        gerr = float(np.linalg.norm(mdl.g[-1] - ref_g[-1]) / (np.linalg.norm(ref_g[-1]) + 1e-12))
        gm_err = float(np.linalg.norm(gm - ref_g[:-1].mean(0)) /
                       (np.linalg.norm(ref_g[:-1].mean(0)) + 1e-12))
        rows.append(dict(
            source="real", index=di, k=len(combo), levels="|".join(f"{v:g}" for v in combo),
            has_zero=int(combo[0] == 0), cmin=float(min(combo)), cmax=float(max(combo)),
            span=float(max(combo) - min(combo)), n_train=int(tr.sum()),
            kappa=rep["kappa"], kappa_intercept=rep["kappa_intercept"],
            leverage50=rep["leverage"][f"{max(TEST):g}"], max_leverage=rep["max_leverage"],
            rmse=met["rmse"], slope=met["slope"], bias=met["bias"], r2=met["r2"],
            train_rmse=float(np.sqrt(np.mean((mdl.predict_c(Xv[tr], brand=brand[tr]) * 100 - y[tr]) ** 2))),
            endmember_err=gerr, matrix_err=gm_err,
            bias_forecast=diagnostics.extrapolation_bias_forecast(y[tr], Xv[tr], float(max(TEST))),
            seconds=time.time() - t0))
        if di % 20 == 0:
            print(f"  real {di}/{len(designs)} k={k} rmse={met['rmse']:.2f} "
                  f"kappa={rep['kappa']:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return rows


def run_synthetic(g_truth, psi, wvn, noise, n_designs=80, nonlinear=0.0, seed=1):
    rng = np.random.default_rng(seed)
    designs = []
    for k in range(2, len(POOL) + 1):
        combos = list(itertools.combinations(POOL, k))
        take = min(len(combos), max(6, n_designs // 6))
        idx = rng.choice(len(combos), size=take, replace=False)
        designs += [combos[i] for i in sorted(idx)]
    rows = []
    for di, combo in enumerate(designs):
        tr_lv = np.array(combo, float)
        levels_tr = np.repeat(tr_lv, 20)          # 5 品牌 × 4 谱
        lev_te = np.repeat(TEST, 20)
        # 品牌循环
        def gen(levels):
            c = levels / 100.0
            ce = c * (1 - nonlinear * c / 0.5)
            b = np.tile(np.arange(5), len(levels) // 5)
            bi = np.repeat(b, 4)[:len(levels)]
            s = 1 + 0.06 * rng.standard_normal(len(levels))
            M = (1 - ce)[:, None] * g_truth[bi] + ce[:, None] * g_truth[-1]
            coef = np.array([0.0, 0.01, 0.004])[:psi.shape[1]]
            beta = rng.standard_normal((len(levels), psi.shape[1])) * coef
            return M * s[:, None] + beta @ psi.T + noise * rng.standard_normal(M.shape), bi
        Xtr, btr = gen(levels_tr)
        Xte, bte = gen(lev_te)
        cfg_syn = unmix.UnmixConfig(baseline_order=3, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
        mdl = unmix.fit(Xtr, levels_tr, wvn, cfg_syn, brand=btr.astype(str))
        p = mdl.predict_c(Xte, brand=bte.astype(str)) * 100.0
        met = stats.metrics(lev_te, p)
        rep = design.condition_report(levels_tr, lev_te)
        gerr = float(np.linalg.norm(mdl.g[-1] - g_truth[-1]) / (np.linalg.norm(g_truth[-1]) + 1e-12))
        rows.append(dict(source="synth", index=di, k=len(combo),
                         levels="|".join(f"{v:g}" for v in combo),
                         has_zero=int(combo[0] == 0), cmin=float(min(combo)),
                         cmax=float(max(combo)), span=float(max(combo) - min(combo)),
                         n_train=len(levels_tr), kappa=rep["kappa"],
                         kappa_intercept=rep["kappa_intercept"],
                         leverage50=rep["leverage"][f"{max(TEST):g}"], max_leverage=rep["max_leverage"],
                         rmse=met["rmse"], slope=met["slope"], bias=met["bias"],
                         r2=met["r2"], train_rmse=float(np.sqrt(np.mean((mdl.predict_c(
                             Xtr, brand=btr.astype(str)) * 100 - levels_tr) ** 2))),
                         endmember_err=gerr, matrix_err=np.nan,
                         bias_forecast=diagnostics.extrapolation_bias_forecast(levels_tr, Xtr, float(max(TEST))),
                         nonlinear=nonlinear, seconds=0.0))
    return rows


def main():
    d = data.load("D2")
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    brand = data.brands("D2", groups)
    Xv = SG.transform(X)
    full = unmix.fit(Xv, y, wvn, CFG, brand=brand)
    ref_g = full.g.copy()
    print("full model fitted; endmember contrast",
          float(np.linalg.norm(ref_g[-1] - ref_g[:-1].mean(0))), flush=True)

    rows = run_real(Xv, y, wvn, groups, brand, ref_g)
    # 半合成：噪声取训练残差的高频稳健估计
    dx = np.diff(Xv, axis=1)
    sigma = 1.4826 * float(np.median(np.abs(dx - np.median(dx, axis=1, keepdims=True)))) / np.sqrt(2)
    print(f"synthetic noise sigma={sigma:.5f}", flush=True)
    psi = F.baseline_basis(wvn, 3)
    rows += run_synthetic(ref_g, psi, wvn, sigma, n_designs=90, nonlinear=0.0, seed=1)
    rows += run_synthetic(ref_g, psi, wvn, sigma * 2, n_designs=60, nonlinear=0.25, seed=2)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp04_design_sweep.csv"), index=False)

    summ = {}
    for src, g in df.groupby("source"):
        for k, gk in g.groupby("k"):
            e = gk[gk.has_zero == 1]
            f = gk[gk.has_zero == 0]
            summ[f"{src}|k={k}"] = dict(
                n=int(len(gk)),
                rmse_median=float(gk.rmse.median()),
                rmse_iqr=float(gk.rmse.quantile(.75) - gk.rmse.quantile(.25)),
                rmse_zero=float(e.rmse.median()) if len(e) else None,
                rmse_nozero=float(f.rmse.median()) if len(f) else None,
                kappa_median=float(gk.kappa.median()))
    with io.open(os.path.join(OUT, "exp04_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(df.groupby(["source", "k", "has_zero"])["rmse"].median().to_string())


if __name__ == "__main__":
    main()
