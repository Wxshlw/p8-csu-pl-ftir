"""exp06 —— 稳健性（半合成 D2）：哪些假设违背真正伤外推？

真值端元取自 D2 全数据拟合（CSU-PL, baseline_order=1）。固定训练设计 0–25 %、
测试 50 %，按单因素扫描：

* noise      —— 白噪声 ×(1…20)（远高于实测 σ）
* scatter    —— 乘性尺度 s~N(1,σ_s²)，σ_s = 0…0.12
* baseline   —— 模型**失配**部分：2–3 阶平滑基线，幅度 = 0…20 % 对比度
                （1 阶部分被模型吸收，不构成失配）
* nonlinear  —— 有效浓度 c_eff = c(1−γc/0.5)，γ = 0…0.4（Beer–Lambert 偏离）
* matrixvar  —— 逐样品基体扰动 δ_i（同品牌不同样品），幅度 = 0…50 % 对比度

所有幅度都以**端元对比度 RMS** ‖g_a−g_m‖ 为标尺 —— 旧版按端元 RMS 标定，
基线失配被放大到对比度的 ~50 %，导致所有条件 RMSE≈23 %（无效）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
CFG_ADD = unmix.UnmixConfig(baseline_order=1, fix_scale=True, scatter_sigma=0.0,
                            resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
TRAIN_LV = [0, 5, 10, 15, 20, 25]
TEST_LV = [50]
N_REP = 20            # 每浓度 5 品牌 × 4 谱
N_BRAND = 5


def make_truth():
    d = data.load("D2")
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    Xv = SG.transform(X)
    br = data.brands("D2", groups)
    m = unmix.fit(Xv, y, wvn, CFG, brand=br)
    psi3 = F.baseline_basis(wvn, 3)
    psi1 = F.baseline_basis(wvn, 1)
    # 与 1 阶空间正交的 2–3 阶基（模型无法吸收的部分），逐列归一
    hi = psi3 - psi1 @ np.linalg.lstsq(psi1, psi3, rcond=None)[0]
    hi = hi / np.maximum(np.sqrt((hi ** 2).mean(0)), 1e-12)
    dx = np.diff(Xv, axis=1)
    sigma = 1.4826 * float(np.median(np.abs(dx - np.median(dx, axis=1, keepdims=True)))) / np.sqrt(2)
    con = float(np.sqrt((((m.g[-1] - m.g[:-1]) ** 2).mean())))
    return m.g.copy(), psi1, hi, wvn, sigma, con


def simulate(g, psi1, hi, wvn, sigma, con, noise_mult=1.0, scatter=0.05,
             base_mult=0.0, nonlinear=0.0, matrix_mult=0.0, seed=0):
    """返回 ((Xtr, brand_tr), (Xte, brand_te))；浓度单位 %。"""
    rng = np.random.default_rng(seed)
    k1, kh = psi1.shape[1], hi.shape[1]

    def gen(levels):
        c_true = levels / 100.0
        c_eff = c_true * (1.0 - nonlinear * c_true / 0.5)  # 有效混合浓度
        bi = np.tile(np.arange(N_BRAND), len(levels) // N_BRAND)
        # 逐样品基体扰动（平滑、与化学浓度无关）
        dlt = (rng.standard_normal((len(levels), kh)) * matrix_mult * con / np.sqrt(kh)) @ hi.T
        M = ((1 - c_eff)[:, None] * (g[bi] + dlt)
             + c_eff[:, None] * g[-1])
        s = 1 + scatter * rng.standard_normal(len(levels)) if scatter > 0 else np.ones(len(levels))
        # 1 阶基线：模型可吸收（幅度取 50 % 对比度，不影响结果）
        b1 = rng.standard_normal((len(levels), k1)) * 0.5 * con / np.sqrt(k1)
        # 2–3 阶失配：模型无法吸收
        b2 = rng.standard_normal((len(levels), kh)) * base_mult * con / np.sqrt(kh)
        X = (M * s[:, None] + b1 @ psi1.T + b2 @ hi.T
             + sigma * noise_mult * rng.standard_normal(M.shape))
        return X, bi.astype(str)

    return gen(np.repeat(TRAIN_LV, N_REP)), gen(np.repeat(TEST_LV, N_REP))


def main():
    g, psi1, hi, wvn, sigma, con = make_truth()
    print(f"truth: sigma={sigma:.3e}, contrast_rms={con:.3e}, sigma/contrast={sigma/con:.3f}",
          flush=True)
    conds = ([("noise", v) for v in (1.0, 5.0, 10.0, 20.0)] +
             [("scatter", v) for v in (0.0, 0.03, 0.06, 0.12)] +
             [("baseline", v) for v in (0.0, 0.05, 0.10, 0.20)] +
             [("nonlinear", v) for v in (0.0, 0.05, 0.10, 0.20, 0.40)] +
             [("matrixvar", v) for v in (0.0, 0.10, 0.25, 0.50)])
    rows = []
    for factor, val in conds:
        kw = dict(noise_mult=1.0, scatter=0.05, base_mult=0.0,
                  nonlinear=0.0, matrix_mult=0.0)
        kw[{"noise": "noise_mult", "scatter": "scatter", "baseline": "base_mult",
            "nonlinear": "nonlinear", "matrixvar": "matrix_mult"}[factor]] = val
        for seed in range(5):
            (Xtr, btr), (Xte, bte) = simulate(g, psi1, hi, wvn, sigma, con, seed=seed, **kw)
            ytr = np.repeat(TRAIN_LV, N_REP)
            yte = np.repeat(TEST_LV, N_REP)
            for meth, cfg in (("CSU-PL", CFG), ("additive-only", CFG_ADD)):
                mdl = unmix.fit(Xtr, ytr, wvn, cfg, brand=btr)
                p = mdl.predict_c(Xte, brand=bte) * 100
                met = stats.metrics(yte, p)
                rows.append(dict(factor=factor, value=val, seed=seed, method=meth,
                                 rmse=met["rmse"], slope=met["slope"], bias=met["bias"],
                                 endmember_err=float(np.linalg.norm(
                                     mdl.g[-1] - g[-1]) / np.linalg.norm(g[-1]))))
        sub = pd.DataFrame(rows)
        sub = sub[(sub.factor == factor) & (sub.value == val)]
        print(f"{factor:9s}={val:<5g} " + "  ".join(
            f"{m}:{sub[sub.method == m].rmse.mean():6.2f}" for m in sub.method.unique()),
            flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp06_robustness.csv"), index=False)


if __name__ == "__main__":
    main()
