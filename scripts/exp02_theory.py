"""exp02 —— 受控合成验证：可辨识性、前向模型正确时的外推、失效模式。

生成三套数据（同一组真值端元）：
  exact      完整前向模型（乘性散射 + 基线 + 噪声）
  additive   同 exact，但用 s≡1 的纯加性模型拟合（模型错设）
  nonlinear  谱响应对标称浓度有二次压缩 c_eff = c(1 - 0.6 c / 50)

并输出 profile 曲线 R(c) 随 Qd 强度的变化（Fig.2）。
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import forward as F, stats, unmix

W = np.arange(400, 4000.001, 2.0)
L = F.line_kernel(4.0, 2.0, "gaussian")
RNG = np.random.default_rng(7)


def peak(c0, w, a):
    return a * np.exp(-0.5 * ((W - c0) / w) ** 2)


EM = peak(1050, 45, 0.55) + peak(1650, 60, 0.75) + peak(2900, 120, 0.35)
EA = peak(1100, 40, 0.60) + peak(1740, 55, 0.95) + peak(2850, 90, 0.50)
G = F.convolve_rows(np.vstack([EM, EA]), L)
PSI = F.baseline_basis(W, 3)
COEF = np.array([0.0, 0.01, 0.004])


def gen(levels, nonlinear=0.0, noise=0.002, scatter=0.15, seed=0):
    rng = np.random.default_rng(seed)
    c = levels / 100.0
    ce = c * (1 - nonlinear * c / 0.5)
    s = 1 + scatter * rng.standard_normal(len(levels))
    M = (1 - ce)[:, None] * G[0] + ce[:, None] * G[1]
    beta = rng.standard_normal((len(levels), PSI.shape[1])) * COEF
    return s[:, None] * M + beta @ PSI.T + noise * rng.standard_normal(M.shape)


def run():
    ctr = np.repeat(np.arange(0, 18.1, 3.0), 20)
    cte = np.repeat(np.arange(21, 30.1, 3.0), 20)
    Xtr = gen(ctr, seed=1)
    scenarios = {
        "exact": dict(Xte=gen(cte, seed=2), cfg=unmix.UnmixConfig(baseline_order=3, resolution=4.0, n_iter=60)),
        "nonlinear": dict(Xte=None, cfg=unmix.UnmixConfig(baseline_order=3, resolution=4.0, n_iter=60)),
        "additive": dict(Xte=gen(cte, seed=2), cfg=unmix.UnmixConfig(baseline_order=3, resolution=4.0, n_iter=60, fix_scale=True)),
    }
    Xtr_nl = gen(ctr, nonlinear=0.6, seed=11)
    scenarios["nonlinear"]["Xtr"] = Xtr_nl
    scenarios["nonlinear"]["Xte"] = gen(cte, nonlinear=0.6, seed=12)

    rows, preds, models = [], [], {}
    for name, sc in scenarios.items():
        Xt = sc.get("Xtr", Xtr)
        m = unmix.fit(Xt, ctr, W, sc["cfg"])
        models[name] = m
        p = m.predict_c(sc["Xte"]) * 100
        met = stats.metrics(cte, p)
        rows.append(dict(scenario=name, **{k: met[k] for k in ("rmse", "slope", "bias", "r2")}))
        for yy, pp in zip(cte, p):
            preds.append(dict(scenario=name, y=float(yy), pred=float(pp)))
        print(f"{name:10s} rmse={met['rmse']:.3f} slope={met['slope']:.3f} bias={met['bias']:+.3f}")

    # profile 曲线：把 Qd 按比例缩放（模拟端元对比越来越接近基线/噪声）
    m = models["exact"]
    Qb = F.projector_out(m.psi)
    c_probe = 25.0
    Xp = gen(np.array([c_probe]), seed=5)
    QA = F.project_out(Xp, Qb)[0]
    Qgm = F.project_out(m.g[0][None, :], Qb)[0]
    Qd = F.project_out((m.g[1] - m.g[0])[None, :], Qb)[0]
    q0 = float(Qgm @ Qgm)
    grid = np.linspace(0, 0.4, 401)
    curves = {}
    for frac in (1.0, 0.5, 0.2, 0.05):
        Qd_s = frac * Qd
        q1 = float(Qgm @ Qd_s)
        q2 = float(Qd_s @ Qd_s)
        a = float(QA @ Qgm)
        b = float(QA @ Qd_s)
        den = q0 + 2 * grid * q1 + grid ** 2 * q2
        R = float(QA @ QA) - (a + grid * b) ** 2 / np.maximum(den, 1e-18)
        curves[f"{frac:g}"] = (R - R.min()).tolist()
    curves["c"] = grid.tolist()

    with io.open(os.path.join(OUT, "exp02_theory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(metrics=rows, predictions=preds, profile=curves,
                       endmembers=dict(gm=G[0].tolist(), ga=G[1].tolist(), wvn=W.tolist())),
                  f, ensure_ascii=False)


if __name__ == "__main__":
    run()
