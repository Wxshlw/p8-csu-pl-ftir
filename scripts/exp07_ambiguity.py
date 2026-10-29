"""exp07 —— 旋转/尺度模糊：无监督 MCR-ALS 多初值 vs 浓度监督解混。

D2，训练 0–25 %，测试 50 %。MCR-ALS 端元与得分同时估计，只能事后用训练浓度
做线性校准；不同初值得到不同解 —— 这就是本文"浓度硬约束"要消除的自由度。
基线采用对其最有利的转导式设置（分解时并入无标签测试谱）。
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

from unmix_ftir import baselines, data, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)


def main():
    d = data.load("D2")
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    Xv = SG.transform(X)
    br = data.brands("D2", groups)
    tr = y <= 25 + 1e-9
    te = np.abs(y - 50) < 1e-9
    ref = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
    pref = ref.predict_c(Xv[te], brand=br[te]) * 100
    mref = stats.metrics(y[te], pref)
    dref = ref.g[-1] - ref.g[:-1].mean(0)
    dref = dref / (np.linalg.norm(dref) + 1e-18)
    Qb = F.projector_out(F.baseline_basis(wvn, 1))
    Rtr = F.project_out(Xv[tr], Qb)
    rows = [dict(method="CSU-PL", init="supervised", seed=-1, **mref,
                 cos_to_ref=1.0, train_recon=np.nan)]
    print(f"CSU-PL rmse={mref['rmse']:.2f}", flush=True)

    inits = [("nndsvda", 0)] + [("random", s) for s in range(6)]
    for init, seed in inits:
        m = baselines.fit_mcr(Xv[tr], y[tr], br[tr], wvn, seed=seed,
                              X_extra=Xv[te], init=init)
        p = baselines.pred_mcr(m, Xv[te])
        met = stats.metrics(y[te], p)
        nmf = m["nmf"]
        dn = nmf.components_[0] - nmf.components_[1]
        dn = dn / (np.linalg.norm(dn) + 1e-18)
        cos = float(abs(dn @ dref))
        # 训练重构误差（未校正基线前空间内）
        Wtr = nmf.transform(np.maximum(Rtr, 0.0))
        rec = np.linalg.norm((Wtr @ nmf.components_) - Rtr) / np.linalg.norm(Rtr)
        rows.append(dict(method="MCR-ALS", init=init, seed=seed, **met,
                         cos_to_ref=cos, train_recon=float(rec)))
        print(f"MCR {init:9s} s={seed} rmse={met['rmse']:6.2f} slope={met['slope']:5.2f} "
              f"bias={met['bias']:+6.2f} cos={cos:+.3f} recon={rec:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp07_ambiguity.csv"), index=False)
    m = df[df.method == "MCR-ALS"]
    print(f"\nMCR-ALS across {len(m)} initialisations: RMSE {m.rmse.min():.2f}-{m.rmse.max():.2f}, "
          f"slope {m.slope.min():.2f}-{m.slope.max():.2f}, "
          f"cos {m.cos_to_ref.min():+.2f}..{m.cos_to_ref.max():+.2f}, "
          f"recon {m.train_recon.min():.3f}..{m.train_recon.max():.3f}")


if __name__ == "__main__":
    main()
