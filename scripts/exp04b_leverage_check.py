"""exp04b —— 杠杆公式的定量验证（半合成、模型自洽）。

只用设计变化，其余条件固定（散射 6 %、1 阶基线、白噪声 σ），因此 RMSE 的
差异只能来自浓度设计。对每个设计、每个测试水平 c*：

    h_b(c*) = 1/n_b + (c* - c̄_b)^2 / S_cc,b          （n_b = 每品牌训练谱数）
    RMSE_pred(c*) = (σ / ||Q d_b||) * sqrt(1 + h_b(c*))

其中 d_b = g_a − g_m,b，Q 投影掉基线子空间。检验 RMSE_obs 与 RMSE_pred 是否
落在 y=x 上（模型正确时应当如此），并给出杠杆解释的方差比例。
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
POOL = [0, 5, 10, 15, 20, 25, 30, 35]
TEST = [40, 45, 50]
N_BRAND, N_REP = 5, 4
SCATTER = 0.06


def main():
    d = data.load("D2")
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    br = data.brands("D2", groups)
    Xv = SG.transform(X)
    full = unmix.fit(Xv, y, wvn, CFG, brand=br)
    g = full.g.copy()
    psi1 = F.baseline_basis(wvn, 1)
    dx = np.diff(Xv, axis=1)
    sigma = 1.4826 * float(np.median(np.abs(dx - np.median(dx, axis=1, keepdims=True)))) / np.sqrt(2)
    Qb = F.projector_out(psi1)
    # 每品牌对比度在基线投影后的范数
    qd = np.array([np.linalg.norm(F.project_out(g[-1] - g[b], Qb)) for b in range(N_BRAND)])
    print(f"sigma={sigma:.3e} |Qd|={qd.round(5)} -> sigma/|Qd|={sigma/qd}", flush=True)

    rng = np.random.default_rng(7)
    designs = []
    for k in range(2, len(POOL) + 1):
        combos = list(itertools.combinations(POOL, k))
        take = min(len(combos), 22)
        idx = rng.choice(len(combos), size=take, replace=False)
        designs += [combos[i] for i in sorted(idx)]
    print(f"{len(designs)} designs", flush=True)

    rows = []
    for di, combo in enumerate(designs):
        tr_lv = np.array(combo, float)
        levels_tr = np.repeat(tr_lv, N_BRAND * N_REP)   # 每水平 5 品牌 × 4 谱
        levels_te = np.repeat(TEST, N_BRAND * N_REP)

        def gen(levels):
            c = levels / 100.0
            bi = np.tile(np.repeat(np.arange(N_BRAND), N_REP), len(levels) // (N_BRAND * N_REP))
            s = 1 + SCATTER * rng.standard_normal(len(levels))
            M = (1 - c)[:, None] * g[bi] + c[:, None] * g[-1]
            beta = rng.standard_normal((len(levels), psi1.shape[1])) * 0.5 * np.sqrt(
                ((g[-1] - g[:-1]) ** 2).mean()) / np.sqrt(psi1.shape[1])
            return (M * s[:, None] + beta @ psi1.T
                    + sigma * rng.standard_normal(M.shape)), bi.astype(str)

        Xtr, btr = gen(levels_tr)
        Xte, bte = gen(levels_te)
        mdl = unmix.fit(Xtr, levels_tr, wvn, CFG, brand=btr)
        rec = dict(index=di, k=len(combo), levels="|".join(f"{v:g}" for v in combo),
                   has_zero=int(combo[0] == 0), cmax=float(max(combo)),
                   span=float(max(combo) - min(combo)), n_brand=int(len(levels_tr) / N_BRAND))
        for cstar in TEST:
            te = np.abs(levels_te - cstar) < 1e-9
            p = mdl.predict_c(Xte[te], brand=bte[te]) * 100
            met = stats.metrics(levels_te[te], p)
            ctr = levels_tr[levels_tr < cstar] / 100.0
            # 每品牌设计（各品牌均衡）：取品牌 0 的浓度子集
            cb = (levels_tr[levels_tr < cstar] / 100.0)[:0]
            # 用实际每品牌样本数
            nb = len(levels_tr) / N_BRAND
            c_all = levels_tr / 100.0
            cbar = float(np.mean(c_all))
            scc = float(np.sum((c_all - cbar) ** 2)) / N_BRAND
            hb = 1.0 / nb + (cstar / 100.0 - cbar) ** 2 / scc
            conc = design.condition_report(levels_tr, levels_te)
            rec[f"rmse{cstar}"] = met["rmse"]
            rec[f"bias{cstar}"] = met["bias"]
            rec[f"h{cstar}"] = hb
            rec[f"h_all{cstar}"] = conc["leverage"][f"{cstar:g}"]
            rec[f"pred{cstar}"] = float(np.mean(sigma / qd)) * np.sqrt(1 + hb) * 100
            rec[f"kappa"] = conc["kappa"]
        rows.append(rec)
        if di % 20 == 0:
            print(f"  {di}/{len(designs)} k={len(combo)} rmse50={rec['rmse50']:.2f} "
                  f"h={rec['h50']:.1f} pred={rec['pred50']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp04b_leverage_check.csv"), index=False)

    from scipy import stats as st

    def powerlaw(h, y):
        """OLS fit of log(y) on log(h); returns (exponent, R2, p, n)."""
        ok = np.isfinite(h) & np.isfinite(y) & (h > 0) & (y > 0)
        lr = st.linregress(np.log(h[ok]), np.log(y[ok]))
        return float(lr.slope), float(lr.rvalue ** 2), float(lr.pvalue), int(ok.sum())

    out = {}
    pool_h, pool_y = [], []
    for cstar in TEST:
        yobs = df[f"rmse{cstar}"].values
        ypred = df[f"pred{cstar}"].values
        hh = df[f"h{cstar}"].values
        ok = np.isfinite(yobs) & np.isfinite(ypred)
        sl = float(np.polyfit(ypred[ok], yobs[ok], 1)[0])
        r2 = float(1 - np.sum((yobs[ok] - ypred[ok]) ** 2) /
                   np.sum((yobs[ok] - yobs[ok].mean()) ** 2))
        ratio = float(np.median(yobs[ok] / ypred[ok]))
        rho = float(st.spearmanr(hh, yobs).statistic)
        ex, exr2, exp_, exn = powerlaw(hh, yobs)
        out[f"c={cstar}"] = dict(slope=sl, r2_vs_identity=r2, median_ratio=ratio,
                                 spearman_h=rho, power_exponent=ex,
                                 power_r2=exr2, power_p=exp_, power_n=exn)
        pool_h.append(hh)
        pool_y.append(yobs)
        print(f"c*={cstar}: obs/pred slope={sl:.2f}, median ratio={ratio:.2f}, "
              f"R2(identity)={r2:.2f}, rho(h,RMSE)={rho:+.2f}, "
              f"RMSE ~ h^{ex:.2f} (R2={exr2:.2f}, p={exp_:.1e}, n={exn})", flush=True)
    ex, exr2, exp_, exn = powerlaw(np.concatenate(pool_h), np.concatenate(pool_y))
    out["pooled"] = dict(power_exponent=ex, power_r2=exr2, power_p=exp_, power_n=exn,
                         n_designs=int(len(df)))
    print(f"pooled over {len(TEST)} test levels: RMSE ~ h^{ex:.2f} "
          f"(R2={exr2:.2f}, p={exp_:.1e}, n={exn} from {len(df)} designs)", flush=True)
    with open(os.path.join(OUT, "exp04b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
