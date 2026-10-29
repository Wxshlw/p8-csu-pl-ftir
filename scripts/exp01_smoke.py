"""exp01_smoke —— 首个可行性检查：D7w / P-out-low 上本文方法 vs 线性基线。

目的：在铺开 10 个基线与 5 个数据集之前，先看两件事：
  1) 前向模型 + profile 推断能不能稳定收敛、端元是否非负且平滑；
  2) 外推段 (1–4 %) 上，物理约束解混与 PLS/Ridge 的差距方向是否正确。

这不是正式结果（无 bootstrap、无全部基线），只用于决定下一步。
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from unmix_ftir import data as D, design as G, forward as F, stats as S, unmix as U  # noqa: E402


def inner_pls(Xtr, ytr, gtr, max_comp=12):
    """训练折内 GroupKFold 选 PLS 成分数（样品级）。"""
    uniq = np.unique(gtr)
    k = min(5, len(uniq))
    gkf = GroupKFold(n_splits=k)
    best, best_rmse = 2, np.inf
    scores = {}
    for nc in range(1, max_comp + 1):
        errs = []
        for tr, va in gkf.split(Xtr, ytr, gtr):
            m = PLSRegression(n_components=nc, scale=False)
            m.fit(Xtr[tr], ytr[tr])
            errs.append(np.sqrt(np.mean((m.predict(Xtr[va]).ravel() - ytr[va]) ** 2)))
        scores[nc] = float(np.mean(errs))
        if scores[nc] < best_rmse:
            best_rmse, best = scores[nc], nc
    return best, scores


def main():
    t0 = time.time()
    nm, proto = "D7w", "low"
    d = D.load(nm)
    tr_lv, te_lv = G.protocol_split(nm, proto)
    y = d["y"]
    itr = np.where(np.isin(np.round(y, 6), tr_lv))[0]
    ite = np.where(np.isin(np.round(y, 6), te_lv))[0]
    X, wvn, groups = d["X"], d["wvn"], d["groups"]
    print(f"{nm} {proto}: train n={len(itr)} levels={sorted(set(y[itr]))}")
    print(f"           test  n={len(ite)} levels={sorted(set(y[ite]))}")
    rep = G.condition_report(y[itr], y[ite])
    print(f"kappa(simplex)={rep['kappa']:.3f}  max leverage={rep['max_leverage']:.3f}")
    print("leverage by level:", {k: round(v, 3) for k, v in rep["leverage"].items()})

    # ---- 本文方法 ----
    res = {}
    for order in (2, 3, 4):
        cfg = U.UnmixConfig(baseline_order=order, resolution=4.0, n_iter=40)
        t = time.time()
        m = U.fit(X[itr], y[itr], wvn, cfg)
        p = m.predict(X[ite])
        met = S.metrics(y[ite], p["c_hat"] * 100)
        met["kappa"] = rep["kappa"]
        met["identifiable_frac"] = p["identifiable_frac"]
        met["fit_s"] = round(time.time() - t, 2)
        met["train_rmse"] = m.train_rmse_
        print(f"[CSU] baseline_order={order}: " +
              " ".join(f"{k}={v:.4g}" for k, v in met.items() if isinstance(v, (int, float))))
        res[f"csu_o{order}"] = met
        if order == 3:
            print("  per-level:", {k: round(v["rmse"], 3) for k, v in
                                    S.by_level(y[ite], p["c_hat"] * 100).items()})
            print("  bias per level:", {k: round(v["bias"], 3) for k, v in
                                         S.by_level(y[ite], p["c_hat"] * 100).items()})
            print(f"  g range: gm [{m.g[0].min():.3f},{m.g[0].max():.3f}] "
                  f"ga [{m.g[1].min():.3f},{m.g[1].max():.3f}]")

    # ---- 线性基线 ----
    preps = {
        "raw": F.Pipeline(),
        "snv": F.Pipeline(F.SNV()),
        "sg1": F.Pipeline(F.SNV(), F.SG(window=11, poly=3, deriv=1)),
    }
    for pname, prep in preps.items():
        Xtr = prep.fit_transform(X[itr])
        Xte = prep.transform(X[ite])
        nc, sc = inner_pls(Xtr, y[itr], groups[itr])
        pls = PLSRegression(n_components=nc, scale=False).fit(Xtr, y[itr])
        met = S.metrics(y[ite], pls.predict(Xte).ravel())
        met["n_comp"] = nc
        print(f"[PLS/{pname}] nc={nc}: rmse={met['rmse']:.4g} bias={met['bias']:.3g} "
              f"slope={met['slope']:.3f} r2={met['r2']:.3f} neg={met['neg_rate']:.2f}")
        res[f"pls_{pname}"] = met
        # Ridge 作为容量匹配对照（同一预处理）
        for alpha in (1e-2, 1.0, 100.0):
            rg = Ridge(alpha=alpha).fit(Xtr, y[itr])
            mr = S.metrics(y[ite], rg.predict(Xte))
            res[f"ridge_{pname}_{alpha}"] = mr
        # 全谱 OLS（物理上等价于 CLS 的逆）
        ols = Ridge(alpha=1e-6).fit(Xtr, y[itr])
        mo = S.metrics(y[ite], ols.predict(Xte))
        print(f"[OLS/{pname}] rmse={mo['rmse']:.4g} bias={mo['bias']:.3g} slope={mo['slope']:.3f}")
        res[f"ols_{pname}"] = mo

    best = min((v["rmse"], k) for k, v in res.items())
    print(f"\nbest: {best[1]} rmse={best[0]:.4f}; CSU o3 rmse={res['csu_o3']['rmse']:.4f}")
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "exp01_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(dict(dataset=nm, protocol=proto, design=rep, results=res,
                       elapsed=round(time.time() - t0, 1)), f, indent=1)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
