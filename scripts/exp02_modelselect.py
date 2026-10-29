"""exp02_modelselect —— 估计器变体扫描（内部用，不进论文）。

问题：首版解混在自由乘性尺度下会利用 "s x c" 退化方向，测试预测变常数。
本脚本对 (baseline_order, scatter_sigma) 做网格，比较外推段表现，并检查端元是否合理。
数据集：D8（真实变化）、D1（高 SNR）、D3（最难）。
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.cross_decomposition import PLSRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from unmix_ftir import data as D, design as G, forward as F, stats as S, unmix as U  # noqa: E402


def run(nm):
    d = D.load(nm)
    y, X, wvn = d["y"], d["X"], d["wvn"]
    tr_lv, te_lv = G.protocol_split(nm, "low")
    itr = np.where(np.isin(np.round(y, 6), tr_lv))[0]
    ite = np.where(np.isin(np.round(y, 6), te_lv))[0]
    rows = {}
    # PLS 参照
    snv = F.Pipeline(F.SNV())
    Xtr, Xte = snv.fit_transform(X[itr]), snv.transform(X[ite])
    pls = PLSRegression(n_components=3, scale=False).fit(Xtr, y[itr])
    mp = S.metrics(y[ite], pls.predict(Xte).ravel())
    rows["PLS3/snv"] = dict(rmse=mp["rmse"], bias=mp["bias"], slope=mp["slope"],
                            train=np.nan, gnorm=np.nan, ifrac=np.nan)
    for order in (2, 3, 4):
        for ss in (0.0, 0.02, 0.05, 0.1, 0.2):
            cfg = U.UnmixConfig(baseline_order=order, resolution=4.0, n_iter=40,
                                scatter_sigma=ss)
            t0 = time.time()
            m = U.fit(X[itr], y[itr], wvn, cfg)
            p = m.predict(X[ite])
            met = S.metrics(y[ite], p["c_hat"] * 100)
            ptr = m.predict(X[itr])
            metr = S.metrics(y[itr], ptr["c_hat"] * 100)
            key = f"o{order}_ss{ss:g}"
            rows[key] = dict(rmse=met["rmse"], bias=met["bias"], slope=met["slope"],
                             train=metr["rmse"], gnorm=float(np.linalg.norm(m.g, axis=1).mean()),
                             ifrac=p["identifiable_frac"],
                             r2=met["r2"], fit=round(time.time() - t0, 1))
    best = sorted(((v["rmse"], k) for k, v in rows.items()))
    print(f"\n=== {nm} low: train {sorted(set(y[itr]))} test {sorted(set(y[ite]))}")
    for k, v in rows.items():
        print(f"  {k:12s} testRMSE={v['rmse']:7.3f} trainRMSE={v['train']:7.4f} "
              f"bias={v['bias']:7.2f} slope={v['slope']:6.2f} |g|={v['gnorm']:.2f} "
              f"ifrac={v['ifrac']:.2f}")
    print("  best:", best[:3])
    return rows


if __name__ == "__main__":
    out = {}
    for nm in ("D8", "D1", "D3", "D2"):
        out[nm] = run(nm)
    with open(os.path.join(ROOT, "out", "exp02_modelselect.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
