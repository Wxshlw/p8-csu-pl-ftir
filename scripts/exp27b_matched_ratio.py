"""exp27b —— 校准样本量匹配下的预指定比较器对照。

exp27(A) 只看了 CSU-PL 自己在匹配样本量后的误差；要主张"外推罚不是样本量
失配造成的"，还需要同一匹配下的比较器。这里把 4 个预指定比较器
（PLS、Ridge、CLS-2pt、MCR-ALS）在同一批 10 次匹配抽样上重跑，给出匹配后
的 CSU-PL/比较器比值，与不匹配时的比值对照。

输出
  out/exp27b_matched_ratio.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import baselines, data, design, forward as F, unmix

OUT = os.path.join(ROOT, "out")
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
KEYS = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
PROTOCOLS = ["low", "interp", "high"]
PRE = ["PLS", "Ridge", "CLS-2pt", "MCR-ALS"]
SG = F.SG(window=15, poly=3, deriv=1)
N_DRAW = 10
RNG = np.random.default_rng(20261028)


def cfg(order=1):
    return unmix.UnmixConfig(baseline_order=order, scatter_sigma=0.0,
                             resolution=np.nan, n_iter=40)


def brands_for(key, groups):
    if key in ("D1", "D2"):
        return np.array([str(v)[0] for v in np.asarray(groups, dtype=object)], dtype=object)
    if key == "D3":
        g = np.asarray(groups, dtype=object)
        return np.array([str(v)[1] if str(v).startswith("p") else str(v)[0] for v in g],
                        dtype=object)
    return None


def rmsep(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    rows = []
    for key in KEYS:
        ds = LABEL[key]
        d = data.load(key)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = brands_for(key, groups)
        Xv = SG.transform(X)
        splits = {}
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(key, prot)
            splits[prot] = (np.isin(y, tr_lv), np.isin(y, te_lv))
        n_min = min(int(tr.sum()) for tr, te in splits.values())
        for prot in PROTOCOLS:
            tr, te = splits[prot]
            idx_tr = np.where(tr)[0]
            yte = y[te]
            r_csu, r_cmp = [], {m: [] for m in PRE}
            for k in range(N_DRAW):
                sel = (np.sort(RNG.choice(idx_tr, size=n_min, replace=False))
                       if len(idx_tr) > n_min else np.sort(idx_tr))
                trm = np.zeros_like(tr)
                trm[sel] = True
                mdl = unmix.fit(Xv[trm], y[trm], wvn, cfg(1),
                                brand=(br[trm] if br is not None else None))
                p = mdl.predict_c(Xv[te], brand=(br[te] if br is not None else None)) * 100.0
                r_csu.append(rmsep(yte, p))
                for m in PRE:
                    q = baselines.fit_predict(m, Xv[trm], y[trm], groups[trm], Xv[te], wvn, seed=0)
                    r_cmp[m].append(rmsep(yte, q))
            rows.append(dict(dataset=ds, protocol=prot, n_cal_matched=n_min,
                             rmsep_csu=float(np.median(r_csu)),
                             **{f"ratio_{m}": float(np.median(r_csu) / np.median(r_cmp[m]))
                                for m in PRE},
                             **{f"rmse_{m}": float(np.median(r_cmp[m])) for m in PRE}))
            print(f"{ds} {prot:6s} csu {np.median(r_csu):6.2f} "
                  + " ".join(f"{m}={np.median(r_csu)/np.median(r_cmp[m]):.2f}" for m in PRE),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp27b_matched_ratio.csv"), index=False)


if __name__ == "__main__":
    main()
