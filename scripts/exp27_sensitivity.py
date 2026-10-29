"""exp27 —— 三项审稿整改敏感性（全部实跑）。

(A) 校准样本量匹配：把每个条件的校准集随机抽到该数据集三个协议中的最小样本量，
    10 次抽取取中位，看 interpolation 与外推的差异是否仍存在。
(B) 未知品牌回退：D1–D3 逐一留出一个品牌（其校准谱全部移除），测试谱按
    "未见品牌 -> 品牌平均基质端元" 预测，与全品牌专属模型对照。
(C) 基线阶数：baseline_order = 1/2/3 在 21 个条件上的 RMSEP。

输出
  out/exp27_matched_size.csv
  out/exp27_brand_fallback.csv
  out/exp27_baseline_order.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import data, design, forward as F, unmix

OUT = os.path.join(ROOT, "out")
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
KEYS = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
PROTOCOLS = ["low", "interp", "high"]
SG = F.SG(window=15, poly=3, deriv=1)


def cfg(order=1):
    return unmix.UnmixConfig(baseline_order=order, scatter_sigma=0.0,
                             resolution=np.nan, n_iter=40)


def brands_for(key, groups):
    if key in ("D1", "D2"):
        g = np.asarray(groups, dtype=object)
        return np.array([str(v)[0] for v in g], dtype=object)
    if key == "D3":
        g = np.asarray(groups, dtype=object)
        return np.array([str(v)[1] if str(v).startswith("p") else str(v)[0] for v in g],
                        dtype=object)
    return None


def rmsep(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def fit_predict(Xv, y, wvn, br, tr, te, order=1, brand_key=None):
    mdl = unmix.fit(Xv[tr], y[tr], wvn, cfg(order), brand=(br[tr] if br is not None else None))
    if brand_key is not None:
        label = np.full(int(te.sum()), str(brand_key), dtype=object)
    else:
        label = br[te] if br is not None else None
    return mdl.predict_c(Xv[te], brand=label) * 100.0


def main():
    rows_m, rows_b, rows_o = [], [], []
    rng = np.random.default_rng(20261028)
    for key in KEYS:
        t0 = time.time()
        d = data.load(key)
        Xv = SG.transform(d["X"])
        y, wvn = d["y"], d["wvn"]
        br = brands_for(key, d["groups"])
        sizes = {}
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(key, prot)
            tr = np.isin(y, tr_lv)
            te = np.isin(y, te_lv)
            sizes[prot] = (int(tr.sum()), tr, te, tr_lv, te_lv)
        n_min = min(v[0] for v in sizes.values())

        for prot, (n_tr, tr, te, tr_lv, te_lv) in sizes.items():
            # (A) 匹配样本量
            tr_idx = np.where(tr)[0]
            vals = []
            for rep in range(10):
                pick = (tr_idx if len(tr_idx) <= n_min
                        else rng.choice(tr_idx, n_min, replace=False))
                m = np.zeros(len(y), bool)
                m[pick] = True
                p = fit_predict(Xv, y, wvn, br, m, te)
                vals.append(rmsep(y[te], p))
            rows_m.append(dict(dataset=LABEL[key], protocol=prot,
                               n_cal_full=int(tr.sum()), n_cal_matched=int(n_min),
                               rmsep_matched=float(np.median(vals)),
                               rmsep_matched_lo=float(np.percentile(vals, 10)),
                               rmsep_matched_hi=float(np.percentile(vals, 90)),
                               n_test=int(te.sum())))

            # (B) 品牌回退（只有多品牌数据集）
            if br is not None:
                for b in sorted(set(br)):
                    te_b = te & (br == b)
                    if te_b.sum() == 0:
                        continue
                    tr_wo = tr & (br != b)
                    if tr_wo.sum() < 10:
                        continue
                    p_spec = fit_predict(Xv, y, wvn, br, tr, te_b)
                    p_fall = fit_predict(Xv, y, wvn, br, tr_wo, te_b, brand_key=b)
                    rows_b.append(dict(dataset=LABEL[key], protocol=prot, brand=str(b),
                                       n_cal=int(tr.sum()), n_cal_wo=int(tr_wo.sum()),
                                       n_test=int(te_b.sum()),
                                       rmsep_brand=rmsep(y[te_b], p_spec),
                                       rmsep_fallback=rmsep(y[te_b], p_fall)))

            # (C) 基线阶数
            p1 = fit_predict(Xv, y, wvn, br, tr, te, order=1)
            rec = dict(dataset=LABEL[key], protocol=prot, rmsep_o1=rmsep(y[te], p1))
            for order in (2, 3):
                rec[f"rmsep_o{order}"] = rmsep(y[te], fit_predict(Xv, y, wvn, br, tr, te,
                                                                 order=order))
            rows_o.append(rec)
        print(f"{LABEL[key]} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows_m).to_csv(os.path.join(OUT, "exp27_matched_size.csv"), index=False)
    pd.DataFrame(rows_b).to_csv(os.path.join(OUT, "exp27_brand_fallback.csv"), index=False)
    pd.DataFrame(rows_o).to_csv(os.path.join(OUT, "exp27_baseline_order.csv"), index=False)
    print("saved")


if __name__ == "__main__":
    main()
