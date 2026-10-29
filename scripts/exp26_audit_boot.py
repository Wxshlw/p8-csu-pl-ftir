"""exp26 —— 审计量自身的不确定性：CNR 与 gamma 的聚类 bootstrap、门限误分类率、
以及 gamma 对水平加权 / 端点水平 / 拟合阶数的敏感性。

重采样单位 = 物理样品（D1–D3 每个样品多条扫描；D4–D7 一样品一谱）。
每个 replicate 重新拟合端元（ALS）、重算 CNR（Eq. 4）与 gamma（Eq. 5），
因此区间同时含"端元估计误差"与"谱间抽样"两部分。

输出
  out/exp26_audit_boot.csv     逐数据集：点估计、2.5/97.5 分位、SD、门限翻转率
  out/exp26_gamma_sens.csv     gamma 对加权 / 去掉端点水平 / 三次项拟合的敏感性
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import data, diagnostics, forward as F, unmix

OUT = os.path.join(ROOT, "out")
N_BOOT = int(os.environ.get("P8_NBOOT", "100"))
SEED = 20261028

# 分析键 -> 正文编号
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
KEYS = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)


def audit_one(d, idx=None):
    X, y = d["X"], d["y"]
    brand = d.get("brand")
    if idx is not None:
        X, y = X[idx], y[idx]
        brand = None if brand is None else np.asarray(brand, dtype=object)[idx]
    mdl = unmix.fit(X, y, d["wvn"], CFG, brand=brand)
    em = diagnostics.endmember_audit(mdl, X, y)
    cv = diagnostics.curvature_diagnostic(y, X)
    return dict(cnr=float(em["cnr"]), cnr_eff=float(em["cnr_partial"]),
                gamma=float(cv["gamma"]), leak=float(em["baseline_leak"]))


def cluster_indices(groups, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    buckets = [np.where(inv == k)[0] for k in range(len(uniq))]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([buckets[k] for k in pick])


def main():
    rng = np.random.default_rng(SEED)
    rows, sens = [], []
    for key in KEYS:
        t0 = time.time()
        d = data.load(key)
        if key in ("D1", "D2", "D3"):
            d["brand"] = _brands(key, d["groups"])
        else:
            d["brand"] = None
        d["X"] = SG.transform(d["X"])
        base = audit_one(d)
        gate0 = base["cnr"] >= 10 and abs(base["gamma"]) <= 1
        cn, ga, ce = [], [], []
        for _ in range(N_BOOT):
            idx = cluster_indices(d["groups"], rng)
            r = audit_one(d, idx)
            cn.append(r["cnr"]); ga.append(r["gamma"]); ce.append(r["cnr_eff"])
        cn, ga, ce = map(np.asarray, (cn, ga, ce))
        g_boot = (cn >= 10) & (np.abs(ga) <= 1)
        rows.append(dict(
            dataset=LABEL[key], n=int(len(d["y"])),
            n_samples=int(len(np.unique(d["groups"]))),
            cnr=base["cnr"], cnr_lo=float(np.percentile(cn, 2.5)),
            cnr_hi=float(np.percentile(cn, 97.5)), cnr_sd=float(cn.std(ddof=1)),
            gamma=base["gamma"], gamma_lo=float(np.percentile(ga, 2.5)),
            gamma_hi=float(np.percentile(ga, 97.5)), gamma_sd=float(ga.std(ddof=1)),
            gate=str(gate0), gate_flip=float(np.mean(g_boot != gate0)),
            n_boot=N_BOOT))
        print(f"{LABEL[key]} done in {time.time()-t0:.0f}s", flush=True)

        # gamma 敏感性（不重拟合，只动水平均值的加权与拟合方式）
        sens += gamma_sens(LABEL[key], d)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp26_audit_boot.csv"), index=False)
    pd.DataFrame(sens).to_csv(os.path.join(OUT, "exp26_gamma_sens.csv"), index=False)
    print("saved")


def _brands(name, groups):
    g = np.asarray(groups, dtype=object)
    if name in ("D1", "D2"):
        return np.array([str(v)[0] for v in g], dtype=object)
    return np.array([str(v)[1] if str(v).startswith("p") else str(v)[0] for v in g],
                    dtype=object)


def gamma_sens(lab, d):
    """gamma 对水平加权、端点水平、拟合阶数的敏感性（点估计，无重采样）。

    内部张成方向与 diagnostics.curvature_diagnostic 完全一致：
    c = (level - mean)/ptp；对比方向 d = sum(c_i M_i)/sum(c_i^2)；t = M d / (d d)。
    """
    from unmix_ftir.diagnostics import level_means

    X, y = d["X"], d["y"]
    lv, means, _ = level_means(y, X)
    lv = np.asarray(lv, float)
    M0 = np.asarray(means, float)
    cnt = np.array([int(np.isclose(y, v).sum()) for v in lv], float)

    def pieces(lv, M):
        c = (lv - lv.mean()) / max(float(np.ptp(lv)), 1e-12)
        M = M - M.mean(0)
        dd = (c[:, None] * M).sum(0) / max(float(c @ c), 1e-12)
        t = M @ dd / max(float(dd @ dd), 1e-12)
        return c, t

    def ratio(c, t, deg=2, w=None):
        ww = np.ones_like(c) if w is None else w
        b1 = float(np.polyfit(c, t, 1, w=np.sqrt(ww))[0])
        b2 = float(np.polyfit(c, t, deg, w=np.sqrt(ww))[deg - 2] if deg == 2
                   else np.polyfit(c, t, deg, w=np.sqrt(ww))[1])
        return float(b2 / b1) if abs(b1) > 0 else np.nan

    c, t = pieces(lv, M0)
    out = [dict(dataset=lab, variant="unweighted (reported)", gamma=ratio(c, t))]
    out.append(dict(dataset=lab, variant="weighted by spectra per level",
                    gamma=ratio(c, t, w=cnt)))
    for name, sl in (("drop lowest level", slice(1, None)),
                     ("drop highest level", slice(None, -1)),
                     ("drop both end levels", slice(1, -1))):
        cc, tt = pieces(lv[sl], M0[sl])
        out.append(dict(dataset=lab, variant=name, gamma=ratio(cc, tt)))
    out.append(dict(dataset=lab, variant="cubic fit, quadratic term",
                    gamma=ratio(c, t, deg=3)))
    return out


if __name__ == "__main__":
    main()
