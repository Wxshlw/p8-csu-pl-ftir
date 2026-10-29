"""exp26b —— 审计量稳定性的分层（按浓度水平）重采样。

两种重采样都试过。按样品有放回抽样会复制谱：低对比系统（D5、D6）的端元拟合
被复制谱重新加权，CNR 的复制分布整体高出点估计约三分之一，区间因此不含点值。
校准设计是固定的（水平由制备决定），所以这里改为在各浓度水平内部无放回抽
80 % 的物理样品，保持水平构成、不复制谱，报告所得分布的分位区间与门判决翻转率。

输出
  out/exp26b_audit_boot_strat.csv
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
SEED = 20261029
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
KEYS = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)


def brands(name, groups):
    g = np.asarray(groups, dtype=object)
    if name in ("D1", "D2"):
        return np.array([str(v)[0] for v in g], dtype=object)
    if name == "D3":
        return np.array([str(v)[1] if str(v).startswith("p") else str(v)[0] for v in g],
                        dtype=object)
    return None


FRAC = 0.8


def strat_indices(y, groups, rng):
    """在各浓度水平内部无放回抽 80 % 的物理样品，保持每个水平的构成、不复制谱。"""
    out = []
    for lv in np.unique(np.round(y, 8)):
        m = np.where(np.abs(y - lv) < 1e-8)[0]
        g = np.asarray(groups, dtype=object)[m]
        uniq = np.unique(g)
        k = max(int(round(FRAC * len(uniq))), 2)
        pick = rng.choice(len(uniq), k, replace=False)
        out.append(np.concatenate([m[g == u] for u in np.asarray(uniq)[pick]]))
    return np.concatenate(out)


def audit_one(d, idx=None):
    X, y, br = d["X"], d["y"], d["brand"]
    if idx is not None:
        X, y = X[idx], y[idx]
        br = None if br is None else np.asarray(br, dtype=object)[idx]
    mdl = unmix.fit(X, y, d["wvn"], CFG, brand=br)
    em = diagnostics.endmember_audit(mdl, X, y)
    cv = diagnostics.curvature_diagnostic(y, X)
    return float(em["cnr"]), float(cv["gamma"])


def summ(name, base, boot):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(dataset=name, point=base, boot_mean=float(b.mean()),
                boot_sd=float(b.std(ddof=1)),
                pct_lo=float(lo), pct_hi=float(hi),
                piv_lo=float(2 * base - hi), piv_hi=float(2 * base - lo),
                n_boot=int(b.size))


def main():
    rng = np.random.default_rng(SEED)
    rows, flips = [], []
    for key in KEYS:
        t0 = time.time()
        d = data.load(key)
        d["brand"] = brands(key, d["groups"])
        d["X"] = SG.transform(d["X"])
        cnr0, g0 = audit_one(d)
        gate0 = (cnr0 >= 10) and (abs(g0) <= 1)
        cn, ga = [], []
        for _ in range(N_BOOT):
            idx = strat_indices(d["y"], d["groups"], rng)
            c, g = audit_one(d, idx)
            cn.append(c); ga.append(g)
        for nm, base, boot in (("CNR", cnr0, cn), ("gamma", g0, ga)):
            r = summ(nm, base, boot)
            r["dataset"] = LABEL[key]
            rows.append(r)
        gb = (np.asarray(cn) >= 10) & (np.abs(np.asarray(ga)) <= 1)
        flips.append(dict(dataset=LABEL[key], n_boot=N_BOOT,
                          gate=str(gate0), gate_flip=float(np.mean(gb != gate0)),
                          gamma_out=float(np.mean(np.abs(ga) > 1))))
        print(f"{LABEL[key]} done in {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp26b_audit_boot_strat.csv"), index=False)
    pd.DataFrame(flips).to_csv(os.path.join(OUT, "exp26b_gate_flips.csv"), index=False)
    print("saved")


if __name__ == "__main__":
    main()
