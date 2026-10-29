"""analyze04 —— 设计扫描汇总：Spearman 相关 + 分 k 统计 + 杠杆形状检验。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")


def main():
    df = pd.read_csv(os.path.join(OUT, "exp04_design_sweep.csv"))
    rows = []
    for src, g in df.groupby("source"):
        for col in ("kappa", "kappa_intercept", "leverage50", "max_leverage", "span"):
            ok = np.isfinite(g[col]) & np.isfinite(g.rmse)
            rho, p = st.spearmanr(g[col][ok], g.rmse[ok])
            rows.append(dict(source=src, metric=col, rho=float(rho), p=float(p), n=int(ok.sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "exp04_correlations.csv"), index=False)
    print(tab.round(3).to_string(index=False))

    # 分 k / 是否含 0 的中位数
    grp = (df[df.source == "real"].groupby(["k", "has_zero"])
           .agg(n=("rmse", "size"), rmse_med=("rmse", "median"),
                q25=("rmse", lambda s: s.quantile(.25)),
                q75=("rmse", lambda s: s.quantile(.75)))
           .reset_index())
    grp.to_csv(os.path.join(OUT, "exp04_by_k.csv"), index=False)
    print("\n" + grp.round(2).to_string(index=False))

    # 杠杆形状（半合成）：RMSE vs sqrt(1+h) 过原点拟合
    b = pd.read_csv(os.path.join(OUT, "exp04b_leverage_check.csv"))
    out = {}
    for c in (40, 45, 50):
        x = b[f"pred{c}"].values
        y = b[f"rmse{c}"].values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        a = float((x @ y) / (x @ x))
        r2 = 1 - np.sum((y - a * x) ** 2) / np.sum((y - y.mean()) ** 2)
        rho = float(st.spearmanr(b[f"h{c}"][ok], y).statistic)
        out[f"c={c}"] = dict(scale=a, r2_origin=float(r2), spearman_h=rho,
                             median_ratio=float(np.median(y / x)))
    with open(os.path.join(OUT, "exp04b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nleverage shape:", json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
