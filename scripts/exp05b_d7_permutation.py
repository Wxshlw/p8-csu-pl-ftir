"""exp05b —— D7 标签可分辨性检验（组水平置换检验）。

问题：D7 肉桂数据的名义浓度标签是否真的对应可学习的光谱差异？
做法：固定 GroupKFold（组 = 样品名），PLS(10) 的 pooled R² 作为统计量；
在**样品级**置换浓度标签 100 次，得到零分布。
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

from unmix_ftir import data, forward as F

SG = F.SG(15, 3, 1)


def cv_r2(X, y, groups, n_comp=10, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.empty_like(y)
    for tr, te in gkf.split(X, y, groups):
        m = PLSRegression(n_components=n_comp, scale=False).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te]).ravel()
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((pred - y) ** 2) / ss), pred


def main():
    rng = np.random.default_rng(0)
    out = {}
    for nm in ["D7w", "D7p", "D7d"]:
        d = data.load(nm)
        X = SG.transform(d["X"])
        y = d["y"]
        g = np.asarray(d["groups"])
        obs, pred = cv_r2(X, y, g)
        null = []
        for _ in range(100):
            # 样品级置换：同一样品（这里一样品一谱）的标签随机重排
            yp = rng.permutation(y)
            null.append(cv_r2(X, yp, g)[0])
        null = np.array(null)
        p = float((1 + np.sum(null >= obs)) / (1 + len(null)))
        out[nm] = dict(observed_r2=obs, null_mean=float(null.mean()),
                       null_p95=float(np.percentile(null, 95)), p_value=p,
                       rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                       label_range=float(y.max() - y.min()))
        print(f"{nm}: R2={obs:+.3f}  null={null.mean():+.3f} [p95 {np.percentile(null,95):+.3f}]  "
              f"p={p:.3f}  RMSE={out[nm]['rmse']:.2f}", flush=True)
    with io.open(os.path.join(OUT, "exp05b_d7_permutation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
