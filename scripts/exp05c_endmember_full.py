"""exp05c 附表 —— 全数据拟合下，每个基质品牌的基质端元与"该品牌 0 % 平均谱"的相似度。

为什么需要它
------------
审计门（CNR / γ）只回答"对比度够不够、响应弯不弯"，**不**担保端元形状是否物理。
正文 §4.1 与 Table S8 里的"中心余弦 0.99–1.00 / D7w 0.66 / D7d 0.49"就出自这张表：
它按品牌逐行给出，因此能区分"某品牌的端元跑偏"与"整套数据都不对"。

口径
----
* 拟合与 exp05 主审计一致：SG 一阶导域、baseline_order=1、自由尺度、非负、40 轮，
  品牌感知地把**全部**光谱用来拟合（不看任何测试划分）。
* 参考谱 = 同一品牌内 y==0 的谱在**导数域**的均值。
* ``cos_raw`` 是端元与参考谱的直接余弦；``cos_centered`` 是各自去均值后的余弦。
  非负约束在导数域裁剪了负瓣并留下一个 DC 偏置，所以 raw 只有 0.08–0.23，
  而形状信息几乎全在 centered 里——这也是正文把 centered 当主指标的原因。
* ``gmin`` 是端元的最小通道值，用来标记非负约束是否真的被触发（0 表示贴边）。

输出
----
out/exp05c_endmember_full.csv   每个 (dataset, brand) 一行：n0 / cos_raw / cos_centered / gmin
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

from unmix_ftir import data, forward as F, unmix

DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def cos_centered(a, b):
    return cos(a - a.mean(), b - b.mean())


def run_dataset(nm, d):
    Xv, y, wvn, groups = SG.transform(d["X"]), d["y"], d["wvn"], d["groups"]
    br = data.brands(nm, groups)
    mdl = unmix.fit(Xv, y, wvn, CFG, brand=br)
    lut = {str(b): i for i, b in enumerate(mdl.brand_levels)} if mdl.brand_levels \
        else {str(sorted(set(br.tolist()))[0]): 0}
    rows = []
    for b, gi in sorted(lut.items(), key=lambda kv: kv[0]):
        m0 = (np.asarray(br, dtype=object).astype(str) == str(b)) & (y == 0)
        if not m0.any():
            continue
        ref = Xv[m0].mean(0)
        gm = mdl.g[gi]
        rows.append(dict(dataset=nm, brand=b, n0=int(m0.sum()),
                         cos_raw=cos(gm, ref), cos_centered=cos_centered(gm, ref),
                         gmin=float(gm.min())))
    return rows


def main():
    rows = []
    for nm in DATASETS:
        r = run_dataset(nm, data.load(nm))
        rows += r
        for x in r:
            print(f"{x['dataset']:4s} brand {x['brand']:>2s} n0={x['n0']:3d} "
                  f"cos_raw={x['cos_raw']:.4f} cos_centered={x['cos_centered']:.4f} "
                  f"gmin={x['gmin']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    # 品牌编号在数据集里是纯数字，历史上这张表按整数写出；
    # 只有在全部都能解析成整数时才转，否则保持文本（本 pandas 版不支持 to_numeric 的 keep/ignore）。
    if df["brand"].astype(str).str.fullmatch(r"-?\d+").all():
        df["brand"] = df["brand"].astype(int)
    df.to_csv(os.path.join(OUT, "exp05c_endmember_full.csv"), index=False)

    s = df.groupby("dataset").agg(n_brands=("brand", "nunique"),
                                  med_raw=("cos_raw", "median"),
                                  med_cen=("cos_centered", "median"),
                                  min_cen=("cos_centered", "min")).reset_index()
    print("\n" + s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
