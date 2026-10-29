"""exp03c —— 0 % 锚点补救：低端协议把未使用的 0 % 样品加进校准集，能挽回多少。

动机
----
三个数据集的低端协议（P-out-low）在校准集中**没有 0 % 样品**：D1 训练 9–30 %、
D2 训练 15–50 %、D7w 训练 6–24 %，而原始数据里 0 % 一栏（D1 102、D2 152、D7w 50 条谱）
一直闲置。低端外推恰好要穿过这段空白，因此最自然的补救不是换模型，而是补设计。

三个变体
--------
``base``                只用地协议训练水平（必须与 exp03 主表逐位一致，脚本自检）
``plus_all_0``          加入该数据集全部 y==0 的谱
``plus_one_per_brand``  每个基质品牌只加入 1 条 0 % 谱（D7 无品牌结构 -> 加 1 条）
                        —— 现实里更便宜的做法，用来判断收益是否需要大量补标样

输出
----
out/exp03c_remedy.csv        3 数据集 × 3 变体 × 3 方法（CSU-PL / PLS / Ridge）的
                             n_train / rmse / bias / slope，喂 Table 3b
out/exp03c_predictions.csv   base 与 plus_all_0 两个变体、CSU-PL 与 PLS 两个方法的
                             逐谱预测，喂 Fig. 3c 的"0 % 锚点恢复"曲线
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

from unmix_ftir import baselines, data, design, forward as F, stats, unmix

DATASETS = ["D1", "D2", "D7w"]
PROTOCOL = "low"
METHODS = ["CSU-PL", "PLS", "Ridge"]
CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)


def train_mask(nm, y, variant):
    """Return a boolean training mask for one remedy variant."""
    tr_lv, _ = design.protocol_split(nm, PROTOCOL)
    tr = np.isin(y, tr_lv)
    if variant == "base":
        return tr
    zero = np.where(y == 0)[0]
    if variant == "plus_all_0":
        tr[zero] = True
        return tr
    raise KeyError(variant)


def add_one_per_brand(nm, y, groups, tr):
    """Add a single 0 % spectrum per matrix brand (single brand -> one spectrum)."""
    br = data.brands(nm, groups)
    tr = tr.copy()
    for b in sorted(set(br.tolist())):
        idx = np.where((y == 0) & (br == b))[0]
        if len(idx):
            tr[idx[0]] = True
    return tr


def run_method(meth, Xtr, ytr, gtr, Xte, wvn, br_tr, br_te):
    if meth == "CSU-PL":
        mdl = unmix.fit(Xtr, ytr, wvn, CFG, brand=br_tr)
        return mdl.predict_c(Xte, brand=br_te) * 100.0
    return baselines.fit_predict(meth, Xtr, ytr, gtr, Xte, wvn, seed=0)


def main():
    met_rows, pred_rows = [], []
    for nm in DATASETS:
        d = data.load(nm)
        Xv, y, wvn, groups = SG.transform(d["X"]), d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        _, te_lv = design.protocol_split(nm, PROTOCOL)
        te = np.isin(y, te_lv)
        base = train_mask(nm, y, "base")
        masks = {"base": base,
                 "plus_all_0": train_mask(nm, y, "plus_all_0"),
                 "plus_one_per_brand": add_one_per_brand(nm, y, groups, base)}
        for variant, tr in masks.items():
            for meth in METHODS:
                pred = run_method(meth, Xv[tr], y[tr], groups[tr], Xv[te], wvn,
                                  br[tr], br[te])
                m = stats.metrics(y[te], pred)
                met_rows.append(dict(dataset=nm, protocol=PROTOCOL, variant=variant,
                                     method=meth, n_train=int(tr.sum()), rmse=m["rmse"],
                                     bias=m["bias"], slope=m["slope"]))
                if variant in ("base", "plus_all_0") and meth in ("CSU-PL", "PLS"):
                    for i, idx in enumerate(np.where(te)[0]):
                        pred_rows.append(dict(dataset=nm, protocol=PROTOCOL,
                                              variant=variant, method=meth,
                                              group=str(groups[idx]),
                                              y=float(y[idx]), pred=float(pred[i])))
                print(f"{nm:4s} {variant:19s} {meth:7s} n_train={int(tr.sum()):4d} "
                      f"rmse={m['rmse']:6.2f} slope={m['slope']:+5.2f}", flush=True)

    rem = pd.DataFrame(met_rows)
    rem.to_csv(os.path.join(OUT, "exp03c_remedy.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(OUT, "exp03c_predictions.csv"),
                                   index=False)

    # --- 自检：base 分支必须复现 exp03 主表的同条件数字 ---
    main_tbl = pd.read_csv(os.path.join(OUT, "exp03_table.csv"))
    j = rem[rem.variant == "base"].merge(
        main_tbl, on=["dataset", "protocol", "method"], how="inner")
    dev = float((j["rmse_x"] - j["rmse_y"]).abs().max())
    print(f"\nself-check: {len(j)} base rows, max |rmse - exp03| = {dev:.4g} pp")
    assert dev < 1e-6, "base branch no longer reproduces exp03; Table 3b invalid"
    csu = rem[rem.method == "CSU-PL"].pivot_table(index="dataset", columns="variant",
                                                 values="rmse")
    print("\ncsu rmse by variant:\n" + csu.round(2).to_string())


if __name__ == "__main__":
    main()
