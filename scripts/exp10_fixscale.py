"""exp10 —— 自由乘性尺度 vs 固定尺度（s ≡ 1）消融：Table 3a 与 Fig. 5 的数据源。

动机
----
式 (1) 里的逐样品乘子 s 是本文与"纯加性"解混（CLS / 普通 ALS）的唯一实质差别。
把 ``UnmixConfig.fix_scale`` 置 True 相当于 λ_s → ∞ 的极限：端元仍由浓度监督求解，
但预测时不再消去每个样品的散射/接触尺度，只消去加性基线。
两种设置跑同一套 7 × 3 条件、同一套划分，因此比值是配对的。

自检
----
``fix_scale=False`` 的每一行必须与 exp03 主实验里 CSU-PL 的同条件结果完全一致
（同一配置、同一划分、同一网格），脚本会把最大偏差打印出来并作为断言；
不一致说明划分或配置被改动过，此时 Table 3a 不可用。

输出
----
out/exp10_fixscale.csv    dataset × protocol × fix_scale 的 rmse / bias / slope
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

from unmix_ftir import data, design, forward as F, stats, unmix

DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
BASE = dict(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40,
            atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)


def run_condition(nm, prot, Xv, y, wvn, br):
    tr_lv, te_lv = design.protocol_split(nm, prot)
    tr, te = np.isin(y, tr_lv), np.isin(y, te_lv)
    out = []
    for fix in (True, False):
        cfg = unmix.UnmixConfig(**dict(BASE, fix_scale=fix))
        mdl = unmix.fit(Xv[tr], y[tr], wvn, cfg, brand=br[tr])
        pred = mdl.predict_c(Xv[te], brand=br[te]) * 100.0
        m = stats.metrics(y[te], pred)
        out.append(dict(dataset=nm, protocol=prot, fix_scale=fix, rmse=m["rmse"],
                        bias=m["bias"], slope=m["slope"]))
    return out


def main():
    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        Xv, y, wvn = SG.transform(d["X"]), d["y"], d["wvn"]
        br = data.brands(nm, d["groups"])
        for prot in PROTOCOLS:
            r = run_condition(nm, prot, Xv, y, wvn, br)
            rows += r
            print(f"{nm:4s} {prot:6s} free={r[1]['rmse']:6.2f} fixed={r[0]['rmse']:6.2f} "
                  f"ratio={r[0]['rmse']/r[1]['rmse']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp10_fixscale.csv"), index=False)

    # --- 自检：自由尺度分支必须复现 exp03 主表的 CSU-PL 数字 ---
    main_tbl = pd.read_csv(os.path.join(OUT, "exp03_table.csv"))
    csu = main_tbl[(main_tbl.method == "CSU-PL") & main_tbl.dataset.isin(DATASETS)]
    j = df[~df.fix_scale].merge(csu, on=["dataset", "protocol"], how="inner")
    if len(j):
        dev = float((j["rmse_x"] - j["rmse_y"]).abs().max())
        print(f"\nself-check: max |free - exp03 CSU-PL| = {dev:.4g} pp over {len(j)} rows")
        assert dev < 1e-6, "free-scale branch no longer reproduces exp03; Table 3a invalid"
    r = (df[df.fix_scale].set_index(["dataset", "protocol"]).rmse /
         df[~df.fix_scale].set_index(["dataset", "protocol"]).rmse)
    print(f"fixed/free ratio: median {float(r.median()):.2f}, "
          f"fixed worse in {int((r > 1).sum())}/{len(r)} conditions")


if __name__ == "__main__":
    main()
