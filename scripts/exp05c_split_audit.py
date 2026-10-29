"""exp05c —— 只用训练 split 的可辨识性审计 + 端元物理性 + 非负消融。

背景：exp05 的 CNR / γ 是在全部谱上算的，而论文把它们写成"只用校准数据"。
本脚本对 21 个外推条件逐个重算：
  * 在训练 split 上拟合 CSU-PL，再用训练谱计算 CNR、γ、泄漏、端元形状；
  * 以训练 split 中每个品牌的 0% 均值谱为参考，计算端元形状的
    原始余弦与去均值（centered）余弦；
  * 同一 split 上跑 nonneg=True/False 两个变体，比较测试 RMSEP/斜率/偏置。

输出
----
out/exp05c_condition_audit.csv
out/exp05c_nonneg.csv
out/exp05c_summary.json
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, diagnostics, forward as F, stats, unmix

DATASETS = ["D1", "D2", "D3", "D8", "D7w", "D7p", "D7d"]
PROTOCOLS = ["low", "high", "interp"]
CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
CFG_NN = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                           n_iter=40, nonneg=False)
SG = F.SG(window=15, poly=3, deriv=1)


def centered_cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def raw_cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def main():
    audit_rows, nn_rows = [], []
    summary = {}
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        Xv = SG.transform(X)
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr = np.isin(y, tr_lv); te = np.isin(y, te_lv)
            mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
            e = diagnostics.endmember_audit(mdl, Xv[tr], y[tr])
            cur = diagnostics.curvature_diagnostic(y[tr], Xv[tr])

            # 端元形状：逐品牌去均值余弦
            levels = mdl.brand_levels or ["0"]
            cos_raw, cos_cen = {}, {}
            zero_tr = tr & (y < 1e-9)
            for bi, b in enumerate(levels):
                mask = zero_tr & (np.asarray([str(v) for v in br]) == str(b))
                if not mask.any():
                    mask = zero_tr                      # 该品牌无 0% -> 用全体 0%
                if not mask.any():
                    continue
                ref = Xv[mask].mean(0)
                gm = mdl.g[bi]
                cos_raw[b] = raw_cos(gm, ref)
                cos_cen[b] = centered_cos(gm, ref)

            p = mdl.predict_c(Xv[te], brand=br[te]) * 100.0
            m = stats.metrics(y[te], p)

            # 非负消融
            mdl_nn = unmix.fit(Xv[tr], y[tr], wvn, CFG_NN, brand=br[tr])
            p_nn = mdl_nn.predict_c(Xv[te], brand=br[te]) * 100.0
            m_nn = stats.metrics(y[te], p_nn)
            nn_rows.append(dict(dataset=nm, protocol=prot,
                                rmse_nn=m["rmse"], slope_nn=m["slope"], bias_nn=m["bias"],
                                rmse_free=m_nn["rmse"], slope_free=m_nn["slope"],
                                bias_free=m_nn["bias"],
                                gmin_nn=float(mdl.g.min()), gmin_free=float(mdl_nn.g.min())))

            audit_rows.append(dict(
                dataset=nm, protocol=prot, n_train=int(tr.sum()), n_test=int(te.sum()),
                cnr=e["cnr"], cnr_partial=e["cnr_partial"], gamma=cur["gamma"],
                p_quad=cur["p_quad"], baseline_leak=e["baseline_leak"],
                cos_raw_mean=float(np.mean(list(cos_raw.values()))) if cos_raw else np.nan,
                cos_cen_mean=float(np.mean(list(cos_cen.values()))) if cos_cen else np.nan,
                cos_cen_min=float(np.min(list(cos_cen.values()))) if cos_cen else np.nan,
                nonneg_frac=e["nonneg_frac"],
                rmse=m["rmse"], slope=m["slope"], bias=m["bias"]))
            print(f"{nm:4s} {prot:6s} n={tr.sum():4d} CNR={e['cnr']:6.2f} "
                  f"gamma={cur['gamma']:+7.3f} leak={e['baseline_leak']:.2f} "
                  f"cosC={audit_rows[-1]['cos_cen_min']:+.3f} "
                  f"RMSEP={m['rmse']:6.2f} (nn-free {m_nn['rmse']:6.2f})", flush=True)

    a = pd.DataFrame(audit_rows)
    n = pd.DataFrame(nn_rows)
    a.to_csv(os.path.join(OUT, "exp05c_condition_audit.csv"), index=False)
    n.to_csv(os.path.join(OUT, "exp05c_nonneg.csv"), index=False)

    a["pass"] = (a.cnr >= 10) & (a.gamma.abs() <= 1)
    summary["n_pass_conditions"] = int(a["pass"].sum())
    summary["pass_by_dataset"] = a.groupby("dataset")["pass"].sum().astype(int).to_dict()
    summary["cnr_range"] = a.groupby("dataset")["cnr"].agg(["min", "max", "median"]).round(3).to_dict("index")
    summary["gamma_range"] = a.groupby("dataset")["gamma"].agg(["min", "max", "median"]).round(4).to_dict("index")
    summary["cos_cen_range"] = a.groupby("dataset")["cos_cen_min"].agg(["min", "max", "median"]).round(3).to_dict("index")
    summary["nonneg_delta_rmse"] = float((n.rmse_free - n.rmse_nn).median())
    summary["nonneg_worse"] = int((n.rmse_free > n.rmse_nn + 1e-9).sum())
    with io.open(os.path.join(OUT, "exp05c_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print(a.groupby(["dataset", "pass"]).size().to_string(), flush=True)
    print("\nnonneg free - nonneg RMSEP by dataset:", flush=True)
    print((n.rmse_free - n.rmse_nn).groupby(n.dataset).agg(["median", "min", "max"]).round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
