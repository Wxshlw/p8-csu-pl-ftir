"""exp19 —— 谱网格敏感性（P1-6）：审计量与外推结论是否依赖 1762 / 2523 点网格。

背景
----
七套数据的通道数分成两族：KAN trio（D1–D3）1,762 点、步长 1.929 cm^-1；
D7/D8 2,523 点、步长 1.427 cm^-1。审计量 CNR 显式含 sqrt(N)，γ 与端元对比
都建立在预处理谱上，因此需要证明"通过/不通过"的分类与 RMSEP 结论不是
网格密度或 SG 窗宽的伪影。

做法
----
每个数据集在四种通道配置上重算：(a) 原生网格 + 原生 SG 窗 15；
(b) 交叉网格（KAN -> D7 细轴，D7/D8 -> KAN 粗轴）+ **物理宽度匹配**的 SG 窗；
(c) 交叉网格 + 固定窗 15（隔离"窗宽"与"网格"两个因素）；
(d) 原生网格隔点抽取 + 匹配窗（只改通道密度，不改插值）。
每一配置重算 CNR / γ / 基线泄漏 / 端元余弦，并跑 P-out-high 协议比较 RMSEP。
分类是否翻转以原生配置为基准。

输出
----
out/exp19_grid.csv        每 数据集 x 配置 一行
out/exp19_grid_rank.csv   每 配置 的 7 数据集秩相关（CNR~RMSEP）与分类一致数
out/exp19_grid_meta.json  轴配对、点数、窗宽
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
DATASETS = ["D1", "D2", "D3", "D8", "D7w", "D7p", "D7d"]
GATE_CNR, GATE_GAMMA = 10.0, 1.0


def resample(X, src, tgt):
    """逐谱线性插值到目标波数轴。src 必须升序，tgt 必须落在 src 覆盖范围内。"""
    X = np.asarray(X, float)
    tgt = np.asarray(tgt, float)
    return np.vstack([np.interp(tgt, src, X[i]) for i in range(X.shape[0])])


def match_window(src, tgt, base=15):
    """保持物理宽度（厘米^-1）不变的 SG 窗：w = base * dnu_src / dnu_tgt，取奇数。"""
    d_src = float(np.median(np.diff(src)))
    d_tgt = float(np.median(np.diff(tgt)))
    w = int(round(base * d_src / d_tgt))
    if w % 2 == 0:
        w += 1
    return max(5, w)


def decimate(X, wvn, step=2):
    return X[:, ::step], np.asarray(wvn, float)[::step]


def evaluate(nm, Xv, y, wvn, br, window):
    """全数据拟合 -> 审计量；P-out-high 用**仅训练段**重新拟合 -> RMSEP。"""
    mdl = unmix.fit(Xv, y, wvn, CFG, brand=br)
    cur = diagnostics.curvature_diagnostic(y, Xv)
    e = diagnostics.endmember_audit(mdl, Xv, y)
    tr_lv, te_lv = design.protocol_split(nm, "high")
    tr = np.isin(y, tr_lv)
    te = np.isin(y, te_lv)
    mdl_tr = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
    p = mdl_tr.predict_c(Xv[te], brand=br[te]) * 100.0
    m = stats.metrics(y[te], p)
    return dict(
        N=int(Xv.shape[1]), window=int(window),
        gamma=cur["gamma"], p_quad=cur["p_quad"],
        cnr=e["cnr"], cnr_partial=e["cnr_partial"], baseline_leak=e["baseline_leak"],
        cos_qd_qgm=e["cos_qd_qgm"], cos_gm_ga=e["cos_gm_ga"], sigma=e["sigma"],
        rmse_high=m["rmse"], slope_high=m["slope"], bias_high=m["bias"],
        n_high_train=int(tr.sum()), n_high_test=int(te.sum()),
    )


def main():
    d_ref = data.load("D1")
    coarse = np.asarray(d_ref["wvn"], float)
    fine_all = np.asarray(data.load("D7w")["wvn"], float)
    fine = fine_all[(fine_all >= coarse.min()) & (fine_all <= coarse.max())]
    meta = dict(coarse=dict(n=len(coarse), lo=float(coarse.min()), hi=float(coarse.max()),
                            dnu=float(np.median(np.diff(coarse)))),
                fine=dict(n=len(fine), lo=float(fine.min()), hi=float(fine.max()),
                          dnu=float(np.median(np.diff(fine)))),
                variants={})

    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        wvn = np.asarray(wvn, float)
        target = coarse if nm in ("D7w", "D7p", "D7d", "D8") else fine
        tgt = target[(target >= wvn.min()) & (target <= wvn.max())]
        w_match = match_window(wvn, tgt)
        Xd, wvnd = decimate(X, wvn, 2)
        w_dec = match_window(wvn, wvnd)

        variants = [
            ("native", X, wvn, 15),
            ("cross_match", resample(X, wvn, tgt), tgt, w_match),
            ("cross_fixed15", resample(X, wvn, tgt), tgt, 15),
            ("decim2_match", Xd, wvnd, w_dec),
        ]
        meta["variants"][nm] = dict(
            cross_n=int(len(tgt)), cross_window=int(w_match), decim_window=int(w_dec),
            native_n=int(X.shape[1]), native_dnu=float(np.median(np.diff(wvn))),
            cross_dnu=float(np.median(np.diff(tgt))))
        print(f"\n=== {nm}: native N={X.shape[1]} dnu={np.median(np.diff(wvn)):.3f}; "
              f"cross N={len(tgt)} dnu={np.median(np.diff(tgt)):.3f} w={w_match}; "
              f"decim N={Xd.shape[1]} w={w_dec} ===", flush=True)

        for vname, Xv_raw, wv, window in variants:
            t0 = time.time()
            Xv = F.SG(window=window, poly=3, deriv=1).transform(Xv_raw)
            rec = evaluate(nm, Xv, y, wv, br, window)
            rec.update(dataset=nm, variant=vname)
            rows.append(rec)
            print(f"  {vname:14s} N={rec['N']:4d} w={window:2d} "
                  f"CNR={rec['cnr']:6.2f} |g|={abs(rec['gamma']):5.2f} "
                  f"leak={rec['baseline_leak']:.2f} RMSEP_high={rec['rmse_high']:6.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["pass_gate"] = (df.cnr >= GATE_CNR) & (df.gamma.abs() <= GATE_GAMMA)
    base = df[df.variant == "native"].set_index("dataset")
    for c in ("cnr", "gamma", "baseline_leak", "rmse_high", "slope_high"):
        df[f"d_{c}"] = df[c] - df.dataset.map(base[c])
    df["d_rmse_rel"] = df.rmse_high / df.dataset.map(base.rmse_high) - 1.0
    df["flip"] = df.pass_gate != df.dataset.map(base.pass_gate)
    df.to_csv(os.path.join(OUT, "exp19_grid.csv"), index=False)

    rank_rows = []
    for vname, g in df.groupby("variant"):
        g = g.sort_values("dataset", kind="stable")
        rho, p = spearmanr(g.cnr, g.rmse_high)
        rho_b, p_b = spearmanr(g.cnr, g.bias_high.abs())
        rank_rows.append(dict(variant=vname, n=len(g), rho_cnr_rmse=float(rho),
                              p=float(p), rho_cnr_absbias=float(rho_b), p_bias=float(p_b),
                              n_same_class=int((~g.flip).sum()),
                              n_pass=int(g.pass_gate.sum())))
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(os.path.join(OUT, "exp19_grid_rank.csv"), index=False)
    with io.open(os.path.join(OUT, "exp19_grid_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=float)

    cols = ["dataset", "variant", "N", "window", "cnr", "gamma", "baseline_leak",
            "rmse_high", "slope_high", "d_cnr", "d_gamma", "d_rmse_rel", "pass_gate", "flip"]
    print("\n" + df[cols].round(3).to_string(index=False))
    print("\n" + rk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
