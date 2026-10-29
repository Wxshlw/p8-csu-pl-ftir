"""exp05 —— 数据/模型自检：曲率、端元物理性、品牌一致性、标签可分辨性。

只使用训练数据 + 已知浓度，输出可作为"外推是否可信"的预警指标。
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
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, diagnostics, forward as F, unmix

NAMES = ["D1", "D2", "D3", "D8", "D7w", "D7p", "D7d"]
CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)


def run():
    out = {}
    for nm in NAMES:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        Xv = SG.transform(X)
        br = data.brands(nm, groups)
        rec = dict(dataset=nm, n=int(X.shape[0]), N=int(X.shape[1]),
                   levels=sorted(set(float(v) for v in y)),
                   curvature=diagnostics.curvature_diagnostic(y, Xv),
                   resolvability=diagnostics.label_resolvability(y, Xv, groups))
        if nm in ("D1", "D2", "D3"):
            rec["brand"] = diagnostics.brand_consistency(y, Xv, br)
        # 全数据拟合（品牌感知）
        mdl = unmix.fit(Xv, y, wvn, CFG, brand=br)
        rec["endmember"] = diagnostics.endmember_audit(mdl, Xv, y)
        rec["train_rmse"] = float(np.sqrt(np.mean(
            (mdl.predict_c(Xv, brand=br) * 100 - y) ** 2)))
        # 各协议的"外推偏差预警"（只用训练段）
        rec["bias_forecast"] = {}
        if nm in design.PROTOCOLS:
            for prot in ("low", "high"):
                tr_lv, te_lv = design.protocol_split(nm, prot)
                tr = np.isin(y, tr_lv)
                rec["bias_forecast"][prot] = {
                    f"{c:g}": diagnostics.extrapolation_bias_forecast(y[tr], Xv[tr], c)
                    for c in sorted(set(te_lv))}
        out[nm] = rec
        e = rec["endmember"]
        print(f"{nm:4s} n={rec['n']:4d} N={rec['N']:4d} gamma={rec['curvature']['gamma']:+.3f} "
              f"p_quad={rec['curvature']['p_quad']:.1e} cnr={e['cnr']:.1f} "
              f"leak={e['baseline_leak']:.3f} resolv_snr={rec['resolvability']['adjacent_snr_med']:.3f} "
              f"trainRMSE={rec['train_rmse']:.2f}", flush=True)

    with io.open(os.path.join(OUT, "exp05_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)

    lines = ["# 数据/模型自检（训练集内部）", "",
             "| 数据集 | 谱数 | 水平 | 曲率 γ | 二次项 p | 端元 CNR | 基线泄漏 | 相邻水平 SNR | 训练 RMSE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for nm, r in out.items():
        c, e, v = r["curvature"], r["endmember"], r["resolvability"]
        lines.append(f"| {nm} | {r['n']} | {len(r['levels'])} | {c['gamma']:+.3f} | "
                     f"{c['p_quad']:.2e} | {e['cnr']:.1f} | {e['baseline_leak']:.3f} | "
                     f"{v['adjacent_snr_med']:.3f} | {r['train_rmse']:.2f} |")
    io.open(os.path.join(OUT, "exp05_audit.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
