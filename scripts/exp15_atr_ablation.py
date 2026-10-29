"""exp15 —— ATR 穿透深度校正前后消融（方案 §4.2 与 §七 风险表中的 🔴 项）。

两个预处理变体
  sg1       : SG 一阶导（正文 §3.2 的现行设置，结果已在 exp03 / exp09 中）
  atr_sg1   : 先按 d_p(ν̃) ∝ ν̃⁻¹ 做穿透深度校正（A ← A·ν̃/median ν̃），再 SG 一阶导
              —— 物理上"先校正再标定"，所以校正作用在**原始吸光度**上，
              而不是用 UnmixConfig.atr_correct 作用在导数谱上。

产出（全部为实跑值，可复算）
  out/exp15_audit.csv        7 数据集 × 2 变体：CNR / CNR_eff / γ / leak / 门判定
  out/exp15_predictions.csv  atr_sg1 变体的逐谱预测（长表，与 exp03/exp09 同构）
  out/exp15_conditions.csv   21 条件 × 方法 × 2 变体：RMSEP / slope / bias
  out/exp15_summary.md       门是否翻转 + 误差与斜率的变化幅度
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import baselines, data, design, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
METHODS_SMALL = ["PLS", "Ridge", "CLS-2pt", "MCR-ALS"]          # D7 与原实验一致
SEEDS = {"MLP": [0, 1, 2, 3, 4], "RF": [0, 1, 2], "HGB": [0, 1, 2]}


def preprocess(X, wvn, variant):
    if variant == "sg1":
        return SG.transform(X)
    if variant == "atr_sg1":
        return SG.transform(F.atr_depth_correct(X, wvn))
    raise KeyError(variant)


def methods_for(nm):
    return METHODS_SMALL if nm.startswith("D7") else baselines.ALL


def seeds_for(meth):
    return SEEDS.get(meth, [0])


# ---------------------------------------------------------------------------
# 审计量（全参考集，与 exp05 同一定义）
# ---------------------------------------------------------------------------
def audit_table():
    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        for variant in ("sg1", "atr_sg1"):
            Xv = preprocess(X, wvn, variant)
            mdl = unmix.fit(Xv, y, wvn, CFG, brand=br)
            e = diagnostics.endmember_audit(mdl, Xv, y)
            c = diagnostics.curvature_diagnostic(y, Xv)
            cnr, gam = e["cnr"], c["gamma"]
            rows.append(dict(dataset=nm, variant=variant, n=int(X.shape[0]), N=int(X.shape[1]),
                             cnr=cnr, cnr_eff=e["cnr_partial"], gamma=gam,
                             leak=e["baseline_leak"], sigma=e["sigma"],
                             train_rmse=float(np.sqrt(np.mean(
                                 (mdl.predict_c(Xv, brand=br) * 100 - y) ** 2))),
                             gate=bool(cnr >= 10 and abs(gam) <= 1)))
            print(f"[audit] {nm:4s} {variant:8s} CNR={cnr:6.2f} CNR_eff={e['cnr_partial']:6.2f} "
                  f"gamma={gam:+.3f} leak={e['baseline_leak']:.3f} "
                  f"gate={'pass' if rows[-1]['gate'] else 'fail'}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp15_audit.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 21 条件重跑（只跑校正变体；未校正结果直接沿用 exp03 / exp09）
# ---------------------------------------------------------------------------
def run_corrected():
    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        Xv = preprocess(X, wvn, "atr_sg1")
        for prot in PROTOCOLS:
            tr_lv, te_lv = design.protocol_split(nm, prot)
            tr, te = np.isin(y, tr_lv), np.isin(y, te_lv)
            t0 = time.time()
            # data.brands 对 D7/D8 返回常数 '0' -> 单基质模型，与 exp03 / exp09 口径一致
            mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
            p = mdl.predict_c(Xv[te], brand=br[te]) * 100
            for i, idx in enumerate(np.where(te)[0]):
                rows.append(dict(dataset=nm, protocol=prot, method="CSU-PL", seed=0,
                                 group=str(groups[idx]), y=float(y[idx]), pred=float(p[i])))
            m = stats.metrics(y[te], p)
            print(f"[run] {nm:4s} {prot:6s} CSU-PL rmse={m['rmse']:6.2f} "
                  f"slope={m['slope']:5.2f} bias={m['bias']:+6.2f}", flush=True)
            for meth in methods_for(nm):
                for sd in seeds_for(meth):
                    q = baselines.fit_predict(meth, Xv[tr], y[tr], groups[tr], Xv[te], wvn, seed=sd)
                    for i, idx in enumerate(np.where(te)[0]):
                        rows.append(dict(dataset=nm, protocol=prot, method=meth, seed=sd,
                                         group=str(groups[idx]), y=float(y[idx]),
                                         pred=float(q[i])))
                    mm = stats.metrics(y[te], q)
                    print(f"       {meth:8s} s={sd} rmse={mm['rmse']:6.2f} "
                          f"slope={mm['slope']:5.2f} bias={mm['bias']:+6.2f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp15_predictions.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 与原（未校正）结果对照
# ---------------------------------------------------------------------------
def load_uncorrected():
    a = pd.read_csv(os.path.join(OUT, "exp03_predictions.csv"))
    b = pd.read_csv(os.path.join(OUT, "exp09_d7_predictions.csv"))
    return pd.concat([a, b], ignore_index=True)


def summarise(pred_new, pred_old):
    def agg(df):
        out = []
        for (nm, prot, meth), g in df.groupby(["dataset", "protocol", "method"]):
            per = [stats.metrics(gs["y"].values, gs["pred"].values)
                   for _, gs in g.groupby("seed")]
            per = pd.DataFrame(per)
            out.append(dict(dataset=nm, protocol=prot, method=meth,
                            rmse=float(per["rmse"].mean()), slope=float(per["slope"].mean()),
                            bias=float(per["bias"].mean())))
        return pd.DataFrame(out)

    old, new = agg(pred_old), agg(pred_new)
    m = old.merge(new, on=["dataset", "protocol", "method"], suffixes=("_sg1", "_atr"))
    m.to_csv(os.path.join(OUT, "exp15_conditions.csv"), index=False)

    def best_baseline(t, col):
        b = t[t.method != "CSU-PL"].groupby(["dataset", "protocol"])[col].idxmin()
        return t.loc[b].set_index(["dataset", "protocol"])

    lines = ["# ATR 穿透深度校正前后消融（exp15，全部为实跑值）", ""]
    aud = pd.read_csv(os.path.join(OUT, "exp15_audit.csv"))
    aud["gate"] = aud["gate"].astype(str).str.strip().str.lower().eq("true")
    w = aud.pivot(index="dataset", columns="variant",
                  values=["cnr", "gamma", "cnr_eff", "leak", "gate"])
    lines += ["## 审计量", "",
              "| 数据集 | CNR sg1 | CNR 校正后 | γ sg1 | γ 校正后 | 门 sg1 | 门 校正后 |",
              "|---|---|---|---|---|---|---|"]
    for nm in DATASETS:
        r = w.loc[nm]
        lines.append(f"| {nm} | {r[('cnr','sg1')]:.1f} | {r[('cnr','atr_sg1')]:.1f} | "
                     f"{r[('gamma','sg1')]:+.2f} | {r[('gamma','atr_sg1')]:+.2f} | "
                     f"{'pass' if r[('gate','sg1')] else 'fail'} | "
                     f"{'pass' if r[('gate','atr_sg1')] else 'fail'} |")
    flip = int((w[("gate", "sg1")].astype(bool) != w[("gate", "atr_sg1")].astype(bool)).sum())
    lines += ["", f"门判定翻转的数据集数：**{flip} / 7**", ""]

    csu = m[m.method == "CSU-PL"].copy()
    bb_old = best_baseline(m, "rmse_sg1")
    bb_new = best_baseline(m, "rmse_atr")
    csu["best_sg1"] = [bb_old.loc[(r.dataset, r.protocol), "rmse_sg1"] for r in csu.itertuples()]
    csu["best_name_sg1"] = [bb_old.loc[(r.dataset, r.protocol), "method"] for r in csu.itertuples()]
    csu["best_atr"] = [bb_new.loc[(r.dataset, r.protocol), "rmse_atr"] for r in csu.itertuples()]
    csu["best_name_atr"] = [bb_new.loc[(r.dataset, r.protocol), "method"] for r in csu.itertuples()]
    csu["ratio_sg1"] = csu.rmse_sg1 / csu.best_sg1
    csu["ratio_atr"] = csu.rmse_atr / csu.best_atr
    gate = aud[aud.variant == "sg1"].set_index("dataset").gate.to_dict()
    csu["gate"] = [gate[n] for n in csu.dataset]
    lines += ["## CSU-PL 逐条件（校正前 → 校正后）", "",
              "| 数据集 | 协议 | 门 | RMSEP sg1 | RMSEP 校正后 | Δ | slope sg1 | slope 校正后 | 比值 sg1 | 比值 校正后 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in csu.itertuples():
        lines.append(f"| {r.dataset} | {r.protocol} | {'pass' if r.gate else 'fail'} | "
                     f"{r.rmse_sg1:.2f} | {r.rmse_atr:.2f} | {r.rmse_atr-r.rmse_sg1:+.2f} | "
                     f"{r.slope_sg1:+.2f} | {r.slope_atr:+.2f} | {r.ratio_sg1:.2f} | "
                     f"{r.ratio_atr:.2f} |")
    ins = csu[csu.gate]
    lines += ["",
              f"门内 9 条件：中位 RMSEP {ins.rmse_sg1.median():.2f} → {ins.rmse_atr.median():.2f}；"
              f"中位比值（CSU/最强基线）{ins.ratio_sg1.median():.2f} → {ins.ratio_atr.median():.2f}",
              f"门外 12 条件：中位 RMSEP {csu[~csu.gate].rmse_sg1.median():.2f} → "
              f"{csu[~csu.gate].rmse_atr.median():.2f}；"
              f"中位比值 {csu[~csu.gate].ratio_sg1.median():.2f} → "
              f"{csu[~csu.gate].ratio_atr.median():.2f}",
              f"全 21 条件 |ΔRMSEP| 中位数 { (csu.rmse_atr-csu.rmse_sg1).abs().median():.2f} pp，"
              f"最大 { (csu.rmse_atr-csu.rmse_sg1).abs().max():.2f} pp", ""]
    lines += ["## 基线族整体（同一校正下重跑）", "",
              "| 方法 | 21 条件中位 RMSEP sg1 | 中位 RMSEP 校正后 | 中位 |Δ| |", "|---|---|---|---|"]
    for meth in sorted(m.method.unique()):
        t = m[m.method == meth]
        lines.append(f"| {meth} | {t.rmse_sg1.median():.2f} | {t.rmse_atr.median():.2f} | "
                     f"{ (t.rmse_atr-t.rmse_sg1).abs().median():.2f} |")
    io.open(os.path.join(OUT, "exp15_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    csu.to_csv(os.path.join(OUT, "exp15_csu_vs_best.csv"), index=False)
    print("\n".join(lines))


def main():
    audit_table()
    new = run_corrected()
    summarise(new, load_uncorrected())


if __name__ == "__main__":
    main()
