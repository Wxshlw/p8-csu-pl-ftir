"""exp22 —— 旋转歧义的可行带对照（回应 P1-4）。

两组分闭合的解族 x_i = s_i[(1-c_i)e1 + c_i e2] 在旋转 R(theta) 下重构质量完全相同
（E -> E R, C -> C R^-1 保持行空间），区别只在解释。因此**满足约束的旋转集合**的大小
就是旋转歧义的大小，集合内各成员给出的 c 的极差就是「同一份重构下掺入量还能取多大范围」。

条件（theta=0 为 CSU 拟合出的参照端元对）：
  D7d / D8 / D7w / D7p 的 high 协议  单基质数据集，K=2 闭合即正文报告的模型；
  D1 / D2 的 high 协议                五基质数据集，此处把基质端元合并为平均基质
                                     （pooled），旋转族只作用于 2-D {基质, 掺入物} 平面。
无标签约束（不含任何浓度信息）：
  近非负 min E(theta) >= -delta * max|E(0)|（delta=0 即严格非负）；
  闭合   c 限制在 [0,1] 网格（模型本身即闭合）；
  正尺度 s_hat > 0。
再叠加浓度标签约束 cal_rmse(theta) <= cal_rmse(0) + tau：
  tau = inf -> 完全不用标签（纯旋转歧义带）；
  tau = 0   -> 「至少和报告拟合一样好地拟合校准数据」的旋转集合。
逐测试水平报告 band = max - min 与 band 内可达最大偏置 max|median c_hat - 真值|。

产出（全部实跑）
  out/exp22_theta.csv      逐 theta：非负裕度、正尺度可行性、校准 RMSE、逐水平中位预测
  out/exp22_band.csv       条件 × delta × tau × 水平：theta 范围、band、band 内最大偏置
  out/exp22_meta.json      条件层：水平表、n、cal RMSE(0)、slack(0)、theta*、CSU 预测、门值
  out/exp22_summary.md     结论摘要（含不利结果）
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, forward as F, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
# (dataset, protocol, model form)：exact = 报告模型本身就是 K=2 闭合；pooled = 平均基质简化
CONDS = [("D7d", "high", "exact"), ("D8", "high", "exact"),
         ("D7w", "high", "exact"), ("D7p", "high", "exact"),
         ("D1", "high", "pooled"), ("D2", "high", "pooled")]
DELTAS = [0.0, 0.02, 0.10, 0.30]                      # 非负松弛：0 = 严格非负
TAUS = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]              # 标签容差（pp）；inf = 不用标签
THETA = np.round(np.arange(-89.5, 90.0, 0.5), 2)       # 度，含 0
NGRID = 401                                           # c 网格点数（0.25 pp 分辨率）
AUDIT = json.load(open(os.path.join(OUT, "exp05_audit.json")))


def level_mask(y, levels):
    y = np.asarray(y, float)
    return np.any(np.isclose(y[:, None], np.asarray(levels, float)[None, :], atol=1e-6), axis=1)


def fit_pair(name, protocol, form):
    """CSU 两端元闭解 + 该协议下的校准/测试谱、标签与 theta=0 的预测。"""
    d = data.load(name)
    X, y, wvn = d["X"], d["y"], d["wvn"]
    Xv = SG.transform(X)
    tr_lv, te_lv = design.PROTOCOLS[name][protocol]
    mtr, mte = level_mask(y, tr_lv), level_mask(y, te_lv)
    br = data.brands(name, d["groups"]) if form == "pooled" else None
    ref = unmix.fit(Xv[mtr], y[mtr], wvn, CFG,
                    brand=None if br is None else br[mtr])
    if form == "pooled":                                   # 五基质 -> 平均基质 + 掺入物
        g = np.vstack([ref.g[:-1].mean(0), ref.g[-1]])
    else:
        g = ref.g.copy()
    mdl = unmix.UnmixModel(g=g, psi=ref.psi, wvn=wvn, cfg=CFG)
    lv = sorted(set(np.round(y[mte], 6)))
    p0 = mdl.predict(Xv[mtr], n_grid=NGRID)
    p1 = mdl.predict(Xv[mte], n_grid=NGRID)
    return dict(e1=g[0].copy(), e2=g[1].copy(), psi=ref.psi.copy(), wvn=wvn,
                Xtr=Xv[mtr], ytr=y[mtr], Xte=Xv[mte], yte=y[mte], lv=lv,
                scale=float(max(np.abs(g[0]).max(), np.abs(g[1]).max())),
                cal_rmse0=float(np.sqrt(np.mean((p0["c_hat"] * 100.0 - y[mtr]) ** 2))),
                pred0={L: float(np.median(p1["c_hat"][level_mask(y[mte], [L])] * 100.0))
                       for L in lv})


def scan(pp):
    """逐 theta：无标签约束 + 校准 RMSE + 逐测试水平的中位预测。"""
    rows = []
    e1, e2 = pp["e1"], pp["e2"]
    for th in THETA:
        a, b = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        g = np.vstack([a * e1 + b * e2, -b * e1 + a * e2])
        mdl = unmix.UnmixModel(g=g, psi=pp["psi"], wvn=pp["wvn"], cfg=CFG)
        ptr = mdl.predict(pp["Xtr"], n_grid=NGRID)
        pte = mdl.predict(pp["Xte"], n_grid=NGRID)
        rec = dict(theta=th, slack=float(g.min()) / pp["scale"],
                   ok_scale=bool((ptr["s_hat"] > 1e-9).all() and (pte["s_hat"] > 1e-9).all()),
                   cal_rmse=float(np.sqrt(np.mean((ptr["c_hat"] * 100.0 - pp["ytr"]) ** 2))))
        pc = pte["c_hat"] * 100.0
        for L in pp["lv"]:
            rec[f"c{L:g}"] = float(np.median(pc[level_mask(pp["yte"], [L])]))
        rows.append(rec)
    return pd.DataFrame(rows)


def band_rows(key, form, th, pp):
    th = th.sort_values("theta").reset_index(drop=True)
    out = []
    for de in DELTAS:
        feas = th[th.ok_scale & (th.slack >= -de)].copy()
        if not len(feas):
            continue
        base = float(th.loc[th.theta == 0.0, "cal_rmse"].iloc[0]) if (th.theta == 0.0).any() \
            else float(feas.cal_rmse.min())
        for tau in TAUS:
            sub = feas if not np.isfinite(tau) else feas[feas.cal_rmse <= base + tau]
            for L in pp["lv"]:
                col = sub[f"c{L:g}"] if len(sub) else pd.Series([np.nan])
                lo, hi = float(col.min()), float(col.max())
                out.append(dict(condition=key, model=form, delta=de, tau=tau, level=L,
                                n_theta=int(len(sub)),
                                contains_zero=bool(len(sub) and (sub.theta == 0.0).any()),
                                theta_min=float(sub.theta.min()) if len(sub) else np.nan,
                                theta_max=float(sub.theta.max()) if len(sub) else np.nan,
                                lo_pp=lo, hi_pp=hi, band_pp=hi - lo,
                                max_bias_pp=float(max(abs(lo - L), abs(hi - L))),
                                csu_pp=float(pp["pred0"][L]),
                                csu_bias_pp=float(pp["pred0"][L] - L)))
    return out


def pivot(sub, metric):
    return sub.pivot_table(index="delta", columns="tau", values=metric, aggfunc="max").round(2)


T0 = time.time()


def main():
    allt, allb, meta = [], [], {}
    for name, proto, form in CONDS:
        key = f"{name}/{proto}"
        pp = fit_pair(name, proto, form)
        th = scan(pp)
        th["condition"], th["model"] = key, form
        allt.append(th)
        allb.extend(band_rows(key, form, th, pp))
        a = AUDIT[name]["endmember"]
        s0 = float(th.loc[th.theta == 0.0, "slack"].iloc[0])
        base = float(th.loc[th.theta == 0.0, "cal_rmse"].iloc[0])
        star = th.loc[th.cal_rmse.idxmin()]
        meta[key] = dict(
            model=form, levels=[float(v) for v in pp["lv"]],
            levels_cal=[float(v) for v in sorted(set(np.round(pp["ytr"], 6)))],
            n_cal=int(len(pp["ytr"])), n_test=int(len(pp["yte"])),
            cal_rmse_theta0=base, slack_theta0=s0,
            n_feas_delta0=int((th.ok_scale & (th.slack >= 0.0)).sum()),
            theta_star=float(star.theta), cal_rmse_star=float(star.cal_rmse),
            csu_pred={str(k): v for k, v in pp["pred0"].items()},
            gate_cnr=float(a["cnr"]), gate_gamma=float(AUDIT[name]["curvature"]["gamma"]))
        print(f"{key} [{form}]: {len(th)} theta in {time.time()-T0:.1f}s  "
              f"levels {pp['lv']}  cal RMSE(0)={base:.2f}  slack(0)={s0:+.6f}  "
              f"feas(delta=0)={meta[key]['n_feas_delta0']}  "
              f"theta*={meta[key]['theta_star']:.2f} (RMSE {meta[key]['cal_rmse_star']:.2f})",
              flush=True)
    tdf = pd.concat(allt, ignore_index=True)
    bdf = pd.DataFrame(allb)
    tdf.to_csv(os.path.join(OUT, "exp22_theta.csv"), index=False)
    bdf.to_csv(os.path.join(OUT, "exp22_band.csv"), index=False)
    json.dump(meta, open(os.path.join(OUT, "exp22_meta.json"), "w"), indent=1, default=float)

    lines = ["# exp22 旋转歧义可行带（P1-4）", "",
             "行 = 非负松弛 delta，列 = 标签容差 tau（inf = 完全不用浓度标签）；"
             "值 = 该条件下所有测试水平中的最坏者；"
             "tau 锚定在 theta=0（CSU 报告拟合）的校准 RMSE 上。", ""]
    for k, v in meta.items():
        sub = bdf[bdf.condition == k]
        lines += [f"## {k}  [{v['model']}]  CNR={v['gate_cnr']:.1f}  gamma={v['gate_gamma']:+.2f}",
                  f"levels {v['levels']}  n_cal {v['n_cal']}  n_test {v['n_test']}  "
                  f"cal RMSE(theta=0) = {v['cal_rmse_theta0']:.2f} pp  "
                  f"slack(theta=0) = {v['slack_theta0']:+.6f}  "
                  f"delta=0 可行 theta 数 = {v['n_feas_delta0']}  "
                  f"theta* = {v['theta_star']:.2f} deg (cal RMSE {v['cal_rmse_star']:.2f})", "",
                  "### band（pp）", pivot(sub, "band_pp").to_string(), "",
                  "### band 内可达最大偏置（pp）", pivot(sub, "max_bias_pp").to_string(), "",
                  "### CSU（theta=0）自身的偏置（pp，最坏水平）",
                  f"{sub.csu_bias_pp.abs().max():.2f}", "",
                  "### 可行 theta 数", pivot(sub, "n_theta").to_string(), "",
                  "### theta=0 是否在可行集内",
                  pivot(sub, "contains_zero").replace({1.0: "yes", 0.0: "no"}).to_string(), ""]
    txt = "\n".join(lines)
    open(os.path.join(OUT, "exp22_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
