"""exp25 —— 审稿整改：预指定比较器逐条检验、复合终点、层级（来源级）相关、簇数。

输出（写入 out/）
  exp25_prespecified.csv    每个条件 × 每个比较器：比值、bootstrap p、Holm（预指定 4 个 / 全部 8 个）
  exp25_condition_counts.csv 每个条件的测试谱数、物理样品数（=聚簇数）
  exp25_endpoint.csv         复合终点：门 + 端元形状 + 校准区间宽度，按数据集
  exp25_endpoint.md          同上，可读版
  exp25_hierarchy.json       来源级（5 个测量情境）Spearman、留一来源、按来源聚簇 bootstrap
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unmix_ftir import stats

OUT = os.path.join(ROOT, "out")
PRED = os.path.join(OUT, "exp03_predictions.csv")

# 分析键 -> 正文编号（relabel 陷阱见 §8.35）
LABEL = {"D1": "D1", "D2": "D2", "D3": "D3", "D7d": "D4", "D7p": "D5",
         "D7w": "D6", "D8": "D7"}
ORDER = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
PROTO_ORDER = ["low", "interp", "high"]

# 8 个比较器中，4 个在任何条件上都跑过（预指定集）；其余 4 个只在 D1–D3、D7 上跑过
PRE = ["PLS", "Ridge", "CLS-2pt", "MCR-ALS"]
ALL = ["PLS", "Ridge", "SVR", "RF", "HGB", "MLP", "CLS-2pt", "MCR-ALS"]


def load_pred():
    """与 analyze03 完全一致的对齐口径：spec 为谱内序号，跨 seed 取均值。"""
    d = pd.read_csv(PRED)
    d["ds"] = d.dataset.map(LABEL)
    d["spec"] = d.groupby(["dataset", "protocol", "method", "seed"]).cumcount()
    return (d.groupby(["ds", "protocol", "method", "group", "y", "spec"],
                      as_index=False)
             .agg(pred=("pred", "mean"), n_seeds=("seed", "nunique")))


def prespecified(df):
    rows = []
    for (ds, proto), g in df.groupby(["ds", "protocol"], sort=False):
        base = g[g.method == "CSU-PL"].set_index(["group", "spec"])
        raw, ratio, metrics = {}, {}, {}
        for meth in sorted(m for m in g.method.unique() if m != "CSU-PL"):
            gm = g[g.method == meth].set_index(["group", "spec"])
            common = base.index.intersection(gm.index)
            y = gm.loc[common, "y"].to_numpy(float)
            p = gm.loc[common, "pred"].to_numpy(float)
            clusters = np.asarray(common.get_level_values("group"))
            d = stats.paired_diff(y, base.loc[common, "pred"].to_numpy(float), p,
                                  clusters, n_boot=1000, metric="rmse", relative=True)
            raw[meth] = d
            ratio[meth] = (stats.metrics(y, base.loc[common, "pred"].to_numpy(float))["rmse"]
                           / stats.metrics(y, p)["rmse"])
            metrics[meth] = (stats.metrics(y, p)["rmse"], len(y), len(set(clusters)))
        have = sorted(raw)
        p_all = stats.holm([raw[m]["p"] for m in have])
        pre_have = [m for m in PRE if m in raw]
        p_pre = (stats.holm([raw[m]["p"] for m in pre_have]) if pre_have else np.array([]))
        for i, m in enumerate(have):
            rmse_m, n, ncl = metrics[m]
            rows.append(dict(
                dataset=ds, protocol=proto, comparator=m,
                comp=rmse_m, ratio=ratio[m],
                p_raw=raw[m]["p"], p_holm_all=float(p_all[i]),
                p_holm_pre=(float(p_pre[pre_have.index(m)]) if m in pre_have else np.nan),
                n=n, n_clusters=ncl))
    return pd.DataFrame(rows)


def counts(df):
    rows = []
    for ds in ORDER:
        for proto in PROTO_ORDER:
            sub = df[(df.ds == ds) & (df.protocol == proto) & (df.method == "CSU-PL")]
            if sub.empty:
                continue
            rows.append(dict(dataset=ds, protocol=proto, n_spectra=int(len(sub)),
                             n_samples=int(sub.group.nunique()),
                             n_levels=int(sub.y.nunique())))
    return pd.DataFrame(rows)


# 表 1 与 out/exp05_audit.json 一致；这里直接读 json，避免手抄
def audit_table():
    a = json.load(open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8"))
    em = pd.read_csv(os.path.join(OUT, "exp05c_endmember_full.csv"))
    out = {}
    for key, lab in LABEL.items():
        v = a[key]
        cosmin = float(em[em.dataset == key].cos_centered.min())
        out[lab] = dict(cnr=float(v["endmember"]["cnr"]),
                        gamma=float(v["curvature"]["gamma"]),
                        cos_min=cosmin)
    return out


def endpoint():
    aud = audit_table()
    cov = pd.read_csv(os.path.join(OUT, "exp08_summary.csv"))
    cov["ds"] = cov.dataset.map(LABEL)
    rows = []
    for lab in ORDER:
        a = aud[lab]
        gate = "pass" if (a["cnr"] >= 10 and abs(a["gamma"]) <= 1) else "fail"
        shp = "ok" if a["cos_min"] >= 0.90 else "bad"
        row = dict(dataset=lab, cnr=a["cnr"], gamma=a["gamma"],
                   cos_min=a["cos_min"], gate=gate, shape_chk=shp)
        for proto in PROTO_ORDER:
            r = cov[(cov.ds == lab) & (cov.protocol == proto)]
            if len(r):
                r = r.iloc[0]
                # 95 % 校准区间宽度（分数）-> pp；与测试跨度比较
                span = float(sub_span(lab, proto))
                row[f"w95_{proto}"] = float(r.width95f) * 100.0
                row[f"span_{proto}"] = span
        rows.append(row)
    d = pd.DataFrame(rows)
    d["verdict"] = np.where(d.gate.eq("fail"), "fail",
                            np.where(d["shape_chk"].eq("bad"), "partial", "pass"))
    return d


def sub_span(lab, proto):
    """测试浓度跨度（pp），由预测表反推，避免另存一份切分。"""
    key = {v: k for k, v in LABEL.items()}[lab]
    d = pd.read_csv(PRED)
    m = (d.dataset == key) & (d.protocol == proto) & (d.method == "CSU-PL")
    y = d[m].y
    return float(y.max() - y.min()) if len(y) else np.nan


def hierarchy():
    """来源层级：5 个测量情境（KAN 仓库 / 肉桂释放 / 姜黄释放 / E1 / E2）。"""
    best = pd.read_csv(os.path.join(OUT, "exp03_vs_best.csv"))
    best["ds"] = best.dataset.map(LABEL)
    err = best.groupby("ds").best_rmse.median()
    aud = audit_table()
    cnr = {k: v["cnr"] for k, v in aud.items()}
    ext = json.load(open(os.path.join(OUT, "exp24_ext_audit.json"), encoding="utf-8"))
    ext_sum = pd.read_csv(os.path.join(OUT, "exp24_ext_summary.csv"))

    # 外部系统的“最优比较器误差”：确定性 4 个里的最小 RMSEP
    names = {("E1", "Sudan IV"): "E1", ("E2", "corn"): "E2a",
             ("E2", "wheat"): "E2b", ("E2", "soybean"): "E2c"}
    ext_err = {}
    for (raw, cond), lab in names.items():
        s = ext_sum[(ext_sum.dataset == raw) & (ext_sum.condition == cond)]
        if len(s):
            ext_err[lab] = float(s.best_rmse.median())
    groups = {"KAN": ["D1", "D2", "D3"], "cinnamon": ["D4", "D5", "D6"],
              "turmeric": ["D7"]}
    pts = []
    for src, ds in groups.items():
        c = np.median([cnr[k] for k in ds])
        e = np.median([err[k] for k in ds])
        pts.append(dict(source=src, cnr=float(c), err=float(e), n=len(ds)))
    pts.append(dict(source="E1", cnr=float(ext["E1_SudanIV"]["cnr"]),
                    err=ext_err.get("E1", np.nan), n=1))
    e2 = [float(ext[k]["cnr"]) for k in ("E2_corn", "E2_wheat", "E2_soybean")]
    e2e = [ext_err[k] for k in ("E2a", "E2b", "E2c") if k in ext_err]
    pts.append(dict(source="E2", cnr=float(np.median(e2)),
                    err=float(np.median(e2e)) if e2e else np.nan, n=3))
    p = pd.DataFrame(pts)
    ok = p[np.isfinite(p.err)]
    rho, pval = sps_spearman(ok.cnr.to_numpy(), ok.err.to_numpy())
    loo = {}
    for i in range(len(ok)):
        m = np.ones(len(ok), bool)
        m[i] = False
        r, _ = sps_spearman(ok.cnr.to_numpy()[m], ok.err.to_numpy()[m])
        loo[ok.source.iloc[i]] = float(r)
    return dict(points=p.to_dict("records"), rho=float(rho), p=float(pval),
                n=int(len(ok)), loo=loo)


def sps_spearman(a, b):
    from scipy import stats as s
    r = s.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def main():
    df = load_pred()
    ps = prespecified(df)
    ps.to_csv(os.path.join(OUT, "exp25_prespecified.csv"), index=False)
    counts(df).to_csv(os.path.join(OUT, "exp25_condition_counts.csv"), index=False)
    ep = endpoint()
    ep.to_csv(os.path.join(OUT, "exp25_endpoint.csv"), index=False)
    hi = hierarchy()
    json.dump(hi, open(os.path.join(OUT, "exp25_hierarchy.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print("== prespecified (Holm over PLS/Ridge/CLS-2pt/MCR-ALS) ==")
    for (ds, proto), s in ps.groupby(["dataset", "protocol"], sort=False):
        pre = s[s.comparator.isin(PRE)]
        nwin = int(((pre.p_holm_pre < 0.05) & (pre.ratio < 1)).sum())
        nloss = int(((pre.p_holm_pre < 0.05) & (pre.ratio > 1)).sum())
        print(f"{ds:>3} {proto:<7} ratio " + " ".join(
            f"{r.comparator}={r.ratio:.2f}(p{r.p_holm_pre:.3f})" for r in pre.itertuples())
            + f"  | win {nwin} loss {nloss}")
    print("\n== endpoint ==")
    print(ep[["dataset", "cnr", "gamma", "cos_min", "gate", "shape_chk", "verdict"]].to_string(index=False))
    print("\n== hierarchy ==")
    print(json.dumps({k: hi[k] for k in ("rho", "p", "n", "loo")}, indent=1))


if __name__ == "__main__":
    main()
