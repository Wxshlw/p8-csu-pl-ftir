"""exp13 —— 独立参考集上的审计稳定性与外推预测力。

回应审稿人最可能提出的质疑：Table 1 的 CNR/γ 用了包含测试浓度的全量程数据，
是否存在信息泄漏？本实验把参考集与评估集做成**物理上不相交**的两半：

* D1–D3（5 个品牌）：按品牌划分。A = 2 个品牌（全量程参考），B = 其余 3 个品牌
  （校准 + 外推测试）。审计只在 A 上算，误差只在 B 上算。这是"用参考品牌预测
  新品牌"的真实迁移场景。
* D7/D8（单一品牌）：按浓度分层把物理样品对半分。A 覆盖全量程，只算审计；
  B 做校准与外推测试。

对每个数据集重复 5 次，记录审计量、B 的 RMSEP/斜率、以及与全量数据审计的
分类一致性。

输出 out/exp13_audit_transfer.csv, out/exp13_summary.json, out/exp13_summary.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")

from unmix_ftir import data, design, diagnostics as D, forward as F, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan,
                        n_iter=40, atr_correct=False)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
REPEATS = 5
RNG = np.random.default_rng(20261024)


def split_by_brand(br, repeats=REPEATS):
    """D1–D3：把品牌随机分成 2 / 3 两组，A 为参考组。"""
    brands = np.array(sorted(set(br)))
    out = []
    combos = []
    n = len(brands)
    for mask in range(1, 1 << n):
        a = [brands[i] for i in range(n) if mask & (1 << i)]
        if len(a) == 2:
            combos.append(a)
    RNG.shuffle(combos)
    for a in combos[:repeats]:
        out.append((np.array(a, dtype=object), np.array([b for b in brands if b not in a], dtype=object)))
    return out


def split_by_sample(y, groups, repeats=REPEATS):
    """单品牌：在每个浓度水平内按物理样品对半分。"""
    levels = np.unique(np.round(y, 8))
    out = []
    for _ in range(repeats):
        A = set()
        B = set()
        for v in levels:
            gv = np.array(sorted(set(groups[np.abs(y - v) < 1e-8])))
            RNG.shuffle(gv)
            half = len(gv) // 2
            A.update(gv[:max(half, 1)])
            B.update(gv[max(half, 1):])
        out.append((A, B))
    return out


def audit_subset(Xv, y, br, wvn):
    m = unmix.fit(Xv, y, wvn, CFG, brand=br)
    e = D.endmember_audit(m, Xv, y)
    c = D.curvature_diagnostic(y, Xv)
    return m, e, c


def main():
    full = json.load(open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8"))
    rows = []
    for nm in DATASETS:
        d = data.load(nm)
        X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
        br = data.brands(nm, groups)
        Xv = SG.transform(X)
        brands = sorted(set(br))
        splits = (split_by_brand(br) if len(brands) >= 3
                  else split_by_sample(y, groups))
        n_full = dict(cnr=full[nm]["endmember"]["cnr"],
                      cnr_eff=full[nm]["endmember"]["cnr_partial"],
                      gamma=full[nm]["curvature"]["gamma"])
        for rep, (A, B) in enumerate(splits):
            if len(brands) >= 3:
                ia = np.array([b in set(A) for b in br])
            else:
                ia = np.array([g in A for g in groups])
            ib = ~ia
            if ia.sum() < 20 or ib.sum() < 20:
                continue
            try:
                _, e, c = audit_subset(Xv[ia], y[ia], br[ia], wvn)
            except Exception as exc:
                print(nm, rep, "audit failed:", exc, flush=True)
                continue
            for prot in PROTOCOLS:
                tr_lv, te_lv = design.protocol_split(nm, prot)
                tr = np.isin(y, tr_lv) & ib
                te = np.isin(y, te_lv) & ib
                if tr.sum() < 10 or te.sum() < 5:
                    continue
                try:
                    m = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
                    p = m.predict_c(Xv[te], brand=br[te]) * 100
                except Exception as exc:
                    print(nm, rep, prot, "fit failed:", exc, flush=True)
                    continue
                rmse = float(np.sqrt(np.mean((p - y[te]) ** 2)))
                slope = float(np.polyfit(y[te], p, 1)[0]) if len(set(y[te])) > 1 else np.nan
                rows.append(dict(
                    dataset=nm, repeat=rep, protocol=prot,
                    n_ref=int(ia.sum()), n_eval=int(ib.sum()),
                    n_train=int(tr.sum()), n_test=int(te.sum()),
                    cnr_ref=float(e["cnr"]), cnr_eff_ref=float(e["cnr_partial"]),
                    gamma_ref=float(c["gamma"]), leak_ref=float(e["baseline_leak"]),
                    cnr_full=float(n_full["cnr"]), cnr_eff_full=float(n_full["cnr_eff"]),
                    gamma_full=float(n_full["gamma"]),
                    pass_ref=int(e["cnr"] >= 10 and abs(c["gamma"]) <= 1),
                    pass_full=int(n_full["cnr"] >= 10 and abs(n_full["gamma"]) <= 1),
                    rmse=rmse, slope=slope, bias=float(np.mean(p - y[te]))))
                print(f"{nm:3s} r{rep} {prot:6s} CNR {e['cnr']:6.1f} "
                      f"g {c['gamma']:+6.2f} -> RMSEP {rmse:6.2f} slope {slope:+5.2f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp13_audit_transfer.csv"), index=False)
    if df.empty:
        return

    # 数据集层：重复 × 协议取中位数，做精确 7! 置换相关
    g = df.groupby("dataset").agg(
        cnr_ref=("cnr_ref", "median"), gamma_ref=("gamma_ref", "median"),
        rmse=("rmse", "median"), slope=("slope", "median"),
        pass_rate=("pass_ref", "mean"), pass_full=("pass_full", "first"),
        cnr_full=("cnr_full", "first"), gamma_full=("gamma_full", "first")).reset_index()
    g["sdist"] = (1 - g.slope).abs()

    def exact_perm(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        obs = float(stats.spearmanr(x, y).statistic)
        cnt = 0
        n = len(x)
        for p in __import__("itertools").permutations(range(n)):
            if abs(stats.spearmanr(x[list(p)], y).statistic) >= abs(obs) - 1e-12:
                cnt += 1
        return obs, cnt / __import__("math").factorial(n)

    corr = {}
    for x, yv, lab in [("cnr_ref", "rmse", "CNR_ref vs RMSEP"),
                       ("gamma_ref", "sdist", "|gamma_ref| vs |1-slope|")]:
        xx = np.abs(g[x].values) if x == "gamma_ref" else g[x].values
        corr[lab] = exact_perm(xx, g[yv].values)

    summary = dict(
        n_rows=len(df),
        corr={k: dict(rho=float(v[0]), p_exact=float(v[1])) for k, v in corr.items()},
        agreement=dict(
            n=int(len(df)),
            agree=int((df.pass_ref == df.pass_full).sum()),
            rate=float((df.pass_ref == df.pass_full).mean())),
        per_dataset=g.to_dict(orient="records"))
    with open(os.path.join(OUT, "exp13_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)

    lines = ["# 独立参考集审计（exp13）", "",
             "| Dataset | CNR_ref (median) | CNR_full | γ_ref (median) | γ_full | pass rate | RMSEP (median) | slope (median) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in g.sort_values("cnr_ref").iterrows():
        lines.append(f"| {r.dataset} | {r.cnr_ref:.1f} | {r.cnr_full:.1f} | "
                     f"{r.gamma_ref:+.2f} | {r.gamma_full:+.2f} | {r.pass_rate:.2f} | "
                     f"{r.rmse:.2f} | {r.slope:+.2f} |")
    lines += ["", f"分类一致率：{summary['agreement']['agree']}/{summary['agreement']['n']} "
              f"({summary['agreement']['rate']:.2f})", "", "相关性（数据集层，精确 7! 置换）："]
    for k, v in summary["corr"].items():
        lines.append(f"- {k}: ρ = {v['rho']:.3f}, p = {v['p_exact']:.4f}")
    open(os.path.join(OUT, "exp13_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
