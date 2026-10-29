"""make_si_md —— 由经过核验的实验输出生成 SI 源文件 manuscript/SI.md。

数据来源（全部为实际运行结果，不做任何人工改写）：
  out/exp03_predictions.csv     各条件的测试水平与样品数
  out/exp09_d7_predictions.csv  D7 三套数据的测试水平与样品数
  out/exp03_splits.json         校准水平的范围与数量
  unmix_ftir/design.py          逐数据集写死的外推协议（精确水平）
  tables/table4_uncertainty.csv 21 条件区间覆盖率与宽度
  out/exp06_robustness.csv      全部因子水平的稳健性结果
  out/exp05_audit.json          端元审计量（CNR、γ、泄漏、SNR、非负比例）
  out/exp05b_d7_permutation.json 标签可分辨性置换检验
  tables/table2_main.csv        测试谱数（与正文表 2 一致）
  out/exp22_band.csv            旋转可行带：条件 × delta × tau × 水平（S20）
  out/exp22_theta.csv           逐 theta 的非负裕度、校准 RMSE、逐水平中位预测
  out/exp22_meta.json           条件层：水平表、theta*、门值（与 exp05 断言一致）
  out/exp18_loq.csv             逐数据集 CV 膨胀因子、区间口径定量限、量程内判定（S21）
  out/exp18_loq_curve.csv       定量限搜索所用的浓度网格（S21）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unmix_ftir.design import PROTOCOLS  # noqa: E402

OUT = ROOT / "out"
SI = ROOT / "manuscript" / "SI.md"

ORDER = [
    ("D1", "high"), ("D1", "interp"), ("D1", "low"),
    ("D2", "high"), ("D2", "interp"), ("D2", "low"),
    ("D3", "high"), ("D3", "interp"), ("D3", "low"),
    ("D7d", "high"), ("D7d", "interp"), ("D7d", "low"),
    ("D7p", "high"), ("D7p", "interp"), ("D7p", "low"),
    ("D7w", "high"), ("D7w", "interp"), ("D7w", "low"),
    ("D8", "high"), ("D8", "interp"), ("D8", "low"),
]

PROT_LABEL = {"high": "high", "interp": "interp", "low": "low"}
PROT_CSV = {"high-end": "high", "interp.": "interp", "low-end": "low"}

# 展示层编号：分析键 D7d/D7p/D7w/D8 → 正文连续编号 D4/D5/D6/D7。
# 数据集在正文中按 D1–D7 连续编号，代码与 out/、tables/ 仍用分析键；
# 该映射只在写入 SI 之前做一次，故裸 D7（三个肉桂数据集的统称）→ "D4–D6"。
DS_MAP = [("D7d", "D4"), ("D7p", "D5"), ("D7w", "D6")]
DS_GROUP = "D4\u2013D6"


def relabel(text: str) -> str:
    """把分析键改写成正文编号，保持全文一致。"""
    for a, b in DS_MAP:
        text = text.replace(a, b)
    text = re.sub(r"(?<![\w])D7(?![\w])", DS_GROUP, text)
    return text.replace("D8", "D7")


def num(v: float) -> str:
    """水平列表：整数不带小数点。"""
    return f"{v:g}"


def level_list(vals) -> str:
    return ", ".join(num(v) for v in vals)


def main() -> None:
    t2 = pd.read_csv(ROOT / "tables" / "table2_main.csv")
    t2["protocol"] = t2["Protocol"].map(PROT_CSV)
    n_of = {(r.Dataset, r.protocol): int(r.n) for r in t2.itertuples()}

    pred = pd.read_csv(OUT / "exp03_predictions.csv")
    pred = pred[pred.method == "CSU-PL"]
    d7 = pd.read_csv(OUT / "exp09_d7_predictions.csv")
    d7 = d7[d7.method == "CSU-PL"]
    pred_all = pd.concat([pred[["dataset", "protocol", "group"]],
                          d7[["dataset", "protocol", "group"]]], ignore_index=True)
    groups_of = {(k[0], k[1]): int(v["group"].nunique())
                 for k, v in pred_all.groupby(["dataset", "protocol"])}

    # ---------------- S1 ----------------
    s1 = ["| Dataset | Protocol | Calibration levels (%) | Test levels (%) | n | Samples |",
          "|---|---|---|---|---|---|"]
    for ds, pr in ORDER:
        tr, te = PROTOCOLS[ds][pr]
        key = (ds, pr)
        if key not in n_of:
            raise KeyError(f"missing n for {key}")
        s1.append(
            f"| {ds} | {PROT_LABEL[pr]} | {level_list(tr)} | {level_list(te)} "
            f"| {n_of[key]} | {groups_of[key]} |"
        )

    # ---------------- S2 ----------------
    t4 = pd.read_csv(ROOT / "tables" / "table4_uncertainty.csv")
    t4["protocol"] = t4["Protocol"].map(PROT_CSV)
    t4["order"] = t4.apply(lambda r: ORDER.index((r.Dataset, r.protocol)), axis=1)
    t4 = t4.sort_values("order")
    s2 = ["| Dataset | Protocol | n | f | RMSEP (%) | cov68 raw | cov95 raw | cov68 cal | cov95 cal | w95 raw (pp) | w95 cal (pp) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t4.itertuples():
        s2.append(
            f"| {r.Dataset} | {PROT_LABEL[r.protocol]} | {int(r.n)} | {r.f:.2f} "
            f"| {r.RMSEP:.2f} | {r.cov68_raw:.2f} | {r.cov95_raw:.2f} "
            f"| {r.cov68_cal:.2f} | {r.cov95_cal:.2f} "
            f"| {r.width95_raw:.2f} | {r.width95_cal:.2f} |"
        )

    # ---------------- S3 ----------------
    rob = pd.read_csv(OUT / "exp06_robustness.csv")
    fac_order = ["noise", "scatter", "baseline", "nonlinear", "matrixvar"]
    fac_name = {
        "noise": "Noise", "scatter": "Multiplicative scattering",
        "baseline": "Baseline mismatch", "nonlinear": "Response nonlinearity",
        "matrixvar": "Matrix variability",
    }
    lev_fmt = {
        "noise": lambda v: f"{num(v)}× σ0",
        "scatter": lambda v: f"CV {num(v * 100)} %",
        "baseline": lambda v: f"{num(v * 100)} % of endmember contrast",
        "nonlinear": lambda v: f"γ = {v:.2f}",
        "matrixvar": lambda v: f"{num(v * 100)} % contrast",
    }
    s3 = ["| Factor | Level | CSU-PL | Fixed scale |", "|---|---|---|---|"]
    for fac in fac_order:
        sub = rob[rob.factor == fac]
        for val in sorted(sub.value.unique()):
            cells = []
            for meth in ("CSU-PL", "additive-only"):
                x = sub[(sub.value == val) & (sub.method == meth)]["rmse"]
                cells.append(f"{x.mean():.2f} ± {x.std(ddof=1):.2f}")
            s3.append(f"| {fac_name[fac]} | {lev_fmt[fac](val)} | {cells[0]} | {cells[1]} |")

    # ---------------- S4 ----------------
    audit = json.loads((OUT / "exp05_audit.json").read_text(encoding="utf-8"))
    perm = json.loads((OUT / "exp05b_d7_permutation.json").read_text(encoding="utf-8"))
    s4 = ["| Dataset | CNR | CNR_eff | \\|γ\\| | Leak | Train RMSE (%) | Adjacent SNR | Nonneg. fraction | Perm. R² | p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ("D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"):
        a = audit[ds]
        cnr = a["endmember"]["cnr"]
        cnre = a["endmember"]["cnr_partial"]
        gam = abs(a["curvature"]["gamma"])
        leak = a["endmember"]["baseline_leak"]
        rmse = a["train_rmse"]
        snr = a["resolvability"]["adjacent_snr_med"]
        nneg = a["endmember"]["nonneg_frac"]
        if ds in perm:
            r2 = f"{perm[ds]['observed_r2']:.3f}"
            p = f"{perm[ds]['p_value']:.3f}"
        else:
            r2 = "—"
            p = "—"
        s4.append(
            f"| {ds} | {cnr:.1f} | {cnre:.1f} | {gam:.2f} | {leak:.2f} | {rmse:.2f} "
            f"| {snr:.1f} | {nneg:.2f} | {r2} | {p} |"
        )

    # S4 说明：按效应量论证，置换 p 只作陈列（p 值取自 100 次置换，下限 1/101）。
    from unmix_ftir import data as _data  # noqa: E402

    y7 = np.asarray(_data.load("D7w")["y"], float)
    sd = float(y7.std())  # 总体标准差：即"用均值预测"时的 RMS 残差，与正文 7.63 一致
    assert all(audit[k]["curvature"]["gamma"] < 0 for k in
               ("D1", "D2", "D3", "D7w", "D7p", "D7d", "D8")), "S4 note claims gamma < 0 for all seven"
    null_means = [perm[k]["null_mean"] for k in perm]
    null_p95 = [perm[k]["null_p95"] for k in perm]

    def neg(v: float) -> str:
        return f"{v:.2f}".replace("-", "−")

    s4_note = (
        "Note: |γ| is reported; the signed values are negative for all seven datasets, so the apparent "
        "contrast decreases as the adulterant fraction increases. The informative quantity in the "
        "permutation columns is the effect size rather than the p-value. Against "
        f"labels spanning {num(perm['D7w']['label_range'])} percentage points with a standard deviation "
        f"of {sd:.2f} percentage points, the out-of-fold RMSE is {perm['D7p']['rmse']:.2f} percentage "
        f"points (R² = {perm['D7p']['observed_r2']:.3f}) for D7p, against "
        f"{perm['D7d']['rmse']:.2f} percentage points (R² = {perm['D7d']['observed_r2']:.3f}) for D7d and "
        f"{perm['D7w']['rmse']:.2f} percentage points (R² = {perm['D7w']['observed_r2']:.3f}) for D7w, "
        "the D5 value being the residual left "
        "by predicting the mean. All three p-values sit at the 1/101 floor of a 100-permutation test "
        f"because the shuffled-label null is strongly negative (null means from {neg(min(null_means))} to "
        f"{neg(max(null_means))}, null 95th percentiles from {neg(min(null_p95))} to "
        f"{neg(max(null_p95))}), so even a positive but negligible observed R² exceeds every permutation "
        "draw. The test records that the labels are not exchangeable with the spectra, not that a usable "
        "contrast exists."
    )

    # ---------------- S5 ----------------
    inf = pd.read_csv(ROOT / "tables" / "table5_inference.csv")

    def pfmt(p) -> str:
        if pd.isna(p):
            return "—"
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    def cifmt(lo, hi) -> str:
        if pd.isna(lo) or pd.isna(hi):
            return "—"
        return f"[{lo:.2f}, {hi:.2f}]"

    s5 = ["| Relation | Level | n | ρ | p | 95 % CI |",
          "|---|---|---|---|---|---|"]
    rel_map = {
        "CNR vs CSU RMSEP": "CNR vs CSU-PL RMSEP",
        "CNR vs best-comparator RMSEP": "CNR vs best-comparator RMSEP",
        "CNR vs |bias|": "CNR vs |bias|",
        "|gamma| vs |1-slope| (CSU)": "|γ| vs |1 − slope| (CSU-PL)",
        "|gamma| vs |1-slope| (best comparator)": "|γ| vs |1 − slope| (best comparator)",
        "|gamma| vs CSU RMSEP": "|γ| vs CSU-PL RMSEP",
        "CNR_eff vs CSU RMSEP": "CNR_eff vs CSU-PL RMSEP",
        "CNR vs CSU/best ratio": "CNR vs CSU-PL/best ratio",
        "CNR vs CSU RMSEP | |gamma|": "CNR vs CSU-PL RMSEP, controlling |γ|",
        "|gamma| vs |1-slope| | CNR": "|γ| vs |1 − slope|, controlling CNR",
    }
    for r in inf.itertuples():
        level = "condition" if r.unit == "condition" else "dataset"
        p = r.p_naive if r.unit == "condition" else r.p_exact
        rel = rel_map.get(str(r.relation), str(r.relation))
        rel = rel.replace("|", "\\|")
        rho = f"{r.statistic:.2f}".replace("-", "−")
        s5.append(
            f"| {rel} | {level} | {int(r.n)} | {rho} "
            f"| {pfmt(p)} | {cifmt(r.ci_lo, r.ci_hi).replace('-', '−')} |"
        )

    # ---------------- S6 ----------------
    # 三组按复合终点（exp25_endpoint.csv 的 verdict）：pass / partial / fail。
    # 逐条件的“最佳比较器”只作描述性上界（取四个预指定比较器中最小者），不给 p 值。
    pre6 = pd.read_csv(OUT / "exp25_prespecified.csv")
    ep6 = pd.read_csv(OUT / "exp25_endpoint.csv").set_index("dataset")["verdict"].to_dict()
    DISP2KEY = {"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D7d",
                "D5": "D7p", "D6": "D7w", "D7": "D8"}
    pre6["ver"] = pre6.dataset.map(ep6)
    assert pre6.ver.notna().all()
    assert pre6.groupby("dataset").ver.nunique().max() == 1
    meth_order = ["PLS", "Ridge", "SVR", "RF", "HGB", "MLP", "CLS-2pt", "MCR-ALS",
                  "Best comparator (oracle)"]
    s6 = ["| Comparator | n pass | Median ratio, pass | Better in pass (sig.) "
          "| n partial | Median ratio, partial | n fail | Median ratio, fail "
          "| Better in fail (sig.) | Worse in fail (sig.) |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def s6_cell(g):
        """一组条件的 (n, 中位比值, 更好的格数, 其中显著的格数, 更差的显著格数)。"""
        if len(g) == 0:
            return int(0), "—", None, None, None
        # oracle 行按测试误差选出，配 p 值等于以选择本身为条件，因此整组 p 缺失时不给出显著格数。
        if g.p_holm_all.isna().all():
            sig_better = sig_worse = None
        else:
            sig_better = int(((g.p_holm_all < 0.05) & (g.ratio < 1)).sum())
            sig_worse = int(((g.p_holm_all < 0.05) & (g.ratio > 1)).sum())
        return (int(len(g)), f"{float(g.ratio.median()):.2f}",
                int((g.ratio < 1).sum()), sig_better, sig_worse)

    # 逐条件“最佳比较器”= 该条件上测试 RMSEP 最小的比较器（在四个预指定比较器与
    # 额外四个学习器中取最小者），因此比值取该条件上的最大值。
    oracle = pre6.loc[pre6.groupby(["dataset", "protocol"]).comp.idxmin(),
                      ["dataset", "protocol", "ratio", "ver"]].copy()
    oracle["comparator"] = "Best comparator (oracle)"
    oracle["p_holm_all"] = np.nan
    s6_src = pd.concat([pre6, oracle], ignore_index=True)
    for m in meth_order:
        g = s6_src[s6_src.comparator == m]
        gp, gm, gf = (g[g.ver == "pass"], g[g.ver == "partial"], g[g.ver == "fail"])
        np_, mp, bp, bp_sig, _ = s6_cell(gp)
        nm, mm, _, _, _ = s6_cell(gm)
        nf, mf, bf, bf_sig, wf_sig = s6_cell(gf)
        fmt = lambda c, s: "—" if c is None else (f"{c} ({s})" if s is not None else f"{c} (—)")
        better_p = fmt(bp, bp_sig)
        better_f = fmt(bf, bf_sig)
        worse_f = "—" if wf_sig is None else str(wf_sig)
        s6.append(f"| {m} | {np_} | {mp} | {better_p} | {nm} | {mm} | {nf} | {mf} "
                  f"| {better_f} | {worse_f} |")
    pre6_p = pre6[pre6.ver == "pass"]
    pre6_f = pre6[pre6.ver == "fail"]
    p4 = pre6_p[pre6_p.comparator.isin(["PLS", "Ridge", "CLS-2pt", "MCR-ALS"])]
    f4 = pre6_f[pre6_f.comparator.isin(["PLS", "Ridge", "CLS-2pt", "MCR-ALS"])]
    assert len(p4) == 24 and len(f4) == 48, (len(p4), len(f4))
    s6_pass = dict(wins=int((p4.ratio < 1).sum()),
                   sig=int(((p4.p_holm_pre < 0.05) & (p4.ratio < 1)).sum()),
                   sig_loss=int(((p4.p_holm_pre < 0.05) & (p4.ratio > 1)).sum()),
                   med=float(p4.ratio.median()))
    s6_fail = dict(wins=int((f4.ratio < 1).sum()),
                   sig=int(((f4.p_holm_pre < 0.05) & (f4.ratio < 1)).sum()),
                   sig_loss=int(((f4.p_holm_pre < 0.05) & (f4.ratio > 1)).sum()),
                   med=float(f4.ratio.median()))
    assert (s6_pass["wins"], s6_pass["sig"], s6_pass["sig_loss"]) == (19, 11, 1), s6_pass
    assert (s6_fail["wins"], s6_fail["sig"], s6_fail["sig_loss"]) == (18, 14, 28), s6_fail
    assert abs(s6_pass["med"] - 0.861) < 5e-4 and abs(s6_fail["med"] - 1.10) < 0.006, (s6_pass, s6_fail)
    orc = s6_src[s6_src.comparator == "Best comparator (oracle)"]
    orc_med = {v: float(orc[orc.ver == v].ratio.median()) for v in ("pass", "partial", "fail")}
    assert abs(orc_med["pass"] - 0.951) < 5e-3, orc_med
    assert abs(orc_med["partial"] - 0.925) < 5e-3, orc_med
    assert abs(orc_med["fail"] - 1.595) < 5e-3, orc_med
    orc_ds = orc.groupby("dataset").ratio.median()
    assert abs(orc_ds["D1"] - 0.899) < 5e-3 and abs(orc_ds["D2"] - 1.003) < 5e-3, orc_ds
    out_ds = orc_ds[["D3", "D5", "D6", "D7"]]
    assert abs(out_ds.min() - 1.054) < 5e-3 and abs(out_ds.max() - 2.232) < 5e-3, out_ds
    s6_rf_p = float(pre6_p[pre6_p.comparator == "RF"].comp.median())
    s6_hgb_p = float(pre6_p[pre6_p.comparator == "HGB"].comp.median())
    csu_cells = (pre6_p.assign(csu=pre6_p.comp * pre6_p.ratio)
                 .groupby(["dataset", "protocol"]).csu.median())
    s6_csu_p = float(csu_cells.median())
    assert abs(s6_csu_p - 3.89) < 0.02, s6_csu_p
    per_comp = []
    for m, lab in (("PLS", "PLS"), ("Ridge", "ridge"), ("CLS-2pt", "the two-point model"),
                   ("MCR-ALS", "MCR-ALS")):
        gp = pre6_p[pre6_p.comparator == m]
        gf = pre6_f[pre6_f.comparator == m]
        per_comp.append(
            f"{lab} in {int((gp.ratio < 1).sum())} of {len(gp)} passing conditions "
            f"(median {gp.ratio.median():.2f}) and in {int((gf.ratio < 1).sum())} of "
            f"{len(gf)} failing ones (median {gf.ratio.median():.2f})")
    note6 = (
        "Note: the partial group holds a single dataset, which passes the gate, fails the "
        "endmember shape check and reaches no concentration reportable at 20 % relative "
        "precision, so its three cells carry no significance count. The four prespecified "
        "comparators were run on all 21 conditions; support-vector regression, random forest, "
        "histogram gradient boosting and the perceptron were run on D1, D2, D3 and D8 only, so "
        "their counts are six rather than 24 or 48. The last row is the per-condition best "
        "comparator: the model with the lowest test RMSEP on that condition, taken from the "
        "four prespecified comparators on D4, D5 and D6 and from those four together with the "
        "four additional learners on D1, D2, D3 and D8. It is selected on the same test error "
        "it is compared against, so it is kept only as a descriptive upper bound and its ratio "
        "cells carry a dash instead of a significance count. Aggregated per dataset its median "
        "ratio is 0.90 for D1 and 1.00 for D2 inside the passing region, and 1.05 to 2.23 for "
        "the four gated-out datasets (the same values are the median error ratios of Table S12). "
        "Over the prespecified four the "
        f"proposed estimator is better in {s6_pass['wins']} of 24 passing cells (median ratio "
        f"{s6_pass['med']:.2f}, {s6_pass['sig']} significant wins against "
        f"{s6_pass['sig_loss']} significant loss) and in {s6_fail['wins']} of 48 failing cells "
        f"(median {s6_fail['med']:.2f}, {s6_fail['sig']} significant wins against "
        f"{s6_fail['sig_loss']} significant losses). Taken comparator by comparator, it is ahead "
        "of " + "; ".join(per_comp) + ". The two ensembles are also the weakest models in the "
        f"passing group (median RMSEP {s6_rf_p:.1f} % for random forest and {s6_hgb_p:.1f} % "
        f"for gradient boosting against {s6_csu_p:.1f} % for the proposed estimator), so their "
        "low ratios reflect that weakness as much as the accuracy of the proposed estimator. The "
        "ratios are computed from predictions averaged over the recorded seeds, as in Table 2 of "
        "the main text; Table S13 panel (c) instead averages the per-seed RMSEs, which moves the "
        "perceptron's medians from 0.70 to 0.65 in the passing group and from 0.98 to 0.95 "
        "outside it."
    )

    # ---------------- S7 ----------------
    gate = pd.read_csv(ROOT / "tables" / "table7_gate_sensitivity.csv")

    def gate_row(g) -> str:
        n_pass = int(g.n_pass)
        n_fail = int(g.n_fail)
        pd_ = str(g.pass_datasets).replace("|", ", ")
        fd = str(g.fail_datasets).replace("|", ", ")
        return (f"| {g.cnr_threshold:g} | {g.gamma_threshold:g} | {n_pass} | {n_fail} "
                f"| {pd_} | {fd} | {g.pass_median:.3f} | {g.fail_median:.3f} "
                f"| {'yes' if int(g.ordered) == 1 else 'no'} |")

    s7a = ["| CNR | \\|γ\\| | n pass | n fail | Passing datasets "
           "| Gate-failing datasets | Median ratio, gate passing | Median ratio, gate failing "
           "| Complete separation |",
           "|---|---|---|---|---|---|---|---|---|"]
    sa = gate[(gate.gamma_threshold - 1.0).abs() < 1e-9]
    for val in (2, 4, 6, 8, 10, 12, 15, 20, 25):
        sub = sa[(sa.cnr_threshold - val).abs() < 1e-9]
        if len(sub) == 0:
            continue
        s7a.append(gate_row(sub.iloc[0]))

    s7b = list(s7a[:2])
    sb = gate[(gate.cnr_threshold - 10).abs() < 1e-9]
    for val in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0):
        sub = sb[(sb.gamma_threshold - val).abs() < 1e-9]
        if len(sub) == 0:
            continue
        s7b.append(gate_row(sub.iloc[0]))

    # ---------------- S8 ----------------
    prot_map = {"high-end": "high", "interp.": "interp", "low-end": "low",
                "high": "high", "interp": "interp", "low": "low"}
    ca = pd.read_csv(OUT / "exp05c_condition_audit.csv")
    ca["protocol"] = ca["protocol"].map(prot_map)
    ca["order"] = ca.apply(lambda r: ORDER.index((r.dataset, r.protocol)), axis=1)
    ca = ca.sort_values("order")
    s8a = ["| Dataset | Protocol | n train | CNR (split) | \\|γ\\| (split) | Gate |",
           "|---|---|---|---|---|---|"]
    for r in ca.itertuples():
        passed = (r.cnr >= 10) and (abs(r.gamma) <= 1)
        s8a.append(
            f"| {r.dataset} | {PROT_LABEL[r.protocol]} | {int(r.n_train)} "
            f"| {r.cnr:.1f} | {abs(r.gamma):.2f} | {'pass' if passed else 'fail'} |"
        )

    em = pd.read_csv(OUT / "exp05c_endmember_full.csv")
    s8b = ["| Dataset | Brands | Median raw cosine | Median centred cosine "
           "| Minimum centred cosine |", "|---|---|---|---|---|"]
    for ds in ("D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"):
        sub = em[em.dataset == ds]
        s8b.append(
            f"| {ds} | {sub.brand.nunique()} | {sub.cos_raw.median():.2f} "
            f"| {sub.cos_centered.median():.3f} | {sub.cos_centered.min():.3f} |"
        )

    nn = pd.read_csv(OUT / "exp05c_nonneg.csv")
    nn["prot"] = nn["protocol"].map(prot_map)
    nn["order"] = nn.apply(lambda r: ORDER.index((r.dataset, r.prot)), axis=1)
    nn = nn.sort_values("order")
    s8c = ["| Dataset | Protocol | RMSEP, non-negative (%) | RMSEP, free (%) | Δ (free − non-negative, pp) |",
           "|---|---|---|---|---|"]
    for r in nn.itertuples():
        delta = r.rmse_free - r.rmse_nn
        s8c.append(
            f"| {r.dataset} | {PROT_LABEL[r.prot]} | {r.rmse_nn:.2f} "
            f"| {r.rmse_free:.2f} | {f'{delta:+.2f}'.replace('-', '−')} |"
        )

    # ---------------- S9 ----------------
    rep = pd.read_csv(ROOT / "tables" / "table8_design_replication.csv")
    s9 = ["| Dataset | Designs | h > 1 | Median h > 1 (%) | Median rest (%) "
          "| Ratio | Max RMSEP (%) | > 2× median | AUC (leverage) | AUC (κ) "
          "| ρ (leverage) | ρ (κ) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep.itertuples():
        auc_l = "—" if pd.isna(r.auc_leverage) else f"{r.auc_leverage:.3f}"
        auc_k = "—" if pd.isna(r.auc_kappa) else f"{r.auc_kappa:.3f}"
        rho_l = f"{r.spearman_leverage:.2f}".replace("-", "−")
        rho_k = f"{r.spearman_kappa:.2f}".replace("-", "−")
        s9.append(
            f"| {r.dataset} | {int(r.n_designs)} | {int(r.n_high_leverage)} "
            f"| {r.median_rmse_high:.2f} | {r.median_rmse_rest:.2f} | {r.ratio_high_rest:.2f} "
            f"| {r.max_rmse:.2f} | {int(r.n_twofold)} | {auc_l} | {auc_k} "
            f"| {rho_l} | {rho_k} |"
        )

    # ---------------- S10 ----------------
    conf = pd.read_csv(OUT / "exp12_conformal.csv")
    s10 = ["| Dataset | Protocol | q95 (pp) | Coverage 95 % conformal | Width 95 % conformal (pp) "
           "| Coverage 95 % Gaussian | Width 95 % Gaussian (pp) | RMSEP (%) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in conf.itertuples():
        s10.append(
            f"| {r.dataset} | {PROT_LABEL[r.protocol]} | {r.q95*100:.2f} "
            f"| {r.cov_conf:.3f} | {r.width_conf*100:.2f} | {r.cov_gauss:.3f} "
            f"| {r.width_gauss*100:.2f} | {r.rmse*100:.2f} |"
        )
    conf_med = float(conf.cov_conf.median())
    conf_ok = int((conf.cov_conf >= 0.75).sum())
    gauss_ok = int((conf.cov_gauss >= 0.75).sum())
    conf_w_med = float(conf.width_conf.median() * 100)
    cal_ok = int((t4.cov95_cal >= 0.75).sum())
    cal_w_med = float(t4.width95_cal.median())
    extra = [conf[(conf.dataset == "D7w") & (conf.protocol == "interp")].bias.iloc[0] * 100,
             conf[(conf.dataset == "D7p") & (conf.protocol == "low")].bias.iloc[0] * 100]

    # ---------------- S11 ----------------
    at = pd.read_csv(OUT / "exp13_audit_transfer.csv")
    s11 = ["| Dataset | n_ref | CNR_ref (median, range) | CNR full "
           "| γ_ref (median, range) | γ full | Pass rate | RMSEP (%) | Slope |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ds in ("D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"):
        g = at[at.dataset == ds]
        row = (
            f"| {ds} | {int(g.n_ref.median())} "
            f"| {g.cnr_ref.median():.1f} ({g.cnr_ref.min():.1f}–{g.cnr_ref.max():.1f}) "
            f"| {g.cnr_full.iloc[0]:.1f} "
            f"| {g.gamma_ref.median():+.2f} ({g.gamma_ref.min():+.2f}–{g.gamma_ref.max():+.2f}) "
            f"| {g.gamma_full.iloc[0]:+.2f} | {g.pass_ref.mean():.2f} "
            f"| {g.rmse.median():.2f} | {g.slope.median():+.2f} |"
        )
        s11.append(row.replace("-", "−"))
    at_agree = int((at.pass_ref == at.pass_full).sum())
    at_n = len(at)

    # ---------------- S12 ----------------
    lodo = json.loads((OUT / "exp14_lodo.json").read_text(encoding="utf-8"))
    s12 = ["| Held-out | CNR thr. | \\|γ\\| thr. | Train bal. acc. "
           "| Median error ratio | Reference label (error-based) | Learned-gate label | Correct |",
           "|---|---|---|---|---|---|---|---|"]
    for f in lodo["folds"]:
        s12.append(
            f"| {f['held_out']} | {f['cnr_thr']:g} | {f['gamma_thr']:g} "
            f"| {f['train_bal_acc']:.3f} | {f['ratio']:.3f} "
            f"| {'pass' if f['truth'] else 'fail'} | {'pass' if f['pred'] else 'fail'} "
            f"| {'yes' if f['correct'] else 'no'} |"
        )

    # ---------------- S13 ----------------
    from scipy import stats as sps

    au = pd.read_csv(OUT / "exp15_audit.csv")
    au["gate"] = au["gate"].astype(str).str.strip().str.lower().eq("true")
    piv_cnr = au.pivot(index="dataset", columns="variant", values="cnr")
    piv_gam = au.pivot(index="dataset", columns="variant", values="gamma")
    piv_gate = au.pivot(index="dataset", columns="variant", values="gate")
    s13a = ["| Dataset | CNR (as used) | CNR (corrected) | γ (as used) | γ (corrected) "
            "| Gate (as used) | Gate (corrected) |", "|---|---|---|---|---|---|---|"]
    for ds in ("D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"):
        s13a.append(
            f"| {ds} | {piv_cnr.at[ds, 'sg1']:.1f} | {piv_cnr.at[ds, 'atr_sg1']:.1f} "
            f"| {abs(piv_gam.at[ds, 'sg1']):.2f} | {abs(piv_gam.at[ds, 'atr_sg1']):.2f} "
            f"| {'pass' if piv_gate.at[ds, 'sg1'] else 'fail'} "
            f"| {'pass' if piv_gate.at[ds, 'atr_sg1'] else 'fail'} |"
        )
    gate_flip = int((piv_gate["sg1"] != piv_gate["atr_sg1"]).sum())
    gam_shrink = int((piv_gam["atr_sg1"].abs() < piv_gam["sg1"].abs()).sum())

    cb = pd.read_csv(OUT / "exp15_csu_vs_best.csv")
    cb["order"] = cb.apply(lambda r: ORDER.index((r.dataset, r.protocol)), axis=1)
    cb = cb.sort_values("order")
    s13b = ["| Dataset | Protocol | Gate | RMSEP (as used, %) | RMSEP (corrected, %) "
            "| Δ (pp) | Best comparator (as used) | Ratio (as used) "
            "| Best comparator (corrected) | Ratio (corrected) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cb.itertuples():
        delta = r.rmse_atr - r.rmse_sg1
        s13b.append(
            f"| {r.dataset} | {PROT_LABEL[r.protocol]} | {'pass' if r.gate else 'fail'} "
            f"| {r.rmse_sg1:.2f} | {r.rmse_atr:.2f} "
            f"| {f'{delta:+.2f}'.replace('-', chr(8722))} "
            f"| {r.best_name_sg1} | {r.ratio_sg1:.2f} | {r.best_name_atr} | {r.ratio_atr:.2f} |"
        )
    s13d_abs = (cb.rmse_atr - cb.rmse_sg1).abs()
    s13d_med = float(s13d_abs.median())
    s13d_max = float(s13d_abs.max())
    s13d_max_key = f"{cb.loc[s13d_abs.idxmax(), 'dataset']} {cb.loc[s13d_abs.idxmax(), 'protocol']}"
    assert abs(s13d_med - 0.61) < 5e-3 and abs(s13d_max - 6.73) < 6e-3, (s13d_med, s13d_max)
    assert s13d_max_key == "D3 interp", s13d_max_key
    ins_cb, out_cb = cb[cb.gate], cb[~cb.gate]
    ins_ratio_raw, ins_ratio_cor = float(ins_cb.ratio_sg1.median()), float(ins_cb.ratio_atr.median())
    out_ratio_raw, out_ratio_cor = float(out_cb.ratio_sg1.median()), float(out_cb.ratio_atr.median())
    ins_win_raw = int((ins_cb.ratio_sg1 < 1).sum())
    ins_win_cor = int((ins_cb.ratio_atr < 1).sum())

    cc = pd.read_csv(OUT / "exp15_conditions.csv")
    csu_row = cc[cc.method == "CSU-PL"].copy()
    csu_row["gate"] = csu_row.dataset.isin(["D1", "D2", "D7d"])
    s13c = ["| Comparator | Median ratio, in-gate (as used) "
            "| Median ratio, in-gate (corrected) | Median ratio, out-of-gate (as used) "
            "| Median ratio, out-of-gate (corrected) |", "|---|---|---|---|---|"]
    fixed = {}
    fixed_out = {}
    for comp in ("PLS", "Ridge", "CLS-2pt", "MCR-ALS", "MLP", "SVR", "HGB", "RF"):
        sub = cc[cc.method == comp][["dataset", "protocol", "rmse_sg1", "rmse_atr"]]
        if not len(sub):
            continue
        jn = csu_row.merge(sub, on=["dataset", "protocol"], suffixes=("", "_c"))
        jn["r_raw"] = jn.rmse_sg1 / jn.rmse_sg1_c
        jn["r_cor"] = jn.rmse_atr / jn.rmse_atr_c
        i_ = jn[jn.gate]
        o_ = jn[~jn.gate]
        fixed[comp] = (float(i_.r_raw.median()), float(i_.r_cor.median()))
        fixed_out[comp] = (float(o_.r_raw.median()), float(o_.r_cor.median()))
        s13c.append(
            f"| {comp} | {i_.r_raw.median():.2f} | {i_.r_cor.median():.2f} "
            f"| {o_.r_raw.median():.2f} | {o_.r_cor.median():.2f} |"
        )
    s13c = s13c[:1] + s13c[1:]
    in_max_raw = max(v[0] for v in fixed.values())
    in_max_cor = max(v[1] for v in fixed.values())
    out_min_cor = min(v[1] for v in fixed_out.values())
    out_max_cor = max(v[1] for v in fixed_out.values())

    # ---------------- S14 ----------------
    sm = pd.read_csv(OUT / "exp16_splitmeta.csv")
    sm_a = sm.groupby(["dataset", "mode"]).agg(
        n_train=("n_train", "median"), n_test=("n_test", "median"),
        g_tr=("n_groups_train", "median"), g_te=("n_groups_test", "median"),
        leak=("leak_rate", "median"), scans=("scans_per_group", "median")).reset_index()
    s14a = ["| Dataset | Split unit | Spectra, calibration | Spectra, test | Groups, calibration "
            "| Groups, test | Scans per group | Test spectra with a calibration sibling |",
            "|---|---|---|---|---|---|---|---|"]
    for ds in ("D1", "D2", "D3"):
        for md in ("sample", "scan"):
            r = sm_a[(sm_a.dataset == ds) & (sm_a["mode"] == md)].iloc[0]
            s14a.append(
                f"| {ds} | {'physical sample' if md == 'sample' else 'single scan'} "
                f"| {int(r.n_train)} | {int(r.n_test)} | {int(r.g_tr)} | {int(r.g_te)} "
                f"| {r.scans:.1f} | {r.leak:.2f} |"
            )

    sp = pd.read_csv(OUT / "exp16_summary.csv")
    s14b = ["| Dataset | Method | RMSE, scan-level (%) | RMSE, sample-level (%) "
            "| Optimism (pp) | Ratio sample/scan |", "|---|---|---|---|---|---|"]
    opt_rows = []
    for ds in ("D1", "D2", "D3"):
        for m_ in ("CSU-PL", "PLS", "Ridge", "SVR", "MLP", "CLS-2pt"):
            sc = sp[(sp.dataset == ds) & (sp["method"] == m_) & (sp["mode"] == "scan")].rmse.iloc[0]
            sa = sp[(sp.dataset == ds) & (sp["method"] == m_) & (sp["mode"] == "sample")].rmse.iloc[0]
            opt = sa - sc
            opt_rows.append(dict(dataset=ds, method=m_, opt=opt, ratio=sa / sc))
            s14b.append(
                f"| {ds} | {m_} | {sc:.2f} | {sa:.2f} "
                f"| {f'{opt:+.2f}'.replace('-', chr(8722))} | {sa / sc:.2f} |"
            )
    opr = pd.DataFrame(opt_rows)
    opt_pos = int((opr.opt > 0).sum())
    opt_n = len(opr)
    opt_med, opt_max = float(opr.opt.median()), float(opr.opt.max())
    neg_rows = opr[opr.opt <= 0]
    n_neg = len(neg_rows)
    neg_ds = ", ".join(sorted(set(neg_rows.dataset))) if n_neg else "none"
    opt_min_pos = min(int((opr[opr.method == m_].opt > 0).sum())
                      for m_ in opr.method.unique())
    s14m = ["| Method | Median optimism (pp) | Positive out of 3 | Median ratio |",
            "|---|---|---|---|"]
    for m_ in ("CSU-PL", "PLS", "Ridge", "SVR", "MLP", "CLS-2pt"):
        g_ = opr[opr.method == m_]
        s14m.append(f"| {m_} | {g_.opt.median():+.2f} | {int((g_.opt > 0).sum())} "
                    f"| {g_.ratio.median():.2f} |")
    csu_rank_shift = []
    for ds in ("D1", "D2", "D3"):
        for md in ("scan", "sample"):
            v = sp[(sp.dataset == ds) & (sp["mode"] == md)].sort_values("rmse")
            csu_rank_shift.append(int(v.index[v["method"] == "CSU-PL"][0]) + 1)

    # ---------------- S15 ----------------
    c17 = pd.read_csv(OUT / "exp17_conditions.csv")
    c17["order"] = c17.apply(lambda r: ORDER.index((r.dataset, r.protocol)), axis=1)
    c17 = c17.sort_values("order")
    s15a = ["| Dataset | Protocol | N | SE of Section 2.3 (pp) | MC SD, white noise (pp) "
            "| SD/SE | Coverage of ±1.96 SE, white | MC SD, residual rows (pp) | SD/SE "
            "| Coverage, residual rows | Observed mean absolute error (pp) | Observed/white SD |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c17.itertuples():
        s15a.append(
            f"| {r.dataset} | {PROT_LABEL[r.protocol]} | {int(r.N)} | {r.se_med:.2f} "
            f"| {r.sd_white:.2f} | {r.ratio_white:.2f} | {r.cov_white:.2f} | {r.sd_resid:.2f} "
            f"| {r.ratio_resid:.1f} | {r.cov_resid:.2f} | {r.obs_mae:.2f} | {r.mismatch:.1f}× |"
        )
    rw = c17.ratio_white
    rw_med, rw_q1, rw_q3 = float(rw.median()), float(rw.quantile(.25)), float(rw.quantile(.75))
    cov_w_med = float(c17.cov_white.median())
    cov_r_med = float(c17.cov_resid.median())
    rr_med = float(c17.ratio_resid.median())
    mm_med, mm_min, mm_max = (float(c17.mismatch.median()), float(c17.mismatch.min()),
                              float(c17.mismatch.max()))
    ins17, out17 = c17[c17.gate == True], c17[c17.gate != True]  # noqa: E712

    mcl = pd.read_csv(OUT / "exp17_mc_long.csv")
    lv = mcl[(mcl["mode"] == "white") & mcl.level.notna()].copy()

    def standardise(frame, sdcol):
        f = frame.copy()
        f["sd_rel"] = f[sdcol] / f.groupby(["dataset", "protocol"])[sdcol].transform("median")
        f["h_rel"] = f["h"] / f.groupby(["dataset", "protocol"])["h"].transform("median")
        return f.dropna(subset=["sd_rel", "h_rel"])

    w_std = standardise(lv, "sd")
    rho_w = sps.spearmanr(w_std.h_rel, w_std.sd_rel)
    rho_w_s = f"{rho_w.correlation:+.3f}".replace("-", chr(8722))

    cal = pd.read_csv(OUT / "exp17_calib_levels.csv")
    merged = lv.groupby(["dataset", "protocol", "level"]).sd.median().rename("sd_white")
    cal = cal.merge(merged.reset_index(), on=["dataset", "protocol", "level"], how="left")
    cal["sd_tot"] = np.sqrt(cal.sd_white ** 2 + cal.sd_calib ** 2)
    t2x = pd.read_csv(ROOT / "tables" / "table2_main.csv")
    t2x["protocol"] = t2x["Protocol"].map(PROT_CSV)
    cal = cal.merge(
        t2x[["Dataset", "protocol", "RMSEP"]].rename(
            columns={"Dataset": "dataset", "RMSEP": "rmsep"}),
        on=["dataset", "protocol"], how="left")
    cal["ratio_obs"] = cal.rmsep / cal.sd_tot
    cal["share"] = (cal.sd_tot / cal.rmsep) ** 2
    c_std = standardise(cal, "sd_calib")
    rho_c = sps.spearmanr(c_std.h_rel, c_std.sd_rel)
    rho_c_s = f"{rho_c.correlation:+.3f}".replace("-", chr(8722))
    rho_w_s = f"{rho_w.correlation:+.3f}".replace("-", chr(8722))
    fit = sps.linregress(np.log(c_std.h_rel.clip(lower=1e-9)), np.log(c_std.sd_rel.clip(lower=1e-9)))
    cal_agg = cal.groupby(["dataset", "protocol"]).agg(
        h_max=("h", "max"), sd_white=("sd_white", "median"), sd_calib=("sd_calib", "median"),
        sd_tot=("sd_tot", "median"), rmsep=("rmsep", "first"), ratio_obs=("ratio_obs", "median"),
        share=("share", "median"), n_lv=("level", "size")).reset_index()
    cal_agg["order"] = cal_agg.apply(lambda r: ORDER.index((r.dataset, r.protocol)), axis=1)
    cal_agg = cal_agg.sort_values("order")
    s15b = ["| Dataset | Protocol | Test levels | Max h(c*) | MC SD, white noise (pp) "
            "| MC SD, endmember re-fit (pp) | Combined SD (pp) | Test RMSEP (%) "
            "| RMSEP / combined SD | Fraction of error variance |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cal_agg.itertuples():
        s15b.append(
            f"| {r.dataset} | {PROT_LABEL[r.protocol]} | {int(r.n_lv)} | {r.h_max:.2f} "
            f"| {r.sd_white:.2f} | {r.sd_calib:.2f} | {r.sd_tot:.2f} | {r.rmsep:.2f} "
            f"| {r.ratio_obs:.0f}× | {r.share * 100:.2f} % |"
        )
    sd_ratio = float((cal.groupby(["dataset", "protocol"]).sd_calib.median()
                      / cal.groupby(["dataset", "protocol"]).sd_white.median()).median())
    tot_med = float(cal_agg.sd_tot.median())
    rmsep_med = float(cal_agg.rmsep.median())
    ratio_obs_med = float(cal_agg.ratio_obs.median())
    share_med = float(cal_agg.share.median())
    keys12 = set(map(tuple, cal[["dataset", "protocol"]].drop_duplicates().to_numpy()))
    csub = c17[[(r.dataset, r.protocol) in keys12 for r in c17.itertuples()]]
    res_med_12 = float(csub.sd_resid.median())
    obs_med_12 = float(csub.obs_mae.median())
    res_over_obs_12 = float((csub.sd_resid / csub.obs_mae).median())
    ins_se, ins_ratio, ins_mm = (float(ins17.se_med.median()), float(ins17.ratio_white.median()),
                                 float(ins17.mismatch.median()))
    out_se, out_ratio, out_mm = (float(out17.se_med.median()), float(out17.ratio_white.median()),
                                 float(out17.mismatch.median()))

    # ---------------- S16 ----------------
    ka = pd.read_csv(OUT / "exp00b_kan_audit.csv")
    km = pd.read_csv(OUT / "exp00b_kan_mirror.csv")
    kl = pd.read_csv(OUT / "exp00b_kan_labelcheck.csv")
    s16a = ["| Dataset | Matrix / adulterant | Sheet (header line) | Raw data rows | Rows excluded "
            "| Exclusion reason | Spectra analysed | Channels | Physical samples | Levels (%) "
            "| Spectra at 0 % | Scans per sample (min, median, max) | Spacing (cm⁻¹) "
            "| Range (cm⁻¹) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    drop_map = [("drop_blank_id", "no sample id"), ("drop_no_matrix", "blank composition"),
                ("drop_no_adulterant", "blank composition"),
                ("drop_nonfinite_spectrum", "non-finite spectrum")]
    t1 = pd.read_csv(ROOT / "tables" / "table1_datasets.csv").set_index("Dataset")
    mat_lab = {d: f"{t1.at[d, 'Matrix']} / {t1.at[d, 'Adulterant']}" for d in ka.dataset}
    for r in ka.itertuples():
        assert int(r.spectra_final) == int(t1.at[r.dataset, "Spectra"]), r.dataset
        assert int(r.physical_samples) == int(t1.at[r.dataset, "Samples"]), r.dataset
        agg = {}
        for c, lab in drop_map:
            if int(getattr(r, c)):
                agg[lab] = agg.get(lab, 0) + int(getattr(r, c))
        reasons = [f"{v} {lab}" for lab, v in agg.items()]
        lv = [float(v) for v in str(r.level_values).split(",")]
        s16a.append(
            f"| {r.dataset} | {mat_lab[r.dataset]} | {r.sheet} (row {int(r.header_row)}) "
            f"| {int(r.raw_data_rows)} | {int(r.rows_dropped)} "
            f"| {'; '.join(reasons) if reasons else 'none'} | {int(r.spectra_final)} "
            f"| {int(r.channels):,} | {int(r.physical_samples)} | {int(r.levels)} "
            f"({lv[0]:g}–{lv[-1]:g}) "
            f"| {int(str(r.spectra_per_level).split(',')[0])} "
            f"| {int(r.scans_per_sample_min)}, {r.scans_per_sample_median:g}, "
            f"{int(r.scans_per_sample_max)} | {r.dnu:.4f} "
            f"| {r.range_lo:.1f}–{r.range_hi:.1f} |")
    kan_sp = int(ka.spectra_final.sum())
    kan_raw = int(ka.raw_data_rows.sum())
    kan_sa = int(ka.physical_samples.sum())
    kan_drop = int(ka.rows_dropped.sum())

    s16b = ["| Dataset | Mirror sheet | Spectrum columns | Distinct sample ids in the sheet "
            "| Columns with no counterpart in the analysed set | Examples "
            "| Analysed spectra | Spectra absent from the mirror |",
            "|---|---|---|---|---|---|---|---|"]
    for r in km.itertuples():
        ex = str(r.extra_examples).strip("[]").replace("'", "")
        s16b.append(
            f"| {r.dataset} | {r.mirror_sheet} | {int(r.spectrum_columns)} "
            f"| {int(r.distinct_ids)} | {int(r.extra_in_mirror)} | {ex} "
            f"| {int(r.flat_spectra)} | {int(r.absent_from_mirror)} |")
    d1_mir = int(km[km.dataset == "D1"].spectrum_columns.iloc[0])
    d2_mir = km[km.dataset == "D2"]
    d2_sum = int(d2_mir.spectrum_columns.sum())
    d2_top = int(d2_mir[d2_mir.mirror_sheet == "Sayfa3"].spectrum_columns.iloc[0])
    d3_mir = int(km[km.dataset == "D3"].spectrum_columns.iloc[0])

    s16c = ["| Sample id | Brand | Level index in the id | Label in the file (%) "
            "| Label implied by the index (%) | Spectra affected |",
            "|---|---|---|---|---|---|"]
    kl_u = kl.groupby("id", as_index=False).agg(
        brand=("brand", "first"), level_index=("level_index", "first"),
        label_in_file=("label_in_file", "first"), label_from_id=("label_from_id", "first"),
        spectra=("id", "size"))
    for r in kl_u.itertuples():
        s16c.append(f"| {r.id} | {int(r.brand)} | {int(r.level_index)} | {r.label_in_file:g} "
                    f"| {r.label_from_id:g} | {int(r.spectra)} |")
    n_bad_ids = int(len(kl_u))
    n_bad_rows = int(kl_u.spectra.sum())
    d2 = ka[ka.dataset == "D2"].iloc[0]
    d2_lab = int(d2.tekrar_labelled)
    d2_unlab = int(d2.tekrar_unlabelled)
    ks = pd.read_csv(OUT / "exp00b_kan_labelsum.csv")
    lvl_ids = {int(r.label_in_file): str(r.ids).split(",") for r in ks.itertuples()}
    sp0 = [[int(v) for v in str(r.spectra_per_level).split(",")] for r in ka.itertuples()]
    n0 = [row[0] for row in sp0]
    zero_max = [row[0] == max(row) for row in sp0]
    sc_min = int(ka.scans_per_sample_min.min())
    sc_max = int(ka.scans_per_sample_max.max())
    d2_s1 = int(d2_mir[d2_mir.mirror_sheet == "Sayfa1"].spectrum_columns.iloc[0])
    d1_ex = int(km[km.dataset == "D1"].extra_in_mirror.iloc[0])
    d3_ex = int(km[km.dataset == "D3"].extra_in_mirror.iloc[0])
    d2_ex = int(d2_mir[d2_mir.mirror_sheet == "Sayfa3"].extra_in_mirror.iloc[0])
    base_absent = not bool(d2.base_present)
    assert all(zero_max) and int(ka['levels'].nunique()) == 1

    # ---------------- S17 ----------------
    # D7/D8 的分发文档没有仪器与日期字段，因此"同源"只能由轴、编号、谱相似度三类可算证据支撑。
    ph = pd.read_csv(OUT / "exp00c_photometry.csv")
    ad = pd.read_csv(OUT / "exp00c_audit_domain.csv")
    ax = pd.read_csv(OUT / "exp00c_axis.csv")
    nm = pd.read_csv(OUT / "exp00c_naming.csv")
    sm17 = pd.read_csv(OUT / "exp00c_similarity.csv")

    s17a = ["| Dataset | Rows in file | Stored minimum | Stored median | Stored maximum "
            "| Fraction above 1 | Mean near the transparent end (≈3900 cm⁻¹) "
            "| Reading of the stored quantity | Domain analysed |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in ph.itertuples():
        s17a.append(
            f"| {r.dataset} | {int(r.rows_in_file)} | {r.stored_min:.4f} | {r.stored_median:.4f} "
            f"| {r.stored_max:.4f} | {r.share_above_1:.3f} | {r.transparent_3900_mean:.4f} "
            f"| {r.stored_quantity} | {'stored values' if r.domain_analysed == 'stored values' else '−log₁₀(stored values)'} |")
    n_conv = int((ph.stored_quantity == "transmittance").sum())
    assert n_conv == 4, n_conv
    assert (ph[ph.stored_quantity == "absorbance"].stored_max < 1.0).all()

    s17b = ["| Dataset | CNR (as stored) | CNR (analysed) "
            "| \\|Δ CNR\\|, % of analysed | \\|γ\\| (as stored) "
            "| \\|γ\\| (analysed) | Leak (as stored) | Leak (analysed) | Gate (as stored) "
            "| Gate (analysed) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    conv = ad[ad.gamma_other.notna()].copy()
    for r in conv.itertuples():
        s17b.append(
            f"| {r.dataset} | {abs(r.cnr_other):.1f} | {abs(r.cnr_analysed):.1f} "
            f"| {r.max_cnr_rel_change * 100:.0f} % | {abs(r.gamma_other):.2f} "
            f"| {abs(r.gamma_analysed):.2f} | {r.leak_other:.2f} | {r.leak_analysed:.2f} "
            f"| {r.gate_other} | {r.gate_analysed} |")
    assert len(s17b) == 6, len(s17b)
    gate_flips17 = int((conv.gate_analysed != conv.gate_other).sum())

    conc = ax[ax.dataset.str.startswith("concordance")].iloc[0]
    inter_row = nm[nm.dataset == "D7∩D8 identifier intersection"].iloc[0]
    n_all = int(inter_row.n_ids)
    inter = int(inter_row.matched)
    m_off = re.search(r"max \|offset\| ([\d.]+) cm\^-1 = ([\d.]+)% of a channel, mean ([\d.]+)",
                      str(conc.common_range))
    assert m_off, conc.common_range
    off_max, off_pct, off_mean = m_off.groups()
    s17c = ["| Check | Result | Detail |", "|---|---|---|"]

    def simv(key):
        return sm17[sm17.test == key].iloc[0]

    def maxof(row):
        return re.search(r"max ([\d.]+)", str(row.detail)).group(1)

    s17c.append(
        f"| Sampling grid | agrees to within {off_pct} % of a channel "
        f"| {int(ax[ax.dataset == 'D7'].channels.iloc[0])} and "
        f"{int(ax[ax.dataset == 'D8'].channels.iloc[0])} ascending channels, step "
        f"{conc.dnu:.4f} cm⁻¹ in both; over the {int(conc.channels)} index-aligned channels the "
        f"largest axis offset is {off_max} cm⁻¹ and the mean is {off_mean} cm⁻¹ |")
    s17c.append(
        f"| Sample identifiers | {inter} of {n_all} identifiers shared "
        f"| the cinnamon files use C[PWD]<brand><sample> and the turmeric file uses "
        f"T<level><sample>, and no identifier occurs in both |")
    cen = simv("D8 -> nearest D7 spectrum, cosine (mean-centred)")
    raw = simv("D8 -> nearest D7 spectrum, cosine (raw absorbance)")
    selfnn = simv("D8 self nearest neighbour cosine (median)")
    s17c.append(
        f"| D8 spectra duplicated in D7 | none "
        f"| cosine of each D8 spectrum with its nearest D7 spectrum: median {cen.value:.4f}, maximum "
        f"{maxof(cen)} mean-centred, and median {raw.value:.4f}, maximum {maxof(raw)} without "
        f"centring. Both stay below the within-release nearest-neighbour scale, whose median is "
        f"{selfnn.value:.4f} inside the turmeric file and "
        f"{simv('D7 self nearest neighbour cosine (median, 1950 spectra)').value:.4f} inside the "
        f"cinnamon files |")
    zero_rows = [simv(k) for k in ("D7 0% walnut vs 0% peanut cosine",
                                   "D7 0% walnut vs 0% wood dust cosine",
                                   "D7 0% peanut vs 0% wood dust cosine")]
    zmax = [float(re.search(r"max ([\d.]+)", str(r.detail)).group(1)) for r in zero_rows]
    dup = sum(int(re.search(r"exact duplicates (\d+)", str(r.detail)).group(1)) for r in zero_rows)
    zc = [float(re.search(r"mean ([\d.]+)", str(r.detail)).group(1)) for r in zero_rows]
    s17c.append(
        f"| Pure-matrix rows within cinnamon | not copies | the 0 % spectra of the three cinnamon "
        f"files have mean pairwise cosines of {min(zc):.4f}–{max(zc):.4f} across the three pairs, a "
        f"largest single pair of {max(zmax):.4f}, and {dup} exact duplicates over the "
        f"{len(zero_rows) * 50} spectra compared, so the three adulterant tables are separate "
        f"measurements rather than one table relabelled |")
    s17c.append("| Instrument, serial number, acquisition dates | not recorded | the cinnamon deposit "
                "description gives 'an ATR-FTIR spectrometer' and 'approximately 400–4000 cm⁻¹'; the "
                "turmeric description gives no instrument sentence; neither names a model, serial number "
                "or date |")
    s17c.append("| Authorship and deposit | same first author | both deposits are Mendeley Data records "
                "cited as [33] and [34] and list the same first author with different co-authors; they "
                "were deposited in April and July 2026, which are release dates rather than acquisition "
                "dates |")

    tr = ph[ph.stored_quantity == "transmittance"]
    ab = ph[ph.stored_quantity == "absorbance"]
    tr_med = (tr.stored_median.min(), tr.stored_median.max())
    tr_max = (tr.stored_max.min(), tr.stored_max.max())
    tr_share = (tr.share_above_1.min(), tr.share_above_1.max())
    tr_edge = (tr.transparent_3900_mean.min(), tr.transparent_3900_mean.max())
    ab_max = (ab.stored_max.min(), ab.stored_max.max())
    ab_edge = (ab.transparent_3900_mean.min(), ab.transparent_3900_mean.max())
    n_neg_min = int((ab.stored_min < 0).sum())
    cnr_rel_max = float(conv.max_cnr_rel_change.max())
    gam_shift = (conv.gamma_other.abs() - conv.gamma_analysed.abs()).abs()
    gam_shift_max = float(gam_shift.max())
    gam_shift_key = str(conv.loc[gam_shift.idxmax(), "dataset"])
    rel_rows = "; ".join(f"{r.dataset} {r.max_cnr_rel_change * 100:.1f} %"
                         for r in conv.itertuples())

    assert all(int(v) == 5 for row in ka.samples_per_level for v in str(row).split(","))
    assert set(ka.raw_axis) == {"descending"}

    # ---------------- S18 (wave-number grid; variables keep the exp19 prefix) ----------------
    ds_order = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
    g19 = pd.read_csv(OUT / "exp19_grid.csv")
    r19 = pd.read_csv(OUT / "exp19_grid_rank.csv").set_index("variant")
    var_order = ["native", "cross_match", "cross_fixed15", "decim2_match"]
    vlabel = {
        "native": "Native grid",
        "cross_match": "Resampled, matched window",
        "cross_fixed15": "Resampled, window 15",
        "decim2_match": "Every second channel, matched window",
    }
    nat = g19[g19.variant == "native"].set_index("dataset")
    audit05 = json.loads((OUT / "exp05_audit.json").read_text(encoding="utf-8"))
    for nm in ds_order:
        assert abs(nat.loc[nm, "cnr"] - audit05[nm]["endmember"]["cnr"]) < 1e-6, nm
        assert abs(nat.loc[nm, "gamma"] - audit05[nm]["curvature"]["gamma"]) < 1e-6, nm
    ref_rmse = {}
    for fname in ("exp03_summary.csv", "exp09_d7_summary.csv"):
        t = pd.read_csv(OUT / fname)
        t = t[(t.method == "CSU-PL") & (t.protocol == "high")]
        for r in t.itertuples():
            ref_rmse[r.dataset] = float(r.rmse)
    for nm in ds_order:
        assert abs(nat.loc[nm, "rmse_high"] - ref_rmse[nm]) < 1e-6, nm

    s19a = ["| Dataset | Configuration | Channels | Window | CNR | \\|γ\\| | Baseline leak "
            "| RMSEP, high (pp) | ΔRMSEP (%) | Gate |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for nm in ds_order:
        for v in var_order:
            r = g19[(g19.dataset == nm) & (g19.variant == v)].iloc[0]
            dv = float(r.d_rmse_rel) * 100.0
            d_txt = "0.0" if abs(dv) < 0.05 else f"{dv:+.1f}".replace("-", "−")
            s19a.append(
                f"| {nm} | {vlabel[v]} | {int(r.N)} | {int(r.window)} | {r.cnr:.2f} "
                f"| {abs(r.gamma):.2f} | {r.baseline_leak:.2f} | {r.rmse_high:.2f} "
                f"| {d_txt} | {'pass' if r.pass_gate else 'fail'} |")
    s19b = ["| Configuration | Same gate outcome | Datasets passing | ρ(CNR, RMSEP) | p |",
            "|---|---|---|---|---|"]
    for v in var_order:
        r = r19.loc[v]
        s19b.append(f"| {vlabel[v]} | {int(r.n_same_class)} of {int(r.n)} "
                    f"| {int(r.n_pass)} | {r.rho_cnr_rmse:.2f}".replace("-", "−") +
                    f" | {r.p:.3f} |")

    nonnat = g19[g19.variant != "native"].copy()
    nonnat["cnr_rel"] = nonnat.cnr / nonnat.dataset.map(nat.cnr) - 1.0
    nonnat["dgam"] = (nonnat.gamma.abs() - nonnat.dataset.map(nat.gamma.abs())).abs()
    match_rows = nonnat[nonnat.variant.isin(["cross_match", "decim2_match"])]
    fixed_rows = nonnat[nonnat.variant == "cross_fixed15"]
    cross_cnr = nonnat[nonnat.variant == "cross_match"].cnr_rel
    decim_cnr = nonnat[nonnat.variant == "decim2_match"].cnr_rel
    fixed_cnr = nonnat[nonnat.variant == "cross_fixed15"].cnr_rel
    cnr_rel_match = (float(cross_cnr.min()), float(cross_cnr.max()))
    cnr_rel_decim = (float(decim_cnr.min()), float(decim_cnr.max()))
    cnr_rel_fixed = (float(fixed_cnr.min()), float(fixed_cnr.max()))
    dgam_match = (float(match_rows.dgam.min()), float(match_rows.dgam.max()))
    dgam_match_key = str(match_rows.loc[match_rows.dgam.idxmax(), "dataset"])
    dgam_fixed = float(fixed_rows.dgam.max())
    dgam_fixed_key = str(fixed_rows.loc[fixed_rows.dgam.idxmax(), "dataset"])
    drmse = (float(nonnat.d_rmse_rel.min()), float(nonnat.d_rmse_rel.max()))
    drmse_key = str(nonnat.loc[nonnat.d_rmse_rel.abs().idxmax(), "dataset"])
    n_flip = int(g19.flip.sum())
    n_rows19 = int(len(g19))
    rho19 = (float(r19.rho_cnr_rmse.min()), float(r19.rho_cnr_rmse.max()))
    cnr_match_txt = (f"{cnr_rel_match[0] * 100:+.0f} % to "
                     f"{cnr_rel_match[1] * 100:+.0f} %").replace("-", "−")
    cnr_decim_txt = (f"{cnr_rel_decim[0] * 100:+.0f} % to "
                     f"{cnr_rel_decim[1] * 100:+.0f} %").replace("-", "−")
    cnr_fixed_txt = (f"{cnr_rel_fixed[0] * 100:+.0f} % to "
                     f"{cnr_rel_fixed[1] * 100:+.0f} %").replace("-", "−")
    drmse_txt = (f"{drmse[0] * 100:+.1f} % to "
                 f"{drmse[1] * 100:+.1f} %").replace("-", "−")
    rho19_txt = f"{rho19[0]:.2f} and {rho19[1]:.2f}".replace("-", "−")

    # ---------------- S19 (capacity-matched comparators) ----------------
    cap = pd.read_csv(OUT / "exp20_capacity.csv")
    caps = pd.read_csv(OUT / "exp20_capacity_summary.csv").set_index("method")
    anc = pd.read_csv(OUT / "exp20_capacity_dof.csv")
    cap_methods = ["PLS-CV"] + [f"PLS-{k}" for k in (2, 3, 4, 5, 6)] + \
                  [f"Ridge-dof{d}" for d in (3, 4, 5)]
    cond_order = [(nm, p) for nm in ["D1", "D2", "D7d"]
                  for p in ("high", "interp", "low")]
    piv = cap.pivot_table(index=["dataset", "protocol"], columns="method", values="rmse")
    s20a = ["| Condition | CSU-PL | PLS-CV | PLS-2 | PLS-3 | PLS-4 | PLS-5 | PLS-6 "
            "| Ridge, dof 3 | Ridge, dof 4 | Ridge, dof 5 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, p in cond_order:
        vals = [f"{piv.loc[(nm, p), m]:.2f}" for m in ["CSU-PL"] + cap_methods]
        s20a.append(f"| {nm} {p} | " + " | ".join(vals) + " |")
    s20b = ["| Comparator | Median proposed/comparator | Conditions proposed better "
            "| Median, D1 | Median, D2 | Median, D7d |",
            "|---|---|---|---|---|---|"]
    for m in cap_methods:
        r = caps.loc[m]
        s20b.append(f"| {m} | {1.0 / r.median_ratio:.2f} | {int(r.n_csu_better)} of "
                    f"{int(r.n_conditions)} | {1.0 / r.median_D1:.2f} "
                    f"| {1.0 / r.median_D2:.2f} | {1.0 / r.median_D7d:.2f} |")
    s20c = ["| Condition | PLS components chosen by CV | Ridge alpha chosen by CV "
            "| Ridge effective dof |", "|---|---|---|---|"]
    for r in anc.itertuples():
        s20c.append(f"| {r.dataset} {r.protocol} | {int(r.pls_components)} "
                    f"| {r.ridge_alpha:.0f} | {r.ridge_dof:.1f} |")

    pls_keys = [f"PLS-{k}" for k in (2, 3, 4, 5, 6)]
    ridge_keys = [f"Ridge-dof{d}" for d in (3, 4, 5)]
    pls_med = (1.0 / float(caps.loc[pls_keys, "median_ratio"].max()),
               1.0 / float(caps.loc[pls_keys, "median_ratio"].min()))
    pls_wins = (int(caps.loc[pls_keys, "n_csu_better"].min()),
                int(caps.loc[pls_keys, "n_csu_better"].max()))
    pls_cv_med = 1.0 / float(caps.loc["PLS-CV", "median_ratio"])
    ridge_med = (1.0 / float(caps.loc[ridge_keys, "median_ratio"].max()),
                 1.0 / float(caps.loc[ridge_keys, "median_ratio"].min()))
    dof_range = (float(anc.ridge_dof.min()), float(anc.ridge_dof.max()))
    k_range = (int(anc.pls_components.min()), int(anc.pls_components.max()))
    k_med = int(anc.pls_components.median())
    loss_cond = {}
    for k in (2, 3, 4, 5, 6):
        sub = cap[(cap.method == f"PLS-{k}") & (cap.ratio < 1.0)]
        for r in sub.itertuples():
            loss_cond.setdefault((r.dataset, r.protocol), []).append(1.0 / float(r.ratio) - 1.0)
    loss_txt = (", ".join(f"{nm} {p} (by {min(v) * 100:.0f}–{max(v) * 100:.0f} %)"
                          for (nm, p), v in sorted(loss_cond.items())) or "none")

    # ---------------- S20 (rotational feasible band) ----------------
    bnd = pd.read_csv(OUT / "exp22_band.csv")
    th22 = pd.read_csv(OUT / "exp22_theta.csv")
    mt22 = json.loads((OUT / "exp22_meta.json").read_text(encoding="utf-8"))
    AMB = ["D1/high", "D2/high", "D7d/high", "D7p/high", "D7w/high", "D8/high"]
    AMB_TAUS = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
    AMB_TAU_HEAD = ["τ = 0", "τ = 0.5", "τ = 1", "τ = 2", "τ = 5", "no labels"]
    AMB_DELTAS = [0.0, 0.02, 0.10, 0.30]
    n_grid = int(th22.groupby("condition").size().max())

    def mn(s):
        """Hyphen-minus to U+2212 for the typeset text."""
        return str(s).replace("-", "\u2212")

    assert int(th22.groupby("condition").size().min()) == n_grid, "θ grid differs per condition"
    for key in AMB:
        aud = audit[key.split("/")[0]]
        assert abs(mt22[key]["gate_cnr"] - aud["endmember"]["cnr"]) < 1e-6, key
        assert abs(mt22[key]["gate_gamma"] - aud["curvature"]["gamma"]) < 1e-6, key
    assert bnd.contains_zero.all(), "θ = 0 must be inside every feasible set it claims"

    def amb_worst(key, dl, tau, col, abs_val=False):
        s = bnd[(bnd.condition == key) & (bnd["delta"] == dl) & (bnd.tau == tau)]
        v = s[col].abs() if abs_val else s[col]
        return float(v.max())

    def amb_ntheta(key, dl, tau):
        s = bnd[(bnd.condition == key) & (bnd["delta"] == dl) & (bnd.tau == tau)]
        return int(s.n_theta.max())

    def amb_lv(v):
        return ", ".join(f"{x:g}" for x in v)

    s21a = ["| Condition | Pair scanned | CNR | \\|γ\\| | n_cal | n_test "
            "| Calibration levels (%) | Test levels (%) | In-range RMSE at θ = 0 (pp) "
            "| θ admitted at δ = 0, no labels | θ* (deg) | In-range RMSE at θ* (pp) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in AMB:
        m = mt22[key]
        tstar = mn("{:.1f}".format(m["theta_star"]))
        mform = {"exact": "fitted pair", "pooled": "pooled matrix"}[m["model"]]
        s21a.append(f"| {key} | {mform} | {m['gate_cnr']:.1f} "
                    f"| {abs(m['gate_gamma']):.2f} | {m['n_cal']} | {m['n_test']} "
                    f"| {amb_lv(m['levels_cal'])} | {amb_lv(m['levels'])} "
                    f"| {m['cal_rmse_theta0']:.2f} | {m['n_feas_delta0']} of {n_grid} "
                    f"| {tstar} | {m['cal_rmse_star']:.2f} |")

    s21b = ["| Condition | δ / τ | " + " | ".join(AMB_TAU_HEAD) + " |",
            "|---|---|---|---|---|---|---|---|"]
    for key in AMB:
        for dl in AMB_DELTAS:
            cells = [f"{amb_worst(key, dl, t, 'band_pp'):.2f}" for t in AMB_TAUS]
            s21b.append(f"| {key} | {dl:.2f} | " + " | ".join(cells) + " |")

    amb_c = {}
    s21c = ["| Condition | θ admitted at δ = 0, τ = 0 | Band at δ = 0, τ = 0 (pp) "
            "| \\|bias\\| of the reported rotation, over test levels (pp) "
            "| Band at δ = 0.10, τ = 1 pp "
            "| Band at δ = 0.10, no labels (pp) | Shrink factor "
            "| Largest \\|bias\\| inside the unlabelled band (pp) |",
            "|---|---|---|---|---|---|---|---|"]
    for key in AMB:
        b_lab = amb_worst(key, 0.10, 1.0, "band_pp")
        b_unl = amb_worst(key, 0.10, np.inf, "band_pp")
        rec = dict(n00=amb_ntheta(key, 0.0, 0.0),
                   band00=amb_worst(key, 0.0, 0.0, "band_pp"),
                   csu=amb_worst(key, 0.0, 0.0, "csu_bias_pp", True),
                   lab=b_lab, unl=b_unl, shrink=b_unl / max(b_lab, 1e-9),
                   bias_unl=amb_worst(key, 0.10, np.inf, "max_bias_pp"))
        amb_c[key] = rec
        s21c.append(f"| {key} | {rec['n00']} of {n_grid} | {rec['band00']:.2f} "
                    f"| {rec['csu']:.2f} | {rec['lab']:.2f} | {rec['unl']:.2f} "
                    f"| {rec['shrink']:.1f} | {rec['bias_unl']:.2f} |")

    # quantities the note states
    lv_cols = [c for c in th22.columns if re.fullmatch(r"c\d+", c)]
    dtheta, gain = {}, {}
    for key in AMB:
        sub = th22[th22.condition == key].set_index("theta")
        z0, z1 = sub.loc[0.0], sub.loc[float(mt22[key]["theta_star"])]
        dd = (z1[lv_cols].astype(float) - z0[lv_cols].astype(float)).abs().dropna()
        dtheta[key] = float(dd.max()) if len(dd) else float("nan")
        gain[key] = float(mt22[key]["cal_rmse_theta0"] - mt22[key]["cal_rmse_star"])
    n_nonneg_only = {key: mt22[key]["n_feas_delta0"] for key in AMB}
    pinned = [k for k, v in n_nonneg_only.items() if v <= 4]
    loose = [k for k in AMB if k not in pinned]
    lab_rng = (min(amb_c[k]["lab"] for k in AMB), max(amb_c[k]["lab"] for k in AMB))
    unl_rng = (min(amb_c[k]["unl"] for k in AMB), max(amb_c[k]["unl"] for k in AMB))
    shr_rng = (min(amb_c[k]["shrink"] for k in AMB), max(amb_c[k]["shrink"] for k in AMB))
    shr_min_key = min(AMB, key=lambda k: amb_c[k]["shrink"])
    band00_rng = (min(amb_c[k]["band00"] for k in AMB), max(amb_c[k]["band00"] for k in AMB))
    csu_big = [k for k in AMB if amb_c[k]["csu"] - amb_c[k]["band00"] > 5.0]
    rot_free = [k for k in AMB if mt22[k]["theta_star"] != 0.0]

    def and_join(v):
        return ", ".join(v[:-1]) + (" and " + v[-1] if len(v) > 1 else v[0])

    byc: dict = {}
    for k in pinned:
        byc.setdefault(n_nonneg_only[k], []).append(k)
    pinned_txt = and_join([
        "{} admit{} {}{}".format(
            and_join(v), "s" if len(v) == 1 else "",
            "a single angle" if c == 1 else f"{c} angles",
            " each" if len(v) > 1 and c == 1 else "")
        for c, v in sorted(byc.items())])
    loose_txt = (f"{n_nonneg_only[loose[0]]} remain on {loose[0]}"
                 + "".join(f", {n_nonneg_only[k]} on {k}" for k in loose[1:]))
    csu_big_txt = ", ".join(f"{amb_c[k]['csu']:.2f} pp on {k}" for k in csu_big)

    z0 = bnd[(bnd["delta"] == 0.0) & (bnd.tau == 0.0)]
    for k in AMB:
        s = z0[z0.condition == k].sort_values("level")
        assert abs(float(s.csu_bias_pp.abs().max()) - amb_c[k]["csu"]) < 1e-9, k
        assert float(s.level.iloc[-1]) == float(s.loc[s.csu_bias_pp.abs().idxmax(), "level"]), k
    d4s = z0[z0.condition == "D7d/high"].sort_values("level")
    d4_lv_txt = ", ".join(f"{v:.2f}".replace("-", "−") for v in d4s.csu_bias_pp)
    d4_lv_rng = f"{d4s.level.iloc[0]:g} % to {d4s.level.iloc[-1]:g} %"
    d4_mean = float(d4s.csu_bias_pp.mean())
    d4_mean_txt = f"{d4_mean:.2f}".replace("-", "−")
    d4_t2 = float(t2.loc[(t2.Dataset == "D7d") & (t2.protocol == "high"), "Bias"].iloc[0])
    assert abs(d4_mean - d4_t2) < 0.01, (d4_mean, d4_t2)
    rot_free_txt = ", ".join(
        "{} at {}° (in-range RMSE lower by {:.2f} pp, test level moved by {:.2f} pp)".format(
            k, mn("{:.1f}".format(mt22[k]["theta_star"])), gain[k], dtheta[k])
        for k in rot_free)
    band00_txt = f"{band00_rng[0]:.2f}–{band00_rng[1]:.2f}"
    lab_txt = f"{lab_rng[0]:.2f}–{lab_rng[1]:.2f} pp"
    unl_txt = f"{unl_rng[0]:.2f}–{unl_rng[1]:.2f} pp"
    shr_txt = f"{shr_rng[0]:.1f}–{shr_rng[1]:.1f}"

    # the passing condition with the largest extrapolation error, from the main analysis
    csu_rmse = pd.concat([
        pd.read_csv(OUT / "exp03_summary.csv").query("method == 'CSU-PL'"),
        pd.read_csv(OUT / "exp09_d7_summary.csv").query("method == 'CSU-PL'"),
    ])[["dataset", "protocol", "rmse"]].drop_duplicates()
    passing = csu_rmse[csu_rmse.dataset.isin(["D1", "D2", "D7d"])]
    assert len(passing) == 9, len(passing)
    worst_pass = str(passing.loc[passing.rmse.idxmax(), "dataset"]) + "/" + \
        str(passing.loc[passing.rmse.idxmax(), "protocol"])
    assert worst_pass == shr_min_key, (worst_pass, shr_min_key)
    worst_pass_rmse = float(passing.rmse.max())
    th_lo, th_hi = float(th22.theta.min()), float(th22.theta.max())
    th_step = float(np.diff(np.sort(th22.theta.unique())).min())
    grid_txt = "{}° to {}° in {}° steps".format(
        mn("{:.1f}".format(th_lo)), mn("{:.1f}".format(th_hi)),
        "{:g}".format(th_step))

    # ---------------- S21 (quantification limits from the calibrated interval) ----------------
    lq = pd.read_csv(OUT / "exp18_loq.csv").set_index("dataset")
    LIMIT_ORDER = ["D1", "D2", "D3", "D7d", "D7p", "D7w", "D8"]
    assert set(lq.index) == set(LIMIT_ORDER), list(lq.index)
    lq = lq.loc[LIMIT_ORDER]
    for nm in LIMIT_ORDER:
        r = lq.loc[nm]
        gate = "pass" if (audit[nm]["endmember"]["cnr"] >= 10
                          and abs(audit[nm]["curvature"]["gamma"]) <= 1) else "fail"
        assert r.audit == gate, (nm, r.audit, gate)
        assert bool(r.loq20_in_range) == bool(
            np.isfinite(r.loq20_pp) and r.loq20_pp <= r.c_max)
        assert bool(r.loq10_in_range) == bool(
            np.isfinite(r.loq10_pp) and r.loq10_pp <= r.c_max)
    cvc = pd.read_csv(OUT / "exp18_loq_curve.csv")
    c_grid = np.sort(cvc[cvc.dataset == "D1"].c_pct.unique())
    c_lo, c_hi = float(c_grid.min()), float(c_grid.max())
    c_step = float(np.diff(c_grid).min())
    chk = (float(lq.synth_over_real.min()), float(lq.synth_over_real.max()))
    assert 0.7 <= chk[0] and chk[1] <= 1.1, chk
    in20 = [nm for nm in LIMIT_ORDER if bool(lq.loc[nm, "loq20_in_range"])]
    in10 = [nm for nm in LIMIT_ORDER if bool(lq.loc[nm, "loq10_in_range"])]
    none20 = [nm for nm in LIMIT_ORDER if not np.isfinite(lq.loc[nm, "loq20_pp"])]
    beyond = [nm for nm in LIMIT_ORDER if np.isfinite(lq.loc[nm, "loq20_pp"])
              and not bool(lq.loc[nm, "loq20_in_range"])]

    def lim(v: float) -> str:
        return "none" if not np.isfinite(v) else f"{v:.1f}"

    def and_join2(v):
        return ", ".join(v[:-1]) + (" and " + v[-1] if len(v) > 1 else v[0])

    s22a = ["| Dataset | Gate | Levels | Measured range (%) | Inflation f "
            "| 95 % calibrated width at the highest prepared level (pp) | Limit at 20 % (pp) "
            "| Limit at 10 % (pp) | 20 % limit in range | 10 % limit in range |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for nm in LIMIT_ORDER:
        r = lq.loc[nm]
        s22a.append(
            f"| {nm} | {r.audit} | {int(r.n_levels)} | 0–{num(r.c_max)} | {r.f:.2f} "
            f"| {r.u95_at_cmax_pp:.2f} | {lim(r.loq20_pp)} | {lim(r.loq10_pp)} "
            f"| {'yes' if r.loq20_in_range else 'no'} "
            f"| {'yes' if r.loq10_in_range else 'no'} |")

    # ---------------- S22：逐比较器的审计相关（exp23） ----------------
    pc = pd.read_csv(OUT / "exp23_percomparator.csv").set_index("method")

    def sgn(v: float, nd: int = 2) -> str:
        """带符号数值，负号用 U+2212，与 SI 其余表格一致。"""
        return f"{v:+.{nd}f}".replace("-", "−")

    s23_name = {"CSU-PL": "Proposed (CSU-PL)", "Best comparator": "Best per condition"}
    s23_prose = {"CSU-PL": "the proposed estimator", "CLS-2pt": "the two-point physical model",
                 "MCR-ALS": "MCR-ALS", "PLS": "PLS", "Ridge": "ridge",
                 "SVR": "support-vector regression", "RF": "random forest",
                 "HGB": "gradient boosting", "MLP": "the perceptron"}
    s23_order = ["CSU-PL", "CLS-2pt", "MCR-ALS", "PLS", "Ridge", "SVR", "RF", "HGB",
                 "MLP", "Best comparator"]
    s23 = ["| Comparator | n | ρ(CNR, RMSEP) | p | Leave-one-out range "
           "| ρ(\\|γ\\|, \\|1 − slope\\|) | p |",
           "|---|---|---|---|---|---|---|"]
    for k in s23_order:
        r = pc.loc[k]
        s23.append(
            f"| {s23_name.get(k, k)} | {int(r.n_datasets)} | {sgn(r.rho_cnr)} | "
            f"{pfmt(r.p_cnr)} | {sgn(r.loo_min)} to {sgn(r.loo_max)} | "
            f"{sgn(r.rho_gamma_slope)} | {pfmt(r.p_gamma_slope)} |")
    comp = pc.drop(index="Best comparator")
    assert (comp.rho_cnr < 0).all(), "S22 note claims a negative contrast coefficient for all"
    assert (comp.rho_gamma_slope > 0).all(), "S22 note claims a positive curvature coefficient for all"
    cnr_lo = comp.rho_cnr.min()
    cnr_lo_name = comp.rho_cnr.idxmin()
    gam_lo, gam_hi = comp.rho_gamma_slope.min(), comp.rho_gamma_slope.max()
    n_four = int((comp.n_datasets == 4).sum())
    assert n_four == 4, "S22 note says four comparators were restricted to four datasets"
    weak = comp.loc[["SVR", "RF", "HGB"], "rho_cnr"]
    assert float(weak.max() - weak.min()) < 1e-9, "S22 note quotes one value for the three"
    four = comp[comp.n_datasets == 4]
    n_flip_four = int((four.loo_max > 0).sum())
    assert n_flip_four == 3, "S22 note says three of the four four-dataset rows flip sign"
    n_flip_txt = {1: "one", 2: "two", 3: "three", 4: "four"}[n_flip_four]
    p_floor = 2 / 24
    note22 = (
        "Note: RMSEP is the mean over seeds at each condition and the dataset value is "
        "the median over its three protocols, as in Table 2; each coefficient is a "
        "dataset-level Spearman correlation with an exact two-sided permutation p-value "
        "over all orderings of the datasets that the comparator covers. The contrast "
        f"coefficient is negative for all nine comparators, from {sgn(cnr_lo)} for "
        f"{s23_prose[cnr_lo_name]} to {sgn(weak.iloc[0])} for support-vector regression, "
        "random forest and gradient boosting, and the curvature–slope coefficient is "
        f"positive for all nine ({sgn(gam_lo)} to {sgn(gam_hi)}), so the direction of the "
        "audit is a property of the spectra rather than of the proposed estimator. The "
        "four comparators restricted to D1–D3 and D8 cannot reach an exact p below "
        f"{p_floor:.3f}, and for {n_flip_txt} of them the leave-one-out range includes "
        "positive values, so their coefficients are descriptive. The two ensembles are "
        "also the least "
        "accurate comparators where the contrast is strongest (median RMSEP "
        f"{comp.loc['RF', 'median_rmsep']:.1f} and {comp.loc['HGB', 'median_rmsep']:.1f} % "
        "over the conditions they cover), so their error is set by model bias rather "
        "than by contrast. The reference row reproduces the ρ = −0.79 of Table S5."
    )

    # ---------------- S23：外部验证（exp24） ----------------
    ext_audit = json.loads((OUT / "exp24_ext_audit.json").read_text(encoding="utf-8"))
    ext_sum = pd.read_csv(OUT / "exp24_ext_summary.csv")
    ext_pred = pd.read_csv(OUT / "exp24_ext_predictions.csv")
    ext_rob = json.loads((OUT / "exp24b_ext_robust.json").read_text(encoding="utf-8"))

    EXT_SYS = [("E1", "Sudan IV", "E1", "Red chilli powder / Sudan IV"),
               ("E2", "corn", "E2a", "Edible insect powder / corn flour"),
               ("E2", "wheat", "E2b", "Edible insect powder / wheat flour"),
               ("E2", "soybean", "E2c", "Edible insect powder / soybean flour")]
    EXT_KEYS = [f"{ds}_{cond.replace(' ', '')}" for ds, cond, _, _ in EXT_SYS]

    s24a = ["| System | Matrix / adulterant | Spectra | Brands | Levels | Range (%) "
            "| Channels | CNR | CNR_eff | γ | Leak | Adjacent SNR | Audit |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, cond, lab, desc), key in zip(EXT_SYS, EXT_KEYS):
        a = ext_audit[key]
        s24a.append(
            f"| {lab} | {desc} | {a['n']} | {a['brands']} | {len(a['levels'])} "
            f"| {num(min(a['levels']))}–{num(max(a['levels']))} | {a['channels']:,} "
            f"| {a['cnr']:.1f} | {a['cnr_eff']:.1f} | {sgn(a['gamma'])} "
            f"| {a['leak']:.2f} | {a['resolv']:.1f} | {a['gate']} |")
    ext_gates = [ext_audit[k]["gate"] for k in EXT_KEYS]
    assert ext_gates == ["fail", "pass", "pass", "pass"], ext_gates
    assert ext_audit["E1_SudanIV"]["cnr"] > 4000, "S23 text calls E1 the largest contrast"
    assert abs(ext_audit["E1_SudanIV"]["gamma"]) > 3, "S23 text quotes γ ≈ −3.6 for E1"

    PROT_ORDER = ["high", "interp", "low"]
    s24b = ["| System | Protocol | Calibration (%) | Test (%) | n cal | n test | RMSEP (%) "
            "| Slope | Bias (pp) | Best comparator | RMSEP best (%) | Ratio |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, cond, lab, _), key in zip(EXT_SYS, EXT_KEYS):
        for pr in PROT_ORDER:
            s = ext_sum[(ext_sum.dataset == ds) & (ext_sum.condition == cond)
                        & (ext_sum.protocol == pr)]
            p = ext_pred[(ext_pred.dataset == ds) & (ext_pred.condition == cond)
                         & (ext_pred.protocol == pr) & (ext_pred.method == "CSU-PL")]
            if len(s) != 1 or len(p) != 1:
                raise KeyError(f"missing external row for {(key, pr)}")
            s, p = s.iloc[0], p.iloc[0]
            tr = f"{num(p.tr_lo)}–{num(p.tr_hi)}"
            te = num(p.te_lo) if p.te_lo == p.te_hi else f"{num(p.te_lo)}–{num(p.te_hi)}"
            slope = "—" if not np.isfinite(s.slope) else sgn(s.slope)
            s24b.append(
                f"| {lab} | {pr} | {tr} | {te} | {int(p.n_tr)} | {int(p.n_te)} "
                f"| {s.cs:.2f} | {slope} | {sgn(s.bias)} | {s.best_method} "
                f"| {s.best_rmse:.2f} | {s.ratio:.2f} |")

    VAR_LABEL = {"base": "Base", "baseline3": "Cubic baseline", "sg11": "SG window 11",
                 "sg21": "SG window 21", "refhalf": "Reference half"}
    s24c = ["| Variant | E1 | E2a | E2b | E2c | Median ratio, E2a–E2c | Median ratio, E1 |",
            "|---|---|---|---|---|---|---|"]
    for v in ("base", "baseline3", "sg11", "sg21", "refhalf"):
        r = ext_rob[v]
        g = [r["audit"][k]["gate"] for k in EXT_KEYS]
        assert g == ["fail", "pass", "pass", "pass"], (v, g)
        s24c.append(f"| {VAR_LABEL[v]} | " + " | ".join(g)
                    + f" | {r['med_pass']:.3f} | {r['med_fail']:.3f} |")
    med_pass = float(ext_rob["base"]["med_pass"])
    med_fail = float(ext_rob["base"]["med_fail"])
    assert abs(med_pass - 0.922) < 5e-4 and abs(med_fail - 1.279) < 5e-4, (med_pass, med_fail)
    pass_lo = min(r["med_pass"] for r in ext_rob.values())
    pass_hi = max(r["med_pass"] for r in ext_rob.values())
    fail_lo = min(r["med_fail"] for r in ext_rob.values())
    fail_hi = max(r["med_fail"] for r in ext_rob.values())
    e1 = ext_audit["E1_SudanIV"]
    assert e1["level_counts"][0] == 24 and min(e1["level_counts"][1:]) == 8
    assert e1["level_counts"][-2] == 13
    note24a = (
        "Note: E1 is released under CC BY 4.0 and E2 under CC BY-NC 4.0, used here for "
        "non-commercial research. The spectra of E1 are transmittance and were converted with "
        "$A=-\\log_{10}(T/100)$; the released adulterated sheet repeats blocks of columns, "
        "and identical duplicates were removed, leaving "
        f"{e1['level_counts'][0]} pure and {sum(e1['level_counts'][1:])} spiked spectra "
        f"({min(e1['level_counts'][1:])}–{max(e1['level_counts'][1:])} per spiked level). "
        "The insect system shares its pure spectra across the three flours, so the "
        f"{3 * ext_audit['E2_corn']['n']} analysed spectra of E2a–E2c are "
        f"{ext_audit['E2_corn']['level_counts'][0] + 3 * sum(ext_audit['E2_corn']['level_counts'][1:])} "
        "distinct measurements. CNR is the baseline-projected contrast-to-noise ratio and γ the "
        "relative quadratic coefficient of the level-mean response (Section 2.3 of the main text); "
        "CNR_eff, leak and the adjacent signal-to-noise ratio (SNR) follow the note to Table S4, "
        "and all are computed from all spectra of the system. The quadratic term of E1 is fitted "
        f"to {len(e1['levels'])} levels with an F-test p of {e1['p_quad']:.2f}, so that verdict "
        "rests on the magnitude of γ, as the gate specifies, rather than on the significance of "
        "the fitted curvature. Levels and ranges are mass fractions of the adulterant (%, w/w)."
    )
    note24b = (
        "Note: both datasets were read after the seven development datasets had been analysed "
        "and nothing in the configuration was changed for them. Ratios are the proposed RMSEP "
        "divided by the lowest comparator RMSEP within the condition, a descriptive upper bound "
        "that carries no p-value; values below one favour the proposed estimator. RMSEP is the root-mean-square "
        "error of prediction and pp stands for percentage points. The low-end protocol of E1 holds "
        "out a single level, so its slope is undefined (—). The comparators are partial least "
        "squares (PLS), ridge regression, two-point classical least squares (CLS-2pt) and "
        "multivariate curve resolution–alternating least squares (MCR-ALS); the proposed estimator "
        "is concentration-supervised unmixing with a profile likelihood (CSU-PL). Over the "
        f"out-of-range protocols the median ratio is {med_pass:.2f} in the three systems that pass "
        f"the audit and {med_fail:.2f} in the one that fails; the interpolation controls give 1.20 "
        "for E1 and 1.16, 1.32 and 2.13 for E2a–E2c, a median of 1.26. Resampling intervals are "
        "not reported for these systems: every test condition holds two levels with four replicate "
        "spectra per level, so a bootstrap over replicates would carry no information about "
        "level-to-level variation, and the ratio is the quantity compared."
    )
    note24c = (
        "Note: every variant repeats the audit and the three protocols with the stated change. "
        "SG is the Savitzky–Golay first derivative of Section 3.2 (window in channels, third-order "
        "polynomial, first derivative). The reference half computes the audit on two of the four "
        "replicates of each level and evaluates on the other two, so the two halves share no "
        "spectrum. The out-of-range median ratio stays within "
        f"{pass_lo:.2f}–{pass_hi:.2f} for the passing systems and {fail_lo:.2f}–{fail_hi:.2f} for "
        "the failing one, and the four verdicts are unchanged in every variant."
    )

    # ---------------- S24：审计量的稳定性（exp26b 分层重采样 + exp26 γ 敏感性） ----------------
    INV = DISP2KEY  # 展示编号 -> 分析键（relabel 写回时再转回展示编号）
    st = pd.read_csv(OUT / "exp26b_audit_boot_strat.csv")
    fl = pd.read_csv(OUT / "exp26b_gate_flips.csv").set_index("dataset")
    au = json.loads((OUT / "exp05_audit.json").read_text(encoding="utf-8"))
    assert len(st) == 14, len(st)
    st["qty"] = [
        "CNR" if abs(r.point - au[INV[r.dataset]]["endmember"]["cnr"]) < 1e-6 else "gamma"
        for r in st.itertuples()]
    assert st.groupby("dataset").qty.nunique().eq(2).all(), "each dataset needs one CNR and one gamma row"
    n_boot = int(st.n_boot.iloc[0])
    s25a = ["| Dataset | CNR | CNR 2.5–97.5 % | γ | γ 2.5–97.5 % | Gate | Gate flips |",
            "|---|---|---|---|---|---|---|"]
    gate_txt = {}
    for disp in st.dataset.unique():
        sub = st[st.dataset == disp]
        c = sub[sub.qty == "CNR"].iloc[0]
        g = sub[sub.qty == "gamma"].iloc[0]
        gate_txt[INV[disp]] = f"{fl.loc[disp, 'gate']} ({int(round(fl.loc[disp, 'gate_flip'] * n_boot))} of {n_boot})"
        s25a.append(
            f"| {INV[disp]} | {c.point:.1f} | {c.pct_lo:.1f}–{c.pct_hi:.1f} "
            f"| {sgn(g.point)} | {sgn(g.pct_lo)}–{sgn(g.pct_hi)} "
            f"| {fl.loc[disp, 'gate']} | {int(round(fl.loc[disp, 'gate_flip'] * n_boot))} of {n_boot} |")
    max_flip = float(fl.gate_flip.max())
    n_cnr_spread = float((st[st.qty == "CNR"].pct_hi / st[st.qty == "CNR"].pct_lo).max())
    cnr_r = st[st.qty == "CNR"].set_index("dataset")
    gam_r = st[st.qty == "gamma"].set_index("dataset")

    def contains(frame):
        return [k for k, r in frame.iterrows() if r.pct_lo <= r.point <= r.pct_hi]

    cnr_in, gam_in = contains(cnr_r), contains(gam_r)
    cnr_above = [k for k in cnr_r.index if k not in cnr_in and cnr_r.loc[k, "pct_lo"] > cnr_r.loc[k, "point"]]
    gam_out_band = [k for k in gam_r.index if k not in gam_in]
    assert sorted(cnr_above) == ["D5", "D6"], cnr_above
    assert sorted(gam_out_band) == ["D6"], gam_out_band
    assert max_flip == 0.0, max_flip
    n_tight = int(((cnr_r.pct_hi / cnr_r.pct_lo) <= 1.2).sum())
    lift = {k: cnr_r.loc[k, "pct_lo"] / cnr_r.loc[k, "point"] for k in cnr_above}

    gs = pd.read_csv(OUT / "exp26_gamma_sens.csv")
    VAR_ORDER = ["unweighted (reported)", "weighted by spectra per level", "drop lowest level",
                 "drop highest level", "drop both end levels", "cubic fit, quadratic term"]
    VAR_SHORT = ["Reported", "Weighted by spectra per level", "Lowest level dropped",
                 "Highest level dropped", "Both end levels dropped", "Cubic fit, quadratic term"]
    s25b = ["| Dataset | " + " | ".join(VAR_SHORT) + " | Gate unchanged |",
            "|---|---|---|---|---|---|---|---|"]
    flip_ds = []
    for disp in st.dataset.unique():
        key = INV[disp]
        row = gs[gs.dataset == disp].set_index("variant").loc[VAR_ORDER, "gamma"]
        base = bool(abs(float(row.iloc[0])) <= 1)
        stable = bool(((row.abs() <= 1) == base).all())
        if not stable:
            flip_ds.append(key)
        s25b.append(f"| {key} | " + " | ".join(sgn(v) for v in row.values)
                    + f" | {'yes' if stable else 'no'} |")
    assert flip_ds == ["D7d"], flip_ds
    note25a = (
        f"Note: each dataset was re-audited on {n_boot} draws in which 80 % of the physical "
        "samples of every concentration level were drawn without replacement, so the level "
        "composition stays fixed and no spectrum is duplicated; a bootstrap that resamples within a "
        "level with replacement was tried first and abandoned because duplicating spectra reweights "
        "the calibration design. The ranges quoted are percentile ranges of that subsample "
        "distribution and are read as a stability band rather than as a confidence interval. The "
        "bands are narrow: the largest ratio of upper to lower bound over the seven CNR entries is "
        f"{n_cnr_spread:.2f} and {n_tight} of the seven lie within 1.2. The band covers the value "
        f"from the full reference set for {len(cnr_in)} of the seven systems for CNR and "
        f"{len(gam_in)} of seven for γ. The exceptions are the two near-degenerate cinnamon systems "
        f"{INV['D5']} and {INV['D6']}, whose CNR bands begin {(lift['D5'] - 1) * 100:.1f} % and "
        f"{(lift['D6'] - 1) * 100:.1f} % above the full-reference values of "
        f"{cnr_r.loc['D5', 'point']:.2f} and {cnr_r.loc['D6', 'point']:.2f} — both read on the "
        f"unrounded bounds, which the one-decimal columns above do not show — and {INV['D6']} again "
        f"in γ, whose band runs {sgn(gam_r.loc['D6', 'pct_lo'])} to "
        f"{sgn(gam_r.loc['D6', 'pct_hi'])} against the reported "
        f"{sgn(gam_r.loc['D6', 'point'])}. The "
        "numerator of Eq. (4) is the norm of an endmember direction refitted on each draw, so where "
        "the contrast is comparable to the estimation noise that distribution is skewed away from "
        "the point value; the noise term in the denominator is measured within each spectrum and "
        "the draw leaves it nearly unchanged. Neither offset changes a verdict: the two CNR values "
        "lie below the threshold of 10 by a factor of four and ten, and the γ value above the "
        "threshold of 1 by a factor of nearly four. CNR is the baseline-projected "
        "contrast-to-noise ratio (Eq. 4 of the main text) and γ the relative quadratic coefficient "
        "of the level-mean response (Eq. 5). The gate column is the decision of Eq. (6) at the full "
        "reference set, which is the verdict carried into the main text, and the last column counts "
        f"the draws that reverse it: none of the {n_boot * len(cnr_r)} draws does."
    )
    note25b = (
        "Note: γ is re-estimated under five alternatives to the reported fit of the level-mean "
        "response. Weighting gives each level a weight proportional to its number of spectra "
        "instead of one; the dropping rows omit the stated calibration level or levels; the cubic "
        "row reads the quadratic coefficient from a cubic fit. These variants test the reported "
        "value of γ against the two choices a reader could reasonably have made differently, the "
        "level spacing and the polynomial order. Five of the seven datasets keep the gate decision "
        "under all variants. The exception is D4, whose |γ| crosses one when the lowest level is "
        "dropped, when both end levels are dropped and when a cubic is fitted; D4 also fails the "
        "endmember shape check, so its outcome is partial or fail under every variant and never "
        "pass. No dataset that passes in Table 1 loses that verdict here."
    )

    # ---------------- S25：MCR-ALS 的 transductive / inductive 两版（exp28） ----------------
    mc = pd.read_csv(OUT / "exp28_mcr_inductive.csv")
    assert len(mc) == 21, len(mc)
    mc["ver"] = mc.dataset.map({"D1": "pass", "D2": "pass", "D3": "fail", "D4": "partial",
                                "D5": "fail", "D6": "fail", "D7": "fail"})
    s26a = ["| Dataset | Protocol | CSU RMSEP (%) | MCR-ALS transductive (%) "
            "| MCR-ALS inductive (%) | Ratio, transductive | Ratio, inductive |",
            "|---|---|---|---|---|---|---|"]
    for r in mc.itertuples():
        s26a.append(
            f"| {INV[r.dataset]} | {r.protocol} | {r.rmse_csu:.2f} | {r.rmse_mcr_transduct:.2f} "
            f"| {r.rmse_mcr_ind:.2f} | {r.ratio_transduct:.2f} | {r.ratio_inductive:.2f} |")
    mp_ = mc[mc.ver == "pass"]
    mf_ = mc[mc.ver == "fail"]
    mc_better = mp_.rmse_mcr_ind <= mp_.rmse_mcr_transduct
    mcr_worse = int((mp_.rmse_mcr_ind > mp_.rmse_mcr_transduct).sum())
    mcr_gain = float((mp_.rmse_mcr_transduct - mp_.rmse_mcr_ind)[mc_better].max())
    mf_worse = int((mf_.rmse_mcr_ind > mf_.rmse_mcr_transduct).sum())
    assert (mcr_worse, int(mc_better.sum()), len(mp_)) == (4, 2, 6)
    assert abs(mcr_gain - 0.0807) < 5e-4 and mf_worse == 7 and len(mf_) == 12
    mcr_note = (
        "Note: the transductive variant decomposes the calibration and test spectra together and "
        "is the conventional use of MCR-ALS; the inductive variant decomposes the calibration "
        "spectra only and projects the test spectra onto the fixed factors, which is what a "
        "method would have to do prospectively. Ratios are the proposed RMSEP divided by the "
        "MCR-ALS RMSEP, so a value below one favours the proposed estimator. Comparing the two "
        "MCR-ALS columns, the inductive factorisation gives the higher RMSEP in "
        f"{mcr_worse} of the {len(mp_)} passing conditions and the lower one in "
        f"{int(mc_better.sum())}, where the gain is below 0.1 % RMSEP, so the "
        "prospective variant is the weaker of the two for MCR-ALS (the median ratio over these "
        f"conditions is {mp_.ratio_transduct.median():.2f} transductive against "
        f"{mp_.ratio_inductive.median():.2f} inductive). Over the failing conditions the two "
        f"variants are close (median {mf_.ratio_transduct.median():.2f} against "
        f"{mf_.ratio_inductive.median():.2f}, inductive higher in {mf_worse} of {len(mf_)}), "
        "so the comparison of Section 4.2, which uses the transductive variant, is the lenient "
        "one for MCR-ALS."
    )

    # ---------------- S26：校准量匹配、品牌回退、基线阶数（exp27 / exp27b） ----------------
    ms = pd.read_csv(OUT / "exp27_matched_size.csv")
    mb = pd.read_csv(OUT / "exp27b_matched_ratio.csv")
    t2b = pd.read_csv(ROOT / "tables" / "table2_main.csv")
    PMAP = {"high-end": "high", "interp.": "interp", "low-end": "low"}
    R2D = {"D7d": "D4", "D7p": "D5", "D7w": "D6", "D8": "D7"}
    t2b["dataset"] = t2b.Dataset.map(lambda x: R2D.get(x, x))
    t2b["protocol"] = t2b.Protocol.map(PMAP)
    mm = ms.merge(t2b[["dataset", "protocol", "RMSEP"]], on=["dataset", "protocol"])
    assert len(mm) == 21, len(mm)
    mm["delta"] = mm.rmsep_matched - mm.RMSEP
    d_med = float(mm.delta.abs().median())
    d_max = float(mm.delta.abs().max())
    d_arg = mm.loc[mm.delta.abs().idxmax()]
    assert abs(d_med - 0.009) < 5e-3 and abs(d_max - 0.54) < 0.006, (d_med, d_max)
    s27a = ["| Dataset | Protocol | Calibration spectra, full | Calibration spectra, matched "
            "| RMSEP, full (%) | RMSEP, matched (%) | Δ (pp) |",
            "|---|---|---|---|---|---|---|"]
    for r in mm.itertuples():
        s27a.append(
            f"| {INV[r.dataset]} | {r.protocol} | {int(r.n_cal_full)} | {int(r.n_cal_matched)} "
            f"| {r.RMSEP:.2f} | {r.rmsep_matched:.2f} | {sgn(r.delta)} |")
    s27b = ["| Dataset | Protocol | Brand | Calibration spectra | RMSEP, brand-specific (%) "
            "| RMSEP, brand-averaged (%) | Ratio |", "|---|---|---|---|---|---|---|"]
    bf = pd.read_csv(OUT / "exp27_brand_fallback.csv")
    br_ratio = bf.rmsep_fallback / bf.rmsep_brand
    assert abs(float(br_ratio.median()) - 0.99) < 0.006, br_ratio.median()
    assert int((br_ratio > 1).sum()) == 22, int((br_ratio > 1).sum())
    for r in bf.itertuples():
        s27b.append(
            f"| {INV[r.dataset]} | {r.protocol} | {r.brand} | {int(r.n_cal_wo)} "
            f"| {r.rmsep_brand:.2f} | {r.rmsep_fallback:.2f} "
            f"| {r.rmsep_fallback / r.rmsep_brand:.2f} |")
    bq1, bq3 = float(br_ratio.quantile(.25)), float(br_ratio.quantile(.75))
    bby = br_ratio.groupby(bf.dataset).median()
    s27c = ["| Dataset | Protocol | Linear baseline (%) | Quadratic baseline (%) "
            "| Cubic baseline (%) | Largest \\|Δ\\| (pp) |", "|---|---|---|---|---|---|"]
    bo = pd.read_csv(OUT / "exp27_baseline_order.csv")
    bo_delta = pd.concat([(bo.rmsep_o2 - bo.rmsep_o1).abs(), (bo.rmsep_o3 - bo.rmsep_o1).abs()])
    assert abs(float((bo.rmsep_o2 - bo.rmsep_o1).abs().max()) - 0.31) < 0.006, "quadratic max"
    assert abs(float((bo.rmsep_o3 - bo.rmsep_o1).abs().max()) - 0.46) < 0.006, "cubic max"
    for r in bo.itertuples():
        s27c.append(
            f"| {INV[r.dataset]} | {r.protocol} | {r.rmsep_o1:.2f} | {r.rmsep_o2:.2f} "
            f"| {r.rmsep_o3:.2f} | {max(abs(r.rmsep_o2 - r.rmsep_o1), abs(r.rmsep_o3 - r.rmsep_o1)):.2f} |")
    PRE4 = ["PLS", "Ridge", "CLS-2pt", "MCR-ALS"]
    MB4 = ["ratio_" + c for c in PRE4]
    mb_med = float(mb[MB4].stack().median())
    mb_ahead = int((mb[MB4] < 1).all(axis=1).sum())
    mb_behind = int((mb[MB4] > 1).all(axis=1).sum())
    mb_mixed = len(mb) - mb_ahead - mb_behind
    mb_cells = int((mb[MB4] < 1).values.sum())
    mb_over = int((mb[MB4] > 1).values.sum())
    assert abs(mb_med - 0.963) < 5e-3, mb_med
    assert (mb_ahead, mb_mixed, mb_behind) == (8, 9, 4), (mb_ahead, mb_mixed, mb_behind)
    assert (mb_cells, mb_over, len(mb) * 4) == (49, 35, 84), (mb_cells, mb_over)
    pu = pre6[pre6.comparator.isin(PRE4)]
    ar_med = float(pu.ratio.median())
    piv_u = pu.pivot_table(index=["dataset", "protocol"], columns="comparator",
                           values="ratio")[PRE4]
    ar_ahead = int((piv_u < 1).all(axis=1).sum())
    ar_behind = int((piv_u > 1).all(axis=1).sum())
    ar_mixed = len(piv_u) - ar_ahead - ar_behind
    ar_cells = int((pu.ratio < 1).sum())
    ar_over = int((pu.ratio > 1).sum())
    assert abs(ar_med - 0.960) < 5e-3, ar_med
    assert (ar_ahead, ar_mixed, ar_behind) == (9, 8, 4), (ar_ahead, ar_mixed, ar_behind)
    assert (ar_cells, ar_over, len(piv_u) * 4) == (49, 35, 84), (ar_cells, ar_over)
    ridge_d2 = (float(piv_u.at[("D2", "interp"), "Ridge"]),
                float(mb.loc[(mb.dataset == "D2") & (mb.protocol == "interp"), "ratio_Ridge"].iloc[0]))
    assert ridge_d2[0] < 1 < ridge_d2[1], ridge_d2
    s27d = ["| Draw | n cal | Median ratio, every comparator | Conditions ahead of all four "
            "| Conditions mixed | Conditions behind all four | Cell ratios below one |",
            "|---|---|---|---|---|---|---|"]
    s27d.append(f"| Matched to the smallest protocol | {int(mb.n_cal_matched.iloc[0])} "
                f"| {mb_med:.3f} | {mb_ahead} of {len(mb)} | {mb_mixed} of {len(mb)} "
                f"| {mb_behind} of {len(mb)} | {mb_cells} of {len(mb) * 4} |")
    s27d.append(f"| As reported, protocol sizes unequal | — | {ar_med:.3f} "
                f"| {ar_ahead} of {len(piv_u)} | {ar_mixed} of {len(piv_u)} "
                f"| {ar_behind} of {len(piv_u)} | {ar_cells} of {len(piv_u) * 4} |")
    note27 = (
        "Note: panel (a) subsamples every condition to the smallest calibration size of its "
        "dataset over ten matched draws and reports the mean RMSEP of those draws; the median "
        f"absolute change is {d_med:.3f} percentage points and the largest {d_max:.2f} at the "
        f"interpolation condition of {INV[d_arg.dataset]}. Panel (b) refits each model without one "
        "brand and predicts that brand with the matrix endmember averaged over the remaining "
        "brands; ratios are brand-averaged over brand-specific RMSEP, and values above one mean "
        f"that the brand-specific endmember matters. The median ratio is {br_ratio.median():.2f} "
        f"(interquartile range {bq1:.2f}–{bq3:.2f}); the degradation is concentrated in D3 "
        f"(median {(bby.get('D3', np.nan)):.2f}) and negligible in D1 and D2 "
        f"({bby.get('D1', np.nan):.2f} and {bby.get('D2', np.nan):.2f}). Panel (c) replaces the "
        "constant baseline of Eq. (1) by a quadratic or a cubic one and leaves the audit gate "
        "outcome of every dataset unchanged. Panel (d) reruns the four prespecified comparators on "
        "the same matched draws as panel (a). Its counts are condition counts: a condition is "
        "ahead of all four when each of its four ratios is below one, behind all four when each is "
        "above one, and mixed otherwise, so the three condition columns add to the 21 conditions "
        "and the last column counts the 84 cells. Matching the calibration sizes changes the cell "
        f"balance not at all ({mb_cells} ratios below one and {mb_over} above, against {ar_cells} "
        f"and {ar_over} as reported) and moves the median ratio from {ar_med:.3f} to "
        f"{mb_med:.3f}. The behind-all-four count stays at {mb_behind}, and one condition, "
        f"the interpolation control of D2, leaves the ahead-of-all-four group because its ridge "
        f"ratio crosses one ({ridge_d2[0]:.4f} as reported, {ridge_d2[1]:.4f} matched). The "
        "extrapolation penalty is therefore not an artefact of unequal calibration sizes. "
        "pp is percentage points."
    )

    text = f"""# Supplementary material

**Beyond the calibration range: identifiability and extrapolation of concentration-supervised Beer–Lambert unmixing for ATR-FTIR adulterant quantification**

This supplement reports the exact extrapolation protocols (Table S1), the full uncertainty analysis (Table S2), the complete robustness surface (Table S3), the per-dataset audit and label-resolvability statistics (Table S4), the inferential analysis with dataset-level permutation and cluster bootstrap (Table S5), the fixed-comparator sensitivity (Table S6), the audit-threshold grid (Table S7), the split-level audit, endmember physicality, and non-negativity ablation (Table S8), the design replication on four datasets (Table S9), the conformal-interval comparison (Table S10), the audit re-estimated on a disjoint reference set (Table S11), the threshold-learning check (Table S12), the attenuated total reflectance (ATR) penetration-depth ablation (Table S13), the scan-level versus sample-level split control (Table S14), the Monte-Carlo validation of the noise bound (Table S15), the dataset inventory read from the three workbooks of the Kolmogorov–Arnold network (KAN) study (Table S16), the photometric domain of the released tables together with the relation between the cinnamon and turmeric deposits (Table S17), the wave-number grid and smoothing-window sensitivity (Table S18), the capacity-matched comparators (Table S19), the rotational feasible band of the two-component closure (Table S20), the quantification limits derived from the calibrated interval (Table S21), the audit–error correlation of every comparator taken separately (Table S22), the external validation on two independently deposited datasets (Table S23), the stability of the audit quantities under subsampling and under alternative curvature fits (Table S24), the transductive and inductive variants of multivariate curve resolution–alternating least squares (Table S25), and the calibration-size, brand-fallback and baseline-order sensitivities (Table S26); Fig. S1 is the condition-number panel of the design sweep, referred to in Section 4.4 of the main text. The dataset labels follow the main text: D1 is olive oil adulterated with sunflower oil, D2 roasted coffee with malt flour, D3 orange juice with apple juice, D7d, D7p and D7w cinnamon powder with wood dust, peanut shell and walnut shell, and D8 turmeric powder with wheat flour. In the external validation of Table S23, E1 is red chilli powder adulterated with Sudan IV and E2a, E2b and E2c are edible insect powder adulterated with corn, wheat and soybean flour. All entries are generated from the same analysis code and public datasets as the main text; the machine-readable versions are provided with the analysis code.

## S1. Extrapolation protocols

**Table S1.** Exact calibration and test concentration levels for the 21 conditions. Levels are mass fractions of the adulterant (%, w/w). Samples are physical samples; n is the number of test spectra analysed.

{chr(10).join(s1)}

Note: the low-end protocols exclude the pure-matrix spectra (0 %) from calibration. In the high-end protocols, one level is left unused between calibration and test to create a gap (18 % for D1; 30 % for D2 and D3; 10 % for the D7 datasets); the D8 high-end protocol is contiguous. The D7 low-end protocols also exclude 0 %.

## S2. Interval coverage and width

**Table S2.** Full uncertainty results for all 21 conditions. f is the cross-validated inflation factor applied to the profile-likelihood half-widths; cov68 and cov95 are the empirical coverages of the 68 % and 95 % intervals before (raw) and after (cal) inflation; w95 is the median 95 % interval width in percentage points (pp). RMSEP is the root-mean-square error of prediction, repeated here to allow the coverage to be read against the error.

{chr(10).join(s2)}

Note: raw intervals use the calibration residual noise level; calibrated intervals multiply the half-widths by f, with the interval clipped to [0, 1]. Coverage is evaluated on the held-out test spectra of each condition.

## S3. Full robustness surface

**Table S3.** Semi-synthetic robustness at every factor level (D2, calibration 0–25 %, test 50 %). Entries are mean ± standard deviation of RMSEP over five seeds (%), for the proposed estimator, concentration-supervised unmixing with a profile likelihood (CSU-PL), and the additive-only variant (fixed scale, s = 1). σ0 is the nominal white-noise level of the base simulation, and CV is the coefficient of variation of the multiplicative scattering factor. Factor levels are defined in Section 3.8.

{chr(10).join(s3)}

Note: the two variants differ only in whether the per-spectrum multiplicative scale is estimated. Noise, baseline mismatch, response nonlinearity and matrix variability are each swept with the other factors left at the base simulation, which already carries multiplicative scattering at CV 5 % and white noise at 1× σ0, so the zero level of those four factors is the same simulation as the 1× σ0 noise row and the four rows coincide. The scattering factor instead sets that background level itself, so its CV 0 % row is a simulation with no multiplicative scale variation at all, and the fixed-scale variant has nothing to correct there. At the largest noise and nonlinearity levels both variants degrade together, whereas baseline mismatch and matrix variability favour the fixed-scale variant.

## S4. Audit and label resolvability

**Table S4.** Per-dataset audit quantities and label-resolvability statistics. CNR is the baseline-projected contrast-to-noise ratio (Eq. 4); CNR_eff is the same ratio after projecting out the component along the matrix endmember, which is absorbed by the per-spectrum scale (Section 2.3); |γ| is the magnitude of the relative quadratic coefficient of the level-mean response (Eq. 5); leak is the fraction of endmember-contrast energy inside the baseline span; train RMSE is the root-mean-square error on the calibration data; adjacent SNR (signal-to-noise ratio) is the median ratio between adjacent calibration levels; nonneg. fraction is the fraction of endmember channels that are non-negative after the physicality audit. Perm. R² and p are the observed cross-validated R² of partial least squares (PLS) on the raw labels and the one-sided permutation p-value (100 sample-level label permutations); the permutation test was run for the three cinnamon datasets, where the same matrix supports three adulterants.

{chr(10).join(s4)}

{s4_note}

## S5. Statistical inference

**Table S5.** Spearman correlations between the audit quantities and the extrapolation outcomes, at the condition level (descriptive) and at the dataset level (inference). Dataset-level p-values are exact over all 7! orderings of the seven datasets; the 95 % confidence interval (CI) is a percentile cluster bootstrap that resamples datasets (5,000 replicates). Partial rows control for the variable named in the row.

{chr(10).join(s5)}

Note: condition-level p-values are shown for description only. The 21 conditions are seven datasets times three protocols, and CNR and |γ| are constant within a dataset, so those p-values treat correlated conditions as independent evidence. The dataset-level sample is seven, and the best-comparator correlation is the only one of the descriptive dataset-level coefficients that reaches the exact p < 0.05 level. The exact two-sided Mann–Whitney test on the per-dataset error ratios of the two passing against the four failing datasets has a minimum attainable p of 0.133, so that comparison is reported as an ordering and the per-cell counts of Table S6 carry the evidence.

## S6. Fixed-comparator sensitivity

**Table S6.** Descriptive comparison against each comparator individually, grouped by the outcome of the three-part endpoint of Table 1 of the main text: pass (D1, D2), partial (D4) and fail (D3, D5, D6, D8). The median ratio is the median of proposed/comparator RMSEP; a value below one favours the proposed estimator. The comparator labels are PLS (partial least squares), support-vector regression (SVR), random forest (RF), histogram gradient boosting (HGB), a two-hidden-layer perceptron (MLP), two-point classical least squares (CLS-2pt) and multivariate curve resolution–alternating least squares (MCR-ALS). Sig. counts are paired clustered-bootstrap results corrected by the Holm procedure within the condition at p < 0.05; the partial group holds one dataset, so its counts are listed without significance. The grouping follows the endpoint outcome, which is a property of the dataset, so the group medians are summaries of the per-cell ratios and carry no test of their own.

{chr(10).join(s6)}

{note6}

## S7. Audit-threshold sensitivity

**Table S7.** Classification of the seven datasets and the median error ratio of the resulting groups as a function of the gate of Eq. (6), the contrast-and-curvature part of the endpoint. Panel (a) varies the CNR threshold at |γ| = 1; panel (b) varies the |γ| threshold at CNR = 10.

(a)

{chr(10).join(s7a)}

(b)

{chr(10).join(s7b)}

Note: the full grid is 11 CNR thresholds (2–25) by 10 |γ| thresholds (0.1–4). Of the 87 combinations that leave at least two datasets on each side, 80 place the passing median below one and the failing median above one. The membership of the passing group is unchanged over CNR 5–15 and |γ| 0.75–3. At the most permissive curvature threshold (|γ| = 4) D8 joins the passing group, and at CNR ≤ 4 D3 does.

## S8. Split-level audit, endmember physicality, and non-negativity

**Table S8.** Split-level audit, endmember physicality, and non-negativity ablation. (a) Audit quantities re-estimated on the calibration subset of each condition, where n train is the number of calibration spectra. (b) Cosine between the estimated matrix endmember and the mean 0 % spectrum of the same brand, computed over the full reference set. (c) Effect of removing the non-negativity constraint.

(a) Audit re-estimated on each calibration split

{chr(10).join(s8a)}

(b) Endmember physicality on the full reference set

{chr(10).join(s8b)}

(c) Non-negativity ablation

{chr(10).join(s8c)}

Note: Table 1 of the main text computes the audit once per dataset from all reference spectra; the split-level values in panel (a) show what a calibration-only calculation would give. D7d passes in one of three splits and D8 in two of three, so a calibration-only audit is unstable and can reject or accept a system depending on the split. What panel (a) changes is the level span: the audit is confined to the calibration spectra of one extrapolation protocol, whose designed range stops short of the test levels, and the curvature of the level-mean response is then fitted over that truncated span. Table S11 changes only the samples, keeping every prepared level in the reference half, and there the gate reproduces itself. The two tables therefore answer different questions, and neither one is the audit of Table 1. In panel (b), the centred cosine subtracts the mean of each vector before the cosine; the raw cosine does not, and is low because non-negativity is imposed in the derivative domain, which clips the negative lobes and adds a constant (DC) offset. D1 and D2 contain no 0 % spectrum in the low-end calibration split, so their split-level endmember cosines are not defined; the values in panel (b) come from the full reference set. In panel (c), free means that non-negativity is removed in the Savitzky–Golay derivative domain, and Δ is the RMSEP change relative to the non-negative fit.

## S9. Design replication

**Table S9.** Design analysis on four datasets. The prediction leverage is evaluated at the test level of each design, and the designs are those enumerated in Section 3.6. The ratio is the median RMSEP of designs with h > 1 divided by the median of the rest. The area under the receiver-operating-characteristic curve (AUC) is the probability that a design with more than twice the median error receives a higher score than a design below that threshold; 0.5 is chance, and values below 0.5 indicate anti-prediction. Spearman ρ is computed across designs within each dataset; h is the prediction leverage and κ the condition number of the concentration design matrix.

{chr(10).join(s9)}

Note: the 640 designs comprise D1 (120), D2 (247), D3 (247), and D8 (26). Across designs within a dataset, leverage orders the error only in D2; in D1 and D3 the condition number is negatively correlated with the error, so a design with a larger condition number is not the one that extrapolates worse. The tail screen is what replicates: designs with h > 1 have 1.14–1.65 times the median error of the remaining designs in all four datasets, and leverage ranks designs with more than twice the median error with AUC 0.96–0.98 where such designs exist. D8 contains none, so its AUC is undefined.

!!FIG figures/figS1_kappa.png 8.2

**Fig. S1.** Condition number of the concentration design matrix against extrapolation error for the 247 D2 designs, the companion panel to the leverage screen of Fig. 5a. κ is the condition number of the concentration design matrix, defined in Section 3.6 of the main text. Navy points are designs without a 0 % anchor and vermillion points designs with one. The black line is the median error over octiles of κ and the grey band the interquartile range within each octile. ρ is the Spearman correlation across designs and p the two-sided p-value of the asymptotic test.

## S10. Conformal intervals

**Table S10.** Split-conformal intervals for all 21 conditions. q95 is the finite-sample 95 % quantile of the absolute out-of-fold residuals; the test interval is the prediction plus or minus q95, clipped to [0, 1]. Coverage and width are empirical values on the held-out test spectra. The Gaussian columns apply 1.96 times the cross-validated RMSE in place of q95, using the same folds. RMSEP is repeated for reference.

{chr(10).join(s10)}

Note: both interval families are built from in-range residuals and use no test labels, so the comparison isolates the effect of the noise assumption from the effect of extrapolation. Median coverage is {conf_med:.2f} for the conformal interval and {float(conf.cov_gauss.median()):.2f} for the Gaussian approximation; {conf_ok} of 21 conditions reach 0.75 for conformal, against {gauss_ok} for the Gaussian approximation and {cal_ok} for the inflated profile interval of Table S2. All six conditions in which the inflated profile interval falls below 0.75 also fail here, and the two additional failures (D7p low-end and D7w interpolation) carry biases of {'−' if extra[1] < 0 else '+'}{abs(extra[1]):.1f} and {'−' if extra[0] < 0 else '+'}{abs(extra[0]):.1f} percentage points. The two families cost about the same width: the median 95 % conformal interval is {conf_w_med:.1f} percentage points against {cal_w_med:.1f} for the inflated profile interval. A calibration built from in-range residuals cannot anticipate a shift of that size, which is why the audit rather than the interval is the primary screen.

## S11. Audit on a disjoint reference set

**Table S11.** Audit quantities re-estimated on a reference subset that is physically disjoint from the evaluation data, over five random splits per dataset. For D1–D3 the reference subset is two of the five brands and the evaluation subset is the remaining three; for D7 and D8 it is a concentration-stratified half of the physical samples. n_ref is the number of spectra in the reference subset; CNR_ref and γ_ref are medians over the five splits, with the range in parentheses; CNR full and γ full are the values in Table 1. Pass rate is the fraction of the 15 dataset–split–protocol combinations whose reference audit passes the gate. RMSEP and slope are medians on the complementary subset.

{chr(10).join(s11)}

Note: the reference audit reproduces the gate classification in {at_agree} of {at_n} dataset–split–protocol combinations; the endpoint of Table 1 also counts the endmember-shape and label-resolvability criteria, which are computed on the full reference set and are not re-estimated here, so D4 passes the gate on either set of spectra and remains partial as an endpoint. The audit is therefore not a fit to the evaluation spectra: it requires reference samples with known concentrations spanning the reporting range, but not the samples on which the out-of-range prediction is later made. Because the reference subset is disjoint at the brand level for D1–D3, the check also covers the transfer from reference brands to new brands.

## S12. Threshold learning

**Table S12.** Leave-one-dataset-out selection of the audit thresholds. For each held-out dataset, the CNR and |γ| thresholds (CNR thr., |γ| thr.) were chosen from the grid of Table S7 to maximize the balanced accuracy on the six training datasets (Train bal. acc.), and the resulting classifier was applied to the held-out dataset. The reference label (error-based) is pass when the median ratio of proposed to best-comparator RMSEP is below one; the learned-gate label is the prediction; the median error ratio is proposed/best-comparator RMSEP for the held-out dataset.

{chr(10).join(s12)}

Note: the reference label of this table is the error-based one, a median ratio of proposed to best-comparator RMSEP below one; the endpoint of Table 1 adds the endmember shape check and the reportable-concentration check, which is the sense in which D2 is a fail here and a pass there. The selection recovers CNR = 10 and |γ| = 1 in six of seven folds. The seventh (D7d held out) chooses |γ| = 0.1 because, without D7d, no training dataset has |γ| between 0.1 and 1 while its ratio is below one, so every threshold in that interval classifies the training data equally well. The learned classifier reproduces the reference label in {lodo['lodo_correct']} of 7 datasets, against {lodo['fixed_correct']} of 7 for the fixed 10/1 gate. The two misses are D2, whose median ratio is 1.003 and therefore a tie at the pass boundary, and D7d, the partial system discussed in Section 4.7.

## S13. ATR penetration-depth ablation

**Table S13.** Ablation of the ATR penetration-depth correction. Panel (a): the audit quantities computed with (corrected) and without (as used) the division of the raw absorbance by wavenumber, which accounts for the wavelength dependence of the sampling depth in an attenuated-total-reflection crystal; the correction is applied before the Savitzky–Golay derivative. Gate is the outcome of the audit rule CNR ≥ 10 and |γ| ≤ 1. Panel (b): test RMSEP of the proposed estimator and its ratio to the best-performing comparator of the same condition, re-estimated under the correction for both the estimator and every comparator. Panel (c): the same comparison against each comparator taken individually rather than the per-condition best.

{chr(10).join(s13a)}

Note: the correction leaves the gate unchanged for all {len(piv_gate)} datasets ({gate_flip} flips). It reduces |γ| in {gam_shrink} of 7 datasets, the largest relative changes being D3 (from {abs(piv_gam.at['D3', 'sg1']):.2f} to {abs(piv_gam.at['D3', 'atr_sg1']):.2f}) and D7d (from {abs(piv_gam.at['D7d', 'sg1']):.2f} to {abs(piv_gam.at['D7d', 'atr_sg1']):.2f}). Because the sampling depth grows with wavenumber, a constant concentration produces a smooth wavenumber-dependent multiplicative scale that differentiation converts into apparent level-dependent curvature; part of the curvature reported in Table 1 is therefore a path-length artefact rather than a chemistry effect. CNR moves in both directions ({piv_cnr.at['D1', 'sg1']:.1f}→{piv_cnr.at['D1', 'atr_sg1']:.1f} for D1, {piv_cnr.at['D2', 'sg1']:.1f}→{piv_cnr.at['D2', 'atr_sg1']:.1f} for D2) because the correction re-weights channels across the whole spectrum.

{chr(10).join(s13b)}

Note: the median absolute change in RMSEP over the 21 conditions is {s13d_med:.2f} percentage points, the largest being {s13d_max:.2f} percentage points ({s13d_max_key}). Median RMSEP in the gate-passing group moves from {ins_cb.rmse_sg1.median():.2f} to {ins_cb.rmse_atr.median():.2f} % and the median ratio to the per-condition best comparator from {ins_ratio_raw:.2f} to {ins_ratio_cor:.2f}; in the gate-failing group the corresponding ratios are {out_ratio_raw:.2f} and {out_ratio_cor:.2f}. The advantage over the best comparator in the gate-passing group therefore depends on the preprocessing: the correction improves some comparators more than it improves the estimator, and the number of conditions where the estimator beats the best comparator falls from {ins_win_raw} to {ins_win_cor} of 9.

{chr(10).join(s13c)}

Note: against a fixed comparator the ordering is unchanged by the correction. The largest in-gate median ratio over the eight comparators is {in_max_raw:.2f} as used and {in_max_cor:.2f} corrected, so the estimator stays ahead of every comparator in the gate-passing group under both variants. In the gate-failing group the same medians lie between {out_min_cor:.2f} and {out_max_cor:.2f} after the correction, against {min(v[0] for v in fixed_out.values()):.2f} and {max(v[0] for v in fixed_out.values()):.2f} as used: behind the regression comparators and level with the naive physical ones, as without the correction. The shift in panel (b) is a property of the oracle comparator, not of the estimator.

## S14. Scan-level versus sample-level splits

**Table S14.** Split-unit leakage control on the three datasets that record several scans per physical sample. Panel (a): the two split units at a fixed 20 % test fraction with the same level coverage. Panel (b): in-range accuracy, where the whole range is available to calibration and the held-out levels are intermediate, for six estimators. Panel (c): the optimism introduced by splitting at the scan.

{chr(10).join(s14a)}

{chr(10).join(s14b)}

{chr(10).join(s14m)}

Note: the two units are compared at matched calibration size, so the only difference is whether a test spectrum shares its physical sample with a calibration spectrum. That happens for {float(sm[sm['mode'] == 'scan'].leak_rate.median()):.2f} of test spectra under the scan-level split and {float(sm[sm['mode'] == 'sample'].leak_rate.median()):.2f} under the sample-level split. Optimism is RMSE(sample) − RMSE(scan), so a positive value means the scan-level split reports the better number. It is positive in {opt_pos} of {opt_n} dataset–method combinations, with median {opt_med:+.2f} percentage points and maximum {opt_max:+.2f} percentage points, and every method is optimistic on at least {opt_min_pos} of the three datasets, which is why grouping by physical sample is used throughout the main text. The {n_neg} negative values{(' fall on ' + neg_ds) if n_neg else ''}: there the sample-level split also leaves fewer independent groups for calibration, so the two effects work in opposite directions.

## S15. Monte-Carlo validation of the noise bound

**Table S15.** Direct test of the noise-limited bound of Section 2.3. Panel (a): for each of the 21 conditions, the analytic standard error (SE) computed from the fitted model, the empirical standard deviation (SD) of the prediction under 200 Monte Carlo (MC) white-noise perturbations of the test spectra at the calibration noise level, the empirical coverage of the ±1.96 SE interval, the same quantities when the perturbation is a randomly drawn calibration residual row instead of white noise, and the observed mean absolute extrapolation error. Panel (b): for the 12 conditions of D1–D3 and D7d, the empirical SD when the calibration spectra are perturbed and the model is refitted (25 replicates), which is the term that carries the design leverage.

{chr(10).join(s15a)}

Note: under white noise the empirical SD matches the analytic bound, with a median SD/SE of {rw_med:.2f} (interquartile range {rw_q1:.2f}–{rw_q3:.2f}) and a median coverage of {cov_w_med:.2f} for the nominal 95 % interval. The linearization behind that bound is therefore exact under its own assumption, in both the passing and the failing group (median {ins_ratio:.2f} and {out_ratio:.2f}). Under the residual-row perturbation the same SD rises to a median of {rr_med:.1f} times the bound and coverage falls to {cov_r_med:.2f}. The observed error is a further factor of {mm_med:.0f} larger (range {mm_min:.1f}–{mm_max:.1f}), and this holds in both groups ({ins_mm:.1f} times in the passing group, {out_mm:.1f} in the failing group). Per test level, the white-noise SD shows no relation to the prediction leverage (ρ = {rho_w_s}, p = {rho_w.pvalue:.3g} over {len(w_std)} levels), because the bound contains no design term: extrapolation costs nothing when the endmembers are known.

{chr(10).join(s15b)}

Note: refitting the model after perturbing the calibration spectra makes the per-level error grow with leverage, as Section 2.4 predicts (ρ = {rho_c_s}, p = {rho_c.pvalue:.3g} over {len(c_std)} levels; log-log slope {fit.slope:+.2f} ± {fit.stderr:.2f}, against the asymptotic value of 0.5 for SD ∝ √(1 + h)). Its amplitude is small: the median endmember-estimation SD is {sd_ratio:.2f} of the noise SD, so the two stochastic terms combine to a median of {tot_med:.2f} percentage points against a median test RMSEP of {rmsep_med:.2f} % in the same 12 conditions. Sampling uncertainty accounts for a median of {share_med * 100:.2f} % of the squared extrapolation error. Perturbing the test spectra with calibration residual rows rather than white noise gives a median SD of {res_med_12:.2f} against a median observed mean absolute error of {obs_med_12:.2f} percentage points, a median per-condition ratio of {res_over_obs_12:.2f}. The scale of the extrapolation error is set by the structure of the calibration residuals, that is, by model mismatch, and not by photon or detector noise.

## S16. Dataset inventory of the three KAN workbooks

**Table S16.** Inventory read from the distributed workbooks of D1–D3. Panel (a): the flat sheet used by the analysis, in which one row is one spectrum, with every row of the sheet assigned either to the analysed set or to one of the four exclusion rules. Panel (b): the transposed sheets of the same workbooks, in which one column is one spectrum. Panel (c): the sample identifiers of D3 whose recorded label differs from the concentration index encoded in the identifier.

{chr(10).join(s16a)}

Note: the {kan_raw} data rows of the three sheets resolve into {kan_sp} analysed spectra and {kan_drop} excluded rows, which are the counts of Table 1. Each file provides {int(ka['levels'].iloc[0])} levels with five physical samples at every level, {sc_min}–{sc_max} scans per sample, and the pure-matrix level carries more spectra than any other level in all three files ({n0[0]}, {n0[1]} and {n0[2]} spectra in D1, D2 and D3). The three workbooks contribute {kan_sp} spectra and {kan_sa} physical samples of the 3,470 spectra and 2,465 samples analysed. All three sheets record the axis in descending wavenumber order, and the loader reverses it before any modelling.

{chr(10).join(s16b)}

Note: these column counts are the origin of the 354, ~515 and 424 spectrum totals that circulated during the data check. D1's yeni sheet has {d1_mir} spectrum columns, {d1_ex} of which carry identifiers absent from the flat sheet analysed here. D2's Sayfa1 sheet has {d2_s1} columns whose identifiers all occur in its Sayfa3 sheet ({d2_top} columns), so adding the two sheets gives {d2_sum} while the union of their identifiers gives {d2_top}; {d2_ex} of the Sayfa3 columns carry identifiers absent from the flat sheet. D3's mirror sheet has {d3_mir} columns, {d3_ex} of them absent from the flat sheet. The analysis uses the flat sheets of panel (a), with {kan_sp} spectra in the three files.

{chr(10).join(s16c)}

Note: D3 identifiers are of the form p<brand><level index> with label = 5 × index (%). The rule holds throughout the file except for {n_bad_ids} identifier, p59, whose {n_bad_rows} spectra are labelled 35 % although index 9 gives 45 %; the identifiers labelled 35 % are {', '.join('%s' % v for v in lvl_ids[35])} and those labelled 45 % are {', '.join('%s' % v for v in lvl_ids[45])}, so brand 5 is absent at 45 % and doubled at 35 %. The loader places those {n_bad_rows} spectra at 45 %. In D2, {int(d2.tekrar_rows)} rows are named 1.kahve 1.1tekrar: {d2_unlab} carry no composition and are excluded, {d2_lab} are labelled 95 % matrix and 5 % adulterant and are kept. The plain identifier 1.kahve1.1 {'does not occur' if base_absent else 'occurs'} in the flat sheet, so the retained repeats are that sample's only spectra.

## S17. Photometric domain of the released tables, and the relation between the two multi-modal releases

**Table S17.** Panel (a): the distribution of the values as they are stored in each released table, the quantity the loader assigns to them, and the domain in which the dataset is analysed. Panel (b): the two audit quantities and the gate outcome computed in both domains for the four datasets stored as single-beam ratios. Panel (c): the checks that bear on whether the cinnamon (D7) and turmeric (D8) releases are independent of one another.

{chr(10).join(s17a)}

Note: the three KAN workbooks and the four cinnamon and turmeric tables differ in the quantity they store. The KAN tables reach maxima of {ab_max[0]:.3f}–{ab_max[1]:.3f} and {n_neg_min} of the three have a negative minimum, which is the distribution of an absorbance; their mean at the transparent end of the axis is {ab_edge[0]:.3f}–{ab_edge[1]:.3f}, where an absorbance falls towards zero. The cinnamon and turmeric tables carry medians of {tr_med[0]:.3f}–{tr_med[1]:.3f}, maxima of {tr_max[0]:.3f}–{tr_max[1]:.3f}, between {tr_share[0] * 100:.1f} % and {tr_share[1] * 100:.1f} % of channels above unity, and a mean at the transparent end of {tr_edge[0]:.3f}–{tr_edge[1]:.3f}, which is the distribution of a single-beam ratio. The cinnamon documentation describes its tables as absorbance spectra, but the stored numbers are ratios, so the four datasets are converted with A = −log₁₀(T), where A is the absorbance and T the transmittance, before preprocessing, as stated in Section 3.2. The conversion leaves Beer–Lambert additivity intact, which is the assumption the closure of Eq. (1) requires, and it is not applied to D1–D3.

{chr(10).join(s17b)}

Note: the conversion is the analysis domain for these four datasets in every figure and table of the paper, including the main-text Table 1. It changes the gate outcome in {gate_flips17} of the four datasets: the three that fail continue to fail and D7d continues to pass. The audit quantities themselves do move, by up to {cnr_rel_max * 100:.0f} % in relative CNR and {gam_shift_max:.2f} in |γ| ({gam_shift_key}), because both are estimated from the spectra the transformation acts on. That relative change is |CNR (as stored) − CNR (analysed)| taken over the analysed value and computed on the unrounded CNR estimates; the CNR columns are printed to one decimal, so the percentage does not follow from the printed pair ({rel_rows} before rounding).

{chr(10).join(s17c)}

Note: the identical sampling grid and the shared first author are consistent with one instrument configuration and one group behind the two releases. They are not re-measurements of one another: no sample identifier is shared, no spectrum of one release coincides with a spectrum of the other, and the pure-matrix rows within the cinnamon files are not copies of one another. The distributed documents do not record an instrument model, a serial number, or acquisition dates, so it cannot be established whether the two measurement sessions were separate. Section 3.1 therefore describes D8 as a related matrix from the same source rather than as an independent external validation. The analysed datasets come from three deposits and seven matrix–adulterant systems, but from two source groups: the KAN collection behind D1–D3 [26,27], and the cinnamon and turmeric releases behind D7 and D8 [28,29], whose relationship is what this panel documents. Within the second group the three cinnamon adulterants remain separated at the level of the physical sample.

## S18. Wave-number grid and smoothing window

**Table S18.** Sensitivity of the audit and of the extrapolation error to the channel grid and the Savitzky–Golay window. Panel (a): each dataset recomputed on four channel configurations, namely the native grid with the 15-channel window, resampling onto the other family's grid (1,762 ↔ 2,381 channels) with a window that preserves the physical width in cm⁻¹, the same resampled grid with the window fixed at 15 channels, and every second native channel with a matched window. ΔRMSEP is relative to the native configuration. Panel (b): dataset-level Spearman correlation between CNR and high-end RMSEP across the seven datasets for each configuration, with the two-sided p-value of the asymptotic test at n = 7 and the number of datasets whose gate outcome agrees with the verdict on the native grid.

{chr(10).join(s19a)}

{chr(10).join(s19b)}

Note: the native rows reproduce the CNR and γ of Table 1 and the RMSEP of Table 2 exactly, and they are asserted against the stored audit output when this supplement is generated. The gate outcome is unchanged in all {n_rows19} dataset–configuration combinations ({n_flip} flips). CNR itself is grid-dependent, because N enters Eq. (4) explicitly and interpolation correlates the channel noise: with a window matched to the physical width it moves by {cnr_match_txt} on resampling and by {cnr_decim_txt} under decimation, while holding the window at 15 channels moves it by {cnr_fixed_txt}. The margin of D1 above the threshold narrows from {nat.loc['D1', 'cnr']:.1f} to {g19[(g19.dataset == 'D1') & (g19.variant == 'decim2_match')].cnr.iloc[0]:.1f} when every second channel is removed. With a physically matched window |γ| changes by at most {dgam_match[1]:.2f} ({dgam_match_key}); with the index window fixed at 15 channels on a different grid it changes by up to {dgam_fixed:.2f} ({dgam_fixed_key}), and D8 stays outside the curvature gate in both cases. RMSEP changes by {drmse_txt} (largest, {drmse_key}), and the dataset-level rank correlation between CNR and RMSEP stays between {rho19_txt}. The audit thresholds therefore act on the analysis grid, but the gate decisions and the ordering of the seven datasets do not depend on which of these grids is used.

## S19. Capacity-matched comparators

**Table S19.** Capacity matching inside the gate. Panel (a): RMSEP (pp) on the nine gate-passing conditions (D1, D2 and D7d, three protocols each; D7d is the partial system of Table 1, so one of the three datasets is not certified by the full endpoint). PLS with a fixed number of components replaces the cross-validated choice: PLS-CV is the cross-validated comparator and PLS-2 to PLS-6 fix the number of components at 2 to 6. The ridge comparator is restricted to 3, 4 or 5 effective degrees of freedom (dof), trace(H) = Σ s_i²/(s_i² + α), the number of free quantities the closure estimates per spectrum (concentration, multiplicative scale and baseline offset). Panel (b): median proposed/comparator RMSEP over the nine conditions, with values below one favouring the proposed estimator, and the number of conditions in which it is lower. Panel (c): the cross-validated comparators of the main text, for reference, with the number of PLS components and the effective degrees of freedom of the ridge selected by grouped cross-validation.

{chr(10).join(s20a)}

{chr(10).join(s20b)}

{chr(10).join(s20c)}

Note: grouped cross-validation already selects {k_range[0]}–{k_range[1]} components for PLS on these conditions (median {k_med}), so the comparator of Table 2 is not a high-capacity model. Fixing the component count leaves the ordering unchanged: over k = 2–6 the proposed estimator is lower in {pls_wins[0]}–{pls_wins[1]} of the nine conditions, with a median proposed/comparator ratio of {pls_med[0]:.2f}–{pls_med[1]:.2f} against {pls_cv_med:.2f} for the cross-validated choice. A fixed-component PLS falls below the proposed estimator at only two of the nine conditions, {loss_txt}. Restricting ridge to the closure's three to five effective degrees of freedom makes it considerably worse (median proposed/comparator ratio {ridge_med[0]:.2f}–{ridge_med[1]:.2f}), while the cross-validated ridge selects {dof_range[0]:.0f}–{dof_range[1]:.0f} effective degrees of freedom and is still behind the closure in seven of the nine conditions (Table S6). Parameter count therefore does not account for the ordering inside the gate: at matched capacity a variance-based model performs worse, and at the capacity selected by cross-validation it performs worse as well.

## S20. Rotational feasible band of the two-component closure

**Table S20.** Rotational ambiguity of the fitted endmember pair, with and without the calibration labels. Panel (a): the six high-end conditions scanned, the pair each one scans, the audit quantities of the fitted endmembers, the number of rotations that non-negativity, closure and a positive scale admit on their own, and the rotation that minimises the in-range error; n_cal and n_test are the numbers of calibration and test spectra, and θ is the rotation angle in degrees. Panel (b): the band of medians reachable at the widest test level, as a function of the non-negativity slack δ and the label tolerance τ. Panel (c): the same band at a permissive slack with and without the labels, next to the largest deviation of the reported rotation over its test levels.

(a) Conditions, fitted pairs and the in-range optimum

{chr(10).join(s21a)}

(b) Worst-level feasible band (pp) over the non-negativity slack δ and the label tolerance τ

{chr(10).join(s21b)}

(c) Labels against non-negativity at δ = 0.10

{chr(10).join(s21c)}

Note: E(θ) = E·R(θ) with C(θ) = C·R(θ)⁻¹ reproduces every spectrum exactly, so the closure alone cannot select a rotation. A rotation is admitted when the rotated pair stays non-negative up to a slack min E(θ) ≥ −δ·max|E(0)|, keeps the closed concentrations in [0, 1] with a positive per-spectrum scale, and, for finite τ, fits the calibration labels no worse than the reported rotation by more than τ percentage points of in-range RMSE; τ = 0 therefore admits only rotations at least as good as θ = 0 inside the calibration range, and the no-labels column removes that condition altogether. The band is the range of the median predicted concentration over the admitted rotations, taken at the test level where it is widest, and θ is scanned over {n_grid} angles, from {grid_txt}. Because the closure confines every concentration to [0, 1], a band cannot exceed 100 pp. D7d, D7p, D7w and D8 are single-brand files, so their fitted pair is the reported model and is scanned exactly; D1 and D2 carry five brands each, whose brand-aware fit returns one matrix endmember per brand and one shared adulterant endmember, so the pooled variant replaces the five matrix rows by their mean and keeps the adulterant row, which is the simplification a two-component rotation requires. The three gate-passing conditions are scanned with the three gate-failing ones, because the quantity of interest is how large the rotation error can be rather than whether the estimator works.

Four readings follow. Strict non-negativity pins the pair by itself in the four conditions whose fitted pair is scanned unchanged, out of the {n_grid} angles scanned: {pinned_txt}. The two pooled conditions leave it open — {loose_txt} — and averaging the five brand matrix endmembers into one row loosens the corner that the non-negativity constraint acts on. Once the slack is permissive the labels do measurable work, because at δ = 0.10 the unlabelled band spans {unl_txt} and the labels at τ = 1 pp reduce it to {lab_txt}, a factor of {shr_txt}. The smallest of those reductions, {amb_c[shr_min_key]['shrink']:.1f}×, falls on {worst_pass.replace('/', ' ')}, the gate-passing condition with the largest extrapolation error of the nine ({worst_pass_rmse:.2f} pp, Table 2). The band is not an error bound: at δ = 0 and τ = 0 it is {band00_txt} pp wide, yet the rotation it selects is itself off by {csu_big_txt}, so those conditions fail through the endmember and curvature categories of Section 4.7 rather than through the rotation. That |bias| column is the largest of the per-level deviations of the reported rotation, and it falls on the highest prepared level in each of the six conditions; the bias column of Table 2 averages the same deviations over all the levels of the condition, so the two quote the same errors over different sets — on D7d/high the level deviations run {d4_lv_txt} pp from {d4_lv_rng} and average {d4_mean_txt} pp, which is the value Table 2 carries for that condition. The reported rotation is also not the in-range optimum in {len(rot_free)} of the six conditions: {rot_free_txt}. The labels therefore buy a rotation that is marginally worse inside the calibration range, and the band is what says how far any other admissible choice could have moved the extrapolated level.

## S21. Quantification limits from the calibrated interval

**Table S21.** Interval-based quantification limits, one row per dataset. The inflation factor $f$ is the cross-validated ratio of Section 3.7, estimated once per dataset from a five-fold grouped cross-validation over all of its levels. The 95 % calibrated width is the inflated interval evaluated at the highest prepared level of that dataset, and it is not the per-protocol median width of Table 4. The limit columns give the lowest concentration at which the inflated one-standard-deviation half-width is within 20 % (screening) or 10 % (quantification) of the concentration, and "none" means the criterion is met at no level on the scanned grid.

{chr(10).join(s22a)}

Note: the limit is defined through the calibrated interval rather than through the calibration curve, because the noise-only interval covers a median 7 % of reference values at the 95 % level (Section 4.6) and omits the endmember and closure error that dominates outside the calibration span (Table S15). At each trial concentration the profile likelihood of Section 2.2 gives a one-standard-deviation half-width from the fitted endmembers, evaluated on synthetic two-component spectra with unit scale and no baseline and taken as the median over the brands of that dataset; the product with $f$ is the half-width used here. The estimator configuration is the reported one: constant baseline, free multiplicative scale, no instrument-line convolution. The scanned grid runs from {c_lo:g} to {c_hi:g} % in {c_step:g} % steps. The synthetic half-width tracks the measured one, their ratio at the prepared levels being {chk[0]:.2f}–{chk[1]:.2f} across the seven datasets, so the curve is not a separate model of the noise.

Under this criterion the limit lies inside the measured range for {and_join2(in20)} at 20 % and for {and_join2(in10)} at 10 %. It is reached at no level for {and_join2(none20)}, and for {and_join2(beyond)} it lies above the highest prepared level. D7d passes the audit gate and has no reportable limit under either fraction: its inflated 95 % interval at the highest prepared level is {lq.loc['D7d', 'u95_at_cmax_pp']:.1f} pp wide against a measured range of 0–{num(lq.loc['D7d', 'c_max'])} %, which is the separation of validity from informativeness that Section 4.8 recommends as the output of the protocol. The limits of {and_join2(in20)} at 20 % ({lq.loc['D1', 'loq20_pp']:.1f} and {lq.loc['D2', 'loq20_pp']:.1f} pp) also sit above the lowest prepared levels of those datasets, so the criterion reports the part of the range that is quantifiable at the stated relative precision rather than a single detection capability for the method.

## S22. Audit–error correlation for each comparator

**Table S22.** Dataset-level correlation of the two audit quantities with the error of every comparator, computed from the same predictions as Table 2. n is the number of datasets on which the comparator was run; support-vector regression, random forest, gradient boosting and the perceptron were run on D1–D3 and D8 only. The leave-one-out range is the interval of the contrast coefficient when one dataset is removed. The last row is the per-condition best comparator of Table 2, included for reference.

{chr(10).join(s23)}

{note22}

## S23. External validation on two independently deposited datasets

**Table S23.** Audit and out-of-range results on two datasets that played no part in the design, the thresholds or the preprocessing choices. (a) Audit quantities per system, computed from all spectra of that system. (b) The three protocols of Section 3.3 applied unchanged, with the proposed estimator and the four comparators of the smaller comparator family; RMSEP is the root-mean-square error of prediction over the test spectra and ratio is proposed divided by best-comparator RMSEP. (c) The audit and the protocols repeated under the stated changes to the preprocessing or to the reference set, with the resulting out-of-range median ratios.

(a)

{chr(10).join(s24a)}

{note24a}

(b)

{chr(10).join(s24b)}

{note24b}

(c)

{chr(10).join(s24c)}

{note24c}

## S24. Stability of the audit quantities

**Table S24.** Stability of the two audit quantities. (a) Each dataset re-audited on 100 draws in which 80 % of the physical samples of every level were drawn without replacement; the 2.5–97.5 % column gives the percentile range of that distribution, and the last two columns give the gate of Eq. (6) at the full reference set and the number of draws that reverse it. (b) γ re-estimated under five alternatives to the reported fit of the level-mean response, with the gate decision of each dataset under every variant.

(a) Subsample distribution of CNR and γ

{chr(10).join(s25a)}

{note25a}

(b) Sensitivity of γ to level weighting, level spacing and polynomial order

{chr(10).join(s25b)}

{note25b}

## S25. Transductive and inductive MCR-ALS

**Table S25.** The proposed estimator and the two variants of MCR-ALS (multivariate curve resolution–alternating least squares) on all 21 conditions. Ratios are the proposed root-mean-square error of prediction (RMSEP) divided by the MCR-ALS RMSEP, so a value below one favours the proposed estimator. CSU is the concentration-supervised unmixing estimator with a profile likelihood.

{chr(10).join(s26a)}

{mcr_note}

## S26. Calibration size, brand fallback and baseline order

**Table S26.** Three sensitivities over the full condition set. (a) Every condition rerun with the calibration set subsampled to the smallest protocol size of its dataset, over ten matched draws. (b) Each brand removed from calibration and predicted with a matrix endmember averaged over the remaining brands. (c) The constant baseline of Eq. (1) replaced by a quadratic or a cubic one. (d) The four prespecified comparators rerun on the same matched draws as panel (a), against the comparison with unequal calibration sizes.

(a) Matched calibration size

{chr(10).join(s27a)}

(b) Brand-averaged matrix endmember

{chr(10).join(s27b)}

(c) Baseline order

{chr(10).join(s27c)}

(d) The prespecified comparison under matched calibration size

{chr(10).join(s27d)}

{note27}
"""
    SI.write_text(relabel(text), encoding="utf-8")
    print(f"wrote {SI}")
    print(f"S1 {len(s1) - 2} rows, S2 {len(s2) - 2}, S3 {len(s3) - 2}, "
          f"S4 {len(s4) - 2}, S5 {len(s5) - 2}, S6 {len(s6) - 2}, "
          f"S7 {len(s7a) - 2}+{len(s7b) - 2}, S8 {len(s8a) - 2}+{len(s8b) - 2}+{len(s8c) - 2}, "
          f"S9 {len(s9) - 2}, S10 {len(s10) - 2}, S11 {len(s11) - 2}, S12 {len(s12) - 2}, "
          f"S13 {len(s13a) - 2}+{len(s13b) - 2}+{len(s13c) - 2}, "
          f"S14 {len(s14a) - 2}+{len(s14b) - 2}+{len(s14m) - 2}, "
          f"S15 {len(s15a) - 2}+{len(s15b) - 2}, "
          f"S16 {len(s16a) - 2}+{len(s16b) - 2}+{len(s16c) - 2}, "
          f"S17 {len(s17a) - 2}+{len(s17b) - 2}+{len(s17c) - 2}, "
          f"S18 {len(s19a) - 2}+{len(s19b) - 2}, "
          f"S19 {len(s20a) - 2}+{len(s20b) - 2}+{len(s20c) - 2}, "
          f"S20 {len(s21a) - 2}+{len(s21b) - 2}+{len(s21c) - 2}, "
          f"S21 {len(s22a) - 2}, S22 {len(s23) - 2}, "
          f"S23 {len(s24a) - 2}+{len(s24b) - 2}+{len(s24c) - 2}, "
          f"S24 {len(s25a) - 2}+{len(s25b) - 2}, S25 {len(s26a) - 2}, "
          f"S26 {len(s27a) - 2}+{len(s27b) - 2}+{len(s27c) - 2}+{len(s27d) - 2}")


if __name__ == "__main__":
    main()
