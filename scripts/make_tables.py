"""make_tables —— 生成正文/补充材料表格（CSV + Markdown）。

Table 1  数据与可辨识性审计
Table 2  主外推结果（CSU-PL vs 最佳基线）
Table 3  设计/消融：自由尺度、0 % 锚点、半合成稳健性
Table 4  不确定度覆盖率
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
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)

PROTO = {"low": "low-end", "interp": "interp.", "high": "high-end"}
ORDER = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]


def _fmt(v, float_fmt):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        return float_fmt.format(v)
    return str(v)


def write(df, name, float_fmt="{:.2f}"):
    p = os.path.join(TAB, name)
    df.to_csv(p + ".csv", index=False)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v, float_fmt) for v in row) + " |")
    io.open(p + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {p}.csv/.md  ({len(df)} rows)")


def table1():
    inv = json.load(io.open(os.path.join(OUT, "exp00_inventory.json"), encoding="utf-8"))
    au = json.load(io.open(os.path.join(OUT, "exp05_audit.json"), encoding="utf-8"))
    rows = []
    for ds in ORDER:
        r = inv[ds]
        a = au[ds]
        e, c = a["endmember"], a["curvature"]
        gate = (e["cnr"] >= 10) and (abs(c["gamma"]) <= 1)
        rows.append(dict(
            Dataset=ds, Matrix=r["matrix"].capitalize(), Adulterant=r["adulterant"],
            Spectra=r["n_spectra"], Samples=r["n_groups"], Levels=r["n_levels"],
            Range=f"{r['levels'][0]:g}–{r['levels'][-1]:g}",
            Channels=r["n_channel"], CNR=e["cnr"], Gamma=abs(c["gamma"]),
            Leak=e["baseline_leak"], TrainRMSE=a["train_rmse"],
            Gate="pass" if gate else "fail"))
    write(pd.DataFrame(rows), "table1_datasets")


def table2():
    tb = pd.read_csv(os.path.join(OUT, "exp03_table.csv"))
    vb = pd.read_csv(os.path.join(OUT, "exp03_vs_best.csv"))
    csu = tb[tb.method == "CSU-PL"].copy()
    rows = []
    for _, r in csu.iterrows():
        v = vb[(vb.dataset == r.dataset) & (vb.protocol == r.protocol)].iloc[0]
        p = float(v.p_holm)
        rows.append(dict(
            Dataset=r.dataset, Protocol=PROTO[r.protocol], n=r.n,
            RMSEP=r.rmse, Lo=r.rmse_lo, Hi=r.rmse_hi,
            Slope=r.slope, Bias=r.bias,
            BestBaseline=v.best, RMSEP_best=v.best_rmse,
            Ratio=r.rmse / v.best_rmse,
            p_Holm=("<0.001" if p < 0.001 else f"{p:.3f}")))
    write(pd.DataFrame(rows), "table2_main")


def table3():
    fx = pd.read_csv(os.path.join(OUT, "exp10_fixscale.csv"))
    rem = pd.read_csv(os.path.join(OUT, "exp03c_remedy.csv"))
    rob = pd.read_csv(os.path.join(OUT, "exp06_robustness.csv"))
    # Panel A：自由尺度 vs 固定尺度（全部条件）
    a = fx.pivot_table(index=["dataset", "protocol"], columns="fix_scale",
                       values="rmse").reset_index()
    a = a.rename(columns={False: "RMSEP_free", True: "RMSEP_fixed"})
    a.columns = ["Dataset", "Protocol", "RMSEP_free", "RMSEP_fixed"]
    a["Ratio_fixed/free"] = a["RMSEP_fixed"] / a["RMSEP_free"]
    a["Protocol"] = a.Protocol.map(PROTO)
    a = a[["Dataset", "Protocol", "RMSEP_free", "RMSEP_fixed", "Ratio_fixed/free"]]
    write(a, "table3a_scale")
    # Panel B：0 % 锚点补救（CSU-PL）
    b = rem[(rem.method == "CSU-PL") & (rem.variant != "plus_one_per_brand")].copy()
    b = b.pivot_table(index=["dataset", "protocol"], columns="variant", values="rmse").reset_index()
    b.columns = ["Dataset", "Protocol", "RMSEP_base", "RMSEP_plus_all_0"]
    b["Protocol"] = b.Protocol.map(PROTO)
    write(b, "table3b_anchor")
    # Panel C：半合成稳健性极端水平
    piv = rob.pivot_table(index=["factor", "value"], columns="method",
                          values="rmse").reset_index()
    picks = [("noise", 20.0), ("scatter", 0.12), ("baseline", 0.20),
             ("nonlinear", 0.20), ("matrixvar", 0.50)]
    c = piv.set_index(["factor", "value"]).loc[picks].reset_index()
    c.columns = ["Factor", "Level", "CSU-PL", "fixed-scale"]
    write(c, "table3c_robustness")
    print("scale: free better in "
          f"{(a['Ratio_fixed/free'] > 1).sum()}/{len(a)}; median ratio "
          f"{a['Ratio_fixed/free'].median():.2f}")


def table4():
    src = os.path.join(OUT, "exp08_summary.csv")
    if not os.path.exists(src):
        return
    s = pd.read_csv(src)
    s["Protocol"] = s.protocol.map(PROTO)
    out = s[["dataset", "Protocol", "n", "f", "rmse", "cov68", "cov95",
             "cov68f", "cov95f", "width95", "width95f"]].copy()
    out.columns = ["Dataset", "Protocol", "n", "f", "RMSEP", "cov68_raw",
                   "cov95_raw", "cov68_cal", "cov95_cal", "width95_raw", "width95_cal"]
    for c in ["width95_raw", "width95_cal"]:
        out[c] = out[c] * 100
    write(out, "table4_uncertainty")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
