"""exp17b —— 把噪声项与校准重拟合项合并到同一尺度，与实测外推误差对照。

exp17 分别给出：白噪声扰动测试谱的经验 SD（测试谱噪声通道）、扰动校准谱并重拟合
端元后的经验 SD（校准/design leverage 通道）。正文 §4.6 与 Table S15 需要的是两者
合并后的随机不确定度占总误差的比例，因此这里按条件合并：

    sd_tot = sqrt(sd_white**2 + sd_calib**2)
    ratio_obs = RMSEP / sd_tot
    exp_share = (sd_tot / RMSEP)**2   —— 随机不确定度解释的误差方差份额

RMSEP 取 tables/table2_main.csv 的 CSU-PL 行（与正文同一口径）。

输出 out/exp17_composite.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
PROT = {"high-end": "high", "interp.": "interp", "low-end": "low"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    mc = pd.read_csv(os.path.join(OUT, "exp17_mc_long.csv"))
    cal = pd.read_csv(os.path.join(OUT, "exp17_calib_levels.csv"))
    cond = pd.read_csv(os.path.join(OUT, "exp17_conditions.csv"))
    t2 = pd.read_csv(os.path.join(ROOT, "tables", "table2_main.csv"))

    # per-condition stochastic terms: white-noise channel and endmember-refit channel.
    # The refit channel exists only for the multi-scan datasets (D1-D3, D7d), so the
    # join to it is inner and the composite table covers those 12 conditions.
    j = mc[mc["mode"] == "cond"][["dataset", "protocol", "se_med", "sd_white",
                                  "sd_resid"]].copy()
    lw = mc[(mc["mode"] == "white") & mc["level"].notna()]
    w = lw.groupby(["dataset", "protocol"]).sd.median().rename("sd_white_lv")
    cl = cal.groupby(["dataset", "protocol"]).agg(
        sd_calib=("sd_calib", "median"),
        se=("se_crlb", "median"),
        n_lv=("level", "nunique"),
        h_max=("h", "max")).reset_index()
    j = (j.merge(w.reset_index(), on=["dataset", "protocol"], how="left")
          .merge(cl, on=["dataset", "protocol"]))          # inner: 12 conditions

    # gate label from the audit table used in exp17, not hard-coded here
    g = cond[["dataset", "protocol", "gate"]].drop_duplicates()
    assert len(g) == len(cond)
    j = j.merge(g, on=["dataset", "protocol"], how="left")

    t2["protocol"] = t2["Protocol"].map(PROT)
    t2["dataset"] = t2["Dataset"]
    j = j.merge(t2[["dataset", "protocol", "RMSEP"]].rename(columns={"RMSEP": "rmsep"}),
                on=["dataset", "protocol"], how="left", suffixes=("", "_t2"))
    j["sd_tot"] = np.sqrt(j.sd_white ** 2 + j.sd_calib ** 2)
    j["ratio_obs"] = j.rmsep / j.sd_tot
    j["exp_share"] = (j.sd_tot / j.rmsep) ** 2

    cols = ["dataset", "protocol", "se_med", "sd_white", "sd_resid", "sd_calib",
            "sd_tot", "rmsep", "ratio_obs", "exp_share", "h_max", "gate"]
    print(j[cols].round(3).to_string(index=False))
    print()
    for g, lab in [(True, "in-gate"), (False, "out-of-gate"), (None, f"all {len(j)}")]:
        s = j if g is None else j[j.gate == g]
        print(f"{lab} (n={len(s)}): SE med {s.se_med.median():.2f} | "
              f"SD_white {s.sd_white.median():.2f} | SD_resid {s.sd_resid.median():.2f} | "
              f"SD_calib {s.sd_calib.median():.2f} | SD_tot {s.sd_tot.median():.2f} | "
              f"RMSEP {s.rmsep.median():.2f} | RMSEP/SD_tot "
              f"{s.ratio_obs.median():.1f}x | variance explained "
              f"{s.exp_share.median() * 100:.1f}%  (IQR "
              f"{s.exp_share.quantile(.25) * 100:.1f}-{s.exp_share.quantile(.75) * 100:.1f})")
    print(f"\nSD_calib/SD_white 中位：{(j.sd_calib / j.sd_white).median():.2f}"
          f"  (SD_calib > SD_white 的比例 {(j.sd_calib > j.sd_white).mean():.2f})")
    print(f"SE 与 sd_white 的一致性：max|diff| = {(j.se_med - j.sd_white).abs().max():.3f} pp")

    j[cols].to_csv(os.path.join(OUT, "exp17_composite.csv"), index=False)


if __name__ == "__main__":
    main()
