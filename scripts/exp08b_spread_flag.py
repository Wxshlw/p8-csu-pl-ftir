"""exp08b —— 区间宽度 / 预测散布 比值（正文 §4.3 的区间侧诊断）。

正文用一句话给出"D7d 低端的中位 95 % 膨胀区间宽度几乎是预测散布的三十倍"，
这个比值此前没有独立的计算脚本。本脚本只做后处理：读取 exp08 已经写出的
逐谱区间结果 out/exp08_uncertainty.csv 与条件汇总 out/exp08_summary.csv，
对 21 个条件各算

  width_pp  中位 95 % 交叉验证膨胀区间宽度（百分点）
  spread_pp 该条件测试集 ĉ 的样本标准差（百分点）
  ratio     width_pp / spread_pp
  ref_span  测试参考浓度的跨度（百分点）
  |1-slope| 取自 tables/table2_main.csv

并汇总比值超过 5 的条件数，供正文引用。不重新拟合模型。

输出
----
out/exp08_spread.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
TAB = os.path.join(ROOT, "tables")


def main():
    per_spec = pd.read_csv(os.path.join(OUT, "exp08_uncertainty.csv"))
    summ = pd.read_csv(os.path.join(OUT, "exp08_summary.csv"))
    t2 = pd.read_csv(os.path.join(TAB, "table2_main.csv")).rename(columns={"Dataset": "dataset"})

    proto_key = {"low-end": "low", "high-end": "high", "interp.": "interp"}
    t2 = t2.assign(protocol=t2.Protocol.map(proto_key))
    assert t2.protocol.notna().all(), "unexpected protocol labels in Table 2"
    t2 = t2.assign(sdist=(1 - t2.Slope).abs())

    rows = []
    for _, s in summ.iterrows():
        te = per_spec[(per_spec.dataset == s.dataset) & (per_spec.protocol == s.protocol)]
        assert len(te) == s.n, f"{s.dataset} {s.protocol}: {len(te)} rows vs n={s.n}"
        width_pp = float(s.width95f) * 100
        spread_pp = float(te.c_hat.std(ddof=1)) * 100
        ref_span = float(te.c_true.max() - te.c_true.min()) * 100
        slope = float(t2[(t2.dataset == s.dataset) & (t2.protocol == s.protocol)].sdist.iloc[0])
        rows.append(dict(dataset=s.dataset, protocol=s.protocol, n=int(s.n),
                         width_pp=width_pp, spread_pp=spread_pp,
                         ratio=width_pp / spread_pp if spread_pp > 0 else np.inf,
                         ref_span_pp=ref_span, abs_1_minus_slope=slope))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "exp08_spread.csv"), index=False)
    print(out.round(3).to_string(index=False), flush=True)

    hi = out[out.ratio > 5]
    print(f"\nratio > 5: {len(hi)} of {len(out)} conditions; "
          f"min |1-slope| among them {hi.abs_1_minus_slope.min():.3f}; "
          f"conditions with |1-slope| >= 0.75: {int((out.abs_1_minus_slope >= 0.75).sum())}",
          flush=True)
    d7d = out[(out.dataset == "D7d") & (out.protocol == "low")].iloc[0]
    print(f"D7d low: width {d7d.width_pp:.2f} pp, reference span {d7d.ref_span_pp:.0f} pp, "
          f"spread {d7d.spread_pp:.2f} pp, ratio {d7d.ratio:.1f}", flush=True)


if __name__ == "__main__":
    main()
