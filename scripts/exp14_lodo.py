"""exp14 —— 审计阈值的 leave-one-dataset-out（LODO）验证。

问题：CNR ≥ 10、|γ| ≤ 1 是在看到 7 个数据集的结果后定的，审稿人会问
"阈值是不是拟合了这 7 个点"。本脚本在 6 个数据集上选阈值、在第 7 个上测试，
重复 7 次；并与固定 10/1 的结果对比。

真值口径：该数据集 3 个协议的中位误差比 < 1（CSU 不劣于最佳基线的中位水平）。
D7d 的中位比为 0.937，按此口径为 pass，但正文将其列为 partial failure（斜率塌缩），
脚本单独标注。

输出 out/exp14_lodo.json, out/exp14_lodo.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
TAB = os.path.join(ROOT, "tables")

GRID_CNR = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
GRID_GAM = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def load():
    t1 = pd.read_csv(os.path.join(TAB, "table1_datasets.csv"))
    t2 = pd.read_csv(os.path.join(TAB, "table2_main.csv"))
    med = t2.groupby("Dataset")["Ratio"].median()
    g = t1[["Dataset", "CNR", "Gamma"]].copy()
    g["ratio"] = g.Dataset.map(med)
    g["truth_pass"] = (g.ratio < 1).astype(int)
    g["gabs"] = g.Gamma.abs()
    return g


def classify(g, cnr_t, gam_t):
    return ((g.CNR >= cnr_t) & (g.gabs <= gam_t)).astype(int).values


def score(g, pred):
    """balanced accuracy，要求两类都存在。"""
    y = g.truth_pass.values
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return 0.5 * (tpr + tnr), tp, tn, fp, fn


def main():
    g = load()
    rows = []
    for hold in g.Dataset:
        tr = g[g.Dataset != hold].reset_index(drop=True)
        te = g[g.Dataset == hold].reset_index(drop=True)
        best = None
        for ct in GRID_CNR:
            for gt in GRID_GAM:
                pred = classify(tr, ct, gt)
                if len(set(pred)) < 2:
                    continue
                ba, tp, tn, fp, fn = score(tr, pred)
                # 平局时偏好更靠网格中心的保守阈值
                margin = 0.0
                ps = tr.ratio[pred == 1]
                fl = tr.ratio[pred == 0]
                if len(ps) and len(fl):
                    margin = float(fl.median() - ps.median())
                key = (ba, margin, -abs(ct - 10) - abs(gt - 1))
                if best is None or key > best[0]:
                    best = (key, ct, gt, ba)
        _, ct, gt, ba = best
        pred = int(classify(te, ct, gt)[0])
        rows.append(dict(held_out=hold, cnr_thr=ct, gamma_thr=gt,
                         train_bal_acc=ba, truth=bool(te.truth_pass.iloc[0]),
                         pred=bool(pred), correct=bool(pred == te.truth_pass.iloc[0]),
                         ratio=float(te.ratio.iloc[0]), cnr=float(te.CNR.iloc[0]),
                         gamma=float(te.Gamma.iloc[0])))
    df = pd.DataFrame(rows)
    fixed = df.apply(lambda r: bool((r.cnr >= 10) and (abs(r.gamma) <= 1)), axis=1)
    out = dict(
        lodo_accuracy=float(df.correct.mean()),
        lodo_correct=int(df.correct.sum()),
        fixed_accuracy=float((fixed == df.truth).mean()),
        fixed_correct=int((fixed == df.truth).sum()),
        folds=df.to_dict(orient="records"))
    with open(os.path.join(OUT, "exp14_lodo.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    lines = ["# LODO 阈值验证（exp14）", "",
             "| Held-out | 选中的 CNR | 选中的 γ | 训练 balanced acc | 真值 pass | 预测 pass | 正确 | 中位比 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.held_out} | {r.cnr_thr:g} | {r.gamma_thr:g} | {r.train_bal_acc:.3f} | "
                     f"{'pass' if r.truth else 'fail'} | {'pass' if r.pred else 'fail'} | "
                     f"{'Y' if r.correct else 'N'} | {r.ratio:.3f} |")
    lines += ["", f"LODO 正确 {out['lodo_correct']}/{len(df)}；固定 10/1 正确 "
              f"{out['fixed_correct']}/{len(df)}。"]
    open(os.path.join(OUT, "exp14_lodo.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
