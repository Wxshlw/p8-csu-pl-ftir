"""exp00_inventory —— 数据集正式盘点表（§3.1 数据描述与 Table S16 的出处）。

跑法：python scripts/exp00_inventory.py
产出：out/exp00_inventory.json
      out/exp00_inventory.md   （可直接进 SI Table S1）
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from unmix_ftir import data as D  # noqa: E402


def homology_check(d7, d8):
    """D7 肉桂 vs D8 姜黄 同源性核查（§3.1 第二段与 Table S17c 的出处）。

    证据分三层：
      1. 波数网格：D8 轴是否等于 D7 轴的前 2522 点（相对偏差 < 1e-5）
      2. 命名规则：都是 <材料码><浓度>S<两位样品号>（CW12S08 / T0S01）
      3. 物理样品：0 % 纯基质谱是否存在完全重复行（同批样品拆分）
    结论只区分"同采集协议"与"独立采集"，不能证明独立。
    """
    a7, a8 = np.asarray(d7["wvn"]), np.asarray(d8["wvn"])
    n = min(len(a7), len(a8))
    overlap_max_abs = float(np.max(np.abs(a7[:n] - a8[:n])))
    overlap_max_rel = float(np.max(np.abs(a7[:n] - a8[:n]) / np.abs(a7[:n])))
    same_grid = overlap_max_rel < 1e-5

    def pure(d):
        return d["X"][np.abs(d["y"]) < 1e-9]

    p7, p8 = pure(d7), pure(d8)
    dup = 0
    if len(p7) and len(p8):
        s8 = {row.tobytes() for row in p8}
        dup = int(sum(1 for row in p7 if row.tobytes() in s8))
    pat = re.compile(r"^([A-Z]+)(\d+)S(\d+)$")
    g7 = [str(v) for v in d7["groups"]]
    g8 = [str(v) for v in d8["groups"]]
    iso = (all(pat.match(g) for g in g7[:200]) and all(pat.match(g) for g in g8[:200])
           and len({pat.match(g).group(3) for g in g7[:200]}) == len({pat.match(g).group(3) for g in g8[:200]}))
    return dict(
        same_wavenumber_grid=bool(same_grid),
        grid_overlap_points=int(n), grid_max_abs_diff=round(overlap_max_abs, 6),
        grid_max_rel_diff=float(f"{overlap_max_rel:.3e}"),
        n_pure_D7=int(len(p7)), n_pure_D8=int(len(p8)),
        exact_duplicate_pure_spectra=int(dup),
        sample_name_examples_D7=g7[:5], sample_name_examples_D8=g8[:5],
        naming_isomorphic=bool(iso),
        verdict=("same acquisition grid and naming convention -> shared protocol; "
                 "no exact duplicate spectra -> distinct physical samples; "
                 "treat as non-independent external validation" if same_grid and iso
                 else "no shared-acquisition evidence"),
    )


def main():
    out = {}
    for nm in D.ALL:
        d = D.load(nm)
        out[nm] = D.inventory(d)
        print(f"[{nm}] n={out[nm]['n_spectra']} N={out[nm]['n_channel']} "
              f"groups={out[nm]['n_groups']} levels={out[nm]['n_levels']} "
              f"dnu={out[nm]['dnu']}")

    d7 = D.load("D7w")
    d8 = D.load("D8")
    out["homology_D7_vs_D8"] = homology_check(d7, d8)

    # 波数轴网格一致性（D7 三个文件之间）
    axes = {}
    for nm in ("D7w", "D7p", "D7d"):
        axes[nm] = D.load(nm)["wvn"]
    out["D7_axes_identical"] = bool(all(
        axes["D7w"].shape == axes[k].shape and np.allclose(axes["D7w"], axes[k], atol=1e-6)
        for k in axes))

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "exp00_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---- Markdown 表（SI） ----
    lines = ["# Table S1. Dataset inventory (read from the distributed files, 2026-10-23)", "",
             "| ID | Matrix | Adulterant | Spectra | Channels | Physical samples | Scans/sample | "
             "Levels | Level values (%) | delta-nu (cm-1) | Range (cm-1) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm in D.ALL:
        r = out[nm]
        lv = ", ".join(f"{v:g}" for v in r["levels"])
        lines.append(
            f"| {nm} | {r['matrix']} | {r['adulterant']} | {r['n_spectra']} | {r['n_channel']} | "
            f"{r['n_groups']} | {r['scans_per_group']['min']}–{r['scans_per_group']['max']} | "
            f"{r['n_levels']} | {lv} | {r['dnu']:g} | {r['wvn_min']:.0f}–{r['wvn_max']:.0f} |")
    h = out["homology_D7_vs_D8"]
    lines += ["", "## D7 / D8 homology audit", "",
              f"- same wavenumber grid: **{h['same_wavenumber_grid']}** "
              f"(overlap {h['grid_overlap_points']} points, max abs diff "
              f"{h['grid_max_abs_diff']} cm-1, max rel diff {h['grid_max_rel_diff']})",
              f"- naming isomorphic: **{h['naming_isomorphic']}**",
              f"- exact duplicate pure-matrix spectra across D7/D8: **{h['exact_duplicate_pure_spectra']}**",
              f"- sample-name examples D7: {h['sample_name_examples_D7']}",
              f"- sample-name examples D8: {h['sample_name_examples_D8']}",
              f"- verdict: {h['verdict']}"]
    with open(os.path.join(ROOT, "out", "exp00_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
