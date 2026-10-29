"""exp00b —— KAN trio（D1/D2/D3）正式盘点表：§3.1 的谱数与样品数口径、Table S16 的素材。

四部分，全部**从分发的 xlsx 实读**：

A  正式口径：所用平面表（D1 Sayfa7 / D2 Sayfa4 / D3 Sayfa2）的原始数据行数、逐条清洗
   规则的丢弃数、最终谱数、物理样品数、每水平样品数与谱数、每样品扫描数分布、
   波数轴长度/原始方向/间隔/首末通道。
B  差源定位：早期沿用口径 354 / ~515 / 424 来自各文件的**转置镜像表**
   （D1 yeni、D2 Sayfa1+Sayfa3、D3 Sayfa1）的"谱列数"，其中含无标签与重复采集。
C  D3 标签纪律：样本名编码为 p<brand><level-index>，level-index×5 % 即标签；
   逐条核对文件标签，定位唯一不一致者（p59），并说明 load_kan 的 45 % 改写依据。
D  D2 tekrar 块：10 条 `1.kahve1.1tekrar` 中 5 条无标签（丢弃）、5 条带 95/5 标签（保留），
   且这 5 条正是该物理样品在平面表中的唯一行。

输出 out/exp00b_kan_audit.csv（A）、out/exp00b_kan_mirror.csv（B）、out/exp00b_kan_audit.md
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unmix_ftir.data import KAN_ROOT, KAN_SHEETS, GROUP_KEYS, _kan_axis, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DID = {"oil": "D1", "coffee": "D2", "juice": "D3"}
# transposed mirror sheets: spectra are columns, one spectrum per column
MIRROR = {"oil": [("kan_oliveoil.xlsx", "yeni", 2)],
          "coffee": [("kan_coffee.xlsx", "Sayfa1", 0), ("kan_coffee.xlsx", "Sayfa3", 0)],
          "juice": [("kan_juice.xlsx", "Sayfa1", 0)]}


def flat(key):
    """Read the flat sheet actually used by load_kan, before and after cleaning."""
    spec = KAN_SHEETS[key]
    path = os.path.join(KAN_ROOT, spec["file"])
    if not os.path.exists(path):
        path = os.path.join(KAN_ROOT, spec["orig"])
    df = pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header"])
    raw = df.to_numpy()
    ids = np.array([str(v) for v in raw[:, 0]], dtype=object)
    auth = pd.to_numeric(pd.Series(raw[:, 2]), errors="coerce").to_numpy(float)
    adul = pd.to_numeric(pd.Series(raw[:, 3]), errors="coerce").to_numpy(float)
    X = np.asarray(raw[:, 4:], dtype=float)
    blank_id = np.array([_norm(i) in ("", "nan", "none") for i in ids])
    no_auth, no_adul = ~np.isfinite(auth), ~np.isfinite(adul)
    bad_spec = ~np.isfinite(X).all(axis=1)
    keep = ~blank_id & ~no_auth & ~no_adul & ~bad_spec
    # disjoint drop accounting, in the order the loader applies its mask
    d_blank = blank_id
    d_auth = no_auth & ~d_blank
    d_adul = no_adul & ~d_blank & ~d_auth
    d_spec = bad_spec & ~d_blank & ~d_auth & ~d_adul
    wvn = _kan_axis(df, X.shape[1])
    return dict(df=df, ids=ids, auth=auth, adul=adul, X=X, keep=keep,
                drop=dict(blank_id=int(d_blank.sum()), no_matrix=int(d_auth.sum()),
                          no_adulterant=int(d_adul.sum()), nonfinite_spectrum=int(d_spec.sum())),
                wvn=wvn, file=os.path.basename(path))


def inventory(key, f):
    assert sum(f["drop"].values()) == len(f["ids"]) - int(f["keep"].sum()), "drop accounting"
    ids, adul, keep = f["ids"], f["adul"], f["keep"]
    y = adul[keep].copy()
    if key == "juice":                       # quirk 4, justified by the level-index check below
        y[np.array([_norm(i) == "p59" for i in ids[keep]])] = 45.0
    groups = np.array([GROUP_KEYS[key](i) for i in ids[keep]], dtype=object)
    scans = pd.Series(groups).value_counts()
    lv = pd.DataFrame({"level": y, "group": groups})
    n_sam = lv.groupby("level")["group"].nunique()
    n_sp = pd.Series(y).value_counts()
    levels = np.unique(y)
    wvn = f["wvn"]
    return dict(
        dataset=DID[key], matrix=KAN_SHEETS[key].get("matrix", key), file=f["file"],
        sheet=KAN_SHEETS[key]["sheet"], header_row=KAN_SHEETS[key]["header"] + 1,
        raw_data_rows=len(ids), rows_dropped=sum(f["drop"].values()), **{f"drop_{k}": v for k, v in f["drop"].items()},
        spectra_final=int(keep.sum()), channels=int(len(wvn)),
        physical_samples=int(len(scans)), levels=int(len(levels)),
        level_values=",".join(f"{v:g}" for v in levels),
        samples_per_level=",".join(str(int(n_sam.get(v, 0))) for v in levels),
        spectra_per_level=",".join(str(int(n_sp.get(v, 0))) for v in levels),
        scans_per_sample_min=int(scans.min()), scans_per_sample_max=int(scans.max()),
        scans_per_sample_median=float(scans.median()),
        raw_axis="descending" if wvn[0] > wvn[-1] else "ascending",
        dnu=float(np.median(np.diff(np.sort(wvn)))),
        range_lo=float(np.sort(wvn)[0]), range_hi=float(np.sort(wvn)[-1]),
    )


def mirror_count(fn, sheet, id_row):
    """Number of spectrum columns in a transposed mirror sheet, with their ids."""
    y = pd.read_excel(os.path.join(KAN_ROOT, fn), sheet_name=sheet, header=None)
    cand = [str(v) for v in y.iloc[id_row, 1:].tolist()
            if str(v).lower() not in ("nan", "none", "")]
    return len(cand), Counter(_norm(c) for c in cand), y.shape


def reconcile(key, f):
    rows = []
    flat_c = Counter(_norm(i) for i in f["ids"][f["keep"]])
    for fn, sheet, id_row in MIRROR[key]:
        n, cnt, shape = mirror_count(fn, sheet, id_row)
        extra_ids = {i: c - flat_c.get(i, 0) for i, c in cnt.items() if c > flat_c.get(i, 0)}
        missing_ids = {i: flat_c[i] - cnt.get(i, 0) for i in flat_c if flat_c[i] > cnt.get(i, 0)}
        rows.append(dict(
            dataset=DID[key], mirror_sheet=sheet, spectrum_columns=n,
            distinct_ids=len(cnt), mirror_rows=shape[0], mirror_cols=shape[1],
            extra_in_mirror=sum(extra_ids.values()),
            extra_ids=len(extra_ids), extra_examples=str(sorted(extra_ids)[:4]),
            absent_from_mirror=sum(missing_ids.values()), absent_ids=len(missing_ids),
            absent_examples=str(sorted(missing_ids)[:4]),
            flat_spectra=int(f["keep"].sum())))
    return rows


def juice_labels(f):
    """level-index encoding check: id p<brand><idx> should carry label idx*5 %."""
    ids, adul = f["ids"], f["adul"]
    rows = []
    for i, (sid, lab) in enumerate(zip(ids, adul)):
        m = re.fullmatch(r"p([1-5])(\d{1,2})", _norm(sid))
        if not m or not np.isfinite(lab):
            continue
        pred = int(m.group(2)) * 5.0
        if pred != lab:
            rows.append(dict(id=str(sid), brand=m.group(1), level_index=int(m.group(2)),
                             label_in_file=float(lab), label_from_id=pred,
                             rows_for_id=int((np.array([_norm(x) for x in ids]) == _norm(sid)).sum())))
    lvl35 = sorted(set(_norm(x) for x, a in zip(ids, adul) if a == 35.0))
    lvl45 = sorted(set(_norm(x) for x, a in zip(ids, adul) if a == 45.0))
    return rows, lvl35, lvl45


def tekrar(key, f):
    if key != "coffee":
        return dict(tekrar_rows=0, tekrar_unlabelled=0, tekrar_labelled=0, tekrar_groups=0)
    is_t = np.array(["tekrar" in _norm(i) for i in f["ids"]])
    lab = np.isfinite(f["auth"]) & np.isfinite(f["adul"])
    groups = np.array([GROUP_KEYS[key](i) for i in f["ids"]])
    return dict(tekrar_rows=int(is_t.sum()), tekrar_unlabelled=int((is_t & ~lab).sum()),
                tekrar_labelled=int((is_t & lab).sum()),
                tekrar_groups=int(len(set(groups[is_t & lab]))),
                base_present=int(any(_norm(g) == "1.kahve1.1" for g in groups[f["keep"]])))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    inv_rows, mir_rows = [], []
    for key in ("oil", "coffee", "juice"):
        f = flat(key)
        inv = inventory(key, f)
        inv.update(tekrar(key, f))
        inv_rows.append(inv)
        mir_rows += reconcile(key, f)
        print(f"{inv['dataset']} {key:6s} rows {inv['raw_data_rows']:3d} -> spectra {inv['spectra_final']:3d}"
              f" (drop {inv['rows_dropped']}: blank_id {inv['drop_blank_id']},"
              f" no matrix % {inv['drop_no_matrix']}, no adulterant % {inv['drop_no_adulterant']},"
              f" non-finite spectrum {inv['drop_nonfinite_spectrum']})"
              f" | samples {inv['physical_samples']} | levels {inv['levels']}"
              f" | scans/sample {inv['scans_per_sample_min']}-{inv['scans_per_sample_max']}"
              f" med {inv['scans_per_sample_median']:g}"
              f" | {inv['channels']} ch, {inv['raw_axis']}, dnu {inv['dnu']:.4f},"
              f" {inv['range_lo']:.1f}-{inv['range_hi']:.1f}")
        print(f"      samples/level {inv['samples_per_level']}")
        print(f"      spectra/level {inv['spectra_per_level']}")
        if key == "coffee":
            print(f"      tekrar rows {inv['tekrar_rows']} = unlabelled {inv['tekrar_unlabelled']}"
                  f" + labelled {inv['tekrar_labelled']} in {inv['tekrar_groups']} group(s)"
                  f" ; plain 1.kahve1.1 present in flat sheet: {bool(inv['base_present'])}")
    for r in mir_rows:
        print(f"MIRROR {r['dataset']} {r['mirror_sheet']:7s} spectrum-columns {r['spectrum_columns']:3d}"
              f" (distinct ids {r['distinct_ids']}) vs flat {r['flat_spectra']};"
              f" extra in mirror {r['extra_in_mirror']}, absent from mirror {r['absent_from_mirror']}")
        print(f"         extra {r['extra_examples']} | absent {r['absent_examples']}")

    f = flat("juice")
    bad, lvl35, lvl45 = juice_labels(f)
    print(f"\nD3 level-index check: {len(bad)} id/label inconsistency(ies)")
    for b in bad:
        print("   ", b)
    print(f"    ids labelled 35%: {lvl35}")
    print(f"    ids labelled 45%: {lvl45}")

    A = pd.DataFrame(inv_rows)
    B = pd.DataFrame(mir_rows)
    A.to_csv(os.path.join(OUT, "exp00b_kan_audit.csv"), index=False)
    B.to_csv(os.path.join(OUT, "exp00b_kan_mirror.csv"), index=False)
    pd.DataFrame(bad).to_csv(os.path.join(OUT, "exp00b_kan_labelcheck.csv"), index=False)
    pd.DataFrame([dict(dataset="D3", label_in_file=35.0, ids=",".join(lvl35)),
                  dict(dataset="D3", label_in_file=45.0, ids=",".join(lvl45))]
                 ).to_csv(os.path.join(OUT, "exp00b_kan_labelsum.csv"), index=False)

    lines = ["# Table S16. KAN trio inventory, read from the distributed workbooks", ""]
    for name, frame in (("A. official counts (flat sheets used by the analysis)", A),
                        ("B. transposed mirror sheets (source of the retired 354 / ~515 / 424)", B)):
        lines += [name, ""]
        lines.append("| " + " | ".join(map(str, frame.columns)) + " |")
        lines.append("|" + "|".join(["---"] * len(frame.columns)) + "|")
        for row in frame.itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    uniq = {b["id"]: b for b in bad}
    lines += ["C. D3 label discipline", "",
              f"- id pattern `p<brand><level-index>` with label = 5 x level-index; "
              f"{len(uniq)} identifier(s) disagree with their index ({len(bad)} spectra): "
              + "; ".join(f"`{b['id']}` labelled {b['label_in_file']:g} % but index {b['level_index']} -> {b['label_from_id']:g} %" for b in uniq.values()),
              f"- ids labelled 35 %: {', '.join(lvl35)}",
              f"- ids labelled 45 %: {', '.join(lvl45)} (brand 5 absent until the correction)",
              ""]
    with open(os.path.join(OUT, "exp00b_kan_audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {OUT}/exp00b_kan_audit.csv|mirror.csv|labelcheck.csv|.md")


if __name__ == "__main__":
    main()
