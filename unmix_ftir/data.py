"""unmix_ftir.data —— 5 套 ATR-FTIR 浓度梯度数据集的统一加载与盘点。

命名（与 §3.1 的数据集命名一致）
--------------------------------
  D1  oil      : KAN 橄榄油 + 葵花籽油    11 水平 0–30 % (3 % 步长) × 5 品牌
  D2  coffee   : KAN 咖啡 + 麦芽粉        11 水平 0–50 % (5 % 步长) × 5 品牌
  D3  juice    : KAN 橙汁 + 苹果汁        11 水平 0–50 % (5 % 步长) × 5 品牌
  D7  cinnamon : 肉桂 + {核桃粉, 花生粉, 木屑}  13 水平 0/1/2/3/4/6/8/10/12/15/18/21/24 %
  D8  turmeric : 姜黄 + 小麦粉            7 水平 0/1/2/3/5/7/9 %

统一返回
--------
  X      (n, N) 吸光度，波数轴**升序**
  y      (n,)   掺杂物质量分数（%）
  groups (n,)   物理样品 ID（泄漏安全分组）
  wvn    (N,)   波数（cm^-1，升序）
  dnu    float  采样间隔（cm^-1，中位数）
  meta   dict   谱数/水平/来源/轴来源/方向等

设计纪律
--------
* 谱数、水平数**只以本模块从原始文件实读为准**（§3.1 与 Table S16）。
* KAN 四个数据 quirk（纯油 Z<k> 合并、18_5_2 重扫描、咖啡 tekrar 无标签、果汁 p59
  误标 65→55）全部复现；不复现会泄漏或错标。
* 波数轴统一升序；不做任何插值（各数据集内部分析，跨数据集比较只用无量纲指标）。
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

SHARED = os.environ.get("P8_SHARED_DATA", "_shared_data")
KAN_ROOT = os.path.join(SHARED, "kan")
CIN_ROOT = os.path.join(SHARED, "cinnamon")
TUR_ROOT = os.path.join(SHARED, "turmeric")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "cache")

# ---------------------------------------------------------------------------
# 元数据（数据集级；谱数等运行期量一律实读，不写死）
# ---------------------------------------------------------------------------
SPECS = {
    "D1": dict(family="kan", key="oil", matrix="olive oil", adulterant="sunflower oil",
               instrument="Bruker Tensor 27 + ATR", resolution=4.0,
               source="KAN-FTIR (GitHub) / Batayhi et al., Foods 2026",
               license="CC BY 4.0"),
    "D2": dict(family="kan", key="coffee", matrix="roasted coffee", adulterant="malt flour",
               instrument="Bruker Tensor 27 + ATR", resolution=4.0,
               source="KAN-FTIR (GitHub) / Batayhi et al., Foods 2026",
               license="CC BY 4.0"),
    "D3": dict(family="kan", key="juice", matrix="orange juice", adulterant="apple juice",
               instrument="Bruker Tensor 27 + ATR", resolution=4.0,
               source="KAN-FTIR (GitHub) / Batayhi et al., Foods 2026",
               license="CC BY 4.0"),
    "D7w": dict(family="cinnamon", key="walnut", matrix="cinnamon", adulterant="walnut",
                instrument="ATR-FTIR (Mendeley rtbsf2r53x)", resolution=np.nan,
                source="Mendeley rtbsf2r53x (DOI 10.17632/rtbsf2r53x.1)",
                license="CC BY 4.0"),
    "D7p": dict(family="cinnamon", key="peanut", matrix="cinnamon", adulterant="peanut",
                instrument="ATR-FTIR (Mendeley rtbsf2r53x)", resolution=np.nan,
                source="Mendeley rtbsf2r53x (DOI 10.17632/rtbsf2r53x.1)",
                license="CC BY 4.0"),
    "D7d": dict(family="cinnamon", key="wood dust", matrix="cinnamon", adulterant="wood dust",
                instrument="ATR-FTIR (Mendeley rtbsf2r53x)", resolution=np.nan,
                source="Mendeley rtbsf2r53x (DOI 10.17632/rtbsf2r53x.1)",
                license="CC BY 4.0"),
    "D8": dict(family="turmeric", key="wheat flour", matrix="turmeric", adulterant="wheat flour",
               instrument="ATR-FTIR (Mendeley cb5kbt7bgv)", resolution=np.nan,
               source="Mendeley cb5kbt7bgv (DOI 10.17632/cb5kbt7bgv.2)",
               license="CC BY 4.0"),
}

KAN_SHEETS = {
    "oil": dict(file="kan_oliveoil.xlsx", orig="zeytinyağı tağşiş.xlsx", sheet="Sayfa7", header=1),
    "coffee": dict(file="kan_coffee.xlsx", orig="Kahve_Tağşiş.xlsx", sheet="Sayfa4", header=0),
    "juice": dict(file="kan_juice.xlsx", orig="meyve suyu tağşiş.xlsx", sheet="Sayfa2", header=0),
}
KAN_LEVELS = {
    "oil": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    "coffee": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "juice": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
}

CIN_FILES = {"walnut": "FTIR_data_Walnut.xlsx", "peanut": "FTIR_data_peanut.xlsx",
             "wood dust": "FTIR_data_wood_dust.xlsx"}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def _orient(wvn, X):
    wvn = np.asarray(wvn, float)
    X = np.asarray(X, float)
    if wvn[0] > wvn[-1]:
        return wvn[::-1].copy(), X[:, ::-1].copy()
    return wvn, X


def _dnu(wvn):
    return float(np.median(np.diff(wvn)))


def _to_absorbance(X):
    """透过率型表格 -> 吸光度 A = -log10(T)；已是吸光度的表格原样返回。

    判据只看存储值本身：单光束比值表的近红外基线略低于 1（本仓库里中位数
    0.96-0.99）、全表不超过 10，而吸光度表的中位数接近 0。转发于
    scripts/exp00c_d7d8_homology.py 的面板 F（out/exp00c_photometry.csv）。
    """
    X = np.asarray(X, float)
    trans = bool(np.nanmax(X) < 10 and np.nanmedian(X) > 0.2)
    return (-np.log10(np.clip(X, 1e-6, None)) if trans else X), trans


# --- KAN trio 分组键（与 KAN 仓库 paper2/parsers.py 一致）------------------
def _oil_group(sid):
    s = _norm(sid)
    m = re.fullmatch(r"z(\d+)", s)
    if m:
        return m.group(1) + "1"          # quirk 1: 纯油块 Z<k> 与 <k>1 同一物理样品
    return re.sub(r"([._]\d+)+$", "", s)  # quirk 2: 18_5_2 -> 18


def _coffee_group(sid):
    return re.sub(r"\.\d+$", "", _norm(sid))


def _juice_group(sid):
    s = _norm(sid)
    m = re.match(r"(p\d+)", s)
    return m.group(1) if m else s


GROUP_KEYS = {"oil": _oil_group, "coffee": _coffee_group, "juice": _juice_group}


def _kan_axis(df, n):
    """从表头解析真实波数；失败返回 None。"""
    vals = []
    for c in df.columns:
        try:
            vals.append(float(str(c).replace(",", ".").strip()))
        except Exception:
            pass
    if len(vals) >= n:
        arr = np.array(vals[-n:], float)
        if 400 < arr.min() and arr.max() < 5000:
            return arr
    return None


def load_kan(key, cache_dir=CACHE):
    spec = KAN_SHEETS[key]
    path = os.path.join(KAN_ROOT, spec["file"])
    if not os.path.exists(path):
        path = os.path.join(KAN_ROOT, spec["orig"])
    os.makedirs(cache_dir, exist_ok=True)
    npz = os.path.join(cache_dir, f"kan_{key}.npz")
    axis_source = "header"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        X, y, groups, ids, wvn = z["X"], z["y"], z["groups"], z["ids"], z["wvn"]
        axis_source = str(z["axis_source"])
    else:
        df = pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header"])
        raw = df.to_numpy()
        ids = np.array([str(v) for v in raw[:, 0]], dtype=object)
        auth = pd.to_numeric(pd.Series(raw[:, 2]), errors="coerce").to_numpy(float)
        adul = pd.to_numeric(pd.Series(raw[:, 3]), errors="coerce").to_numpy(float)
        X = np.asarray(raw[:, 4:], dtype=float)
        keep = np.isfinite(auth) & np.isfinite(adul) & np.isfinite(X).all(axis=1)
        ids, adul, X = ids[keep], adul[keep], X[keep]   # quirk 3: 咖啡 tekrar 无标签行在此丢弃
        if key == "juice":
            # quirk 4: p59 is labelled 35 % in the distributed sheet, but the ids follow
            # p<brand><level-index> with label = 5 x index, so index 9 belongs at 45 %;
            # the 45 % level otherwise lacks brand 5 (p19,p29,p39,p49) while 35 % carries
            # six ids. Verified by scripts/exp00b_kan_audit.py (Table S16c).
            hit = np.array([_norm(i) == "p59" for i in ids])
            if hit.any():
                adul[hit] = 45.0
        groups = np.array([GROUP_KEYS[key](i) for i in ids], dtype=object)
        wvn = _kan_axis(df, X.shape[1])
        if wvn is None:
            raise RuntimeError(f"{key}: wavenumber axis not parsed from header")
        y = adul
        np.savez(npz, X=X, y=adul, groups=groups, ids=ids, wvn=wvn, axis_source=axis_source)
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    if wvn[0] < 100 or wvn[-1] > 5000:
        raise RuntimeError(f"{key}: implausible wavenumber axis {wvn[0]}..{wvn[-1]}")
    return dict(X=X, y=np.asarray(y, float), groups=np.asarray(groups, dtype=object),
                wvn=wvn, dnu=_dnu(wvn), ids=np.asarray(ids, dtype=object),
                meta=dict(axis_source=axis_source, raw_axis_descending=raw_desc))


def load_cinnamon(key, cache_dir=CACHE):
    stem = os.path.splitext(CIN_FILES[key])[0]
    npz = os.path.join(cache_dir, f"cinnamon_{stem}.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        X, cat, groups, wvn = z["X"], z["y"], z["groups"], z["wvn"]
    else:
        os.makedirs(cache_dir, exist_ok=True)
        df = pd.read_excel(os.path.join(CIN_ROOT, CIN_FILES[key]), sheet_name=0,
                           header=0, engine="openpyxl")
        names = df.iloc[:, 0].to_numpy()
        wvn = df.columns[1:-1].to_numpy(float)
        X = df.iloc[:, 1:-1].to_numpy(float)
        raw = df.iloc[:, -1]
        if raw.dtype == object or str(raw.dtype).startswith("str"):
            cat = np.array([float(str(v).strip().rstrip("%")) / 100.0 for v in raw])
        else:
            cat = raw.to_numpy(float)
        groups = np.asarray(names, dtype=object)
        np.savez(npz, X=X, y=cat, groups=groups, wvn=wvn)
    keep = np.isfinite(X).all(axis=1)
    X, cat, groups = X[keep], np.asarray(cat, float)[keep], np.asarray(groups, dtype=object)[keep]
    # 分发表格是单光束比值（3900 cm^-1 处约 0.998，带向下），转为吸光度后
    # 才与 §2 的 Beer-Lambert 闭合式同量纲。缓存里保持分发原值。
    X, trans = _to_absorbance(X)
    y = cat * 100.0                                  # 质量分数 -> %
    lvl = np.array([int(n[2:n.index("S")]) for n in groups])   # 名称内嵌浓度级
    if not np.allclose(lvl, y):
        raise RuntimeError(f"D7 {key}: category 与 sample_name 浓度级不一致")
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    return dict(X=X, y=y, groups=groups, wvn=wvn, dnu=_dnu(wvn), ids=groups,
                meta=dict(axis_source="header", raw_axis_descending=raw_desc,
                          from_transmittance=trans,
                          pct="category(fraction)*100, name-checked"))


def load_turmeric():
    df = pd.read_csv(os.path.join(TUR_ROOT, "FTIR_Data.csv"))
    names = np.array([str(v).strip() for v in df.iloc[:, 0]], dtype=object)
    tcol = list(df.columns).index("Target")
    wvn = df.columns[1:tcol].to_numpy(float)
    X = df.iloc[:, 1:tcol].to_numpy(float)
    y = np.array([float(str(v).replace("%", "").strip()) for v in df.iloc[:, tcol]])
    keep = np.isfinite(X).all(axis=1)
    X, y, names = X[keep], y[keep], names[keep]
    X, trans = _to_absorbance(X)            # 分发表 T~0.38-1.10 -> A = -log10(T)
    raw_desc = bool(wvn[0] > wvn[-1])
    wvn, X = _orient(wvn, X)
    return dict(X=X, y=y, groups=names, wvn=wvn, dnu=_dnu(wvn), ids=names,
                meta=dict(axis_source="header", raw_axis_descending=raw_desc,
                          from_transmittance=trans))


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def brands(name, groups):
    """品牌（基质来源）标签。

    KAN trio 的样品 ID 编码 ``<brand><level>``（D3 前置 ``p``），因此品牌 = 首位字符。
    其余数据集没有品牌结构 -> 返回常数 '0'（单基质）。
    """
    g = np.asarray(groups, dtype=object)
    if name in ("D1", "D2"):
        return np.array([str(v)[0] for v in g], dtype=object)
    if name == "D3":
        return np.array([str(v)[1] if str(v).startswith("p") else str(v)[0] for v in g],
                        dtype=object)
    return np.full(len(g), "0", dtype=object)


def load(name: str, **kw):
    """name ∈ {D1,D2,D3,D7w,D7p,D7d,D8}"""
    spec = SPECS[name]
    if spec["family"] == "kan":
        d = load_kan(spec["key"], **kw)
    elif spec["family"] == "cinnamon":
        d = load_cinnamon(spec["key"], **kw)
    elif spec["family"] == "turmeric":
        d = load_turmeric()
    else:                                   # pragma: no cover
        raise KeyError(name)
    d["name"] = name
    d["spec"] = spec
    d["y_frac"] = d["y"] / 100.0
    d["meta"].update(dict(name=name, dataset=name, n_spectra=int(d["X"].shape[0]),
                          n_channel=int(d["X"].shape[1]),
                          n_groups=int(len(np.unique(d["groups"]))),
                          resolution=spec["resolution"],
                          matrix=spec["matrix"], adulterant=spec["adulterant"],
                          source=spec["source"], license=spec["license"]))
    return d


ALL = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]


def inventory(d, level_tol=1e-6):
    """单数据集盘点：谱数、样品数、每样品扫描数、水平与各水平样本数、轴信息。"""
    y = np.asarray(d["y"], float)
    g = np.asarray(d["groups"], dtype=object)
    wvn = np.asarray(d["wvn"], float)
    levels = np.unique(np.round(y / level_tol) * level_tol)
    per_level = {float(v): int((np.abs(y - v) < 1e-6).sum()) for v in levels}
    per_level_groups = {float(v): int(len(np.unique(g[np.abs(y - v) < 1e-6]))) for v in levels}
    groups, counts = np.unique(g, return_counts=True)
    n_scans_total = int(d["X"].shape[0])
    # 每样品是否只含一个浓度（泄漏检查）
    mixed = 0
    for gg in groups:
        if len(np.unique(np.round(y[g == gg], 6))) > 1:
            mixed += 1
    return dict(
        name=d["name"], matrix=d["spec"]["matrix"], adulterant=d["spec"]["adulterant"],
        n_spectra=n_scans_total, n_channel=int(d["X"].shape[1]),
        n_groups=int(len(groups)), scans_per_group=dict(
            min=int(counts.min()), median=float(np.median(counts)), max=int(counts.max()),
            mean=round(float(counts.mean()), 2)),
        n_levels=int(len(levels)), levels=[float(v) for v in levels],
        scans_per_level={f"{k:g}": v for k, v in per_level.items()},
        groups_per_level={f"{k:g}": v for k, v in per_level_groups.items()},
        wvn_min=float(wvn.min()), wvn_max=float(wvn.max()),
        dnu=round(float(np.median(np.diff(wvn))), 4),
        wvn_stored_descending=bool(d["wvn"][0] > d["wvn"][-1]),
        axis_source=d["meta"].get("axis_source", "n/a"),
        resolution=d["spec"]["resolution"],
        absorbance=dict(min=round(float(d["X"].min()), 4),
                        max=round(float(d["X"].max()), 4),
                        p001=round(float(np.percentile(d["X"], 0.1)), 4),
                        p999=round(float(np.percentile(d["X"], 99.9)), 4)),
        nonfinite=int((~np.isfinite(d["X"])).sum()),
        groups_with_multiple_levels=int(mixed),
        duplicate_spectra=int(n_scans_total - len(np.unique(d["X"], axis=0))),
        meta=dict(source=d["spec"]["source"], license=d["spec"]["license"],
                  instrument=d["spec"]["instrument"]),
    )


if __name__ == "__main__":
    import json
    out = {}
    for nm in ALL:
        d = load(nm)
        inv = inventory(d)
        out[nm] = inv
        print(f"{nm:5s} n={inv['n_spectra']:4d} N={inv['n_channel']:4d} "
              f"groups={inv['n_groups']:3d} levels={inv['n_levels']:2d} "
              f"dnu={inv['dnu']:.3f} range={inv['wvn_min']:.0f}-{inv['wvn_max']:.0f} "
              f"scans/grp={inv['scans_per_group']['min']}-{inv['scans_per_group']['max']}")
    print(json.dumps(out, indent=1)[:400])
