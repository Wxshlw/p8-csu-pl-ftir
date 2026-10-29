"""exp00c —— D7（肉桂）与 D8（姜黄）同源性核查：§3.1 第二段与 Table S17 的素材。

七条独立证据，全部从本地分发文件、随文 README 与正文参考文献表实读：

A  波数轴：两套数据各自的通道数、首末通道、采样间隔；两条轴**按通道序号对齐**后
   的最大/平均偏移（cm⁻¹ 及占一个通道宽度的百分比）。两套数据的轴各自以不同
   精度存储（间隔 1.42670 vs 1.42669 cm⁻¹），浮点精确集合交集会误判为"不共轴"。
B  样品命名与浓度结构：`C<P/W/D><level>S<nn>` 与 `T<level>S<nn>` 的模式、
   水平集合、每水平样品数、编号范围、命名空间是否相交。
C  谱级重复：D8 每条谱对 D7 全部谱的最大余弦相似度（原始吸光度域，
   零均值中心化前后各算一次）；D7 三个掺杂物子集的 0 % 谱之间的
   同源检查（若是同一批纯肉桂样品重复测量，应接近 1）；
   以及各自"最近邻"相似度作为同基质参照。
D  随文文档：cinnamon 的 Readme.pdf / Dataset Description.pdf 与 turmeric 的
   README.docx 里关于仪器、模态、水平、样品数、命名与数据质控的原文字段
   （用于确认"仪器型号/序列号/采集日期"这三项在原始文档中是否真的存在）。
E  归属：从 manuscript.md 参考文献表读取两条 Mendeley 数据条目，抽取作者串，
   判断两套数据是否共享第一作者（"同一课题组"这一结论的证据来源）。

F  量纲：分发文件存的是吸光度还是透过率。判据是近透明区（约 3900 cm^-1）的取值——
   吸光度表那里≈0、透过率表那里≈1，全表取值范围再独立佐证一次。约 1030 cm^-1 处
   的带对比度斜率只作描述量：肉桂基质与这三种掺杂物在糖类区都有吸收，其符号反映的
   是哪个相单位体积吸收更多，而不是文件存的是哪个量。
G  该选择对审计门槛的代价：用 exp05_audit.py 的同一套配置（SG 一阶导、基线一阶、
   自由尺度、品牌感知拟合），在"按存储值"与"按 A = -log10(T)"两个量纲下各重算一次
   曲率 gamma、端元 CNR、基线泄漏与 pass/fail 判定，并标出加载器实际交付的是哪个。

输出 out/exp00c_axis.csv、out/exp00c_naming.csv、out/exp00c_similarity.csv、
     out/exp00c_photometry.csv、out/exp00c_audit_domain.csv、out/exp00c_provenance.json、out/exp00c_d7d8.md
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unmix_ftir.data import CIN_FILES, CIN_ROOT, TUR_ROOT, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def sources(cin_root, tur_root):
    """(dataset, stored-format, path) for the four non-KAN releases, same files the loader reads."""
    ids = {"walnut": "D7w", "peanut": "D7p", "wood dust": "D7d"}
    return [(ids[k], "xlsx", os.path.join(cin_root, CIN_FILES[k]))
            for k in ("walnut", "peanut", "wood dust")] + \
           [("D8", "csv", os.path.join(tur_root, "FTIR_Data.csv"))]


# ---------------------------------------------------------------- A 轴
def axis_rows(sets):
    """Index-aligned grid comparison.

    Both releases are ascending uniform grids that start at ~399.47 cm^-1 with a
    ~1.4267 cm^-1 step, so channel k of one release is channel k of the other; the
    question is how far the two stored axes drift apart over those channels.
    """
    rows = []
    w7 = np.asarray(sets["D7w"]["wvn"], float)
    w8 = np.asarray(sets["D8"]["wvn"], float)
    step = float(np.median(np.diff(w7)))
    tol = 0.01 * step
    n = min(len(w7), len(w8))
    off = np.abs(w7[:n] - w8[:n])
    for name, w in (("D7", w7), ("D8", w8)):
        d = np.diff(w)
        rows.append(dict(dataset=name, channels=int(len(w)), lo=float(w[0]), hi=float(w[-1]),
                         dnu=float(np.median(d)), dnu_spread=float(d.max() - d.min()),
                         matched_within_0p01_channel="", unmatched="",
                         common_range=f"{max(w7[0], w8[0]):.2f}-{min(w7[-1], w8[-1]):.2f}"))
    rows.append(dict(
        dataset=f"concordance, first {n} index-aligned channels", channels=int(n),
        lo=float(w7[0]), hi=float(w8[-1]), dnu=step,
        dnu_spread=float(np.abs(np.median(np.diff(w7)) - np.median(np.diff(w8)))),
        matched_within_0p01_channel=int((off <= tol).sum()),
        unmatched=int(abs(len(w7) - len(w8))),
        common_range=(f"max |offset| {off.max():.4f} cm^-1 = {off.max()/step*100:.2f}% of a "
                      f"channel, mean {off.mean():.4f}")))
    return rows, int(len(w7) - n), int(len(w8) - n), int(n)


# ---------------------------------------------------------------- B 命名
def naming_rows(sets):
    rows = []
    spec = {"D7": (r"^C([PWD])(\d+)S(\d+)$", 2, 3), "D8": (r"^T(\d+)S(\d+)$", 1, 2)}
    for group, key in (("D7", "D7w"), ("D7", "D7p"), ("D7", "D7d"), ("D8", "D8")):
        rx, lv_g, nn_g = spec[group]
        ids = [str(v).strip() for v in sets[key]["ids"]]
        ok = [m for m in (re.fullmatch(rx, s) for s in ids) if m]
        lv = [int(m.group(lv_g)) for m in ok]
        nn = [int(m.group(nn_g)) for m in ok]
        levels = sorted(set(lv))
        rows.append(dict(dataset=f"{group}:{key}", n_ids=len(ids), matched=len(ok), pattern=rx,
                         levels=",".join(map(str, levels)), n_levels=len(levels),
                         samples_per_level=",".join(str(lv.count(L)) for L in levels),
                         sample_no_min=int(min(nn)), sample_no_max=int(max(nn)),
                         duplicate_ids=int(len(ids) - len(set(ids))),
                         namespace="".join(sorted({s[0] for s in ids}))))
    d7ns = {str(v) for k in ("D7w", "D7p", "D7d") for v in sets[k]["ids"]}
    d8ns = {str(v) for v in sets["D8"]["ids"]}
    rows.append(dict(dataset="D7∩D8 identifier intersection", n_ids=len(d7ns | d8ns),
                     matched=len(d7ns & d8ns), pattern="-", levels="-", n_levels=np.nan,
                     samples_per_level="-", sample_no_min=np.nan, sample_no_max=np.nan,
                     duplicate_ids=np.nan, namespace=str(len(d7ns & d8ns))))
    return rows


# ---------------------------------------------------------------- C 谱级相似
def cos_matrix(A, B):
    an = A / np.linalg.norm(A, axis=1, keepdims=True)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def similarity_rows(sets):
    d7 = np.vstack([sets[k]["X"] for k in ("D7w", "D7p", "D7d")])
    d8 = sets["D8"]["X"]
    n_ch = min(d7.shape[1], d8.shape[1])
    A7, A8 = d7[:, :n_ch], d8[:, :n_ch]
    rows = [dict(test="channels used (index-aligned prefix)", value=int(n_ch),
                 detail=f"first {n_ch} stored channels of each release; axes agree to "
                        f"well under one channel width (panel A)")]
    for tag, M in (("raw absorbance", cos_matrix(A8, A7)),
                   ("mean-centred", cos_matrix(A8 - A8.mean(1, keepdims=True),
                                               A7 - A7.mean(1, keepdims=True)))):
        mx = M.max(axis=1)
        rows.append(dict(test=f"D8 -> nearest D7 spectrum, cosine ({tag})",
                         value=float(np.median(mx)),
                         detail=(f"p5 {np.percentile(mx, 5):.4f}, p95 {np.percentile(mx, 95):.4f}, "
                                 f"max {mx.max():.4f}; n>0.999 {int((mx > 0.999).sum())}, "
                                 f"n>0.99 {int((mx > 0.99).sum())}, n>0.95 {int((mx > 0.95).sum())} "
                                 f"of {len(mx)}")))
    # 最近邻来自哪个数据集（对称双向）
    c_dd = cos_matrix(d8, d8)
    c_d7d7 = cos_matrix(d7, d7)
    np.fill_diagonal(c_dd, -1.0)
    np.fill_diagonal(c_d7d7, -1.0)
    rows.append(dict(test="D8 self nearest neighbour cosine (median)",
                     value=float(np.median(c_dd.max(1))),
                     detail=f"p95 {np.percentile(c_dd.max(1), 95):.4f}, max {c_dd.max():.4f}"))
    rows.append(dict(test=f"D7 self nearest neighbour cosine (median, {d7.shape[0]} spectra)",
                     value=float(np.median(c_d7d7.max(1))),
                     detail=f"p5 {np.percentile(c_d7d7.max(1), 5):.4f}, "
                            f"p95 {np.percentile(c_d7d7.max(1), 95):.4f}, "
                            f"max {c_d7d7.max():.4f}"))
    # D7 内部：三个子集的 0 % 谱是否同一批纯肉桂
    keys = ["D7w", "D7p", "D7d"]
    z = {k: sets[k]["X"][np.isclose(sets[k]["y"], 0.0)] for k in keys}
    lab = {k: sets[k]["spec"]["adulterant"] for k in keys}
    for a in range(3):
        for b in range(a + 1, 3):
            ka_, kb_ = keys[a], keys[b]
            M = cos_matrix(z[ka_], z[kb_])
            rows.append(dict(test=f"D7 0% {lab[ka_]} vs 0% {lab[kb_]} cosine",
                             value=float(np.median(M.max(1))),
                             detail=(f"max {M.max():.4f}, mean {M.mean():.4f}, "
                                     f"exact duplicates {int((M.max(1) > 0.99999).sum())} of "
                                     f"{M.shape[0]}")))
    # 基质可分性：D7 0 % 与 D8 0 % 之间的最近邻
    z7 = np.vstack([sets[k]["X"][np.isclose(sets[k]["y"], 0.0)] for k in ("D7w", "D7p", "D7d")])
    z8 = sets["D8"]["X"][np.isclose(sets["D8"]["y"], 0.0)]
    M = cos_matrix(z8[:, :n_ch], z7[:, :n_ch])
    rows.append(dict(test="D8 0% vs D7 0% nearest cosine", value=float(np.median(M.max(1))),
                     detail=f"max {M.max():.4f}, mean {M.mean():.4f} ({len(z8)} vs {len(z7)} spectra)"))
    return rows


# ---------------------------------------------------------------- F 光度量纲
def _raw_family(kind, path):
    """Axis, values and labels exactly as stored, before any loader rewrite.

    Both Mendeley files carry their wavenumber axis in their own column headers, so
    the absorbance-versus-transmittance readout is self-contained.
    """
    if kind == "csv":
        df = pd.read_csv(path)
        t = list(df.columns).index("Target")
        w = df.columns[1:t].to_numpy(float)
        X = df.iloc[:, 1:t].to_numpy(float)
        y = np.array([float(str(v).replace("%", "").strip()) for v in df.iloc[:, t]])
    else:
        df = pd.read_excel(path, sheet_name=0, header=0, engine="openpyxl")
        w = df.columns[1:-1].to_numpy(float)
        X = df.iloc[:, 1:-1].to_numpy(float)
        names = [str(v).strip() for v in df.iloc[:, 0]]
        y = np.array([int(n[2:n.index("S")]) for n in names], float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return w, X[keep], y[keep]


def _raw_kan(name):
    """Stored sheet values for a KAN release, parsed the way load_kan parses them."""
    from unmix_ftir import data as D
    spec = D.KAN_SHEETS[name]
    path = os.path.join(D.KAN_ROOT, spec["file"])
    if not os.path.exists(path):
        path = os.path.join(D.KAN_ROOT, spec["orig"])
    df = pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header"])
    raw = df.to_numpy()
    auth = pd.to_numeric(pd.Series(raw[:, 2]), errors="coerce").to_numpy(float)
    adul = pd.to_numeric(pd.Series(raw[:, 3]), errors="coerce").to_numpy(float)
    X = np.asarray(raw[:, 4:], dtype=float)
    keep = np.isfinite(auth) & np.isfinite(adul) & np.isfinite(X).all(axis=1)
    w = D._kan_axis(df, X.shape[1])
    if w is None:
        raise SystemExit(f"{name}: wavenumber axis not parsed from header")
    w, X = D._orient(w, X)
    return w, X[keep], adul[keep]


def photometry_rows(sets, cin_root, tur_root):
    """Is each release storing absorbance or transmittance, and does it matter?

    Run over all seven analysis datasets, because that is what bounds the finding: if
    only D7 stores a transmittance-like quantity, the whole absorbance-versus-
    transmittance issue is confined to the failing cinnamon conditions.

    The decisive readout, taken from the file as distributed, is the level in a
    near-transparent region (~3900 cm^-1): an absorbance table stores ~0 there, a
    transmittance table stores ~1. The whole-file range settles it independently -
    an absorbance table of these films spans roughly 0-2 with a mode near 0, whereas
    the stored tables here sit just below 1. The slope of the ~1030 cm^-1 contrast
    along the adulterant gradient is reported only as a descriptor: both the cinnamon
    matrix and these adulterants absorb in the carbohydrate region, so its sign says
    which phase carries more absorber per unit volume, not which quantity is stored.
    The consequence is then measured on the manuscript's own linearity diagnostic
    gamma, recomputed after A = -log10(T).
    """
    from unmix_ftir import diagnostics
    from unmix_ftir.forward import SG

    sg = SG(window=15, poly=3, deriv=1)

    def row(key, d, w, X, y):
        band = int(np.argmin(np.abs(w - 1030)))
        transp = int(np.argmin(np.abs(w - 3900)))
        lv = np.unique(np.round(y, 6))
        ml = X[np.abs(y - lv[0]) < 1e-6].mean(0)
        mt = X[np.abs(y - lv[-1]) < 1e-6].mean(0)
        depth = np.array([X[np.abs(y - v) < 1e-6].mean(0)[transp]
                          - X[np.abs(y - v) < 1e-6].mean(0)[band] for v in lv])
        slope = float(np.polyfit(lv, depth, 1)[0])
        base = float(np.nanmean(X[:, transp]))
        as_stored = bool(np.allclose(d["X"], X, equal_nan=True))
        as_conv = bool(np.allclose(d["X"], -np.log10(np.clip(X, 1e-6, None)), equal_nan=True))
        applies = base > 0.5            # -log10 is only a candidate re-reading for T-like tables
        # Labels are taken as analysed: D3's sheet labels p59 at 35 % while the loader
        # ships the documented 45 % correction (Table S16c), and the two differ here.
        y_used = np.asarray(d["y"], float) if X.shape == d["X"].shape else y
        gam = diagnostics.curvature_diagnostic(y_used, sg.transform(X))["gamma"]
        gam_conv = (diagnostics.curvature_diagnostic(
            y_used, sg.transform(-np.log10(np.clip(X, 1e-6, None))))["gamma"]
            if applies else np.nan)
        return dict(
            dataset=key, rows_in_file=int(len(X)), stored_min=float(np.nanmin(X)),
            stored_median=float(np.nanmedian(X)), stored_max=float(np.nanmax(X)),
            share_above_1=float(np.nanmean(X > 1.0)),
            transparent_3900_mean=base,
            stored_quantity=("transmittance" if base > 0.5 else
                             "absorbance" if base < 0.2 else "AMBIGUOUS"),
            log10_transform_applicable=("yes" if applies else
                                        "no (values already absorbance-like, some negative)"),
            domain_analysed=("stored values" if as_stored else
                             "-log10(stored)" if as_conv else "AMBIGUOUS"),
            band_1030_low_label=float(ml[band]), band_1030_high_label=float(mt[band]),
            band_depth_slope=slope,
            labels_as_stored=str(bool(np.array_equal(y_used, np.asarray(y, float)))),
            gamma_as_stored=float(gam), gamma_after_minus_log10=float(gam_conv))

    rows = [row(key, sets[key], *_raw_kan(name))
            for key, name in (("D1", "oil"), ("D2", "coffee"), ("D3", "juice"))]
    for key, kind, path in sources(cin_root, tur_root):
        rows.append(row(key, sets[key], *_raw_family(kind, path)))
    return rows


# ---------------------------------------------------------------- G 审计量对量纲的敏感度
def audit_rows(sets, cin_root, tur_root):
    """Re-estimate the Table 1 audit gate quantities in both photometric domains.

    Panel F establishes *which* quantity the files store; this panel establishes what
    it *costs* to keep it. The gate quantities (curvature gamma, endmember CNR,
    baseline leak) are recomputed with the exact plumbing of exp05_audit.py -
    Savitzky-Golay first derivative, baseline order 1, free scale, brand-aware fit -
    once on the stored values and once after A = -log10(T). Whichever of the two
    equals the loader output is the domain the manuscript's numbers were produced in,
    so the as-analysed column of all seven rows is also an end-to-end cross-check of
    Table 1. The KAN trio has no alternative domain to compare against: panel F
    shows it is already stored as absorbance.
    """
    from unmix_ftir import data as D
    from unmix_ftir import diagnostics, forward as F, unmix

    cfg = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0,
                           resolution=np.nan, n_iter=40)
    sg = F.SG(window=15, poly=3, deriv=1)

    def gate(cur, e):
        return "pass" if (e["cnr"] >= 10.0 and abs(cur["gamma"]) <= 1.0) else "fail"

    def quantify(X, y, br, wvn):
        Xv = sg.transform(np.asarray(X, float))
        cur = diagnostics.curvature_diagnostic(y, Xv)
        mdl = unmix.fit(Xv, y, wvn, cfg, brand=br)
        e = diagnostics.endmember_audit(mdl, Xv, y)
        return cur, e

    rows = []
    for key in ("D1", "D2", "D3"):
        d = sets[key]
        cur, e = quantify(d["X"], d["y"], D.brands(key, d["groups"]), d["wvn"])
        rows.append(dict(
            dataset=key, domain_analysed="stored values (already absorbance, panel F)",
            gamma_analysed=float(cur["gamma"]), cnr_analysed=float(e["cnr"]),
            cnr_partial_analysed=float(e["cnr_partial"]),
            leak_analysed=float(e["baseline_leak"]), gate_analysed=gate(cur, e),
            gamma_other=np.nan, cnr_other=np.nan, cnr_partial_other=np.nan,
            leak_other=np.nan, gate_other="n/a (no transmittance reading applies)",
            gate_flips="n/a", max_cnr_rel_change=np.nan))
    for key, kind, path in sources(cin_root, tur_root):
        d = sets[key]
        w, Xs, ys = _raw_family(kind, path)
        if Xs.shape != d["X"].shape or not np.allclose(ys, d["y"]):
            raise SystemExit(f"{key}: stored file {Xs.shape}/{len(ys)} does not match "
                             f"loader {d['X'].shape}/{len(d['y'])}")
        Xa = -np.log10(np.clip(Xs, 1e-6, None))
        as_stored = bool(np.allclose(d["X"], Xs, equal_nan=True))
        as_log10 = bool(np.allclose(d["X"], Xa, equal_nan=True))
        if as_stored == as_log10:
            raise SystemExit(f"{key}: cannot tell which domain the loader returns")
        br = D.brands(key, d["groups"])
        cs, es = quantify(Xs, d["y"], br, w)
        ca, ea = quantify(Xa, d["y"], br, w)
        used_cur, used_e = (cs, es) if as_stored else (ca, ea)
        alt_cur, alt_e = (ca, ea) if as_stored else (cs, es)
        rows.append(dict(
            dataset=key, domain_analysed=("stored" if as_stored else "-log10(stored)"),
            gamma_analysed=float(used_cur["gamma"]), cnr_analysed=float(used_e["cnr"]),
            cnr_partial_analysed=float(used_e["cnr_partial"]),
            leak_analysed=float(used_e["baseline_leak"]),
            gate_analysed=gate(used_cur, used_e),
            gamma_other=float(alt_cur["gamma"]), cnr_other=float(alt_e["cnr"]),
            cnr_partial_other=float(alt_e["cnr_partial"]),
            leak_other=float(alt_e["baseline_leak"]), gate_other=gate(alt_cur, alt_e),
            gate_flips=str(gate(used_cur, used_e) != gate(alt_cur, alt_e)),
            max_cnr_rel_change=float(abs(alt_e["cnr"] - used_e["cnr"]) /
                                     max(used_e["cnr"], 1e-9))))
    return rows


# ---------------------------------------------------------------- D 文档证据
DOC_PATTERNS = {
    "instrument": r"(spectrometer|Tensor|Nicolet|Frontier|Perkin|Bruker|FTIR instrument)",
    "range": r"(400[–-]4000|4000[–-]600|spectral range|wavenumber)",
    "variables": r"(2,?523|2,?522|spectral variables|variables)",
    "modalities": r"(colorimet|HunterLab|RGB|image|multimodal)",
    "levels": r"(thirteen|seven|levels|adulteration levels)",
    "samples": r"(50 (independently|samples)|samples per|Total samples)",
    "naming": r"(CP0S01|CW15S18|CD24S50|T0S01|identifier|naming)",
    "qc": r"(validation|duplicate|quality assurance|verification)",
    "serial": r"(serial|instrument ID|sequence number)",
    "date": r"(collect|acquisition|date|measured on)",
}


def read_pdf(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def read_docx(path):
    from docx import Document
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.append(" | ".join(c.text for c in r.cells))
    return "\n".join(parts)


def provenance():
    texts = {}
    for tag, path, kind in (
            ("cinnamon_Readme.pdf", os.path.join(CIN_ROOT, "Readme.pdf"), "pdf"),
            ("cinnamon_Dataset_Description.pdf", os.path.join(CIN_ROOT, "Dataset Description.pdf"), "pdf"),
            ("turmeric_README.docx", os.path.join(TUR_ROOT, "README.docx"), "docx")):
        raw = (read_pdf(path) if kind == "pdf" else read_docx(path))
        raw = re.sub(r"\s+", " ", raw)
        texts[tag] = raw
    hits = {}
    for key, rx in DOC_PATTERNS.items():
        found = re.compile(rx, re.I)
        hits[key] = {tag: [s for s in re.split(r"(?<=[.;])\s+", t)
                           if found.search(s)][:3]
                     for tag, t in texts.items()}
    meta = {tag: {"chars": len(t)} for tag, t in texts.items()}
    refs = authorship_from_reference_list()
    return meta, hits, {tag: t[:400] for tag, t in texts.items()}, refs


def authorship_from_reference_list():
    """D7/D8 attribution read from the manuscript reference list, not restated by hand."""
    md = open(os.path.join(ROOT, "manuscript", "manuscript.md"), encoding="utf-8").read()
    out = {}
    for tag, doi in (("D7_cinnamon", "10.17632/rtbsf2r53x"), ("D8_turmeric", "10.17632/cb5kbt7bgv")):
        line = next((ln.strip() for ln in md.splitlines() if doi in ln), None)
        if line is None:
            raise SystemExit(f"reference for {doi} not found in manuscript.md")
        body = re.sub(r"^\[\d+\]\s*", "", line)
        authors = body.split(", Multimodal")[0] if ", Multimodal" in body else body[:60]
        out[tag] = dict(reference=line, authors=authors, doi=doi,
                        first_author=authors.split(",")[0].strip(),
                        n_authors=len([a for a in re.split(r",| and ", authors) if a.strip()]))
    out["shared_first_author"] = (out["D7_cinnamon"]["first_author"]
                                  == out["D8_turmeric"]["first_author"])
    def tok(s):
        return {t.strip() for t in re.split(r",| and ", s) if t.strip()}
    out["shared_authors"] = sorted(tok(out["D7_cinnamon"]["authors"])
                                   & tok(out["D8_turmeric"]["authors"]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sets = {k: load(k) for k in ("D1", "D2", "D3", "D7w", "D7p", "D7d", "D8")}
    for k in ("D7p", "D7d"):
        assert np.array_equal(sets[k]["wvn"], sets["D7w"]["wvn"]), f"{k} axis differs from D7w"
    ar, only7, only8, inter = axis_rows(sets)
    pd.DataFrame(ar).to_csv(os.path.join(OUT, "exp00c_axis.csv"), index=False)
    nr = naming_rows(sets)
    pd.DataFrame(nr).to_csv(os.path.join(OUT, "exp00c_naming.csv"), index=False)
    sr = similarity_rows(sets)
    pd.DataFrame(sr).to_csv(os.path.join(OUT, "exp00c_similarity.csv"), index=False)
    pr = photometry_rows(sets, CIN_ROOT, TUR_ROOT)
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "exp00c_photometry.csv"), index=False)
    gr = audit_rows(sets, CIN_ROOT, TUR_ROOT)
    pd.DataFrame(gr).to_csv(os.path.join(OUT, "exp00c_audit_domain.csv"), index=False)
    meta, hits, heads, refs = provenance()
    with open(os.path.join(OUT, "exp00c_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(files=meta, patterns=hits, heads=heads, attribution=refs),
                  fh, ensure_ascii=False, indent=1)

    lines = ["# D7 / D8 homology check (exp00c)", ""]
    for title, rows in (("A. wavenumber axis", ar), ("B. identifiers and design", nr),
                        ("C. spectrum-level similarity", sr),
                        ("F. photometric quantity of the stored files", pr),
                        ("G. audit gate quantities in both photometric domains", gr)):
        lines += [title, ""]
        cols = list(rows[0])
        lines.append("| " + " | ".join(map(str, cols)) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
        lines.append("")
    lines += ["D. documentation fields", ""]
    for key, per in hits.items():
        got = {t: v for t, v in per.items() if v}
        lines.append(f"- **{key}**: " + ("; ".join(f"{t}: {v[0][:150]}" for t, v in got.items())
                                        if got else "no sentence in either README matches this pattern"))
    lines += ["", "E. attribution from the reference list", ""]
    for tag in ("D7_cinnamon", "D8_turmeric"):
        lines.append(f"- **{tag}**: {refs[tag]['reference']}")
    lines.append(f"- shared first author: {refs['shared_first_author']}")
    lines.append(f"- authors in common: {', '.join(refs['shared_authors']) or 'none'}")
    lines.append("")
    with open(os.path.join(OUT, "exp00c_d7d8.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    for r in ar:
        print(f"{r['dataset']}: {r['channels']} ch, {r['lo']:.2f}-{r['hi']:.2f}, "
              f"dnu {r['dnu']:.5f} (spread {r['dnu_spread']:.5f}), "
              f"matched {r['matched_within_0p01_channel']}, extra {r['unmatched']} "
              f"[{r['common_range']}]")
    print(f"extra channels beyond the shared prefix: D7 {only7}, D8 {only8}, "
          f"index-aligned channels compared {inter}")
    for r in nr:
        print(f"{r['dataset']}: n {r['n_ids']}, matched {r['matched']}, levels {r['levels']}, "
              f"sample no {r['sample_no_min']}-{r['sample_no_max']}, dup {r['duplicate_ids']}")
    for r in sr:
        print(f"[{r['test']}] {r['value']} | {r['detail']}")
    for r in pr:
        g2 = ("n/a" if np.isnan(r["gamma_after_minus_log10"])
              else f"{r['gamma_after_minus_log10']:+.3f}")
        print(f"{r['dataset']}: stored [{r['stored_min']:.4f}, {r['stored_median']:.4f}, "
              f"{r['stored_max']:.4f}], share>1 {r['share_above_1']:.4f}, "
              f"3900 mean {r['transparent_3900_mean']:.4f} -> {r['stored_quantity']}, "
              f"band contrast slope {r['band_depth_slope']:+.5f}, "
              f"loader delivers {r['domain_analysed']}, "
              f"gamma stored {r['gamma_as_stored']:+.3f} -> after -log10 {g2}")
    for r in gr:
        print(f"{r['dataset']} analysed as {r['domain_analysed']}: "
              f"gamma {r['gamma_analysed']:+.3f}, CNR {r['cnr_analysed']:.1f} "
              f"(partial {r['cnr_partial_analysed']:.1f}), leak {r['leak_analysed']:.3f}, "
              f"gate {r['gate_analysed']} || other domain: gamma {r['gamma_other']:+.3f}, "
              f"CNR {r['cnr_other']:.1f} (partial {r['cnr_partial_other']:.1f}), "
              f"leak {r['leak_other']:.3f}, gate {r['gate_other']} "
              f"[flips={r['gate_flips']}]")
    comp = [r for r in gr if r["gate_flips"] != "n/a"]
    print(f"domain-sensitivity verdict: verdicts that flip = "
          f"{sum(r['gate_flips'] == 'True' for r in comp)} of {len(comp)} comparable rows; "
          f"largest relative CNR change = {np.nanmax([r['max_cnr_rel_change'] for r in comp]):.2f}")
    print(f"\nwrote {OUT}/exp00c_axis.csv, exp00c_naming.csv, exp00c_similarity.csv, "
          f"exp00c_photometry.csv, exp00c_audit_domain.csv, exp00c_provenance.json, "
          f"exp00c_d7d8.md")


if __name__ == "__main__":
    main()
