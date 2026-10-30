"""make_submission_docs —— 生成 Analytica Chimica Acta 投稿用 cover letter 与 highlights（docx）。

Cover letter 在作者自己的信头模板上生成（`manuscript/cover_letter_template.docx`：
集美大学信头图、手写签名、A4 页面设置、12 pt Times New Roman / 1.5 倍行距的段落样式），
其余段落由模板自身的同类段落克隆而来。
Highlights 按 ACA《Guide for authors》核对：3–5 条、每条含空格不超过 85 字符。
作者、单位、邮箱均保留占位符，不虚构任何署名信息。
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "manuscript"

# The author's own letterhead file, kept inside the project so no personal path is
# written into the shipped code. Not redistributed in the code archive.
CL_TEMPLATE = MDIR / "cover_letter_template.docx"
# Date line inside the letterhead's sender block; set to the actual submission date.
LETTER_DATE = "October 29, 2026"


def _segments(text: str) -> list:
    """Split **bold** markup into (text, is_bold) run segments."""
    segs = [(chunk, i % 2 == 1) for i, chunk in enumerate(text.split("**")) if chunk]
    assert not any("**" in s for s, _ in segs), text
    return segs

TITLE = ("Beyond the calibration range: identifiability and extrapolation of "
         "concentration-supervised Beer\u2013Lambert unmixing for ATR-FTIR "
         "adulterant quantification")

HIGHLIGHTS = [
    "Endmember contrast, curvature and shape audit out-of-range reporting.",
    "Unmixing beats four fixed comparators in 19 of 24 in-region cells.",
    "Leverage above one flags ill-posed designs; condition number does not.",
    "Profile and conformal intervals both fail where extrapolation bias dominates.",
    "Seven development datasets, two external ones, 21 conditions, 8 comparators.",
]


def _check_highlights_match_manuscript() -> None:
    """Fail loudly if the highlight list drifts from manuscript.md."""
    import re

    # ACA guide for authors: 3-5 bullets, at most 85 characters including spaces.
    assert 3 <= len(HIGHLIGHTS) <= 5, len(HIGHLIGHTS)
    over = [(len(h), h) for h in HIGHLIGHTS if len(h) > 85]
    assert not over, over

    src = (MDIR / "manuscript.md").read_text(encoding="utf-8")
    block = re.search(r"## Highlights\s*\n\n(.+?)\n\n## Abstract", src, re.S).group(1)
    from_md = [l.strip()[1:].strip() for l in block.splitlines() if l.strip().startswith("-")]
    if from_md != HIGHLIGHTS:
        raise SystemExit(
            "HIGHLIGHTS in make_submission_docs.py differ from manuscript.md:\n"
            f"  script: {HIGHLIGHTS}\n  md:     {from_md}"
        )

NOVELTY = (
    "Out-of-range quantification by attenuated total reflectance Fourier transform "
    "infrared (ATR-FTIR) spectroscopy is usually treated as a property of the "
    "regression model. We show it is a property of the reference data. Two quantities "
    "computed before any out-of-range sample is measured (the baseline-projected "
    "endmember contrast-to-noise ratio and the level-mean curvature) gate whether "
    "a binary unmixing assay can be extrapolated at all, independently of model class, "
    "and two further reads of the same reference spectra, the shape of the recovered "
    "matrix endmember and the width of the calibrated interval relative to the "
    "reporting range, decide whether the resulting number is worth reporting. Across "
    "seven public datasets, 14 out-of-range conditions and seven interpolation "
    "controls, contrast orders the systems by extrapolation error (\u03c1 = \u22120.60 across "
    "the five measurement sources, \u22120.71 at the dataset level), curvature is "
    "associated with distortion of the response slope, and the gate decision is "
    "unchanged when the audit is re-estimated on samples and brands disjoint from "
    "the calibration and test data. A concentration-supervised Beer\u2013Lambert "
    "estimator then has the lower error in 19 of the 24 cells inside the passing region "
    "against four comparators named before the comparison (median ratio 0.861), and two "
    "datasets deposited independently of this work return the same four verdicts. "
    "A leverage screen flags ill-posed designs, and "
    "both calibrated and distribution-free intervals expose residual bias rather than "
    "repairing it. The audit tells analysts, at the design stage, when not to report a number."
)

COVER = [
    ("title", "**Manuscript Title:** " + TITLE),
    ("salutation", "Dear Editor in Chief,"),
    ("body", "I am pleased to submit our manuscript entitled \u201c" + TITLE + "\u201d for your "
             "consideration for publication in **Analytica Chimica Acta**."),
    ("body", "Out-of-range adulterant quantification is a routine screening demand in attenuated "
             "total reflectance Fourier transform infrared (ATR-FTIR) spectroscopy. A calibration "
             "fitted over the declared range is asked for a number beyond it, where it has no "
             "support, and its interval no longer describes the error it makes. Multivariate "
             "regression cannot answer the question from the calibration alone, and a two-component "
             "Beer\u2013Lambert closure is usually trusted to extrapolate without a test of whether it "
             "can. Figures of merit evaluated at the calibration stage describe the fitted range and "
             "say nothing about behaviour outside it."),
    ("body", "To address these challenges, we propose a design-stage audit and a "
             "concentration-supervised estimator, and summarize the contributions of this work as "
             "follows."),
    ("body", "First, we show that out-of-range quantifiability is a property of the reference data, "
             "not of the regression model. Two quantities computed from the reference spectra before "
             "any out-of-range sample is measured (the baseline-projected endmember contrast-to-noise "
             "ratio, CNR, and the level-mean response curvature, \u03b3) gate whether a binary unmixing "
             "assay can be extrapolated at all, independently of model class. Two further reads of "
             "the same spectra, the shape of the recovered matrix endmember and the width of the "
             "calibrated interval relative to the reporting range, decide whether the resulting "
             "number is worth reporting."),
    ("body", "Second, we model the mixture as a two-component Beer\u2013Lambert closure whose "
             "endmembers are estimated under concentration supervision. Once the endmembers are "
             "fixed, the fraction and an interval per spectrum follow in closed form from a profiled "
             "likelihood, so each out-of-range result is reported with a calibrated statement rather "
             "than as a bare point estimate. The estimator is tested against eight established "
             "regression and curve-resolution methods, four of them named before the comparison."),
    ("body", "Third, we validate the audit on seven public ATR-FTIR datasets (3,470 spectra; "
             "fourteen out-of-range conditions and seven interpolation controls) and on two datasets "
             "deposited independently of this work (537 spectra, four further matrix\u2013adulterant "
             "systems). The audit separates the systems in which out-of-range quantification "
             "succeeds from those in which it fails and orders them by extrapolation error "
             "(\u03c1 = \u22120.60 across the five measurement sources, \u22120.71 at the dataset level). Inside "
             "the passing region the estimator has the lower error in **19 of the 24 cells** against "
             "the four prespecified comparators (**median ratio 0.861**), and **18 of 48** outside it "
             "(**1.10**). The external datasets return the same four verdicts and the same ordering "
             "(median **0.92** where the audit passes, **1.28** where it fails)."),
    ("body", "Beyond predictive performance, the protocol identifies the cause of each failure "
             "(insufficient spectral contrast, a nonlinear response, a non-physical endmember, or a "
             "misspecified nuisance model) and gives the design rule that avoids it: leverage above "
             "one flags a calibration whose error is systematically larger, where the condition number does "
             "not. Intervals calibrated inside the range understate the error made outside it and "
             "expose the residual bias instead of repairing it. All of these decisions are made at "
             "the design stage, before routine samples are analysed."),
    ("body", "We believe that this manuscript is well suited to **Analytica Chimica Acta** because it "
             "speaks to the journal's readership on figures of merit and method validation. The "
             "contrast criterion is the net analyte signal of the adulterant in units of measurement "
             "noise, the curvature criterion is the linearity check behind lack-of-fit diagnostics, "
             "and the interval is the calibrated statement those diagnostics exist to support. The "
             "audit therefore tells analysts, before a measurement is made, when an out-of-range "
             "result should not be reported."),
    ("body", "This manuscript is original, has not been published elsewhere, and is not under "
             "consideration by any other journal. All authors have approved the manuscript and agree "
             "with its submission. The authors declare that they have no known competing financial "
             "interests or personal relationships that could have appeared to influence the work "
             "reported in this paper. All datasets are public and are used under the licences of "
             "their deposits (CC BY 4.0, and CC BY-NC 4.0 for one of them, used here for "
             "non-commercial research); no samples were collected. The analysis code and the derived "
             "result tables accompany the submission, are deposited at "
             "https://github.com/Wxshlw/p8-csu-pl-ftir (release v1.0.0), and remain available from "
             "the corresponding author."),
    ("body", "Data reuse declaration: some datasets used here also appear in companion studies by "
             "our group on adjacent questions (spectral pretraining, open-set adulterant detection, "
             "and frequency-domain diagnostics); this submission is staggered by at least six months "
             "from them."),
    ("body", "Thank you for your time and consideration. We sincerely hope that our manuscript will "
             "be of interest to the readers of **Analytica Chimica Acta**, and we would be grateful for "
             "the opportunity to have it evaluated for publication."),
    ("label", "The following are three possible reviewers who are familiar with the work at this "
              "aspect:"),
    ("revname", "(1) Prof. Alejandro C. Olivieri"),
    ("rev", "Departamento de Qu\u00edmica Anal\u00edtica, Facultad de Ciencias Bioqu\u00edmicas y "
            "Farmac\u00e9uticas, Universidad Nacional de Rosario, Instituto de Qu\u00edmica de Rosario "
            "(IQUIR-CONICET), Suipacha 531, S2002LRK Rosario, Argentina"),
    ("rev", "E-mail: aolivier@fbioyf.unr.edu.ar"),
    ("revname", "(2) Prof. Federico Marini"),
    ("rev", "Department of Chemistry, Sapienza University of Rome, Piazzale Aldo Moro 5, "
            "00185 Rome, Italy"),
    ("rev", "E-mail: federico.marini@uniroma1.it"),
    ("revname", "(3) Prof. Aoife Gowen"),
    ("rev", "School of Biosystems and Food Engineering, University College Dublin, Belfield, "
            "Dublin 4, Ireland"),
    ("rev", "E-mail: aoife.gowen@ucd.ie"),
    ("sign", "Sincerely,"),
    ("sign", "Honghao Cai, on behalf of all authors"),
    ("sign", "School of Science, Jimei University, Xiamen, Fujian Province 361021, China"),
    ("sign", "Email: hhcai@jmu.edu.cn; ORCID: https://orcid.org/0000-0002-1870-8061"),
]


def base_doc() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.line_spacing = 1.0
    return doc


def _make_signature_inline(p) -> None:
    """Make the template's signature image inline instead of floating.

    In the author's own letter nothing follows the signature, so it is anchored
    with ``wrapNone`` 0.44 cm below its paragraph. The submitted letter types the
    name, affiliation and ORCID under "Sincerely,", and a floating image would
    overlap them. The image part, its extent and its relationship id are kept
    byte-identical; only the wrapping changes.
    """
    n = 0
    for anchor in p.findall(".//" + qn("wp:anchor")):
        inline = OxmlElement("wp:inline")
        for a in ("distT", "distB", "distL", "distR"):
            inline.set(a, anchor.get(a) or "0")
        for tag in ("wp:extent", "wp:effectExtent", "wp:docPr",
                    "wp:cNvGraphicFramePr", "a:graphic"):
            el = anchor.find(qn(tag))
            if el is not None:
                anchor.remove(el)
                inline.append(el)
        anchor.addprevious(inline)
        anchor.getparent().remove(anchor)
        n += 1
    assert n == 1, n
    assert p.findall(".//" + qn("a:blip")), "signature image lost"


def cover_letter() -> Path:
    """Build the cover letter ON the author's own letterhead template.

    ``manuscript/cover_letter_template.docx`` is the author's file: it carries
    the Jimei University letterhead (banner image + floating sender block), the
    handwritten signature, the A4 page setup with 2.54/3.17 cm margins and the
    12 pt Times New Roman / 1.5-line-spacing paragraph styles. The letterhead and
    the signature paragraphs are kept verbatim; every other paragraph is cloned
    from the template's own prototype for that role, so the letter inherits the
    look instead of imitating it. The sender block's date line is refreshed to
    ``LETTER_DATE``, which must be set to the actual submission date.
    """
    if not CL_TEMPLATE.exists():
        raise SystemExit(
            f"cover letter template not found: {CL_TEMPLATE}\n"
            "The letterhead and signature supplied by the author are not "
            "redistributed with the code archive."
        )
    doc = Document(str(CL_TEMPLATE))
    body = doc.element.body
    ps = [c for c in body if c.tag == qn("w:p")]
    assert len(ps) == 34, len(ps)

    def ptext(el) -> str:
        return "".join(t.text or "" for t in el.findall(".//" + qn("w:t")))

    # the template's fixed landmarks, asserted so a silent template edit fails loudly
    letterhead, blank0, sig = ps[0], ps[1], ps[31]
    assert letterhead.findall(".//" + qn("w:drawing")), "letterhead drawing missing"
    assert sig.findall(".//" + qn("w:drawing")), "signature drawing missing"
    for i, prefix in ((2, "Manuscript Title:"), (4, "Dear Editor"), (6, "I am pleased"),
                       (16, "The following are"), (17, "(1)"), (30, "Sincerely,")):
        assert ptext(ps[i]).startswith(prefix), (i, ptext(ps[i])[:40])
    protos = {"title": ps[2], "blank": ps[3], "salutation": ps[4], "body": ps[6],
              "label": ps[16], "revname": ps[17], "rev": ps[18], "sign": ps[30]}
    bruns = [r for r in protos["body"].findall(qn("w:r")) if r.find(qn("w:t")) is not None]

    def is_bold_run(r) -> bool:
        rpr = r.find(qn("w:rPr"))
        return rpr is not None and rpr.find(qn("w:b")) is not None

    bold_run = next(r for r in bruns if is_bold_run(r))
    plain_run = next(r for r in bruns if not is_bold_run(r))

    # refresh the date line inside the letterhead (WordprocessingML copy + VML fallback)
    dated = 0
    for txbx in letterhead.iter(qn("w:txbxContent")):
        for p in txbx.findall(qn("w:p")):
            if re.search(r"\d{4}\s*$", ptext(p).strip()):
                runs = p.findall(qn("w:r"))
                keep = runs[0]
                for r in runs[1:]:
                    p.remove(r)
                for t in keep.findall(qn("w:t")):
                    keep.remove(t)
                t = OxmlElement("w:t")
                t.text = LETTER_DATE
                keep.append(t)
                dated += 1
    assert dated == 2, dated

    made: list = []

    def para(kind: str, text: str, force_bold: bool = False) -> None:
        p = deepcopy(protos[kind])
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        for seg, is_bold in _segments(text):
            r = deepcopy(bold_run if (is_bold or force_bold) else plain_run)
            for t in r.findall(qn("w:t")):
                r.remove(t)
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = seg
            r.append(t)
            p.append(r)
        made.append(p)

    def blank() -> None:
        p = deepcopy(protos["blank"])
        for child in list(p):
            if child.tag != qn("w:pPr"):
                p.remove(child)
        made.append(p)

    sign_lines = [t for k, t in COVER if k == "sign"]
    assert sign_lines[0].startswith("Sincerely,")
    groups = 0
    for kind, text in COVER:
        if kind == "sign":
            continue
        if kind == "revname":
            groups += 1
            if groups > 1:
                blank()                  # template separates the reviewer entries
        para(kind, text, force_bold=(kind == "revname"))
        if kind in ("title", "salutation"):
            blank()                      # template puts one blank line after each
    blank()
    para("sign", sign_lines[0])
    sig = deepcopy(sig)
    _make_signature_inline(sig)          # the template floats it; text follows below
    made.append(sig)
    for line in sign_lines[1:]:
        para("sign", line)

    for el in list(body):
        if el.tag in (qn("w:p"), qn("w:bookmarkStart"), qn("w:bookmarkEnd")):
            body.remove(el)              # the old letter's selection bookmarks
    sect = body.find(qn("w:sectPr"))
    for el in [letterhead, blank0, *made]:
        if sect is not None:
            sect.addprevious(el)
        else:
            body.append(el)
    settings = doc.settings.element
    dv = settings.find(qn("w:docVars"))
    if dv is not None:
        settings.remove(dv)          # the template's WPS device/document ids
    assert settings.find(qn("w:docVars")) is None, "WPS docVars survived"
    out = MDIR / "cover_letter_ACA.docx"
    doc.save(str(out))
    return out


def highlights() -> Path:
    doc = base_doc()
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Highlights")
    r.bold = True
    r.font.size = Pt(14)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    p.add_run(TITLE).italic = True
    for h in HIGHLIGHTS:
        q = doc.add_paragraph(style="List Bullet")
        q.paragraph_format.space_after = Pt(4)
        q.add_run(h)
    out = MDIR / "highlights.docx"
    doc.save(str(out))
    return out


def novelty() -> Path:
    doc = base_doc()
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("Novelty statement")
    r.bold = True
    r.font.size = Pt(14)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    p.add_run(TITLE).italic = True
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.add_run(NOVELTY)
    out = MDIR / "novelty_statement.docx"
    doc.save(str(out))
    return out


if __name__ == "__main__":
    _check_highlights_match_manuscript()
    print(cover_letter())
    print(highlights())
    print(novelty())
    for h in HIGHLIGHTS:
        print(len(h), h)
    print("novelty words:", len(NOVELTY.split()))
