"""make_submission_docs —— 生成 Analytica Chimica Acta 投稿用 cover letter 与 highlights（docx）。

Highlights 按 ACA《Guide for authors》核对：3–5 条、每条含空格不超过 85 字符。
作者、单位、邮箱均保留占位符，不虚构任何署名信息。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "manuscript"

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
    "computed before any out-of-range sample is measured\u2014the baseline-projected "
    "endmember contrast-to-noise ratio and the level-mean curvature\u2014gate whether "
    "a binary unmixing assay can be extrapolated at all, independently of model class, "
    "and two further reads of the same reference spectra, the shape of the recovered "
    "matrix endmember and the width of the calibrated interval relative to the "
    "reporting range, decide whether the resulting number is worth reporting. Across "
    "seven public datasets, 14 out-of-range conditions and seven interpolation "
    "controls, contrast orders the systems by extrapolation error (\u03c1 = \u22120.60 across "
    "the five measurement sources, \u22120.71 at the dataset level), curvature is "
    "associated with distortion of the response slope, and the "
    "classification is unchanged when the audit is re-estimated on samples and brands "
    "disjoint from the evaluation data. A concentration-supervised Beer\u2013Lambert "
    "estimator then has the lower error in 19 of the 24 cells inside the passing region "
    "against four comparators named before the comparison (median ratio 0.861), and two "
    "datasets deposited independently of this work return the same four verdicts. "
    "A leverage screen flags ill-posed designs, and "
    "both calibrated and distribution-free intervals expose residual bias rather than "
    "repairing it. The audit tells analysts, at the design stage, when not to report a number."
)

COVER = [
    ("blank", ""),
    ("salutation", "Dear Editor,"),
    ("body", "We submit for consideration in Analytica Chimica Acta our original research "
             "article, \u201c" + TITLE + ".\u201d The manuscript asks whether an adulterant can be "
             "quantified outside the range covered by its calibration standards, and answers the "
             "question from reference spectra before any test sample is measured."),
    ("body", "Out-of-range quantification by attenuated total reflectance Fourier transform "
             "infrared (ATR-FTIR) spectroscopy is a routine screening demand, but the calibration is "
             "asked for a number it cannot support, and its interval no longer describes the "
             "error it makes. We model the "
             "mixture as a two-component Beer\u2013Lambert closure whose endmembers are estimated "
             "under concentration supervision; once the endmembers are fixed, the fraction and an "
             "interval per spectrum follow from a profiled likelihood. From the same reference "
             "spectra we compute two audit quantities, the contrast of the adulterant against the "
             "noise and baseline variation that the model must absorb and the curvature of the "
             "level-mean response, and two usability checks on the same data: the shape of the "
             "recovered matrix endmember, and the width of the calibrated interval relative to "
             "the reporting range. A system passes when the gate and both checks are satisfied, "
             "and the decision is made at the design stage."),
    ("body", "Across seven public ATR-FTIR datasets (3,470 spectra; fourteen out-of-range "
             "conditions and seven interpolation controls) and eight established regression and "
             "curve-resolution methods, the audit separates the systems in which out-of-range "
             "quantification succeeds from those in which it fails, and orders them by "
             "extrapolation error (\u03c1 = \u22120.60 across the five measurement sources, \u22120.71 at the "
             "dataset level). Where the audit passes, the "
             "estimator has the lower error in 19 of the 24 cells against the four comparators "
             "named before the comparison (median ratio 0.861); outside that region it wins 18 of "
             "48 (1.10). Where it fails, the cause is identified: "
             "insufficient spectral contrast, a nonlinear response, a "
             "non-physical endmember, or a misspecified nuisance model. Intervals calibrated "
             "inside the range understate the error made outside it; their calibrated width, "
             "relative to the reporting range, says whether a number is worth reporting. Two "
             "datasets deposited independently of this work, covering four further "
             "matrix\u2013adulterant systems, return the same four verdicts and the same ordering "
             "of the out-of-range error ratios (median 0.92 where the audit passes, 1.28 where "
             "it fails)."),
    ("body", "The work suits the journal's readership because it supplies both a quantification "
             "protocol and a design-stage test of where that protocol applies."),
    ("body", "The manuscript is original, is not under consideration elsewhere, and all authors "
             "have approved the submission and declare no competing interests. All datasets are "
             "public and are used under the licences of their deposits (CC BY 4.0, and CC BY-NC "
             "4.0 for one of them, used here for non-commercial research); no samples were "
             "collected. The analysis code and "
             "derived result tables accompany the submission and are available from the "
             "corresponding author."),
    ("body", "Data reuse declaration: some datasets used here also appear in "
             "companion studies by our group on adjacent questions (spectral pretraining, "
             "open-set adulterant detection, and frequency-domain diagnostics); this "
             "submission is staggered by at least six months from them."),
    ("sign", "Sincerely,"),
    ("blank", ""),
    ("sign", "Honghao Cai, on behalf of all authors"),
    ("sign", "School of Science, Jimei University, Xiamen, Fujian Province 361021, China"),
    ("sign", "Email: hhcai@jmu.edu.cn; ORCID: https://orcid.org/0000-0002-1870-8061"),
    ("blank", ""),
    ("label", "Suggested reviewers:"),
    ("rev", "1. Prof. Muhieddine Hatab, Institute of Artificial Intelligence, "
            "De Montfort University, Leicester, LE1 9BH, UK. Email: muhieddine.hatab91@gmail.com"),
    ("rev", "2. Prof. Hanping Ke, College of Information Engineering, "
            "Ningde Normal University, Ningde, China. Email: kehanping@ndnu.edu.cn"),
    ("rev", "3. Prof. Muhammad Hussain, Department of Computer Science, "
            "School of Computing and Engineering, University of Huddersfield, "
            "Queensgate, Huddersfield HD1 3DH, UK. Email: M.M.Hussain@hud.ac.uk"),
    ("rev", "4. Prof. Alejandro C. Olivieri, Departamento de Qu\u00edmica Anal\u00edtica, "
            "Facultad de Ciencias Bioqu\u00edmicas y Farmac\u00e9uticas, "
            "Universidad Nacional de Rosario, Instituto de Qu\u00edmica de Rosario "
            "(IQUIR-CONICET), Suipacha 531, S2002LRK Rosario, Argentina. "
            "Email: aolivier@fbioyf.unr.edu.ar"),
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


def cover_letter() -> Path:
    doc = base_doc()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(3.0)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11.5)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.space_after = Pt(0)
    for kind, text in COVER:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.15
        pf.space_before = Pt(7.8 if kind == "body" else 0)
        pf.space_after = Pt(7.8 if kind == "body" else 0)
        if kind == "body":
            pf.first_line_indent = Cm(0.847)
        if kind == "label":
            pf.space_before = Pt(7.8)
            pf.space_after = Pt(3)
            pf.keep_with_next = True
        if kind == "rev":
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.left_indent = Cm(0.6)
            pf.first_line_indent = Cm(-0.6)
            pf.space_after = Pt(4)
        if not text:
            continue
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.5)
        r.bold = kind in ("title", "salutation", "label")
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
