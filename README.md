# Out-of-range ATR-FTIR adulterant quantification — code and derived results

Code, numbered result tables, intermediate results and figures for

**Beyond the calibration range: identifiability and extrapolation of concentration-supervised Beer–Lambert unmixing for ATR-FTIR adulterant quantification**
Yang Yang, Yuxuan Jiang, Yalu Cai, Hanping Ke, Honghao Cai

School of Science, Jimei University, Xiamen, Fujian 361021, China — Yang Yang, Yuxuan Jiang, Yalu Cai, Honghao Cai

College of Information Engineering, Ningde Normal University, Ningde, Fujian 352100, China — Hanping Ke

Repository: <https://github.com/Wxshlw/p8-csu-pl-ftir>

An adulterant mass fraction is estimated from ATR-FTIR spectra with a two-component
Beer–Lambert closure in which the concentration enters linearly. The matrix endmember, a
shared adulterant endmember, a per-spectrum multiplicative scale and an additive baseline
are fitted jointly, and the fraction and its profile-likelihood interval are read from the
profiled residual. From the reference spectra alone the code also computes four quantities
that decide, before any sample is measured, whether a concentration outside the calibration
range can be reported: the endmember contrast-to-noise ratio (CNR), the curvature γ of the
level-mean response, the centred cosine between the recovered matrix endmember and the
measured matrix spectrum, and the calibrated profile interval width. The repository covers
the estimator, eight comparators, the audit, the calibration-design leverage calculation,
the interval and coverage analyses, and every experiment and figure of the manuscript.

Release `v1.0.0` is the deposit the manuscript's Data availability section points to. The code,
tables and results are exactly those the manuscript was built from; the repository adds the
scaffolding around them — this README, `docs/`, `figures/`, the licence and citation files,
`.gitignore`, and a `requirements.txt` pinned to the versions used.

## Contents

| Path | Contents |
|---|---|
| `unmix_ftir/` | the analysis package: dataset loader, forward model, unmixing, baseline families, diagnostics, calibration design, statistics, plotting style |
| `scripts/` | one script per experiment (`exp00` onward), plus the table, figure and document builders |
| `tables/` | the numbered tables of the manuscript as CSV (the `.md` companions are the rendered versions) |
| `out/` | every intermediate result the manuscript reports or cites |
| `figures/` | the manuscript figures as shipped: PNG at 600 dpi and PDF vector; see [`figures/README.md`](figures/README.md) |
| `docs/DATASETS.md` | the seven development datasets and the two external ones: sources, licences, file names, expected layout, dataset-code crosswalk |
| `docs/REPRODUCING.md` | running order, and the script and `out/` file behind each manuscript table and figure |

## Requirements

The analyses were run in Python 3.13.14 with NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1,
scikit-learn 1.9.1, matplotlib 3.11.2 and openpyxl 3.1.5. `requirements.txt` pins those
versions. `lxml` and `python-docx` are needed only by the document assembly scripts.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

No spectra are redistributed here. All nine datasets are public and are downloaded from
their deposits; `docs/DATASETS.md` lists the links, the file names and the directory layout
the loader expects. Point the loader at that directory with `P8_SHARED_DATA`:

```bash
export P8_SHARED_DATA="$PWD/_shared_data"        # Windows: $env:P8_SHARED_DATA = "$PWD\_shared_data"
```

The external-validation files are read from `P8_EXT_DATA` (default `ext_data/`).

## Quick start

```bash
python scripts/exp05_audit.py            # CNR and γ of all seven datasets
python scripts/exp03_extrapolation.py    # the 21 out-of-range and interpolation conditions
python scripts/exp25_revision.py         # per-condition tests, composite endpoint, source hierarchy
```

Each script is self-contained, writes into `out/` and prints a short summary; the loader
reads the spectrum and level counts from the deposited workbooks rather than assuming them.
Script docstrings and inline comments are in Chinese; the README and `docs/` are in English.

## Reproducing every number and figure

`docs/REPRODUCING.md` gives the running order of all experiment scripts and names the file
in `out/` behind each table and figure. The figure scripts are named after the file they
write, which is not always the manuscript number: `scripts/make_fig2.py` produces
`figures/fig2_audit.png`, which is Fig. 3 of the manuscript. The mapping is in
`figures/README.md`.

Document assembly (`build_docx.py`, `build_si_docx.py`) additionally needs Microsoft Word
for the PDF export step and an Elsevier Word template; it is not needed to reproduce any
number.

## Licence

The code is released under the MIT licence (`LICENSE`). The derived tables in `tables/`
and the intermediate results in `out/` are released under CC BY 4.0, except the files
derived from the E2 deposit (`out/exp24*`), which remain CC BY-NC 4.0 because that dataset
is licensed for non-commercial use. The figures in `figures/` reproduce the manuscript and
are included so that the reported panels can be checked against the code; they are not
offered under a separate reuse licence. The datasets keep the licences of their deposits;
see `docs/DATASETS.md`.

## Citation

If you use this code, please cite the article and this repository; `CITATION.cff` is
provided for reference managers.

## Contact

Honghao Cai — hhcai@jmu.edu.cn — ORCID [0000-0002-1870-8061](https://orcid.org/0000-0002-1870-8061)
