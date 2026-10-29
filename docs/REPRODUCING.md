# Reproducing the results

Scripts resolve all paths from their own location, so they can be run from any working
directory, and they write into `out/` and `tables/` at the repository root. Set
`P8_SHARED_DATA` (and `P8_EXT_DATA` for the external datasets) as described in
[DATASETS.md](DATASETS.md) before starting.

## Running order

The experiment scripts are self-contained and can be run in numeric order.

| Script | Content |
|---|---|
| `exp00_inventory.py` | dataset inventory and checks |
| `exp00b_kan_audit.py` | workbook inventory of D1–D3: raw-row accounting, transposed-sheet reconciliation, D3 label and D2 repeat checks (Table S16) |
| `exp00c_d7d8_homology.py` | cinnamon/turmeric release check (axis, identifiers, spectrum duplication, stored photometric quantity, audit in both domains; Table S17) |
| `exp01_smoke.py` | forward-model smoke test |
| `exp02_modelselect.py`, `exp02_theory.py` | preprocessing and profile-likelihood checks; CRLB comparison |
| `exp03_extrapolation.py`, `exp03b_mcr.py`, `exp03c_anchor_remedy.py` | 21 out-of-range conditions; MCR-ALS ambiguity; effect of adding a 0 % anchor |
| `exp04_design_sweep.py`, `exp04b_leverage_check.py`, `exp04c_design_other.py` | 640-design leverage sweep |
| `exp05_audit.py`, `exp05b_d7_permutation.py`, `exp05c_split_audit.py`, `exp05c_endmember_full.py` | CNR / curvature audit; D7 label permutation; endmember physicality on all seven datasets |
| `exp06_robustness.py` | nuisance-model robustness envelope |
| `exp07_ambiguity.py` | rotational ambiguity vs supervision |
| `exp08_uncertainty.py` | profile and inflated intervals, coverage |
| `exp08b_spread_flag.py` | predicted-spread flag against the observed error, per condition |
| `exp10_fixscale.py` | fixed-scale (λ_s → ∞) ablation against the free-scale fit (Table 3a) |
| `exp09_d7_external.py` | D7 as a held-out matrix system (Table S17 records the relation between the cinnamon and turmeric releases) |
| `exp11_inference.py` | dataset-level permutation, bootstrap, Holm |
| `exp12_conformal.py` | split-conformal and Gaussian interval comparison |
| `exp13_audit_transfer.py` | audit on a disjoint reference half (brand / sample splits) |
| `exp14_lodo.py` | leave-one-dataset-out threshold learning |
| `exp15_atr_ablation.py` | ATR path-length correction ablation (Table S13) |
| `exp16_scan_vs_sample.py` | scan-level vs sample-level split leakage control (Table S14) |
| `exp17_crlb_mc.py` | CRLB interval validation and leverage decomposition (Table S15); stages `white`, `resid`, `calib`, `report` |
| `exp17b_composite.py` | combines the noise and endmember-refit terms against the observed error (Table S15, note) |
| `exp18_loq.py` | quantification limits derived from the calibrated interval of Section 3.7 (Table S21) |
| `exp19_grid.py` | wave-number grid and Savitzky-Golay-window sensitivity of the audit and of the extrapolation error (Table S18) |
| `exp20_capacity.py` | capacity-matched PLS and ridge comparators inside the gate (Table S19) |
| `exp22_ambiguity_band.py` | rotational feasible band of the fitted two-component pair, with and without the calibration labels (Table S20) |
| `exp23_percomparator.py` | dataset-level audit correlation of every comparator taken separately (Table S22) |
| `exp24_external.py` | audit and extrapolation protocols on two independently deposited datasets (Table S23): red chilli powder with Sudan IV, edible insect powder with three cereal flours |
| `exp25_revision.py` | per-condition tests against the four prespecified comparators with Holm correction, the three-part endpoint and the measurement-source hierarchy (Tables S5 and S6, Fig. 3) |
| `exp26_audit_boot.py` | level-weighting, level-spacing and polynomial-order sensitivity of the curvature statistic (Table S24b); its with-replacement bootstrap of the audit quantities is superseded by `exp26b_audit_boot_strat.py` |
| `exp26b_audit_boot_strat.py` | stratified draws of 80 % of the physical samples within each concentration level, giving the stability band of CNR and γ and the count of draws that reverse the gate (Table S24a) |
| `exp27_sensitivity.py` | calibration-size matched reruns, brand-fallback and baseline-order sensitivity (Tables S26a–S26c) |
| `exp27b_matched_ratio.py` | comparator ratios on the size-matched calibration draws (Table S26d) |
| `exp28_mcr_inductive.py` | transductive and inductive variants of multivariate curve resolution-alternating least squares (Table S25) |
| `exp29_band_contrast.py` | band positions and signed amplitudes of the level-mean absorbance difference between the highest prepared level and 0 %, with a linear baseline through two band-free windows (Section 4.1) |
| `analyze03.py`, `analyze04.py`, `analyze04c.py` | summarise the extrapolation and design runs into tables |
| `make_fig1.py`, `make_fig1_scheme.py`, `make_fig2.py`–`make_fig5.py`, `make_graphical_abstract.py` | manuscript figures |
| `make_tables.py`, `make_si_md.py`, `build_si_docx.py`, `build_docx.py` | numbered tables, supplement and manuscript assembly |

`unmix_ftir/unmix.py` alternates endmember and nuisance updates; the nuisance solve is
batched over concentration levels (`batched=True`, the default) because it is linear and
separable per channel. Setting `batched=False` reproduces the level-by-level reference
solve bit-for-bit on the reported metrics, so the two routes are interchangeable.

## Figures

Each figure script is named after the file it writes, and the file name is not always the
manuscript number.

| Script | Writes | Manuscript figure |
|---|---|---|
| `make_fig1_scheme.py` | `figures/fig1_scheme.*` | Fig. 1 |
| `make_fig1.py` | `figures/fig1_forward.*` | Fig. 2 |
| `make_fig2.py` | `figures/fig2_audit.*` | Fig. 3 |
| `make_fig3.py` | `figures/fig3_extrapolation.*` | Fig. 4 |
| `make_fig4.py` | `figures/fig4_design.*`, `figures/figS1_kappa.*` | Fig. 5, Fig. S1 |
| `make_fig5.py` | `figures/fig5_robustness.*` | Fig. 6 |
| `make_graphical_abstract.py` | `figures/graphical_abstract.*` | graphical abstract |

## Result files

`out/exp05_audit.json` holds the audit values used throughout. `out/exp13_summary.json`
and `out/exp14_lodo.json` hold the disjoint-reference and LODO results; `out/exp15_*.csv`,
`out/exp16_*.csv` and `out/exp17_*.csv` hold the ablation, leakage-control and
interval-validation results behind Tables S13–S15; `out/exp00b_*.csv` holds the workbook
inventory behind Table S16; `out/exp00c_*.csv` and `out/exp00c_d7d8.md` hold the release
comparison and the two-domain audit behind Table S17. `out/exp19_*.csv` and
`out/exp20_*.csv` hold the grid-sensitivity and capacity-matching results behind Tables S18
and S19; `out/exp22_theta.csv`, `out/exp22_band.csv` and `out/exp22_meta.json` hold the
rotation scan behind Table S20; `out/exp18_loq.csv` and `out/exp18_loq_curve.csv` hold the
interval-width search behind Table S21; `out/exp23_percomparator.csv` holds the
per-comparator coefficients behind Table S22; `out/exp24_ext_predictions.csv`,
`out/exp24_ext_summary.csv` and `out/exp24_ext_audit.json` hold the external-validation
predictions, the per-condition comparison against the deterministic comparators and the
audit quantities behind Table S23, and `out/exp24b_ext_robust.json` holds its preprocessing
and reference-half sensitivity. `out/exp25_prespecified.csv`, `out/exp25_endpoint.csv`,
`out/exp25_condition_counts.csv` and `out/exp25_hierarchy.json` hold the per-comparator
ratios with their Holm-adjusted p-values, the endpoint verdict of each dataset, the
per-condition cell counts and the source-level correlations behind Tables S5 and S6;
`out/exp26b_audit_boot_strat.csv` and `out/exp26b_gate_flips.csv` hold the subsample
distributions and the gate-reversal counts behind Table S24a, and
`out/exp26_gamma_sens.csv` the curvature variants behind Table S24b, while
`out/exp26_audit_boot.csv` is the superseded with-replacement bootstrap kept for
traceability; `out/exp28_mcr_inductive.csv` holds the two MCR-ALS variants behind Table
S25; `out/exp27_matched_size.csv`, `out/exp27_brand_fallback.csv`,
`out/exp27_baseline_order.csv` and `out/exp27b_matched_ratio.csv` hold the sensitivity
runs behind Table S26; `out/exp29_band_contrast.csv` holds, for each dataset, the band
positions and signed amplitudes of the difference between the highest prepared level and
0 % that Section 4.1 reads. Tables in `tables/` are the numbered tables of the manuscript;
`out/exp*.md` files are the human-readable summaries of individual experiments.

## Document assembly

`build_docx.py` and `build_si_docx.py` reproduce the manuscript and the supplement from
the Markdown sources. They need Microsoft Word for the PDF export step and an Elsevier Word
template (`Elsevier-template.docx` in the working directory, and `MML2OMML.XSL` from the
Office installation). The manuscript and supplement sources are not part of this
repository; document assembly is not needed to reproduce any number.
