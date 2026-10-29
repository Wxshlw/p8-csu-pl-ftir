# Figures

The seven figures of the manuscript as shipped, each as PNG (600 dpi, the size embedded in
the manuscript) and PDF (vector). They are drawn by the scripts in `scripts/`; each script
is named after the file it writes, which is not always the manuscript number.

| File | Manuscript | Content | Script |
|---|---|---|---|
| `fig1_scheme` | Fig. 1 | study design: data and sample-level splits, the a priori audit, profile-likelihood estimation, the decision to report or withhold, and the evaluation | `make_fig1_scheme.py` |
| `fig1_forward` | Fig. 2 | matrix and adulterant endmembers with a 25 % mixture, predicted against nominal concentration for the three closure variants, and the profile-likelihood objective as a function of the partial contrast | `make_fig1.py` |
| `fig2_audit` | Fig. 3 | regime map of CNR against the calibration curvature, error ratios to the four prespecified comparators, calibration slopes, and label resolvability for D4–D6 | `make_fig2.py` |
| `fig3_extrapolation` | Fig. 4 | predicted against reference concentration for four representative conditions, with the comparator of each panel | `make_fig3.py` |
| `fig4_design` | Fig. 5 | calibration-design sweep on the D2 coffee–malt system: leverage against error, the semi-synthetic designs with a log–log fit, and error against the number of calibration levels | `make_fig4.py` |
| `figS1_kappa` | Fig. S1 | condition number of the design matrix across the same sweep | `make_fig4.py` |
| `fig5_robustness` | Fig. 6 | operating envelope: one perturbation at a time, free- against fixed-scale error, and raw against inflated coverage | `make_fig5.py` |

The graphical abstract (`graphical_abstract.*` in the submitted package) is drawn by
`make_graphical_abstract.py` and is not included here.
