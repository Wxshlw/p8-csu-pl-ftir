# Datasets

No spectra are redistributed in this repository. All datasets below are public and are
downloaded from their deposits. `unmix_ftir/data.py` reads them from the directory given by
the environment variable `P8_SHARED_DATA`; when it is unset the loader looks for a
`_shared_data` directory at the repository root.

## Expected layout

```
_shared_data/
  kan/
    kan_oliveoil.xlsx    (or zeytinyağı tağşiş.xlsx),  sheet Sayfa7, header row 2
    kan_coffee.xlsx      (or Kahve_Tağşiş.xlsx),       sheet Sayfa4, header row 1
    kan_juice.xlsx       (or meyve suyu tağşiş.xlsx),  sheet Sayfa2, header row 1
  cinnamon/
    FTIR_data_Walnut.xlsx
    FTIR_data_peanut.xlsx
    FTIR_data_wood_dust.xlsx
  turmeric/
    FTIR_Data.csv
```

The KAN workbooks are read from the sheet names listed above; the loader falls back to the
original Turkish file names when the renamed copies are absent.

The external-validation datasets are read from `P8_EXT_DATA` (default `ext_data/` at the
repository root), which holds `FTIR_DATA_Pure_RcP_Sudan(IV)_AdulteratedRcP.xlsx` for E1 and
`redu_insect1.xlsx` for E2, the latter renamed from the deposited "Dataset 1" file.

## Sources and licences

| System | Dataset | Source | Licence |
|---|---|---|---|
| D1 | olive oil / sunflower oil | [KAN-FTIR food adulteration repository](https://github.com/abdulhamidbatayhi123/kan-ftir-food-adulteration) | CC BY 4.0 |
| D2 | roasted coffee / malt flour | same repository | CC BY 4.0 |
| D3 | orange juice / apple juice | same repository | CC BY 4.0 |
| D4 | cinnamon / wood dust | [10.17632/rtbsf2r53x.1](https://doi.org/10.17632/rtbsf2r53x.1) | CC BY 4.0 |
| D5 | cinnamon / peanut shell | same deposit | CC BY 4.0 |
| D6 | cinnamon / walnut shell | same deposit | CC BY 4.0 |
| D7 | turmeric / wheat flour | [10.17632/cb5kbt7bgv.2](https://doi.org/10.17632/cb5kbt7bgv.2) | CC BY 4.0 |
| E1 | red chilli powder / Sudan IV | [10.17632/gwgn2th6mg.1](https://doi.org/10.17632/gwgn2th6mg.1) | CC BY 4.0 |
| E2 | edible insect powder / corn, wheat, soybean flour | [10.25824/redu/zjn5fi](https://doi.org/10.25824/redu/zjn5fi) | CC BY-NC 4.0 |

E2 is licensed for non-commercial research use. Only aggregate error statistics derived
from it are reported in the manuscript, and the files derived from it here (`out/exp24*`)
carry the same restriction.

D1–D3 were acquired on a Bruker Tensor 27 spectrometer with an ATR accessory at a nominal
resolution of 4 cm⁻¹ over 4000–600 cm⁻¹, with 16 scans per spectrum. Only the spectral
workbooks of the KAN-FTIR repository are used; its analysis code and manuscript-preparation
files are not part of this study. The CC BY 4.0 terms relied on are those stated on the
repository page.

## Dataset codes

The manuscript numbers the seven systems D1–D7 consecutively, whereas the code, the `out/`
files and the `tables/` files keep the keys `D7w`, `D7p`, `D7d` and `D8`:

| Key in code | Manuscript |
|---|---|
| `D1` | D1 (olive oil / sunflower oil) |
| `D2` | D2 (coffee / malt flour) |
| `D3` | D3 (orange juice / apple juice) |
| `D7d` | D4 (cinnamon / wood dust) |
| `D7p` | D5 (cinnamon / peanut shell) |
| `D7w` | D6 (cinnamon / walnut shell) |
| `D8` | D7 (turmeric / wheat flour) |

`make_si_md.py` applies this mapping when it writes the supplement, and the figure scripts
`make_fig1_scheme.py` and `make_fig2.py` carry it in their label maps.

## Stored photometric quantity

The cinnamon and turmeric files hold transmittance-like values (median near 0.96, with
2.7–10.8 % of the entries above 1), and `unmix_ftir/data.py` converts them to absorbance
(`A = -log10(T)`) before any analysis. The three KAN workbooks are already stored as
absorbance and are used as they are. E1 is released as transmittance in per cent and was
converted with `A = -log10(T/100)`; E2 is released as absorbance and was used without
conversion. Table S17 of the supplement lists the distribution of the stored values, the
quantity assigned to them, and the audit quantities recomputed in both domains.
