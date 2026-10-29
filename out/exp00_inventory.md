# Table S1. Dataset inventory (read from the distributed files, 2026-10-23)

| ID | Matrix | Adulterant | Spectra | Channels | Physical samples | Scans/sample | Levels | Level values (%) | delta-nu (cm-1) | Range (cm-1) |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | olive oil | sunflower oil | 352 | 1762 | 55 | 5–22 | 11 | 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 | 1.929 | 600–3996 |
| D2 | roasted coffee | malt flour | 405 | 1762 | 55 | 4–31 | 11 | 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 | 1.929 | 600–3996 |
| D3 | orange juice | apple juice | 413 | 1762 | 55 | 5–36 | 11 | 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 | 1.929 | 600–3996 |
| D7w | cinnamon | walnut | 650 | 2523 | 650 | 1–1 | 13 | 0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24 | 1.4267 | 399–3998 |
| D7p | cinnamon | peanut | 650 | 2523 | 650 | 1–1 | 13 | 0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24 | 1.4267 | 399–3998 |
| D7d | cinnamon | wood dust | 650 | 2523 | 650 | 1–1 | 13 | 0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24 | 1.4267 | 399–3998 |
| D8 | turmeric | wheat flour | 350 | 2522 | 350 | 1–1 | 7 | 0, 1, 2, 3, 5, 7, 9 | 1.4267 | 399–3996 |

## D7 / D8 homology audit

- same wavenumber grid: **True** (overlap 2522 points, max abs diff 0.02199 cm-1, max rel diff 5.522e-06)
- naming isomorphic: **True**
- exact duplicate pure-matrix spectra across D7/D8: **0**
- sample-name examples D7: ['CW0S01', 'CW0S02', 'CW0S03', 'CW0S04', 'CW0S05']
- sample-name examples D8: ['T0S01', 'T0S02', 'T0S03', 'T0S04', 'T0S05']
- verdict: same acquisition grid and naming convention -> shared protocol; no exact duplicate spectra -> distinct physical samples; treat as non-independent external validation
