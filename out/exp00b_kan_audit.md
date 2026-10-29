# Table S16. KAN trio inventory, read from the distributed workbooks

A. official counts (flat sheets used by the analysis)

| dataset | matrix | file | sheet | header_row | raw_data_rows | rows_dropped | drop_blank_id | drop_no_matrix | drop_no_adulterant | drop_nonfinite_spectrum | spectra_final | channels | physical_samples | levels | level_values | samples_per_level | spectra_per_level | scans_per_sample_min | scans_per_sample_max | scans_per_sample_median | raw_axis | dnu | range_lo | range_hi | tekrar_rows | tekrar_unlabelled | tekrar_labelled | tekrar_groups | base_present |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | oil | kan_oliveoil.xlsx | Sayfa7 | 2 | 352 | 0 | 0 | 0 | 0 | 0 | 352 | 1762 | 55 | 11 | 0,3,6,9,12,15,18,21,24,27,30 | 5,5,5,5,5,5,5,5,5,5,5 | 102,25,25,25,25,25,25,25,25,25,25 | 5 | 22 | 5.0 | descending | 1.9289999999999736 | 599.82 | 3996.231 | 0 | 0 | 0 | 0 | nan |
| D2 | coffee | kan_coffee.xlsx | Sayfa4 | 1 | 410 | 5 | 0 | 5 | 0 | 0 | 405 | 1762 | 55 | 11 | 0,5,10,15,20,25,30,35,40,45,50 | 5,5,5,5,5,5,5,5,5,5,5 | 152,25,25,26,26,25,25,25,25,27,24 | 4 | 31 | 5.0 | descending | 1.9289999999999736 | 599.82 | 3996.231 | 10 | 5 | 5 | 1 | 0.0 |
| D3 | juice | kan_juice.xlsx | Sayfa2 | 1 | 413 | 0 | 0 | 0 | 0 | 0 | 413 | 1762 | 55 | 11 | 0,5,10,15,20,25,30,35,40,45,50 | 5,5,5,5,5,5,5,5,5,5,5 | 162,25,25,26,25,25,25,25,25,25,25 | 5 | 36 | 5.0 | descending | 1.9289999999999736 | 599.82 | 3996.231 | 0 | 0 | 0 | 0 | nan |

B. transposed mirror sheets (source of the retired 354 / ~515 / 424)

| dataset | mirror_sheet | spectrum_columns | distinct_ids | mirror_rows | mirror_cols | extra_in_mirror | extra_ids | extra_examples | absent_from_mirror | absent_ids | absent_examples | flat_spectra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | yeni | 354 | 281 | 1766 | 355 | 2 | 2 | ['28_2', '33_'] | 0 | 0 | [] | 352 |
| D2 | Sayfa1 | 95 | 91 | 1764 | 96 | 5 | 5 | ['1.kahve1.1', '1.kahve1.2', '1.kahve1.3', '1.kahve1.4'] | 315 | 308 | ['2.kahve0.19', '2.kahve0.2', '2.kahve0.20', '2.kahve0.21'] | 405 |
| D2 | Sayfa3 | 420 | 402 | 1764 | 421 | 15 | 9 | ['1.kahve1.1', '1.kahve1.2', '1.kahve1.3', '1.kahve1.4'] | 0 | 0 | [] | 405 |
| D3 | Sayfa1 | 424 | 60 | 1764 | 425 | 11 | 9 | ['p1.1', 'p1.1_2', 'p14', 'p20'] | 0 | 0 | [] | 413 |

C. D3 label discipline

- id pattern `p<brand><level-index>` with label = 5 x level-index; 1 identifier(s) disagree with their index (5 spectra): `p59` labelled 35 % but index 9 -> 45 %
- ids labelled 35 %: p17, p27, p37, p47, p57, p59
- ids labelled 45 %: p19, p29, p39, p49 (brand 5 absent until the correction)
