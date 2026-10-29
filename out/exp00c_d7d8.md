# D7 / D8 homology check (exp00c)

A. wavenumber axis

| dataset | channels | lo | hi | dnu | dnu_spread | matched_within_0p01_channel | unmatched | common_range |
|---|---|---|---|---|---|---|---|---|
| D7 | 2523 | 399.47517 | 3997.60513 | 1.4266999999999825 | 1.0000000656873453e-05 |  |  | 399.48-3996.16 |
| D8 | 2522 | 399.47297 | 3996.15644 | 1.4266900000000078 | 1.0000000656873453e-05 |  |  | 399.48-3996.16 |
| concordance, first 2522 index-aligned channels | 2522 | 399.47517 | 3996.15644 | 1.4266999999999825 | 9.999999974752427e-06 | 1538 | 1 | max |offset| 0.0220 cm^-1 = 1.54% of a channel, mean 0.0121 |

B. identifiers and design

| dataset | n_ids | matched | pattern | levels | n_levels | samples_per_level | sample_no_min | sample_no_max | duplicate_ids | namespace |
|---|---|---|---|---|---|---|---|---|---|---|
| D7:D7w | 650 | 650 | ^C([PWD])(\d+)S(\d+)$ | 0,1,2,3,4,6,8,10,12,15,18,21,24 | 13 | 50,50,50,50,50,50,50,50,50,50,50,50,50 | 1 | 50 | 0 | C |
| D7:D7p | 650 | 650 | ^C([PWD])(\d+)S(\d+)$ | 0,1,2,3,4,6,8,10,12,15,18,21,24 | 13 | 50,50,50,50,50,50,50,50,50,50,50,50,50 | 1 | 50 | 0 | C |
| D7:D7d | 650 | 650 | ^C([PWD])(\d+)S(\d+)$ | 0,1,2,3,4,6,8,10,12,15,18,21,24 | 13 | 50,50,50,50,50,50,50,50,50,50,50,50,50 | 1 | 50 | 0 | C |
| D8:D8 | 350 | 350 | ^T(\d+)S(\d+)$ | 0,1,2,3,5,7,9 | 7 | 50,50,50,50,50,50,50 | 1 | 50 | 0 | T |
| D7∩D8 identifier intersection | 2300 | 0 | - | - | nan | - | nan | nan | nan | 0 |

C. spectrum-level similarity

| test | value | detail |
|---|---|---|
| channels used (index-aligned prefix) | 2522 | first 2522 stored channels of each release; axes agree to well under one channel width (panel A) |
| D8 -> nearest D7 spectrum, cosine (raw absorbance) | 0.962211446965652 | p5 0.9427, p95 0.9731, max 0.9735; n>0.999 0, n>0.99 0, n>0.95 257 of 350 |
| D8 -> nearest D7 spectrum, cosine (mean-centred) | 0.9253440564117104 | p5 0.8778, p95 0.9471, max 0.9478; n>0.999 0, n>0.99 0, n>0.95 0 of 350 |
| D8 self nearest neighbour cosine (median) | 0.9871409083496618 | p95 0.9988, max 0.9989 |
| D7 self nearest neighbour cosine (median, 1950 spectra) | 0.9986984475055878 | p5 0.9625, p95 0.9995, max 0.9995 |
| D7 0% walnut vs 0% peanut cosine | 0.9347092737616473 | max 0.9381, mean 0.9343, exact duplicates 0 of 50 |
| D7 0% walnut vs 0% wood dust cosine | 0.8650997294575513 | max 0.8677, mean 0.8600, exact duplicates 0 of 50 |
| D7 0% peanut vs 0% wood dust cosine | 0.8622194028915466 | max 0.8629, mean 0.8565, exact duplicates 0 of 50 |
| D8 0% vs D7 0% nearest cosine | 0.9599268190474269 | max 0.9621, mean 0.9016 (50 vs 150 spectra) |

F. photometric quantity of the stored files

| dataset | rows_in_file | stored_min | stored_median | stored_max | share_above_1 | transparent_3900_mean | stored_quantity | log10_transform_applicable | domain_analysed | band_1030_low_label | band_1030_high_label | band_depth_slope | labels_as_stored | gamma_as_stored | gamma_after_minus_log10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 352 | -0.0040438 | 0.0025917 | 0.26262 | 0.0 | -0.00038223749343750007 | absorbance | no (values already absorbance-like, some negative) | stored values | 0.0395284105882353 | 0.04036883999999999 | -3.80119711084017e-05 | True | -0.08771083133779457 | nan |
| D2 | 405 | 0.0045416 | 0.032693 | 0.26316 | 0.0 | 0.008855579503703703 | absorbance | no (values already absorbance-like, some negative) | stored values | 0.18671855263157897 | 0.20684166666666662 | -0.0004276548700211174 | True | -0.16403887693483626 | nan |
| D3 | 413 | -0.0025395 | 0.049361 | 0.51146 | 0.0 | -0.0007633278184019371 | absorbance | no (values already absorbance-like, some negative) | stored values | 0.0819927654320988 | 0.07387052 | 7.162366223068285e-05 | False | -0.3682292448747345 | nan |
| D7w | 650 | 0.8113404272077364 | 0.9867941871209369 | 1.009319950454091 | 0.06977285892862588 | 0.998355476460945 | transmittance | yes | -log10(stored) | 0.9089499153658861 | 0.9064669359414061 | -0.0030104278837656427 | True | -4.018935026832343 | -3.8989925105128713 |
| D7p | 650 | 0.7185234613400087 | 0.9571801971048857 | 1.01903691800631 | 0.10789536266349584 | 1.0033624293114303 | transmittance | yes | -log10(stored) | 0.7290176987934717 | 0.741261095634556 | 9.909154576040071e-06 | True | -0.07832460214601993 | -0.07228816923885079 |
| D7d | 650 | 0.7228273430409257 | 0.9692047877735714 | 1.005078338768043 | 0.04345437360895149 | 0.9983964739767848 | transmittance | yes | -log10(stored) | 0.9526244227028381 | 0.9040582111553576 | 0.0006112068140327388 | True | -0.4431344452816523 | -0.5121402163601874 |
| D8 | 350 | 0.37680947 | 0.9562426975 | 1.095204933 | 0.027091877194969978 | 0.9993608139857143 | transmittance | yes | -log10(stored) | 0.65494748896 | 0.68538807394 | -0.0027944598615859476 | True | -2.8827229348196264 | -3.0229448589016537 |

G. audit gate quantities in both photometric domains

| dataset | domain_analysed | gamma_analysed | cnr_analysed | cnr_partial_analysed | leak_analysed | gate_analysed | gamma_other | cnr_other | cnr_partial_other | leak_other | gate_other | gate_flips | max_cnr_rel_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | stored values (already absorbance, panel F) | -0.08771083133779457 | 18.031499009268916 | 12.981959837527738 | 0.8097225010170596 | pass | nan | nan | nan | nan | n/a (no transmittance reading applies) | n/a | nan |
| D2 | stored values (already absorbance, panel F) | -0.16403887693483626 | 34.484861576211514 | 33.10852328632854 | 0.2769341130063496 | pass | nan | nan | nan | nan | n/a (no transmittance reading applies) | n/a | nan |
| D3 | stored values (already absorbance, panel F) | -0.3682292448747345 | 4.030625914709285 | 3.7159400122005044 | 0.1658080605786466 | fail | nan | nan | nan | nan | n/a (no transmittance reading applies) | n/a | nan |
| D7w | -log10(stored) | -3.8989925105128713 | 2.3503274782637558 | 2.0713290329656036 | 0.33169511474553715 | fail | -4.018935026832343 | 2.381457320420713 | 2.252488052644768 | 0.8355515814403621 | fail | False | 0.01324489563469402 |
| D7p | -log10(stored) | -0.07228816923885079 | 1.030246550869665 | 0.8788661020737927 | 0.6333441972749403 | fail | -0.07832460214601993 | 0.8777264199496233 | 0.8731819758597472 | 0.01178916721461587 | fail | False | 0.1480423601431079 |
| D7d | -log10(stored) | -0.5121402163601874 | 21.39530314226607 | 20.36652606949413 | 0.9168080421880844 | pass | -0.4431344452816523 | 15.199138837339877 | 15.19083216822993 | 0.9242303467558237 | pass | False | 0.28960395016258367 |
| D8 | -log10(stored) | -3.0229448589016537 | 25.305855965300374 | 22.919864485479152 | 0.9315147154964959 | fail | -2.8827229348196264 | 17.661601869088624 | 17.444383141569002 | 0.9452505100416386 | fail | False | 0.30207451218775694 |

D. documentation fields

- **instrument**: cinnamon_Readme.pdf: FTIR Spectral Data The FTIR datasets contain absorbance spectra acquired using an ATR-FTIR spectrometer over the spectral range of approximately 400–4; cinnamon_Dataset_Description.pdf: 4.2 FTIR Spectral Dataset The FTIR dataset contains absorbance spectra acquired using an ATR-FTIR spectrometer over the spectral range of approximatel
- **range**: cinnamon_Readme.pdf: FTIR Spectral Data The FTIR datasets contain absorbance spectra acquired using an ATR-FTIR spectrometer over the spectral range of approximately 400–4; cinnamon_Dataset_Description.pdf: 4.2 FTIR Spectral Dataset The FTIR dataset contains absorbance spectra acquired using an ATR-FTIR spectrometer over the spectral range of approximatel
- **variables**: cinnamon_Readme.pdf: Each sample includes: • Sample identifier • 2,523 spectral variables • Adulteration category RGB Image Data The image datasets consist of RGB images a; cinnamon_Dataset_Description.pdf: Each sample contains: • Sample identifier • 2,523 spectral variables • Adulteration category 3 The spectra capture chemical variations introduced by d
- **modalities**: cinnamon_Readme.pdf: 1 README Multimodal Dataset for Cinnamon Powder Adulteration Detection Using Colorimetry, FTIR Spectroscopy, and RGB Images Version: Version 1.0 1.; cinnamon_Dataset_Description.pdf: 1 Dataset Description Multimodal Dataset for Cinnamon Powder Adulteration Detection Using Colorimetry, FTIR Spectroscopy, and RGB Images 1.; turmeric_README.docx: README: Multimodal Turmeric Adulteration Dataset Overview This dataset provides a multimodal representation for detecting and quantifying turmeric adu
- **levels**: cinnamon_Readme.pdf: Data were collected from cinnamon powder adulterated with peanut shell powder, walnut shell powder, and wood dust powder at thirteen predefined adulte; cinnamon_Dataset_Description.pdf: Three economically important adulterants were considered: • Peanut shell powder • Walnut shell powder • Wood dust powder Each adulterant was mixed wit; turmeric_README.docx: Dataset Contents Image_Data/: Raw images organized by adulteration levels EfficientNetB0_Image_Features.csv: 1280-dimensional image features FTIR_Data
- **samples**: cinnamon_Readme.pdf: Dataset Statistics Description Value Number of adulterants 3 Number of modalities 3 Adulteration levels 13 Samples per adulteration level 50 Samples p; cinnamon_Dataset_Description.pdf: Adulterant Colorimetry FTIR Spectra RGB Images Total Samples Peanut Shell Powder 650 650 650 1,950 Walnut Shell Powder 650 650 650 1,950 Wood Dust Pow; turmeric_README.docx: Dataset Contents Image_Data/: Raw images organized by adulteration levels EfficientNetB0_Image_Features.csv: 1280-dimensional image features FTIR_Data
- **naming**: cinnamon_Readme.pdf: Data Modalities Colorimetric Data The colorimetric datasets contain six color attributes measured using a HunterLab UltraScan VIS spectro-colorimeter:; cinnamon_Dataset_Description.pdf: Each sample contains: • Sample identifier • 2,523 spectral variables • Adulteration category 3 The spectra capture chemical variations introduced by d; turmeric_README.docx: Dataset Contents Image_Data/: Raw images organized by adulteration levels EfficientNetB0_Image_Features.csv: 1280-dimensional image features FTIR_Data
- **qc**: cinnamon_Dataset_Description.pdf: Data Quality Assurance Prior to publication, the complete dataset underwent systematic validation to ensure data integrity and consistency.
- **serial**: no sentence in either README matches this pattern
- **date**: cinnamon_Readme.pdf: Data were collected from cinnamon powder adulterated with peanut shell powder, walnut shell powder, and wood dust powder at thirteen predefined adulte

E. attribution from the reference list

- **D7_cinnamon**: All datasets are public and used under CC BY 4.0. D1–D3: https://github.com/abdulhamidbatayhi123/kan-ftir-food-adulteration. D7: https://doi.org/10.17632/rtbsf2r53x.1. D8: https://doi.org/10.17632/cb5kbt7bgv.2. No new experimental data were generated in this study. The analysis code, the derived result tables (Tables 1–4 and S1–S12), and the split and fold definitions are provided as a supplementary archive with this submission and are available from the corresponding author.
- **D8_turmeric**: All datasets are public and used under CC BY 4.0. D1–D3: https://github.com/abdulhamidbatayhi123/kan-ftir-food-adulteration. D7: https://doi.org/10.17632/rtbsf2r53x.1. D8: https://doi.org/10.17632/cb5kbt7bgv.2. No new experimental data were generated in this study. The analysis code, the derived result tables (Tables 1–4 and S1–S12), and the split and fold definitions are provided as a supplementary archive with this submission and are available from the corresponding author.
- shared first author: True
- authors in common: All datasets are public, used under CC BY 4.0. D1–D3: htt
