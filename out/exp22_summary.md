# exp22 旋转歧义可行带（P1-4）

行 = 非负松弛 delta，列 = 标签容差 tau（inf = 完全不用浓度标签）；值 = 该条件下所有测试水平中的最坏者；tau 锚定在 theta=0（CSU 报告拟合）的校准 RMSE 上。

## D7d/high  [exact]  CNR=21.4  gamma=-0.51
levels [12.0, 15.0, 18.0, 21.0, 24.0]  n_cal 350  n_test 250  cal RMSE(theta=0) = 5.25 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 1  theta* = 10.50 deg (cal RMSE 4.31)

### band（pp）
tau    0.0    0.5    1.0    2.0    5.0    inf
delta                                        
0.00   0.0   0.00   0.00   0.00   0.00   0.00
0.02   9.5  10.22  10.93  10.93  10.93  10.93
0.10   9.5  10.22  10.93  12.32  16.27  16.90
0.30   9.5  10.22  10.93  12.32  16.27  29.79

### band 内可达最大偏置（pp）
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   14.5  14.5  14.5  14.5  14.5  14.5
0.02   24.0  24.0  24.0  24.0  24.0  24.0
0.10   24.0  24.0  24.0  24.0  24.0  24.0
0.30   24.0  24.0  24.0  24.0  24.0  24.0

### CSU（theta=0）自身的偏置（pp，最坏水平）
14.50

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     1    1    1    1    1    1
0.02    25   26   27   27   27   27
0.10    63   64   65   67   73   74
0.30    63   64   65   67   73   97

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

## D8/high  [exact]  CNR=25.3  gamma=-3.02
levels [5.0, 7.0, 9.0]  n_cal 200  n_test 150  cal RMSE(theta=0) = 1.73 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 1  theta* = 2.00 deg (cal RMSE 1.19)

### band（pp）
tau     0.0   0.5   1.0   2.0   5.0    inf
delta                                     
0.00   0.00  0.00  0.00  0.00  0.00   0.00
0.02   2.93  3.75  3.75  5.34  6.12   6.12
0.10   2.93  3.75  3.75  5.34  8.39  13.31
0.30   2.93  3.75  3.75  5.34  8.39  26.93

### band 内可达最大偏置（pp）
tau     0.0   0.5   1.0   2.0   5.0    inf
delta                                     
0.00   7.18  7.18  7.18  7.18  7.18   7.18
0.02   9.00  9.00  9.00  9.00  9.00   9.00
0.10   9.00  9.00  9.00  9.00  9.00   9.00
0.30   9.00  9.00  9.00  9.00  9.00  21.93

### CSU（theta=0）自身的偏置（pp，最坏水平）
7.18

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     1    1    1    1    1    1
0.02     8  179  179  181  182  182
0.10     8  181  181  183  187  194
0.30     8  181  181  183  187  217

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

## D7w/high  [exact]  CNR=2.4  gamma=-3.90
levels [12.0, 15.0, 18.0, 21.0, 24.0]  n_cal 350  n_test 250  cal RMSE(theta=0) = 1.95 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 4  theta* = -0.50 deg (cal RMSE 1.84)

### band（pp）
tau     0.0   0.5   1.0   2.0    5.0    inf
delta                                      
0.00   0.83  2.45  2.45  2.45   2.45   2.45
0.02   0.83  3.30  4.95  6.91   9.14   9.14
0.10   0.83  3.30  4.95  6.91  10.58  15.95
0.30   0.83  3.30  4.95  6.91  10.58  29.03

### band 内可达最大偏置（pp）
tau      0.0    0.5    1.0    2.0    5.0    inf
delta                                          
0.00   21.84  21.84  21.84  21.84  21.84  21.84
0.02   21.84  22.69  23.54  24.00  24.00  24.00
0.10   21.84  22.69  23.54  24.00  24.00  24.00
0.30   21.84  22.69  23.54  24.00  24.00  24.00

### CSU（theta=0）自身的偏置（pp，最坏水平）
21.84

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     2    4    4    4    4    4
0.02     2    5    7   10  130  130
0.10     2    5    7   10  190  198
0.30     2    5    7   10  190  221

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

## D7p/high  [exact]  CNR=1.0  gamma=-0.07
levels [12.0, 15.0, 18.0, 21.0, 24.0]  n_cal 350  n_test 250  cal RMSE(theta=0) = 1.83 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 1  theta* = 0.00 deg (cal RMSE 1.83)

### band（pp）
tau    0.0  0.5   1.0   2.0   5.0    inf
delta                                   
0.00   0.0  0.0  0.00  0.00  0.00   0.00
0.02   0.0  2.5  5.01  6.71  9.67  15.14
0.10   0.0  2.5  5.01  6.71  9.67  43.96
0.30   0.0  2.5  5.01  6.71  9.67  57.13

### band 内可达最大偏置（pp）
tau      0.0    0.5    1.0    2.0    5.0    inf
delta                                          
0.00   21.17  21.17  21.17  21.17  21.17  21.17
0.02   21.17  22.85  24.00  24.00  24.00  24.00
0.10   21.17  22.85  24.00  24.00  24.00  31.96
0.30   21.17  22.85  24.00  24.00  24.00  45.13

### CSU（theta=0）自身的偏置（pp，最坏水平）
21.17

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     1    1    1    1    1    1
0.02     1    4    7   13   28   36
0.10     1    4    7   13   90  154
0.30     1    4    7   13  109  203

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

## D1/high  [pooled]  CNR=18.0  gamma=-0.09
levels [21.0, 24.0, 27.0, 30.0]  n_cal 227  n_test 100  cal RMSE(theta=0) = 3.13 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 76  theta* = 1.50 deg (cal RMSE 2.63)

### band（pp）
tau     0.0   0.5   1.0   2.0    5.0    inf
delta                                      
0.00   3.03  4.29  5.58  7.57  26.86  26.86
0.02   3.03  4.87  6.74  9.32  30.97  61.69
0.10   3.03  4.87  6.74  9.32  30.97  74.03
0.30   3.03  4.87  6.74  9.32  30.97  84.58

### band 内可达最大偏置（pp）
tau     0.0   0.5  1.0   2.0   5.0    inf
delta                                    
0.00   5.85  6.96  8.1  9.85  30.0  30.00
0.02   5.85  6.96  8.1  9.85  30.0  35.45
0.10   5.85  6.96  8.1  9.85  30.0  47.34
0.30   5.85  6.96  8.1  9.85  30.0  57.38

### CSU（theta=0）自身的偏置（pp，最坏水平）
3.14

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     6    8   10   13   76   76
0.02     6    9   12   16   89  157
0.10     6    9   12   16   98  191
0.30     6    9   12   16  119  230

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

## D2/high  [pooled]  CNR=34.5  gamma=-0.16
levels [35.0, 40.0, 45.0, 50.0]  n_cal 279  n_test 101  cal RMSE(theta=0) = 1.88 pp  slack(theta=0) = +0.000000  delta=0 可行 theta 数 = 18  theta* = 0.00 deg (cal RMSE 1.88)

### band（pp）
tau    0.0   0.5   1.0   2.0    5.0     inf
delta                                      
0.00   0.0  1.44  1.93  3.41   8.03    8.56
0.02   0.0  1.92  2.88  5.30  12.23   15.50
0.10   0.0  1.92  2.88  5.30  12.23   44.05
0.30   0.0  1.92  2.88  5.30  12.23  100.00

### band 内可达最大偏置（pp）
tau     0.0  0.5   1.0   2.0   5.0    inf
delta                                    
0.00   1.09  2.4  2.87  4.27  8.62   9.12
0.02   1.09  2.4  2.87  4.27  8.62  11.67
0.10   1.09  2.4  2.87  4.27  8.62  23.33
0.30   1.09  2.4  2.87  4.27  8.62  50.00

### CSU（theta=0）自身的偏置（pp，最坏水平）
1.09

### 可行 theta 数
tau    0.0  0.5  1.0  2.0  5.0  inf
delta                              
0.00     1    4    5    8   17   18
0.02     1    5    7   12   26   32
0.10     1    5    7   12   26   89
0.30     1    5    7   12   26  189

### theta=0 是否在可行集内
tau     0.0   0.5   1.0   2.0   5.0   inf
delta                                    
0.00   True  True  True  True  True  True
0.02   True  True  True  True  True  True
0.10   True  True  True  True  True  True
0.30   True  True  True  True  True  True

