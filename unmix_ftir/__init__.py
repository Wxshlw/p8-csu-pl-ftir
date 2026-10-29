"""unmix_ftir —— 物理约束解混：可辨识性与校准域外外推。

模块划分
--------
data      : 5 套 ATR-FTIR 浓度梯度数据集统一加载（D1–D3 KAN trio, D7 肉桂×3, D8 姜黄）
design    : 外推协议、浓度设计矩阵、kappa(C) 与杠杆
forward   : Beer–Lambert 前向模型（端元 ⊛ 仪器线型, 乘性散射, 加性基线）
unmix     : 浓度监督的物理约束解混（ALS）+ 带约束推断器
baselines : 黑箱基线族 + CLS + 无监督 MCR-ALS + 容量匹配对照
stats     : 样品级固定划分、指标、聚类 bootstrap、Holm
"""
__version__ = "0.1.0"
