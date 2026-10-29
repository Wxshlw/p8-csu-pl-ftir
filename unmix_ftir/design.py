"""unmix_ftir.design —— 外推协议、浓度设计矩阵与可辨识性诊断量。

核心量（对应论文 Theory §2.3）
------------------------------
二元体系（基体 + 一种掺杂物，闭合 c_matrix + c_adulterant = 1）的单样品前向模型可写成

    A_i = s_i [ (1 - c_i) g_m + c_i g_a ] + Psi_i beta_i + eps_i
        = s_i [ g_m + c_i (g_a - g_m) ] + Psi_i beta_i + eps_i

训练集上 c 已知 → 端元 (g_m, g_a) 由**浓度设计矩阵**

    C = [[1-c_1, c_1], ..., [1-c_n, c_n]]   (n x 2)

的最小二乘解给出（在给定 nuisance 后），其误差放大由

    kappa(C) = sigma_max(C) / sigma_min(C)

控制。kappa 只取决于浓度水平的**位置与展布**，与光谱无关 → 它就是一个纯粹的
"实验设计"诊断量。二元体系下 kappa 随校准水平向窄区间收缩而单调增大，
在单一水平（无展布）时发散。

外推预测的杠杆
--------------
把 c 视作设计变量、端元视作待估参数，则预测点 c* 的杠杆为

    h(c*) = [1, c*] (D^T D)^{-1} [1, c*]^T ,  D = [1, c]

（这里 D 只是 C 的列变换；使用 D 形式是因为它直接对应 "g_m + c D" 的参数化，
两者 kappa 只差一个固定 2x2 变换，见模块内 kappa_transform_note()。）

    h(c*) = 1/n + (c* - mean(c))^2 / sum_i (c_i - mean(c))^2

→ 外推误差的**方差**部分随 (c* - mean)^2 / 展布 增长。这是经典杠杆公式，
本文的贡献是把它接到"端元未知 + 基线/散射 nuisance"的解混问题上，并用实验验证。

统计口径纪律（方案 §4.6）
------------------------
* 划分是**固定的样品级划分**（GroupKFold 组分配对顺序置换不变），
  禁止写"随机化重复划分"；不确定性一律用聚类 bootstrap（重采样单位 = 物理样品）。
"""
from __future__ import annotations

import itertools

import numpy as np

# ---------------------------------------------------------------------------
# 外推协议（每个数据集显式写死，便于 Methods 逐条复述）
# ---------------------------------------------------------------------------
PROTOCOLS = {
    # 肉桂 D7：与方案 §4.2 完全一致。P-out-low 训练 6–24 %，测试 1–4 %；
    # P-out-high 训练 0–8 %，测试 12–24 %（10 % 整个留空，作为 gap 不外推）。
    "D7w": dict(low=([6, 8, 10, 12, 15, 18, 21, 24], [1, 2, 3, 4]),
                high=([0, 1, 2, 3, 4, 6, 8], [12, 15, 18, 21, 24]),
                interp=([0, 1, 2, 3, 4, 6, 8, 21, 24], [10, 12, 15, 18])),
    "D7p": dict(low=([6, 8, 10, 12, 15, 18, 21, 24], [1, 2, 3, 4]),
                high=([0, 1, 2, 3, 4, 6, 8], [12, 15, 18, 21, 24]),
                interp=([0, 1, 2, 3, 4, 6, 8, 21, 24], [10, 12, 15, 18])),
    "D7d": dict(low=([6, 8, 10, 12, 15, 18, 21, 24], [1, 2, 3, 4]),
                high=([0, 1, 2, 3, 4, 6, 8], [12, 15, 18, 21, 24]),
                interp=([0, 1, 2, 3, 4, 6, 8, 21, 24], [10, 12, 15, 18])),
    # KAN trio：11 个等间距水平。低端外推取最低的 2 个非零水平作测试（每个水平 5 个物理样品）。
    "D1": dict(low=([9, 12, 15, 18, 21, 24, 27, 30], [3, 6]),
               high=([0, 3, 6, 9, 12, 15], [21, 24, 27, 30]),
               interp=([0, 3, 6, 27, 30], [12, 15, 18, 21, 24])),
    "D2": dict(low=([15, 20, 25, 30, 35, 40, 45, 50], [5, 10]),
               high=([0, 5, 10, 15, 20, 25], [35, 40, 45, 50]),
               interp=([0, 5, 10, 45, 50], [20, 25, 30, 35, 40])),
    "D3": dict(low=([15, 20, 25, 30, 35, 40, 45, 50], [5, 10]),
               high=([0, 5, 10, 15, 20, 25], [35, 40, 45, 50]),
               interp=([0, 5, 10, 45, 50], [20, 25, 30, 35, 40])),
    "D8": dict(low=([3, 5, 7, 9], [1, 2]),
               high=([0, 1, 2, 3], [5, 7, 9]),
               interp=([0, 1, 2, 7, 9], [3, 5])),
}

DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PRIMARY = ["D7w", "D7p", "D7d", "D1", "D2", "D3", "D8"]


def split_levels(y, levels, tol=1e-6):
    """按浓度水平切分索引。返回 (idx_train, idx_test)。"""
    y = np.asarray(y, float)
    lv = np.asarray(levels, float)
    tr = np.zeros_like(y, dtype=bool)
    te = np.zeros_like(y, dtype=bool)
    for v in lv:
        tr |= np.abs(y - v) < tol
    te = ~tr
    return np.where(tr)[0], np.where(te)[0]


def protocol_split(name, protocol):
    tr, te = PROTOCOLS[name][protocol]
    return np.asarray(tr, float), np.asarray(te, float)


# ---------------------------------------------------------------------------
# 可辨识性诊断量
# ---------------------------------------------------------------------------
def design_matrix(c, form="simplex"):
    """浓度设计矩阵。form='simplex' -> [[1-c, c]]; form='intercept' -> [[1, c]]。"""
    c = np.asarray(c, float).ravel()
    if form == "simplex":
        return np.column_stack([1.0 - c, c])
    if form == "intercept":
        return np.column_stack([np.ones_like(c), c])
    raise ValueError(form)


def kappa(c, form="simplex"):
    """浓度设计矩阵的条件数；单一水平 -> inf。"""
    C = design_matrix(c, form)
    s = np.linalg.svd(C, compute_uv=False)
    if s[-1] <= 1e-12 * max(1.0, s[0]):
        return np.inf
    return float(s[0] / s[-1])


def leverage(c_train, c_star, form="intercept"):
    """预测点 c* 的杠杆 h(c*) = x*(D^T D)^-1 x*^T（含截距时）。"""
    D = design_matrix(c_train, form)
    x = np.array([[1.0, c_star]]) if form == "intercept" else np.array([[1 - c_star, c_star]])
    G = D.T @ D
    try:
        h = float(np.asarray(x @ np.linalg.solve(G, x.T)).ravel()[0])
    except np.linalg.LinAlgError:
        h = np.inf
    return h


def condition_report(c_train, c_test):
    """一个划分的设计诊断：kappa、杠杆（各测试水平）、训练展布。"""
    c_train = np.asarray(c_train, float)
    c_test = np.asarray(c_test, float)
    return dict(
        kappa=kappa(c_train, "simplex"),
        kappa_intercept=kappa(c_train, "intercept"),
        c_train_min=float(c_train.min()), c_train_max=float(c_train.max()),
        c_train_mean=float(c_train.mean()), c_train_std=float(c_train.std(ddof=0)),
        n_train=int(len(c_train)),
        leverage={f"{v:g}": leverage(c_train, v) for v in sorted(set(c_test.tolist()))},
        max_leverage=float(max(leverage(c_train, v) for v in sorted(set(c_test.tolist())))),
    )


def kappa_transform_note():
    """说明 simplex 形式与 intercept 形式的 kappa 关系（写作时引用）。

    [1-c, c] = [1, c] @ T,  T = [[1, 0], [-1, 1]]  =>  kappa 至多相差 cond(T)=2.618。
    两种形式都随"浓度水平向窄区间收缩"单调增大，结论不变。
    """
    T = np.array([[1.0, 0.0], [-1.0, 1.0]])
    return float(np.linalg.cond(T))


# ---------------------------------------------------------------------------
# 实验设计扫描（G3 主实验）
# ---------------------------------------------------------------------------
def sweep_designs(pool, k, distribution, include_zero=False, rng=None, max_designs=1):
    """从候选水平池中挑 k 个训练水平，按分布类型。

    pool          可用的非零训练水平（升序）
    k             训练水平数
    distribution  'uniform' | 'low_dense' | 'high_dense' | 'two_ends' | 'all'
    include_zero  是否额外加入 0 %
    max_designs   每个 (k, distribution) 组合产出多少个不同子集；'all' 时枚举

    说明：'uniform' = 在池内等间距取 k 个；'low_dense' = 几何偏向低端；
    'high_dense' = 几何偏向高端；'two_ends' = 两端各半。
    """
    pool = np.asarray(sorted(set(np.round(pool, 6))), float)
    rng = rng or np.random.default_rng(0)
    out = []

    def add(vals):
        vals = np.asarray(sorted(set(np.round(vals, 6))), float)
        if include_zero and 0.0 not in vals:
            vals = np.concatenate([[0.0], vals])
        if len(vals) == k + int(include_zero):
            out.append(vals)

    if k <= 0 or k > len(pool):
        return out
    if k == 1:
        add([pool[0] if distribution == "low_dense" else
             pool[-1] if distribution == "high_dense" else pool[len(pool) // 2]])
        return out

    if distribution == "all":
        combos = list(itertools.combinations(pool, k))
        if len(combos) > max_designs:
            idx = rng.choice(len(combos), size=max_designs, replace=False)
            combos = [combos[i] for i in sorted(idx)]
        for cb in combos:
            add(cb)
        return out

    if distribution == "uniform":
        idx = np.linspace(0, len(pool) - 1, k)
        add(pool[np.round(idx).astype(int)])
    elif distribution == "low_dense":
        t = np.linspace(0, 1, k) ** 1.8
        idx = np.round(t * (len(pool) - 1)).astype(int)
        add(pool[idx])
    elif distribution == "high_dense":
        t = 1 - np.linspace(0, 1, k) ** 1.8
        idx = np.round(t * (len(pool) - 1)).astype(int)
        add(pool[idx])
    elif distribution == "two_ends":
        n_lo = k // 2
        n_hi = k - n_lo
        lo = pool[:max(1, n_lo)]
        hi = pool[-max(1, n_hi):]
        add(np.concatenate([lo, hi]))
    elif distribution == "random":
        for _ in range(max_designs):
            add(rng.choice(pool, size=k, replace=False))
    else:
        raise ValueError(distribution)
    return out


def design_pool(name):
    """设计扫描的候选池：训练域（高于最低测试水平）内的非零水平。"""
    tr, te = PROTOCOLS[name]["low"]
    return sorted(set(tr))
