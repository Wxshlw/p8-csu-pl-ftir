"""unmix_ftir.baselines —— 黑箱/经典基线族的统一封装。

纪律
----
* 所有超参数只在**训练集内部**用 GroupKFold（组 = 物理样品）选择；
  外推测试段完全不参与任何选择。
* 随机模型（RF / HGB / MLP）用固定种子集合，报告跨种子均值与符号稳定性。
* 预处理（SG 一阶导）对所有方法相同，保证可比。
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import forward as F


def _cv(groups, n_splits=5):
    n = len(groups)
    k = min(n_splits, len(np.unique(groups)))
    return GroupKFold(n_splits=max(2, k))


def fit_pls(X, y, g, seed=0):
    grid = {"n_components": [2, 3, 4, 6, 8, 10, 12, 15, 20]}
    gs = GridSearchCV(PLSRegression(scale=False), grid, cv=_cv(g),
                      scoring="neg_root_mean_squared_error", n_jobs=1)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


def fit_ridge(X, y, g, seed=0):
    grid = {"ridge__alpha": np.logspace(-2, 4, 13)}
    pipe = SkPipeline([("sc", StandardScaler()), ("ridge", Ridge())])
    gs = GridSearchCV(pipe, grid, cv=_cv(g), scoring="neg_root_mean_squared_error")
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


def fit_svr(X, y, g, seed=0):
    grid = {"svr__C": [10.0, 100.0, 1000.0],
            "svr__gamma": ["scale", 1e-4],
            "svr__epsilon": [0.5, 1.0]}
    pipe = SkPipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    gs = GridSearchCV(pipe, grid, cv=_cv(g), scoring="neg_root_mean_squared_error")
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


def fit_rf(X, y, g, seed=0):
    """树模型在 PCA(30) 得分上拟合：控制容量，同时避免 1762 维上的无意义分裂。"""
    grid = {"m__min_samples_leaf": [1, 4]}
    pipe = SkPipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=30, random_state=0)),
                       ("m", RandomForestRegressor(n_estimators=300, random_state=seed,
                                                   n_jobs=-1))])
    gs = GridSearchCV(pipe, grid, cv=_cv(g), scoring="neg_root_mean_squared_error")
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


def fit_hgb(X, y, g, seed=0):
    grid = {"m__max_leaf_nodes": [15, 31]}
    pipe = SkPipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=30, random_state=0)),
                       ("m", HistGradientBoostingRegressor(
                           max_iter=250, l2_regularization=1.0, random_state=seed))])
    gs = GridSearchCV(pipe, grid, cv=_cv(g), scoring="neg_root_mean_squared_error")
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


def fit_mlp(X, y, g, seed=0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", alpha=1e-3,
                       learning_rate_init=1e-3, max_iter=800, early_stopping=True,
                       n_iter_no_change=30, random_state=seed)
    net.fit(Z, y)
    return dict(kind="mlp", net=net, mu=mu, sd=sd)


def pred_mlp(model, X):
    return model["net"].predict((X - model["mu"]) / model["sd"])


# ---------------------------------------------------------------------------
# 物理/经典基线
# ---------------------------------------------------------------------------
def fit_cls2(X, y, g, wvn=None, order=3):
    """两点 CLS：以训练浓度最低 / 最高水平的均值谱作为两个锚点（朴素物理基线）。"""
    y = np.asarray(y, float)
    psi = F.baseline_basis(wvn, order)
    Qb = F.projector_out(psi)
    c0, c1 = float(y.min()), float(y.max())
    m0 = X[np.abs(y - c0) < 1e-9].mean(0)
    m1 = X[np.abs(y - c1) < 1e-9].mean(0)
    p0, p1 = F.project_out(m0, Qb), F.project_out(m1, Qb)
    return dict(kind="cls2", p0=p0, p1=p1, psi=psi, Qb=Qb, c0=c0, c1=c1)


def pred_cls2(model, X):
    X = np.asarray(X, float)
    QA = F.project_out(X, model["Qb"])
    d = model["p1"] - model["p0"]
    # 解 min ||QA - p0 - t d||^2 -> t = d^T(QA-p0)/||d||^2
    t = (QA @ d - model["p0"] @ d) / max(float(d @ d), 1e-12)
    return model["c0"] + t * (model["c1"] - model["c0"])


def fit_mcr(X, y, g, wvn=None, seed=0, X_extra=None, init="nndsvda", n_comp=2):
    """无监督 MCR-ALS（非负矩阵分解）+ 事后线性校准（旋转模糊对照）。

    无监督分解本身是转导式的：MCR-ALS 的常规用法就是把校准谱与待测谱一起
    分解，只在校准步使用浓度。这里给基线**最有利**的设置——把 X_extra
    （测试谱，无标签）并入分解；监督解混反而只用训练谱。若这样仍外推失败，
    说明问题出在缺少浓度硬约束，而不是信息量不足。
    """
    from sklearn.decomposition import NMF
    psi = F.baseline_basis(wvn, 1)
    Qb = F.projector_out(psi)
    R0 = F.project_out(np.asarray(X, float), Qb)
    parts = [R0]
    if X_extra is not None:
        parts.append(F.project_out(np.asarray(X_extra, float), Qb))
    Rall = np.vstack(parts)
    nmf = NMF(n_components=n_comp, init=init, random_state=seed,
              max_iter=800, tol=1e-5)
    Wall = nmf.fit_transform(np.maximum(Rall, 0.0))
    Wtr = Wall[:len(R0)]
    A = np.column_stack([Wtr, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(kind="mcr", nmf=nmf, coef=coef, Qb=Qb)


def pred_mcr(model, X, wvn=None):
    R = F.project_out(np.asarray(X, float), model["Qb"])
    W = model["nmf"].transform(np.maximum(R, 0.0))
    return np.column_stack([W, np.ones(len(W))]) @ model["coef"]


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
def fit_predict(name, Xtr, ytr, gtr, Xte, wvn, seed=0):
    if name == "PLS":
        m = fit_pls(Xtr, ytr, gtr, seed); return m.predict(Xte).ravel()
    if name == "Ridge":
        m = fit_ridge(Xtr, ytr, gtr, seed); return m.predict(Xte).ravel()
    if name == "SVR":
        m = fit_svr(Xtr, ytr, gtr, seed); return m.predict(Xte).ravel()
    if name == "RF":
        m = fit_rf(Xtr, ytr, gtr, seed); return m.predict(Xte).ravel()
    if name == "HGB":
        m = fit_hgb(Xtr, ytr, gtr, seed); return m.predict(Xte).ravel()
    if name == "MLP":
        m = fit_mlp(Xtr, ytr, gtr, seed); return pred_mlp(m, Xte)
    if name == "CLS-2pt":
        m = fit_cls2(Xtr, ytr, gtr, wvn); return pred_cls2(m, Xte)
    if name == "MCR-ALS":
        m = fit_mcr(Xtr, ytr, gtr, wvn, seed, X_extra=Xte); return pred_mcr(m, Xte)
    raise KeyError(name)


BLACKBOX = ["PLS", "Ridge", "SVR", "RF", "HGB", "MLP"]
ALL = BLACKBOX + ["CLS-2pt", "MCR-ALS"]
