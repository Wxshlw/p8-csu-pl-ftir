"""unmix_ftir.forward —— 前向模型的物理构件（不是学习出来的部分）。

  * ATR 穿透深度校正：d_p ∝ 1/nu  ->  吸光度乘以 nu 归一化
  * 仪器线型 L：由标称分辨率与切趾类型解析给定（高斯近似），卷积到端元上
  * 加性基线空间 Psi：低阶多项式（常数/线性/二次/三次）
  * 乘性散射：逐样品标量（接触/光程），在推断时被 profile 掉
  * 常用黑箱预处理（SNV / SG 导数），供基线族共用；**只在训练折内 fit**
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------
def atr_depth_correct(X, wvn, normalize=True):
    """ATR 穿透深度校正：A_corr = A * nu / median(nu)（等价于除以 d_p ∝ 1/nu）。

    论文 §4.2 要求：先校正再标定，并做校正前后消融。
    """
    X = np.asarray(X, float)
    wvn = np.asarray(wvn, float)
    f = wvn / np.median(wvn)
    if not normalize:
        f = wvn
    return X * f[None, :]


# ---------------------------------------------------------------------------
# 仪器线型
# ---------------------------------------------------------------------------
def line_kernel(resolution, dnu, kind="gaussian", truncate=3.0):
    """仪器线型核（FWHM = resolution，单位 cm^-1；采样间隔 dnu）。

    返回归一化一维核。resolution 为 nan/0 时返回 [1]（恒等）。
    """
    if resolution is None or not np.isfinite(resolution) or resolution <= 0:
        return np.array([1.0])
    fwhm_samples = float(resolution) / float(dnu)
    if fwhm_samples <= 0.5:
        return np.array([1.0])
    if kind == "gaussian":
        sigma = fwhm_samples / 2.354820045
    elif kind == "boxcar":
        w = max(1, int(round(fwhm_samples)))
        k = np.ones(w, float)
        return k / k.sum()
    else:
        raise ValueError(kind)
    half = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve_rows(A, kernel):
    """沿最后一维做 same 卷积（边缘反射填充）。"""
    A = np.asarray(A, float)
    k = np.asarray(kernel, float)
    if len(k) == 1:
        return A.copy()
    half = len(k) // 2
    pad = np.pad(A, ((0, 0), (half, half)), mode="reflect")
    if A.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)


# ---------------------------------------------------------------------------
# 基线空间
# ---------------------------------------------------------------------------
def baseline_basis(wvn, order=3):
    """低阶多项式基线基（Legendre，数值稳定）。order = 项数（1=常数）。

    order<=0 -> 空基（N x 0）。
    """
    wvn = np.asarray(wvn, float)
    if order is None or order <= 0:
        return np.zeros((len(wvn), 0))
    v = (wvn - wvn.mean()) / (np.ptp(wvn) / 2.0 + 1e-12)
    cols = [np.ones_like(v)]
    if order >= 2:
        cols.append(v)
    if order >= 3:
        cols.append(1.5 * v ** 2 - 0.5)
    if order >= 4:
        cols.append(2.5 * v ** 3 - 1.5 * v)
    if order >= 5:
        cols.append((35 * v ** 4 - 30 * v ** 2 + 3) / 8)
    return np.column_stack(cols[:order])


def projector_out(B):
    """返回正交补投影矩阵 Q = I - B B^+（B 的列数为 0 时返回 None 表示恒等）。"""
    B = np.asarray(B, float)
    if B.size == 0:
        return None
    Qb, _ = np.linalg.qr(B)
    return Qb


def project_out(A, Qb):
    """把 A 的行（谱）投影到 span(B) 的正交补：A_proj = A - A Qb Qb^T。"""
    if Qb is None:
        return A
    return A - (A @ Qb) @ Qb.T


# ---------------------------------------------------------------------------
# 黑箱族共享的预处理（fit 只在训练折调用）
# ---------------------------------------------------------------------------
class SNV:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, ddof=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (X - mu) / sd


class SG:
    """Savitzky–Golay 平滑/导数（逐谱，不需要 fit）。"""

    def __init__(self, window=11, poly=3, deriv=1):
        self.window, self.poly, self.deriv = window, poly, deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        w = min(self.window, X.shape[1] - (1 - X.shape[1] % 2))
        if w % 2 == 0:
            w -= 1
        w = max(w, self.poly + 1 + (self.poly + 1) % 2)
        return savgol_filter(X, window_length=w, polyorder=self.poly,
                             deriv=self.deriv, axis=1)


class Pipeline:
    """顺序预处理（如 SNV -> SG1）。"""

    def __init__(self, *steps):
        self.steps = steps

    def fit(self, X, y=None):
        for s in self.steps:
            s.fit(X, y)
        return self

    def transform(self, X):
        for s in self.steps:
            X = s.transform(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
