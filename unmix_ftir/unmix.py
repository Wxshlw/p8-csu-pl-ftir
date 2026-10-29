"""unmix_ftir.unmix —— 浓度监督的物理约束解混（CSU-PL）。

模型（二元闭合体系，论文 §2.1）
    A_i(nu) = s_i [ (1 - c_i) g_m(nu) + c_i g_a(nu) ] + Psi(nu) beta_i + eps
    g_k      = E_k ⊛ L        （L 解析给定，见 forward.line_kernel）
    s_i >= 0                  逐样品乘性散射/接触因子（profile 掉）
    Psi beta_i                逐样品低阶加性基线（profile 掉）

估计（训练，c 已知）
    交替最小二乘：
      (i)  固定端元 -> 逐样品最小二乘解 (s_i, beta_i)
      (ii) 固定 nuisance -> 逐通道 2 变量加权最小二乘解 (g_m, g_a)
    约束：g_k >= 0；端元在 L 的带限子空间内（每轮与 L 卷积）；
    **浓度 c 以硬约束进入第 (ii) 步**——这正是与 MCR-ALS 的本质差别
    （MCR-ALS 的得分矩阵是估出来的，存在旋转模糊）。

推断（测试，c 未知）
    固定端元，对 c 做一维 profile 似然（s 与 beta 解析消去）：

        Q = I - Psi Psi^+                 （投影到基线空间的正交补）
        a_i = <Q A_i, Q g_m>,  b_i = <Q A_i, Q d>,  d = g_a - g_m
        q0 = ||Q g_m||^2, q1 = <Q g_m, Q d>, q2 = ||Q d||^2
        s_i(c) = max(0, a_i + c b_i) / (q0 + 2 c q1 + c^2 q2)
        R_i(c) = ||Q A_i||^2 - max(0, a_i + c b_i)^2 / (q0 + 2 c q1 + c^2 q2)

    在 c 的网格上全局取极小 + 抛物线细化。可见：
      * **只有 Q d（端元对比在基线正交补内的分量）携带浓度信息**；
        若 Q d = 0 则 R(c) 与 c 无关 -> 完全不可辨识（定理 1 的可计算形式）。
      * 预测对 c 是线性投影的逆，**不做任何向训练均值的收缩** ->
        校准范围外仍是一致估计（定理 2 的推论），代价是方差随杠杆 h(c*) 增大。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import forward as F
from .design import design_matrix


@dataclass
class UnmixConfig:
    baseline_order: int = 3          # 项数：1=常数,2=+线性,3=+二次
    resolution: float = np.nan       # cm^-1；nan -> 不卷积（L=identity）
    line_kind: str = "gaussian"
    n_iter: int = 60
    nonneg: bool = True
    smooth_sigma: float = 0.0        # 额外的端元平滑（采样点），0=不做
    ridge: float = 1e-8
    scatter_sigma: float = 0.05      # 乘性尺度先验 s ~ N(1, sigma^2)；None/<=0 -> 无先验（自由 s）
    fix_scale: bool = False          # True -> s 恒等于 1（纯加性模型，lambda_s -> inf 的极限）
    atr_correct: bool = False
    batched: bool = True               # True -> ALS nuisance 批量闭式解；False -> 原逐样品 lstsq 循环（等价性审计用）
    tol: float = 1e-6
    verbose: bool = False


@dataclass
class UnmixModel:
    g: np.ndarray                    # (K, N)；K=2 单基质 [g_m, g_a]；K=B+1 多基质 [g_m,1..B, g_a]
    psi: np.ndarray                  # (N, P) 基线基
    wvn: np.ndarray
    cfg: UnmixConfig
    n_iter_run: int = 0
    history: dict = field(default_factory=dict)
    brand_levels: list | None = None   # 多基质时长度 B 的品牌标签（与 g 前 B 行对应）

    @property
    def n_matrix(self):
        return 1 if self.brand_levels is None else len(self.brand_levels)

    def _brand_idx(self, brand, n):
        if self.n_matrix == 1:
            return np.zeros(n, dtype=int)
        if brand is None:
            raise ValueError("multi-matrix model requires brand labels at predict time")
        lut = {b: i for i, b in enumerate(self.brand_levels)}
        idx = np.array([lut.get(str(b), -1) for b in np.asarray(brand, dtype=object)])
        # 未见过的品牌 -> 回退到平均基质端元（在 predict 中标记为 -1 会越界，故用 B 行）
        return idx

    # ---------------- 推断 ----------------
    def _profile_terms(self, X, brand=None):
        X = np.asarray(X, float)
        if self.cfg.atr_correct:
            X = F.atr_depth_correct(X, self.wvn)
        Qb = F.projector_out(self.psi)
        QA = F.project_out(X, Qb)
        gm = self.g[:self.n_matrix]                 # (B, N)
        ga = self.g[-1][None, :]
        Qgm = F.project_out(gm, Qb)                 # (B, N)
        Qd = F.project_out(ga - gm, Qb)             # (B, N)
        bi = self._brand_idx(brand, len(X))
        unseen = bi < 0
        if unseen.any():                            # 未见品牌 -> 用品牌平均基质
            gm_mean = gm.mean(0, keepdims=True)
            Qgm_mean = F.project_out(gm_mean, Qb)[0]
            Qd_mean = F.project_out(ga - gm_mean, Qb)[0]
            Qgm = np.vstack([Qgm, Qgm_mean])
            Qd = np.vstack([Qd, Qd_mean])
            bi = np.where(unseen, self.n_matrix, bi)
        q0 = np.einsum("ij,ij->i", Qgm, Qgm)[bi]
        q1 = np.einsum("ij,ij->i", Qgm, Qd)[bi]
        q2 = np.einsum("ij,ij->i", Qd, Qd)[bi]
        a = np.einsum("ij,ij->i", QA, Qgm[bi])
        b = np.einsum("ij,ij->i", QA, Qd[bi])
        const = np.einsum("ij,ij->i", QA, QA)
        return dict(a=a, b=b, q0=q0, q1=q1, q2=q2, const=const,
                    Qd=Qd[bi], Qgm=Qgm[bi], Qb=Qb, QA=QA, brand_idx=bi,
                    lam_s=getattr(self, "lam_s", 0.0))

    def predict(self, X, brand=None, lo=0.0, hi=1.0, n_grid=1001, refine=True):
        """profile 推断 c（乘性尺度带先验 s~N(1,sigma^2)，解析消去）。

        目标（逐样品，Psi beta 投影掉之后）：
            ||Q A_i||^2 + lam_s - (corr_i(c)+lam_s)^2 / (den(c)+lam_s)
        其中 corr_i(c)=a_i+c b_i, den(c)=q0+2c q1+c^2 q2。lam_s=0 退化为自由尺度；
        lam_s=inf 退化为 s≡1（纯加性模型）。
        """
        t = self._profile_terms(X, brand=brand)
        lam = float(t["lam_s"])
        grid = np.linspace(lo, hi, n_grid)
        corr = t["a"][:, None] + grid[None, :] * t["b"][:, None]
        den0 = (t["q0"][:, None] + 2 * grid[None, :] * t["q1"][:, None]
                + grid[None, :] ** 2 * t["q2"][:, None])
        if self.cfg.fix_scale:
            # s == 1（lambda_s -> inf 的极限，闭式）：R(c)=const-2(a+cb)+den0
            curve = t["const"][:, None] - 2.0 * corr + den0
            den = np.maximum(den0, 1e-12)
        else:
            den = np.maximum(den0 + lam, 1e-12)
            curve = t["const"][:, None] - np.maximum(corr + lam, 0.0) ** 2 / den
        j = np.argmin(curve, axis=1)
        c_hat = grid[j].copy()
        if refine:
            dg = grid[1] - grid[0]
            jm = np.clip(j - 1, 0, len(grid) - 1)
            jp = np.clip(j + 1, 0, len(grid) - 1)
            y0, y1, y2 = (curve[np.arange(len(j)), jm], curve[np.arange(len(j)), j],
                          curve[np.arange(len(j)), jp])
            denom = (y0 - 2 * y1 + y2)
            with np.errstate(divide="ignore", invalid="ignore"):
                delta = np.where(np.abs(denom) > 1e-18, 0.5 * (y0 - y2) / denom, 0.0)
            c_hat = np.clip(c_hat + np.clip(delta, -1, 1) * dg, lo, hi)
        if self.cfg.fix_scale:
            s_hat = np.ones_like(c_hat)
        else:
            s_hat = (t["a"] + c_hat * t["b"] + lam) / np.maximum(
                t["q0"] + 2 * c_hat * t["q1"] + c_hat ** 2 * t["q2"] + lam, 1e-12)
            s_hat = np.clip(s_hat, 0.0, 5.0)
        gd = self.g[-1] - self.g[np.minimum(t["brand_idx"], self.n_matrix - 1)]
        return dict(c_hat=c_hat, s_hat=s_hat, grid=grid, curve=curve,
                    identifiable_frac=float(np.linalg.norm(t["Qd"]) /
                                            (np.linalg.norm(gd) + 1e-12)),
                    contrast=gd, brand_idx=t["brand_idx"])

    def predict_c(self, X, brand=None, **kw):
        return self.predict(X, brand=brand, **kw)["c_hat"]

    def reconstruct(self, X, brand=None):
        """给定 X 与推断出的 (c, s, beta) 的重建（用于审计/残差图）。"""
        X = np.asarray(X, float)
        if self.cfg.atr_correct:
            X = F.atr_depth_correct(X, self.wvn)
        c = self.predict_c(X, brand=brand)
        bi = self._brand_idx(brand, len(X))
        bi = np.clip(bi, 0, self.n_matrix - 1)
        n = X.shape[0]
        Ahat = np.empty_like(X)
        for i in range(n):
            m = (1 - c[i]) * self.g[bi[i]] + c[i] * self.g[-1]
            Z = np.column_stack([m, self.psi]) if self.psi.size else m[:, None]
            th, *_ = np.linalg.lstsq(Z, X[i], rcond=None)
            Ahat[i] = Z @ th
        return Ahat


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _smooth(G, kernel, extra_sigma=0.0, dnu=1.0):
    G = F.convolve_rows(G, kernel)
    if extra_sigma and extra_sigma > 0:
        k = F.line_kernel(2.354820045 * extra_sigma * dnu, dnu)
        G = F.convolve_rows(G, k)
    return G


def _design_multi(c, bi, n_matrix, s):
    """多基质设计矩阵：列 = [s(1-c)·1{b=0}, ..., s(1-c)·1{b=B-1}, s·c]。"""
    n = len(c)
    Z = np.zeros((n, n_matrix + 1))
    Z[np.arange(n), bi] = s * (1.0 - c)
    Z[:, -1] = s * c
    return Z


def _nuisance_batched(M, X, psi, lam=0.0, prior=False):
    """逐样品 nuisance (s_i, beta_i) 的批量闭式解。

    与 fit() 里原来的逐样品 lstsq 循环**同一法方程**，只是把 n 次 (N x (P+1))
    最小二乘换成 n 个 (P+1) x (P+1) 的批量求解：
        Z_i = [m_i | Psi]
        Z_iᵀ Z_i = [[||m_i||², m_iᵀ Psi], [Psiᵀ Psi (共享)]]
        Z_iᵀ x_i = [m_iᵀ x_i, Psiᵀ x_i]
    尺度先验在循环里是追加一行 [sqrt(lam), 0..0]、目标 sqrt(lam)，
    等价于 Gram[0,0] += lam、rhs[0] += lam（行与目标的乘积是 lam 本身）。
    返回 (s, beta)。
    """
    n, N = M.shape
    P = int(psi.shape[1]) if psi.size else 0
    A = np.zeros((n, P + 1, P + 1))
    b = np.zeros((n, P + 1))
    mm = np.einsum("ij,ij->i", M, M)
    mx = np.einsum("ij,ij->i", M, X)
    A[:, 0, 0] = mm
    b[:, 0] = mx
    if P:
        mp = M @ psi                      # (n, P)
        pp = psi.T @ psi                  # (P, P)
        A[:, 0, 1:] = mp
        A[:, 1:, 0] = mp
        A[:, 1:, 1:] = pp
        b[:, 1:] = X @ psi
    if lam and lam > 0:
        A[:, 0, 0] += lam
        b[:, 0] += lam
    try:
        th = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        th = np.einsum("nij,nj->ni", np.linalg.pinv(A), b)
    if prior:
        s = np.clip(th[:, 0], 1e-6, 5.0)
    else:
        s = np.maximum(th[:, 0], 1e-6)
    beta = th[:, 1:] if P else np.zeros((n, 0))
    return s, beta


def fit(X, c, wvn, cfg: UnmixConfig | None = None, brand=None):
    """浓度监督解混。X:(n,N), c:(n,) 质量分数 %, wvn:(N,) 升序。

    brand 为 None -> 单基质（2 端元）；给出品牌标签 -> 每个品牌一个基质端元
    + 共享掺杂物端元（B+1 端元）。品牌是样品标签，不是浓度信息。
    """
    cfg = cfg or UnmixConfig()
    X = np.asarray(X, float)
    wvn = np.asarray(wvn, float)
    c = np.asarray(c, float).ravel() / 100.0
    n, N = X.shape
    if cfg.atr_correct:
        X = F.atr_depth_correct(X, wvn)
    dnu = float(np.median(np.diff(wvn)))
    kernel = F.line_kernel(cfg.resolution, dnu, cfg.line_kind)
    psi = F.baseline_basis(wvn, cfg.baseline_order)

    if brand is None:
        brand_levels = None
        bi = np.zeros(n, dtype=int)
        n_matrix = 1
    else:
        brand_levels = sorted({str(b) for b in np.asarray(brand, dtype=object)})
        lut = {b: i for i, b in enumerate(brand_levels)}
        bi = np.array([lut[str(b)] for b in np.asarray(brand, dtype=object)])
        n_matrix = len(brand_levels)

    # ---- 初始化：逐品牌 CLS（s=1, beta=0）----
    G = np.zeros((n_matrix + 1, N))
    ga_list = []
    for b in range(n_matrix):
        m = bi == b
        if m.sum() < 2:
            continue
        Cb = design_matrix(c[m], "simplex")
        Gb = np.linalg.lstsq(Cb, X[m], rcond=None)[0]
        G[b] = Gb[0]
        ga_list.append(Gb[1])
    if ga_list:
        G[-1] = np.mean(ga_list, axis=0)
    else:  # pragma: no cover
        G[-1] = G[:-1].mean(0)
    if cfg.nonneg:
        G = np.maximum(G, 0.0)
    G = _smooth(G, kernel, cfg.smooth_sigma, dnu)

    # 乘性尺度先验强度 lam_s = sigma_noise^2 / sigma_s^2（残差用 MAD 稳健估计）
    C = design_matrix(c, "simplex") if n_matrix == 1 else None
    mu0 = np.where(n_matrix == 1, 1.0, 1.0)
    M0 = np.where(n_matrix == 1, (1 - c)[:, None] * G[0] + c[:, None] * G[1],
                  (1 - c)[:, None] * G[bi] + c[:, None] * G[-1])
    resid0 = X - M0
    sigma_hat = 1.4826 * float(np.median(np.abs(resid0 - np.median(resid0, axis=1, keepdims=True))))
    # lambda_s 与"整条谱的残差能量"同量纲：逐通道噪声方差 x 通道数 / 先验方差
    lam_s = 0.0 if cfg.scatter_sigma is None or cfg.scatter_sigma <= 0 else float(
        N * sigma_hat ** 2 / cfg.scatter_sigma ** 2)

    hist = dict(rmse=[], dG=[])
    s = np.ones(n)
    beta = np.zeros((n, psi.shape[1]))
    if cfg.fix_scale:
        lam_s = np.inf                       # s 固定为 1：纯加性模型
    it = 0
    for it in range(1, cfg.n_iter + 1):
        G_old = G.copy()
        # (i) 逐样品 nuisance：(s_i, beta_i)
        if cfg.fix_scale:
            # s ≡ 1：只估基线系数（闭式，逐样品）
            M = (1 - c)[:, None] * G[bi] + c[:, None] * G[-1]
            if psi.size:
                beta, *_ = np.linalg.lstsq(psi, (X - M).T, rcond=None)
                beta = beta.T
        elif lam_s <= 0:
            if cfg.batched:
                M = (1 - c)[:, None] * G[bi] + c[:, None] * G[-1]
                s, beta = _nuisance_batched(M, X, psi, lam=0.0, prior=False)
            else:
                for i in range(n):
                    m = (1 - c[i]) * G[bi[i]] + c[i] * G[-1]
                    Z = np.column_stack([m, psi]) if psi.size else m[:, None]
                    th, *_ = np.linalg.lstsq(Z, X[i], rcond=None)
                    s[i] = max(th[0], 1e-6)
                    beta[i] = th[1:]
        else:
            # 带先验的逐样品 LS：augmented rows [sqrt(lam)*e_s, sqrt(lam)*1]
            r = np.sqrt(lam_s)
            if cfg.batched:
                M = (1 - c)[:, None] * G[bi] + c[:, None] * G[-1]
                s, beta = _nuisance_batched(M, X, psi, lam=lam_s, prior=True)
            else:
                for i in range(n):
                    m = (1 - c[i]) * G[bi[i]] + c[i] * G[-1]
                    Z = np.column_stack([m, psi]) if psi.size else m[:, None]
                    Za = np.vstack([Z, np.concatenate([[r], np.zeros(psi.shape[1])])])
                    ba = np.concatenate([X[i], [r]])
                    th, *_ = np.linalg.lstsq(Za, ba, rcond=None)
                    s[i] = float(np.clip(th[0], 1e-6, 5.0))
                    beta[i] = th[1:]
        # (ii) 固定 nuisance 更新端元（浓度硬约束在此）
        R = X - beta @ psi.T if psi.size else X
        D = _design_multi(c, bi, n_matrix, s)
        A = D.T @ D + cfg.ridge * np.eye(n_matrix + 1)
        G = np.linalg.solve(A, D.T @ R)
        if cfg.nonneg:
            G = np.maximum(G, 0.0)
        G = _smooth(G, kernel, cfg.smooth_sigma, dnu)
        # 固定全局尺度（s 均值 = 1），不影响 c
        if not cfg.fix_scale:
            ms = float(np.mean(s))
            if ms > 0:
                s = s / ms
                G = G * ms
        Mu = (1 - c)[:, None] * G[bi] + c[:, None] * G[-1]
        rmse = float(np.sqrt(np.mean((X - (s[:, None] * Mu + beta @ psi.T)) ** 2)))
        hist["rmse"].append(rmse)
        hist["dG"].append(float(np.linalg.norm(G - G_old) / (np.linalg.norm(G_old) + 1e-12)))
        if cfg.verbose and (it % 10 == 0 or it == 1):
            print(f"  [unmix] iter {it:3d} rmse={rmse:.5f} dG={hist['dG'][-1]:.2e}")
        if hist["dG"][-1] < cfg.tol and it > 3:
            break

    model = UnmixModel(g=G, psi=psi, wvn=wvn, cfg=cfg, n_iter_run=it, history=hist,
                       brand_levels=brand_levels)
    model.scale_ = s
    model.beta_ = beta
    model.lam_s = lam_s
    model.sigma_hat = sigma_hat
    model.train_rmse_ = hist["rmse"][-1] if hist["rmse"] else np.nan
    return model
