"""exp17 —— CRLB 预测标准误与蒙特卡洛实测标准差的对照（P0-3）。

正文 §2.3 给了一个**只用训练集**就能算出的噪声下界

    Var(ĉ) = σ² / ‖d_⊥‖² ,  CNR_eff = ‖d_⊥‖/(σ√N)  ->  SE(ĉ) = 1/(√N·CNR_eff)

但从未验证过它是否成立。本实验直接验证：

mode=white  给测试谱加白噪声 ε ~ N(0, σ²)（σ = 一阶差分稳健估计），
            在 K 次重复下重跑 profile 推断，得经验 SD。
            这一步检验的是**解析线性化本身**：ratio = SD/SE 应 ≈ 1。
mode=resid  把训练集残差行整行随机加到测试谱上（真实的、经 SG 相关的噪声）。
            与 white 的差 = 噪声相关结构带来的额外不确定性。
mode=calib  对训练谱加白噪声后**重新拟合**端元，再预测原始测试谱。
            经验 SD 里含端元估计误差 → 它才随预测点杠杆 h(c*) 增长（§2.4）。
            与 white 的差 = 校准不确定性的贡献。

同时报告观测误差 mean|ĉ−y|：它远大于 white SD 就说明误差由模型失配主导，
这正是 §2.3 末尾"不是噪声而是失配设定误差"的可检验版本。

复用 unmix.UnmixModel._profile_terms 的解析量，自己实现网格 argmin，
并在每个条件上先用 assert 验证它与 predict_c 完全一致后才做蒙特卡洛。
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

from unmix_ftir import data, design, diagnostics, forward as F, stats, unmix

CFG = unmix.UnmixConfig(baseline_order=1, scatter_sigma=0.0, resolution=np.nan, n_iter=40)
SG = F.SG(window=15, poly=3, deriv=1)
DATASETS = ["D1", "D2", "D3", "D7w", "D7p", "D7d", "D8"]
PROTOCOLS = ["low", "high", "interp"]
K_MC = 200
CHUNK = 20
K_FIT = 25
CALIB_DATASETS = ["D1", "D2", "D3", "D7d"]     # 重新拟合代价高：门内三套 + D3 失败案例
MAX_TEST_PER_LEVEL = 12
GRID = np.linspace(0.0, 1.0, 1001)


# ---------------------------------------------------------------------------
# profile 目标（与 unmix.UnmixModel.predict 同一闭式，自由尺度 lam=0）
# ---------------------------------------------------------------------------
def argmin_c(a, b, q0, q1, q2, lam=0.0, grid=GRID, fix_scale=False, const=None):
    """a,b 形状 (..., n) -> 逐谱 argmin c（含抛物线细化）。"""
    a = np.asarray(a, float)[..., None]
    b = np.asarray(b, float)[..., None]
    g = grid[None, :]
    corr = a + g * b
    den0 = np.asarray(q0)[..., None] + 2 * g * np.asarray(q1)[..., None] + \
        g ** 2 * np.asarray(q2)[..., None]
    if fix_scale:
        curve = np.zeros(corr.shape) - 2.0 * corr + den0
        if const is not None:
            curve = curve + np.asarray(const)[..., None]
    else:
        den = np.maximum(den0 + lam, 1e-12)
        curve = -np.maximum(corr + lam, 0.0) ** 2 / den
    j = np.argmin(curve, axis=-1)
    out = np.take_along_axis(np.broadcast_to(grid, curve.shape), j[..., None], -1)[..., 0]
    # 抛物线细化
    dg = grid[1] - grid[0]
    jm = np.clip(j - 1, 0, len(grid) - 1)
    jp = np.clip(j + 1, 0, len(grid) - 1)
    ar = np.arange(curve.shape[-1])
    flat = curve.reshape(-1, len(grid))
    jj = j.ravel()
    idx = np.arange(flat.shape[0])
    y0, y1, y2 = flat[idx, jm.ravel()], flat[idx, jj], flat[idx, jp.ravel()]
    denom = y0 - 2 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-18, 0.5 * (y0 - y2) / denom, 0.0)
    out = np.clip(out.ravel() + np.clip(delta, -1, 1) * dg, grid[0], grid[-1])
    return out.reshape(j.shape)


def model_terms(mdl, X, brand):
    return mdl._profile_terms(X, brand=brand)


def analytic_se(t, sigma):
    """逐谱 SE（浓度分数单位）= sigma / ‖d_perp‖。"""
    Qgm, Qd = t["Qgm"], t["Qd"]
    qgm2 = np.einsum("ij,ij->i", Qgm, Qgm)
    proj = np.einsum("ij,ij->i", Qd, Qgm)
    coef = np.where(qgm2 > 1e-18, proj / np.maximum(qgm2, 1e-18), 0.0)
    dperp = Qd - coef[:, None] * Qgm
    nrm = np.linalg.norm(dperp, axis=1)
    return sigma / np.maximum(nrm, 1e-18), nrm


def mc_perturb(t, eps_gen, sigma, fix_scale=False, chunk=CHUNK, k=K_MC):
    """逐次实现扰动：返回 (K, n) 的 ĉ 矩阵。eps_gen(i, n) 给第 i 批的 (n, N) 噪声。"""
    a0, b0 = t["a"], t["b"]
    q0, q1, q2, lam = t["q0"], t["q1"], t["q2"], float(t["lam_s"])
    Qgm, Qd = t["Qgm"], t["Qd"]
    out = []
    done = 0
    while done < k:
        m = min(chunk, k - done)
        a = np.empty((m, len(a0)))
        b = np.empty((m, len(a0)))
        for r in range(m):
            e = eps_gen(done + r, len(a0))
            a[r] = a0 + np.einsum("ij,ij->i", e, Qgm)
            b[r] = b0 + np.einsum("ij,ij->i", e, Qd)
        out.append(argmin_c(a, b, q0, q1, q2, lam=lam, fix_scale=fix_scale))
        done += m
    return np.vstack(out)


# ---------------------------------------------------------------------------
# 每个条件的准备
# ---------------------------------------------------------------------------
def prepare(nm, prot):
    d = data.load(nm)
    X, y, wvn, groups = d["X"], d["y"], d["wvn"], d["groups"]
    br = data.brands(nm, groups)
    Xv = SG.transform(X)
    tr_lv, te_lv = design.protocol_split(nm, prot)
    tr, te = np.isin(y, tr_lv), np.isin(y, te_lv)
    mdl = unmix.fit(Xv[tr], y[tr], wvn, CFG, brand=br[tr])
    # 测试谱：每水平至多 MAX_TEST_PER_LEVEL 条（分层，按索引顺序确定性抽样）
    keep = []
    for lv in np.unique(y[te]):
        idx = np.where(te)[0][np.abs(y[te] - lv) < 1e-6]
        keep.extend(idx[:MAX_TEST_PER_LEVEL])
    keep = np.array(sorted(keep), dtype=int)
    t = model_terms(mdl, Xv[keep], brand=br[keep])
    # 噪声 σ：与 diagnostics 同一稳健估计（训练谱一阶差分）
    e = diagnostics.endmember_audit(mdl, Xv[tr], y[tr])
    sigma = e["sigma"]
    # 训练残差行（真实的、SG 相关后的噪声形状）
    y_tr = y[tr]
    b_tr = mdl._brand_idx(br[tr], int(tr.sum()))
    b_tr = np.clip(b_tr, 0, mdl.n_matrix - 1)
    Mtr = (1 - y_tr / 100.0)[:, None] * mdl.g[b_tr] + (y_tr / 100.0)[:, None] * mdl.g[-1]
    Rtr = Xv[tr] - (mdl.scale_[:, None] * Mtr + mdl.beta_ @ mdl.psi.T)
    c_hat = mdl.predict_c(Xv[keep], brand=br[keep])
    # 自检：闭式重算必须与 predict_c 一致
    c_re = argmin_c(t["a"], t["b"], t["q0"], t["q1"], t["q2"], lam=float(t["lam_s"]),
                    fix_scale=CFG.fix_scale, const=t["const"])
    assert np.max(np.abs(c_re - c_hat)) < 5e-4, (nm, prot, np.max(np.abs(c_re - c_hat)))
    se, nrm = analytic_se(t, sigma)
    lev = {f"{v:g}": design.leverage(y[tr], v, "intercept") for v in sorted(set(y[keep]))}
    h = np.array([lev[f"{v:g}"] for v in y[keep]])
    return dict(nm=nm, prot=prot, mdl=mdl, Xv=Xv, y=y, wvn=wvn, groups=groups, br=br,
                tr=tr, te=te, keep=keep, t=t, sigma=sigma, Rtr=Rtr, c_hat=c_hat,
                se=se, dperp=nrm, h=h, tr_lv=tr_lv, te_lv=te_lv,
                n_test=int(len(keep)), n_train=int(tr.sum()), N=int(Xv.shape[1]),
                audit=e)


# ---------------------------------------------------------------------------
# stage 1：white / resid（不重新拟合，便宜）
# ---------------------------------------------------------------------------
def stage_mc():
    rows = []
    for nm in DATASETS:
        for prot in PROTOCOLS:
            P = prepare(nm, prot)
            t, keep = P["t"], P["keep"]
            rng = np.random.default_rng(0)
            N = P["N"]

            def white(i, n):
                return rng.normal(0.0, P["sigma"], size=(n, N))

            rs = np.random.default_rng(7)

            def resid(i, n):
                pick = rs.integers(0, P["Rtr"].shape[0], n)
                sgn = rs.choice([-1.0, 1.0], n)[:, None]
                return P["Rtr"][pick] * sgn

            res = dict(dataset=nm, protocol=prot, n_test=P["n_test"], n_train=P["n_train"],
                       N=N, sigma=P["sigma"], se_med=float(np.median(P["se"]) * 100),
                       cnr_eff_med=float(np.median(1.0 / (np.sqrt(N) * P["se"]))),
                       obs_mae=float(np.mean(np.abs(P["c_hat"] * 100 - P["y"][keep]))))
            for name, gen in (("white", white), ("resid", resid)):
                C = mc_perturb(t, gen, P["sigma"], fix_scale=CFG.fix_scale)
                sd = C.std(axis=0, ddof=1) * 100
                res[f"sd_{name}"] = float(np.median(sd))
                res[f"ratio_{name}"] = float(np.median(sd / np.maximum(P["se"] * 100, 1e-12)))
                # ±1.96 SE 区间的经验覆盖率
                z = np.abs(C - P["c_hat"][None, :]) <= 1.96 * P["se"][None, :]
                res[f"cov_{name}"] = float(z.mean())
                # 逐水平
                for lv in sorted(set(P["y"][keep])):
                    m = P["y"][keep] == lv
                    rows.append(dict(dataset=nm, protocol=prot, level=float(lv),
                                     mode=name, n=int(m.sum()),
                                     h=float(P["h"][m][0]), se=float(np.median(P["se"][m]) * 100),
                                     sd=float(np.median(sd[m])),
                                     obs_mae=float(np.mean(np.abs(P["c_hat"][m] * 100 -
                                                                  P["y"][keep][m])))))
            rows.append(dict(dataset=nm, protocol=prot, level=np.nan, mode="cond",
                             n=P["n_test"], h=np.nan, **{k: v for k, v in res.items()
                                                         if k not in ("dataset", "protocol", "n_test")}))
            print(f"[mc] {nm:4s} {prot:6s} SE={res['se_med']:.2f}pp "
                  f"SD_white={res['sd_white']:.2f} (×{res['ratio_white']:.2f}) "
                  f"SD_resid={res['sd_resid']:.2f} (×{res['ratio_resid']:.2f}) "
                  f"obsMAE={res['obs_mae']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp17_mc_long.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# stage 2：calib（训练谱扰动后重新拟合，检验杠杆标度）
# ---------------------------------------------------------------------------
def calib_block(nm, prot):
    rows = []
    P = prepare(nm, prot)
    Xv, tr, keep = P["Xv"], P["tr"], P["keep"]
    y_tr, br_tr = P["y"][tr], P["br"][tr]
    rng = np.random.default_rng(11)
    C = np.empty((K_FIT, len(keep)))
    t0 = time.time()
    for k in range(K_FIT):
        Xp = Xv.copy()
        Xp[tr] += rng.normal(0.0, P["sigma"], size=(int(tr.sum()), P["N"]))
        mdl = unmix.fit(Xp[tr], y_tr, P["wvn"], CFG, brand=br_tr)
        C[k] = mdl.predict_c(Xp[keep], brand=P["br"][keep])
    sd = C.std(axis=0, ddof=1) * 100
    for lv in sorted(set(P["y"][keep])):
        m = P["y"][keep] == lv
        rows.append(dict(dataset=nm, protocol=prot, level=float(lv),
                         n=int(m.sum()), h=float(P["h"][m][0]),
                         se_crlb=float(np.median(P["se"][m]) * 100),
                         sd_calib=float(np.median(sd[m])),
                         sd_calib_mean=float(sd[m].mean())))
    print(f"[calib] {nm:4s} {prot:6s} SE={np.median(P['se'])*100:.2f}pp "
          f"SD_calib={np.median(sd):.2f}pp ({time.time()-t0:.0f}s)", flush=True)
    return rows


def stage_calib(n_jobs=6):
    jobs = [(nm, prot) for nm in CALIB_DATASETS for prot in PROTOCOLS]
    if n_jobs == 1:
        res = [calib_block(*j) for j in jobs]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(calib_block)(*j) for j in jobs)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp17_calib_levels.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def markdown(mc, calib=None):
    cond = mc[mc["mode"] == "cond"].copy()
    cond["mismatch"] = cond.obs_mae / np.maximum(cond.sd_white, 1e-9)
    lines = ["# CRLB 与蒙特卡洛实测 SD 对照（exp17，全部实跑值）", "",
             "正文 §2.3 的噪声下界 SE = 1/(√N·CNR_eff) 在 21 个条件上的直接检验；"
             "white = 加白噪声，resid = 加训练残差整行，obs MAE = 实际外推误差。",
             "下表 CNR_eff|fit 由**本条件的拟合模型 + 该谱品牌**的 ‖d_⊥‖ 反算，"
             "口径与 Table 1（全参考集、品牌平均基质端元）不同，只用来核对 SE 的数量级。", "",
             "| 数据集 | 协议 | N | CNR_eff\\|fit | SE (pp) | SD white | ×比 | SD resid | ×比 | 实测 MAE | 失配倍数 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cond.itertuples():
        lines.append(f"| {r.dataset} | {r.protocol} | {int(r.N)} | {r.cnr_eff_med:.1f} | "
                     f"{r.se_med:.2f} | {r.sd_white:.2f} | {r.ratio_white:.2f} | "
                     f"{r.sd_resid:.2f} | {r.ratio_resid:.2f} | {r.obs_mae:.2f} | "
                     f"{r.mismatch:.1f}× |")
    lines += ["",
              f"白噪声比 SD/SE：中位 {cond.ratio_white.median():.2f}，"
              f"四分位 [{cond.ratio_white.quantile(.25):.2f}, {cond.ratio_white.quantile(.75):.2f}]；"
              f"残差扰动比：中位 {cond.ratio_resid.median():.2f}",
              f"±1.96·SE 区间在白噪声下的经验覆盖：中位 {cond.cov_white.median():.2f}，"
              f"在残差扰动下：中位 {cond.cov_resid.median():.2f}",
              f"实测外推误差 / 白噪声 SD（失配倍数）：中位 {cond.mismatch.median():.1f}×，"
              f"范围 [{cond.mismatch.min():.1f}×, {cond.mismatch.max():.1f}×]", ""]
    gate = {"D1": True, "D2": True, "D7d": True}
    cond["gate"] = [gate.get(n, False) for n in cond.dataset]
    ins, outs = cond[cond.gate], cond[~cond.gate]
    lines += [f"门内条件（D1/D2/D7d，n={len(ins)}）：SE 中位 {ins.se_med.median():.2f} pp，"
              f"SD_white/SE 中位 {ins.ratio_white.median():.2f}，失配倍数中位 "
              f"{ins.mismatch.median():.1f}×",
              f"门外条件（n={len(outs)}）：SE 中位 {outs.se_med.median():.2f} pp，"
              f"SD_white/SE 中位 {outs.ratio_white.median():.2f}，失配倍数中位 "
              f"{outs.mismatch.median():.1f}×", ""]
    def standardise(frame, sdcol):
        """把每个条件内的 SD 与杠杆各自除以该条件的中位数，再跨条件合并。"""
        f = frame.copy()
        key = ["dataset", "protocol"]
        f["sd_rel"] = f[sdcol] / f.groupby(key)[sdcol].transform("median")
        f["h_rel"] = f["h"] / f.groupby(key)["h"].transform("median")
        return f.dropna(subset=["sd_rel", "h_rel"])

    lv = mc[(mc["mode"] == "white") & mc["level"].notna()]
    w = standardise(lv, "sd")
    rho_w = sps.spearmanr(w.h_rel, w.sd_rel)
    lines += [f"白噪声 SD（条件内相对值，{len(w)} 个水平）与杠杆 h(c*) 的秩相关："
              f"ρ={rho_w.correlation:+.3f}, p={rho_w.pvalue:.3g} —— 端元固定时 CRLB "
              "不含杠杆项，故外推距离本身不放大噪声误差；杠杆只通过端元估计误差起作用。", ""]
    if calib is not None and len(calib):
        cal = calib.copy()
        merged = lv.groupby(["dataset", "protocol", "level"]).sd.median().rename("sd_white")
        cal = cal.merge(merged, left_on=["dataset", "protocol", "level"], right_index=True,
                        how="left")
        cal["tot"] = np.sqrt(cal.sd_white ** 2 + cal.sd_calib ** 2)
        c = standardise(cal, "sd_calib")
        rr = sps.spearmanr(c.h_rel, c.sd_rel)
        lo = np.log(np.maximum(c.h_rel.values, 1e-9))
        ls = np.log(np.maximum(c.sd_rel.values, 1e-9))
        slope, inter, rpear, pfit, se_ = sps.linregress(lo, ls)
        lines += ["## 端元估计误差（重新拟合扰动）", "",
                  "| 数据集 | 协议 | 测试水平 | n | h(c*) | SD 白噪声 | SD 端元重拟合 | 合成 SD |",
                  "|---|---|---|---|---|---|---|---|"]
        for r_ in cal.itertuples():
            lines.append(f"| {r_.dataset} | {r_.protocol} | {r_.level:g} | {int(r_.n)} | "
                         f"{r_.h:.2f} | {r_.sd_white:.2f} | {r_.sd_calib:.2f} | {r_.tot:.2f} |")
        lines += ["",
                  f"端元重拟合 SD（条件内相对值，{len(c)} 个水平）与杠杆的秩相关："
                  f"ρ={rr.correlation:+.3f}, p={rr.pvalue:.3g}；"
                  f"log-log 斜率 {slope:+.2f} ± {se_:.2f}"
                  f"（§2.4 的线性模型给出 SD ∝ √(1+h)，h≫1 端斜率趋近 0.5）"]
    io.open(os.path.join(OUT, "exp17_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    cond.to_csv(os.path.join(OUT, "exp17_conditions.csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "mc"
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"
    mc = pd.read_csv(os.path.join(OUT, "exp17_mc_long.csv")) \
        if os.path.exists(os.path.join(OUT, "exp17_mc_long.csv")) else stage_mc()
    calib_path = os.path.join(OUT, "exp17_calib_levels.csv")
    cal = None
    if stage == "calib":
        nj = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        cal = stage_calib(n_jobs=nj)
    elif stage == "report":
        if not os.path.exists(calib_path):
            raise SystemExit("no exp17_calib_levels.csv yet; run `calib` first")
        cal = pd.read_csv(calib_path)
    markdown(mc, cal)
