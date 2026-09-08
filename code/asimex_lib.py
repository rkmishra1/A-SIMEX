"""
Adaptive SIMEX simulation library for Poisson regression with classical
measurement error.

Implements:
  - batched Poisson IRLS (vectorised across SIMEX replicates)
  - standard SIMEX (grid {0,0.5,1,1.5,2}, quadratic extrapolant, B=100)
  - Adaptive SIMEX (CV over (lambda_max, degree) via held-out Poisson
    deviance on the observed scale, 1-SE parsimony rule; ridge second
    stage of Section 3.2 of the manuscript)
  - Nakamura (1990) corrected-score estimator (Gaussian error, known var)
  - regression calibration
  - analytic sandwich variances for SIMEX-family estimators
    (data component + O(1/B) simulation component; see corrected-proofs
    document, Theorem 2 / Corollary 1)

All randomness flows from numpy SeedSequence so every Monte Carlo
repetition is reproducible.
"""
from __future__ import annotations

import time

import numpy as np



def _safe_solve(A, b, scale=1.0):
    """Solve with escalating ridge; fall back to least squares."""
    for ridge in (0.0, 1e-10, 1e-6, 1e-3):
        try:
            return np.linalg.solve(
                A + ridge * scale * np.eye(A.shape[-1]), b)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _safe_inv(A, scale=1.0):
    for ridge in (0.0, 1e-10, 1e-6, 1e-3):
        try:
            return np.linalg.inv(A + ridge * scale * np.eye(A.shape[-1]))
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(A)


# ---------------------------------------------------------------------------
# Batched Poisson IRLS
# ---------------------------------------------------------------------------

def poisson_irls_batched(Y, W, beta_init=None, tol=1e-9, max_iter=80):
    """Batched IRLS for Poisson GLMs with log link.

    Y : (n,) response, shared across replicates.
    W : (B, n, p) pseudo-covariates (one row-block per SIMEX replicate).
    Returns beta (B, p), converged (B,) bool.
    """
    B, n, p = W.shape
    Yb = np.broadcast_to(Y, (B, n))
    if beta_init is None:
        beta = np.zeros((B, p))
    else:
        beta = np.broadcast_to(beta_init, (B, p)).copy()

    converged = np.zeros(B, dtype=bool)

    def _deviance(b):
        eta = np.einsum("bnp,bp->bn", W, b)
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        d = np.where(Yb > 0, Yb * np.log(np.maximum(Yb, 1e-10) / mu), 0.0)
        return 2.0 * (d - (Yb - mu)).sum(axis=1)

    dev = _deviance(beta)
    for it in range(max_iter):
        eta = np.einsum("bnp,bp->bn", W, beta)
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        # working response z = eta + (y - mu)/mu ; weights mu
        z = eta + (Yb - mu) / np.maximum(mu, 1e-10)
        z = np.clip(z, -50.0, 50.0)
        Wm = mu[..., None] * W                       # (B, n, p) weighted
        XtWX = np.einsum("bnp,bnq->bpq", Wm, W)
        XtWz = np.einsum("bnp,bn->bp", Wm, z)
        # tiny ridge for numerical safety
        XtWX[:, np.arange(p), np.arange(p)] += 1e-10
        beta_new = _safe_solve(XtWX, XtWz[..., None],
                               scale=np.mean(np.diagonal(XtWX, axis1=1, axis2=2)
                             / p))[..., 0]
        # step halving to guarantee deviance non-increase
        dev_new = _deviance(beta_new)
        halve = dev_new > dev - 1e-12
        for _ in range(5):
            if not halve.any():
                break
            beta_new[halve] = 0.5 * (beta_new[halve] + beta[halve])
            dev_new = _deviance(beta_new)
            halve = dev_new > dev - 1e-12
        step = np.max(np.abs(beta_new - beta), axis=1)
        improved = dev_new < dev - 1e-12
        beta = np.where(improved[:, None], beta_new, beta)
        dev = np.where(improved, dev_new, dev)
        # systems whose step cannot improve are stalled: stop iterating them
        converged = (step < tol) | (~improved)
        if converged.all():
            break
    return beta, converged


def poisson_irls(Y, X, beta_init=None, tol=1e-9, max_iter=80):
    """Single-fit convenience wrapper."""
    beta, _ = poisson_irls_batched(
        Y, X[None, :, :], beta_init=beta_init, tol=tol, max_iter=max_iter)
    return beta[0]


def poisson_sandwich(Y, X, beta):
    """Robust (model-assisted) sandwich variance for a Poisson GLM fit."""
    mu = np.exp(np.clip(X @ beta, -30.0, 30.0))
    XtWX = X.T @ (mu[:, None] * X)
    XtWX[np.diag_indices_from(XtWX)] += 1e-10
    score = X * (Y - mu)[:, None]
    B = score.T @ score
    Hinv = np.linalg.inv(XtWX)
    return Hinv @ B @ Hinv


# ---------------------------------------------------------------------------
# SIMEX machinery
# ---------------------------------------------------------------------------

def _simex_batch(Y, W, me_cols, sigma2_vec, lam, noise, beta_init=None):
    """One grid point with PRE-DRAWN noise (shared across the curve).

    noise: (reps, n, len(me_cols)) standard-normal draws, to be scaled by
    sqrt(sigma2) and sqrt(lam). Fitting one lambda of the curve for all
    replicates. Returns beta_reps (reps, p) and per-replicate information
    matrices I_b (reps, p, p) at the fitted pseudo-parameter.
    """
    n, p = W.shape
    reps = noise.shape[0]
    sc = np.sqrt(sigma2_vec)[None, None, :]
    Wp = np.broadcast_to(W, (reps, n, p)).copy()
    Wp[:, :, me_cols] += noise * sc * np.sqrt(lam)
    beta, _ = poisson_irls_batched(Y, Wp, beta_init=beta_init)
    mu = np.exp(np.clip(np.einsum("rnp,rp->rn", Wp, beta), -30, 30))
    Wm = mu[..., None] * Wp
    Info = np.einsum("rnp,rnq->rpq", Wm, Wp)
    Info[:, np.arange(p), np.arange(p)] += 1e-10 * np.maximum(
        1.0, np.trace(Info, axis1=1, axis2=2)[:, None] / p)
    return beta, Info


def simex_curve(Y, W, me_cols, sigma2_vec, lam_grid, B, rng, beta_init=None):
    """Mean SIMEX curve with noise shared across lambda (Algorithm 1).

    Returns curve (K, p), Info_mean (K, p, p), beta_all (K, B, p).
    """
    K, p = len(lam_grid), W.shape[1]
    noise = rng.standard_normal((B, W.shape[0], len(me_cols)))
    curve = np.empty((K, p))
    info = np.empty((K, p, p))
    beta_all = np.empty((K, B, p))
    warm = beta_init
    for k, lam in enumerate(lam_grid):
        beta, Info = _simex_batch(Y, W, me_cols, sigma2_vec, lam, noise,
                                  beta_init=warm)
        curve[k] = beta.mean(axis=0)
        info[k] = Info.mean(axis=0)
        beta_all[k] = beta
        warm = curve[k]                      # warm start along the curve
    return curve, info, beta_all


def extrap_weights(lam_grid, degree):
    """LS weights c_j such that p(-1) = sum_j c_j * curve_j."""
    Phi = np.vander(lam_grid, degree + 1, increasing=True)   # (K, d+1)
    phi_m1 = np.array([(-1.0) ** k for k in range(degree + 1)])
    return phi_m1 @ np.linalg.pinv(Phi)                       # (K,)


def extrapolate(lam_grid, curve, degree):
    """Polynomial extrapolation to lambda = -1."""
    c = extrap_weights(lam_grid, degree)
    return c @ curve



def combined_weights(lam_grid, degree, gamma=0.01, extra_degree=2):
    """Total weights of the two-stage estimator: primary extrapolation
    plus the ridge stage-2 correction (both linear in the curve)."""
    K = len(lam_grid)
    Phi1 = np.vander(lam_grid, degree + 1, increasing=True)
    c_prim = extrap_weights(lam_grid, degree)
    d2 = degree + extra_degree
    Phi2 = np.vander(lam_grid, d2 + 1, increasing=True)
    phi_m1 = np.array([(-1.0) ** k for k in range(d2 + 1)])
    A = Phi2.T @ Phi2 + gamma * np.eye(d2 + 1)
    # r(-1) = phi(-1)' A^{-1} Phi2' (I - P1) curve, P1 = primary projection
    w_ridge = phi_m1 @ np.linalg.solve(A, Phi2.T) @ (
        np.eye(K) - Phi1 @ np.linalg.pinv(Phi1))
    return c_prim + w_ridge


def ridge_second_stage(lam_grid, curve, degree_primary, gamma=0.01,
                       extra_degree=2):
    """Stage-2 ridge correction of Section 3.2 of the manuscript.

    Fits a ridge polynomial of degree `degree_primary + extra_degree` to
    the residuals of the primary fit on the observed range and evaluates
    the correction at lambda = -1.
    """
    d2 = degree_primary + extra_degree
    Phi = np.vander(lam_grid, d2 + 1, increasing=True)        # (K, d2+1)
    P1 = np.vander(lam_grid, degree_primary + 1, increasing=True)
    res = curve - P1 @ np.linalg.pinv(P1) @ curve
    A = Phi.T @ Phi + gamma * np.eye(d2 + 1)
    coef = np.linalg.solve(A, Phi.T @ res)
    phi_m1 = np.array([(-1.0) ** k for k in range(d2 + 1)])
    return phi_m1 @ coef


# ---------------------------------------------------------------------------
# Analytic sandwich variance for SIMEX-family estimators
# (corrected-proofs document, Theorem 2 / Corollary 1)
# ---------------------------------------------------------------------------

def _tilt(b, P):
    """Gaussian tilt integrals for E[exp(b'U) {1, U, UU'}], U~N(0,P)."""
    e = np.exp(0.5 * b @ P @ b)
    Pb = P @ b
    a1 = Pb * e
    a2 = (P + np.outer(Pb, Pb)) * e
    return e, a1, a2


def simex_sandwich(Y, W, me_cols, sigma2_vec, lam_grid, curve, Info_mean,
                   beta_all, degree, weights=None):
    """Exact analytic sandwich for a SIMEX-family extrapolated estimator.

    Var(sqrt(n)(beta_hat - beta0)) ~= C V^data C' + (1/B) C V^sim C',
    C = [c_0 H_0^{-1}, ..., c_{K-1} H_{K-1}^{-1}]  (p x Kp).
    """
    n, p = W.shape
    K = len(lam_grid)
    B = beta_all.shape[1]
    c = extrap_weights(lam_grid, degree) if weights is None else weights
    Sig = np.zeros((p, p))               # error cov embedded in full space
    me_cols = list(me_cols)
    Sig[me_cols, me_cols] = sigma2_vec

    w = W  # (n, p)
    y = Y
    # ---- data component: sample covariance of stacked psi_i ----
    psi = np.empty((n, K, p))
    for j, lam in enumerate(lam_grid):
        bj = curve[j]
        tilt = np.exp(w @ bj + 0.5 * lam * (bj @ Sig @ bj))
        psi[:, j, :] = y[:, None] * w - tilt[:, None] * (
            w + lam * (Sig @ bj)[None, :])
    psi_flat = psi.reshape(n, K * p)
    Vdata = psi_flat.T @ psi_flat / n
    mflat = psi_flat.mean(axis=0)
    Vdata -= np.outer(mflat, mflat)

    # ---- simulation component: conditional covariance given (y, w) ----
    Vsim = np.zeros((K * p, K * p))
    if np.any(np.asarray(sigma2_vec) > 0) and K > 1:
        P = Sig.copy()                   # covariance of U (full space)
        bvec = np.sqrt(lam_grid)[:, None] * curve            # (K, p)
        for j in range(K):
            for k in range(j, K):
                bj, bk = bvec[j], bvec[k]
                lamj, lamk = lam_grid[j], lam_grid[k]
                bsum = bj + bk
                e0j, a1j, a2j = _tilt(bj, P)
                _, a1jk, a2jk = _tilt(bsum, P)
                S0 = np.einsum("np,nq->npq", w, w)
                ES = S0 + np.sqrt(lamj * lamk) * P[None, :, :]
                # E[exp(b_j'U) S_jk], e^{beta_j'w} factored out
                g1 = (e0j * S0
                      + np.sqrt(lamk) * np.einsum("np,q->npq", w, a1j)
                      + np.sqrt(lamj) * np.einsum("nq,p->npq", w, a1j)
                      + np.sqrt(lamj * lamk) * a2j[None, :, :])
                g1 *= np.exp(w @ curve[j])[:, None, None]
                # E[exp((b_j+b_k)'U) S_jk], e^{(beta_j+beta_k)'w} factored out
                g2 = (np.exp(0.5 * bsum @ P @ bsum) * S0
                      + np.sqrt(lamk) * np.einsum("np,q->npq", w, a1jk)
                      + np.sqrt(lamj) * np.einsum("nq,p->npq", w, a1jk)
                      + np.sqrt(lamj * lamk) * a2jk[None, :, :])
                g2 *= np.exp(w @ (curve[j] + curve[k]))[:, None, None]
                Emm = ((y * y)[:, None, None] * ES
                       - y[:, None, None] * (g1 + g1.transpose(0, 2, 1))
                       + g2)
                cross = Emm - np.einsum("np,nq->npq", psi[:, j, :],
                                        psi[:, k, :])
                Vsim[j * p:(j + 1) * p, k * p:(k + 1) * p] = \
                    cross.mean(axis=0)
                if k != j:
                    Vsim[k * p:(k + 1) * p, j * p:(j + 1) * p] = \
                        cross.mean(axis=0).T

    C = np.empty((p, K * p))
    for j in range(K):
        H_j = Info_mean[j] / n          # per-observation Hessian
        C[:, j * p:(j + 1) * p] = c[j] * _safe_inv(
            H_j, scale=np.trace(H_j) / H_j.shape[0])
    V = (C @ Vdata @ C.T) + (1.0 / B) * (C @ Vsim @ C.T)
    return V, {"v_data": C @ Vdata @ C.T, "v_sim": C @ Vsim @ C.T}


# ---------------------------------------------------------------------------
# Competitors
# ---------------------------------------------------------------------------

def nakamura_corrected_score(Y, W, sigma2_vec, me_cols, beta_init=None):
    """Nakamura (1990) corrected-score estimator for Poisson regression.

    S*(beta) = sum_i [ y_i w_i - exp(u_i)(w_i - Sig beta) ],
    u_i = beta'w_i - 0.5 beta' Sig beta,   Sig = diag(sigma2) on ME cols.
    Sandwich variance via A_i = exp(u_i)(v_i v_i' - Sig).
    """
    n, p = W.shape
    Sig = np.zeros((p, p))
    Sig[me_cols, me_cols] = sigma2_vec
    if beta_init is None:
        beta_init = poisson_irls(Y, W)
    beta = beta_init.copy()
    for _ in range(80):
        u = W @ beta - 0.5 * beta @ Sig @ beta
        v = W - (Sig @ beta)[None, :]
        e = np.exp(np.clip(u, -500, 300))
        U = Y[:, None] * W - e[:, None] * v                       # score (n,p)
        A = (e[:, None, None] * (np.einsum("np,nq->npq", v, v)
                                 - Sig[None, :, :])).sum(axis=0)
        step = _safe_solve(A, U.sum(axis=0), scale=np.trace(A) / p)
        beta = beta + step
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e4:
            raise FloatingPointError("corrected score diverged")
        if np.max(np.abs(step)) < 1e-10:
            break
    u = W @ beta - 0.5 * beta @ Sig @ beta
    v = W - (Sig @ beta)[None, :]
    e = np.exp(np.clip(u, -500, 300))
    U = Y[:, None] * W - e[:, None] * v
    A = (e[:, None, None] * (np.einsum("np,nq->npq", v, v)
                             - Sig[None, :, :])).sum(axis=0)
    Ainv = _safe_inv(A, scale=np.trace(A) / p)
    var = Ainv @ (U.T @ U) @ Ainv
    return beta, var




def regression_calibration(Y, W, sigma2_vec, me_cols):
    """Regression calibration: plug in E[X|W] under Gaussian assumptions."""
    Xc = W.copy()
    for j, col in enumerate(me_cols):
        s2 = sigma2_vec[j]
        vw = np.var(W[:, col])
        Xc[:, col] = W[:, col] - s2 * (W[:, col] - W[:, col].mean()) / max(
            vw, 1e-12)
    beta = poisson_irls(Y, Xc)
    var = poisson_sandwich(Y, Xc, beta)
    return beta, var


# ---------------------------------------------------------------------------
# Method wrappers used by the Monte Carlo driver
# ---------------------------------------------------------------------------

STD_LAM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
LAM_MAX_SEQ = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
DEG_SEQ = [1, 2, 3]


def cv_select(Y, W, me_cols, sigma2_vec, rng, folds=5, B_cv=25,
              lam_max_seq=LAM_MAX_SEQ, deg_seq=DEG_SEQ):
    """CV over (lambda_max, degree) with held-out Poisson deviance.

    Uses the prefix property of SIMEX: within a fold, ONE curve is fitted
    on the union grid {0, 0.5, ..., max(lam_max_seq)} and each candidate
    lambda_max is evaluated by extrapolating the corresponding prefix.
    Noise draws are shared across lambda_max within a fold.
    """
    n = len(Y)
    idx = rng.permutation(n)
    fold_id = np.array_split(idx, folds)
    union = np.arange(0.0, lam_max_seq.max() + 1e-9, 0.5)
    cv_scores = np.full((len(lam_max_seq), len(deg_seq)), np.nan)
    fold_scores = np.full((folds, len(lam_max_seq), len(deg_seq)), np.nan)

    for k, test_idx in enumerate(fold_id):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Ytr, Wtr = Y[train_mask], W[train_mask]
        Yte, Wte = Y[test_idx], W[test_idx]
        curve, _, _ = simex_curve(Ytr, Wtr, me_cols, sigma2_vec, union,
                                  B_cv, rng)
        for a, lm in enumerate(lam_max_seq):
            sub = union[union <= lm + 1e-9]
            cpart = curve[:len(sub)]
            for di, deg in enumerate(deg_seq):
                beta_ex = extrapolate(sub, cpart, deg)
                mu = np.exp(np.clip(Wte @ beta_ex, -30, 30))
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.where(Yte > 0,
                                    Yte * np.log(np.maximum(Yte, 1e-10) / mu),
                                    0.0)
                fold_scores[k, a, di] = 2.0 * np.sum(term - (Yte - mu))
    cv_scores = fold_scores.mean(axis=0)
    se = fold_scores.std(axis=0, ddof=1) / np.sqrt(folds)
    best = np.unravel_index(np.argmin(cv_scores), cv_scores.shape)
    thresh = cv_scores[best] + se[best]
    ok = cv_scores <= thresh
    order = sorted(
        [(deg_seq[di], lam_max_seq[a], a, di)
         for a in range(cv_scores.shape[0])
         for di in range(cv_scores.shape[1]) if ok[a, di]])
    _, _, a_1se, di_1se = order[0]
    a_am, di_am = int(best[0]), int(best[1])
    return (lam_max_seq[a_1se], deg_seq[di_1se],
            lam_max_seq[a_am], deg_seq[di_am], cv_scores, se)


def run_methods(Y, W, sigma2_vec, me_cols, rng, B=100, B_cv=25,
                true_beta=None, use_adaptive=True, adaptive_override=None,
                only=None):
    """Run (a subset of) the method suite for one Monte Carlo repetition.

    only : str or None -- if given, run just that method ("naive", "rc",
    "cs", "std_simex", "adaptive"). Each call is independent; failures
    propagate to the caller for per-method fault isolation.
    Returns dict: method -> (beta, se) plus per-method timings.
    """
    n, p = W.shape
    out = {}
    want = (lambda m: True) if only is None else (lambda m: m == only)

    if want("naive"):
        t0 = time.perf_counter()
        b_naive = poisson_irls(Y, W)
        var_naive = poisson_sandwich(Y, W, b_naive)
        out["naive"] = (b_naive, np.sqrt(np.diag(var_naive)))
        out["naive_time"] = time.perf_counter() - t0
    else:
        b_naive = poisson_irls(Y, W)

    if want("rc"):
        t0 = time.perf_counter()
        b_rc, var_rc = regression_calibration(Y, W, sigma2_vec, me_cols)
        out["rc"] = (b_rc, np.sqrt(np.diag(var_rc)))
        out["rc_time"] = time.perf_counter() - t0

    if want("cs"):
        t0 = time.perf_counter()
        b_cs, var_cs = nakamura_corrected_score(Y, W, sigma2_vec, me_cols,
                                               beta_init=b_naive)
        out["cs"] = (b_cs, np.sqrt(np.diag(var_cs)))
        out["cs_time"] = time.perf_counter() - t0

    if want("std_simex"):
        t0 = time.perf_counter()
        curve, info, beta_all = simex_curve(Y, W, me_cols, sigma2_vec,
                                            STD_LAM, B, rng,
                                            beta_init=b_naive)
        d_std = combined_weights(STD_LAM, 2)
        b_std = d_std @ curve
        v_std, _ = simex_sandwich(Y, W, me_cols, sigma2_vec, STD_LAM, curve,
                                  info, beta_all, 2, weights=d_std)
        out["std_simex"] = (b_std, np.sqrt(np.diag(v_std) / n))
        out["std_simex_time"] = time.perf_counter() - t0

    if use_adaptive and want("adaptive"):
        t0 = time.perf_counter()
        if adaptive_override is None:
            lm_1se, dg_1se, lm_am, dg_am, cv_scores, cv_se = cv_select(
                Y, W, me_cols, sigma2_vec, rng, B_cv=B_cv)
        else:
            lm_1se, dg_1se = adaptive_override
            lm_am, dg_am = adaptive_override
        lm_top = max(lm_1se, lm_am)
        grid = np.arange(0.0, lm_top + 1e-9, 0.5)
        curve, info, beta_all = simex_curve(Y, W, me_cols, sigma2_vec,
                                            grid, B, rng,
                                            beta_init=b_naive)
        # 1-SE (parsimony) variant -- the manuscript's rule
        k1 = int(np.sum(grid <= lm_1se + 1e-9))
        g1 = grid[:k1]
        d1 = combined_weights(g1, dg_1se)
        b1 = d1 @ curve[:k1]
        v1, _ = simex_sandwich(Y, W, me_cols, sigma2_vec, g1, curve[:k1],
                               info[:k1], beta_all[:k1], dg_1se,
                               weights=d1)
        out["adaptive"] = (b1, np.sqrt(np.diag(v1) / n))
        # argmin variant (diagnostic: no parsimony rule)
        if (lm_am, dg_am) != (lm_1se, dg_1se):
            ka = int(np.sum(grid <= lm_am + 1e-9))
            ga = grid[:ka]
            da = combined_weights(ga, dg_am)
            ba = da @ curve[:ka]
            va, _ = simex_sandwich(Y, W, me_cols, sigma2_vec, ga,
                                   curve[:ka], info[:ka], beta_all[:ka],
                                   dg_am, weights=da)
            out["adaptive_am"] = (ba, np.sqrt(np.diag(va) / n))
        else:
            out["adaptive_am"] = (b1, np.sqrt(np.diag(v1) / n))
        out["adaptive_time"] = time.perf_counter() - t0
        out["adaptive_pick"] = (lm_1se, dg_1se, lm_am, dg_am)

    return out


def oracle_fit(Y, X):
    b = poisson_irls(Y, X)
    var = poisson_sandwich(Y, X, b)
    return b, np.sqrt(np.diag(var))
