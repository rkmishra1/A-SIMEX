# Analytic sandwich for the extrapolated SIMEX estimator (Section 3.3 /
# Corollary 1): Var(sqrt(n)(beta_hat - beta0)) = C (V^data + V^sim/B) C',
# C = [c_j H_j^{-1}]. Exact port of asimex_lib.simex_sandwich, including
# the Gaussian tilt integrals. Arrays are indexed (i, p, q).

tilt_integrals <- function(b, P) {
  e <- as.numeric(exp(0.5 * drop(t(b) %*% P %*% b)))
  Pb <- as.numeric(P %*% b)
  list(e = e, a1 = Pb * e, a2 = (P + tcrossprod(Pb)) * e)
}

simex_sandwich <- function(Y, W, me_vars, sigma2_vec, lam_grid, curve,
                           Info_mean, beta_all, weights = NULL) {
  n <- nrow(W); p <- ncol(W); K <- length(lam_grid); B <- dim(beta_all)[2]
  cc <- if (is.null(weights)) extrap_weights(lam_grid, length(curve) - 1)
        else as.numeric(weights)
  Sig <- matrix(0, p, p)
  Sig[cbind(me_vars, me_vars)] <- sigma2_vec
  w <- W; y <- Y

  # data component: covariance of stacked conditional influence means
  psi <- array(NA_real_, c(n, K, p))
  for (j in seq_along(lam_grid)) {
    lam <- lam_grid[j]; bj <- curve[j, ]
    tilt <- as.numeric(exp(w %*% bj + 0.5 * lam * drop(t(bj) %*% Sig %*% bj)))
    psi[, j, ] <- y * w - tilt * (w + lam * matrix(Sig %*% bj, n, p, byrow = TRUE))
  }
  psi_flat <- matrix(psi, n, K * p)
  Vdata <- crossprod(psi_flat) / n
  Vdata <- Vdata - tcrossprod(colMeans(psi_flat))

  # simulation component: conditional covariance given (Y, w)
  Vsim <- matrix(0, K * p, K * p)
  if (any(sigma2_vec > 0) && K > 1) {
    P <- Sig
    bvec <- sqrt(lam_grid) * curve
    Xp <- array(rep(as.vector(w), times = p), c(n, p, p))   # (i,p,q)=w[i,p]
    Xq <- aperm(Xp, c(1, 3, 2))                              # (i,p,q)=w[i,q]
    S0 <- Xp * Xq                                            # w[i,p] w[i,q]
    for (j in 1:K) for (k in j:K) {
      lamj <- lam_grid[j]; lamk <- lam_grid[k]
      bj <- bvec[j, ]; bk <- bvec[k, ]; bsum <- bj + bk
      tj <- tilt_integrals(bj, P)
      tjk <- tilt_integrals(bsum, P)
      ES <- S0; ES[] <- ES[] +
        rep(sqrt(lamj * lamk) * P, each = n)
      # E[exp(bj'U) S_jk] and E[exp((bj+bk)'U) S_jk]
      g1 <- tj$e * S0 +
        sqrt(lamk) * (Xp * rep(tj$a1, each = n * p)) +
        sqrt(lamj) * (Xq * rep(tj$a1, each = n)) +
        sqrt(lamj * lamk) * rep(tj$a2, each = n)
      g1 <- g1 * rep(as.numeric(exp(w %*% curve[j, ])), each = p * p)
      g2 <- exp(0.5 * drop(t(bsum) %*% P %*% bsum)) * S0 +
        sqrt(lamk) * (Xp * rep(tjk$a1, each = n * p)) +
        sqrt(lamj) * (Xq * rep(tjk$a1, each = n)) +
        sqrt(lamj * lamk) * rep(tjk$a2, each = n)
      g2 <- g2 * rep(as.numeric(exp(w %*% (curve[j, ] + curve[k, ]))),
                     each = p * p)
      Emm <- (y * y) * ES - y * (g1 + aperm(g1, c(1, 3, 2))) + g2
      pj <- psi[, j, ]; pk <- psi[, k, ]
      pouter <- array(rep(as.vector(pj), times = p), c(n, p, p)) *
        aperm(array(rep(as.vector(pk), times = p), c(n, p, p)), c(1, 3, 2))
      crossm <- apply(Emm - pouter, c(2, 3), mean)
      Vsim[((j - 1) * p + 1):(j * p), ((k - 1) * p + 1):(k * p)] <- crossm
      if (k != j)
        Vsim[((k - 1) * p + 1):(k * p), ((j - 1) * p + 1):(j * p)] <- t(crossm)
    }
  }
  C <- matrix(0, p, K * p)
  for (j in 1:K) {
    Hj <- Info_mean[j, , ] / n
    C[, ((j - 1) * p + 1):(j * p)] <-
      cc[j] * safe_invert(Hj, scale = sum(diag(Hj)) / p)
  }
  V <- C %*% Vdata %*% t(C) + (1 / B) * C %*% Vsim %*% t(C)
  list(vcov = V, v_data = C %*% Vdata %*% t(C), v_sim = C %*% Vsim %*% t(C))
}

# Pairs bootstrap standard errors: redraw observations and simulated
# errors NB times, refit the two-stage estimator.
bootstrap_se <- function(Y, W, me_vars, sigma2_vec, lam_grid, degree,
                         NB = 200, seed = NULL) {
  if (!is.null(seed)) set.seed(seed + 1L)
  n <- nrow(W); p <- ncol(W)
  est <- matrix(NA_real_, NB, p)
  for (b in seq_len(NB)) {
    idx <- sample.int(n, n, replace = TRUE)
    Yb <- Y[idx]; Wb <- W[idx, , drop = FALSE]
    bn <- poisson_irls(Yb, Wb)
    sc <- simex_curve(Yb, Wb, me_vars, sigma2_vec, lam_grid, B = 50,
                      seed = NULL, beta_init = bn)
    d <- combined_weights(lam_grid, degree)
    est[b, ] <- as.numeric(d %*% sc$curve)
  }
  apply(est, 2, sd, na.rm = TRUE)
}
