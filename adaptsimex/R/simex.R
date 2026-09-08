# SIMEX curve, extrapolation weights, and the two-stage ridge correction.
# Faithful port of asimex_lib (Python): noise is drawn once per replicate
# and shared across lambda (Algorithm 1 of the manuscript).

vander <- function(lam, degree) outer(lam, 0:degree, "^")

extrap_weights <- function(lam_grid, degree) {
  Phi <- vander(lam_grid, degree)
  phi_m1 <- (-1)^(0:degree)
  as.numeric(phi_m1 %*% MASS_inv(Phi))
}

# Combined primary + ridge (Section 3.2) extrapolation weights.
combined_weights <- function(lam_grid, degree_primary, gamma = 0.01,
                             extra_degree = 2) {
  K <- length(lam_grid)
  c0 <- extrap_weights(lam_grid, degree_primary)
  d2 <- degree_primary + extra_degree
  Phi_p <- vander(lam_grid, degree_primary)
  Phi_r <- vander(lam_grid, d2)
  Pproj <- Phi_p %*% MASS_inv(Phi_p)
  R <- diag(K) - Pproj
  A <- crossprod(Phi_r, R %*% Phi_r) + gamma * diag(d2 + 1)
  Wm <- safe_solve(A, crossprod(Phi_r, R), scale = mean(diag(A)))
  phi_m1 <- (-1)^(0:d2)
  c0 + as.numeric(phi_m1 %*% Wm)
}

# One SIMEX curve: returns mean curve, mean replicate information
# matrices, and all replicate estimates.
simex_curve <- function(Y, W, me_vars, sigma2_vec, lam_grid, B = 100,
                        seed = NULL, beta_init = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(W); p <- ncol(W)
  noise <- matrix(rnorm(B * n * length(me_vars)), B, n * length(me_vars))
  curve <- matrix(NA_real_, length(lam_grid), p)
  info <- array(NA_real_, c(length(lam_grid), p, p))
  beta_all <- array(NA_real_, c(length(lam_grid), B, p))
  warm <- beta_init
  for (k in seq_along(lam_grid)) {
    lam <- lam_grid[k]
    Wp <- array(rep(W, B), c(n, p, B))
    Wp <- aperm(Wp, c(3, 1, 2))
    for (j in seq_along(me_vars)) {
      col <- me_vars[j]
      add <- noise[, ((j - 1) * n + 1):(j * n)] *
        sqrt(sigma2_vec[j] * lam)              # (B, n)
      Wp[, , col] <- Wp[, , col] + add
    }
    betas <- matrix(NA_real_, B, p)
    infos <- array(NA_real_, c(B, p, p))
    for (b in seq_len(B)) {
      Wb <- Wp[b, , ]
      betas[b, ] <- poisson_irls(Y, Wb, beta_init = warm)
      mu <- as.numeric(exp(pmin(pmax(Wb %*% betas[b, ], -30), 30)))
      infos[b, , ] <- crossprod(mu * Wb, Wb) +
        1e-10 * max(1, sum(diag(crossprod(mu * Wb, Wb))) / p) * diag(p)
    }
    curve[k, ] <- colMeans(betas)
    info[k, , ] <- colMeans(infos)
    beta_all[k, , ] <- betas
    warm <- curve[k, ]
  }
  list(curve = curve, info = info, beta_all = beta_all)
}

# Evaluate a degree-d polynomial extrapolation of the curve at lambda = -1.
extrapolate <- function(lam_grid, curve, degree) {
  as.numeric(extrap_weights(lam_grid, degree) %*% curve)
}

# Ridge second stage evaluated at -1 (kept for API parity; estimation
# uses combined_weights so point estimate and variance share weights).
ridge_second_stage <- function(lam_grid, curve, degree_primary,
                               gamma = 0.01, extra_degree = 2) {
  d2 <- degree_primary + extra_degree
  Phi_p <- vander(lam_grid, degree_primary)
  res <- curve - Phi_p %*% qr.solve(Phi_p, curve)
  Phi_r <- vander(lam_grid, d2)
  A <- crossprod(Phi_r) + gamma * diag(d2 + 1)
  coef <- safe_solve(A, crossprod(Phi_r, res), scale = mean(diag(A)))
  as.numeric((-1)^(0:d2) %*% coef)
}
