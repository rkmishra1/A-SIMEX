# Nakamura (1990) corrected-score estimator and regression calibration,
# the competitors used in the paper's comparisons.

nakamura_corrected_score <- function(Y, W, sigma2_vec, me_vars,
                                     beta_init = NULL, tol = 1e-10,
                                     max_iter = 80) {
  n <- nrow(W); p <- ncol(W)
  Sig <- matrix(0, p, p); Sig[cbind(me_vars, me_vars)] <- sigma2_vec
  if (is.null(beta_init)) beta_init <- poisson_irls(Y, W)
  beta <- as.numeric(beta_init)
  score_block <- function(beta) {
    u <- as.numeric(W %*% beta) - 0.5 * drop(t(beta) %*% Sig %*% beta)
    v <- W - matrix(Sig %*% beta, n, p, byrow = TRUE)
    e <- exp(pmin(pmax(u, -500), 300))
    U <- Y * W - e * v
    A <- matrix(0, p, p)
    for (i in 1:n) A <- A + e[i] * (tcrossprod(v[i, ]) - Sig)
    list(U = colSums(U), Umat = U, A = A)
  }
  # Damped Newton on the corrected-score moment U*(beta) = 0. A is the
  # Jacobian of U*, which is not a gradient field, so steps are halved
  # until the moment norm decreases and convergence is declared on
  # ||U*||, not on step size (plain Newton can cycle).
  sb <- score_block(beta)
  Un <- sqrt(sum(sb$U^2))
  for (it in seq_len(max_iter)) {
    if (!all(is.finite(beta)) || max(abs(beta)) > 1e4)
      stop("corrected score diverged")
    if (Un < 1e-8) break
    step <- safe_solve(sb$A, sb$U, scale = sum(diag(sb$A)) / p)
    lambda <- 1; improved <- FALSE
    for (k in seq_len(30)) {
      cand <- beta + lambda * step
      if (all(is.finite(cand)) && max(abs(cand)) <= 1e4) {
        cand_block <- score_block(cand)
        Un2 <- sqrt(sum(cand_block$U^2))
        if (is.finite(Un2) && Un2 < Un) { improved <- TRUE; break }
      }
      lambda <- lambda / 2
    }
    if (!improved) break
    beta <- cand; sb <- cand_block; Un <- Un2
  }
  sb <- score_block(beta)
  Ainv <- safe_invert(sb$A, scale = sum(diag(sb$A)) / p)
  var <- Ainv %*% (crossprod(sb$Umat)) %*% Ainv
  list(beta = beta, vcov = var)
}

regression_calibration <- function(Y, W, sigma2_vec, me_vars) {
  n <- nrow(W); p <- ncol(W)
  Xc <- W
  for (j in seq_along(me_vars)) {
    col <- me_vars[j]; s2 <- sigma2_vec[j]
    Xc[, col] <- W[, col] - s2 * (W[, col] - mean(W[, col])) / max(var(W[, col]), 1e-12)
  }
  beta <- poisson_irls(Y, Xc)
  list(beta = beta, vcov = poisson_sandwich(Y, Xc, beta))
}
