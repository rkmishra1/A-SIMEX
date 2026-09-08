# Batched-equivalent IRLS for Poisson regression with log link.
# Port of asimex_lib.poisson_irls (Python); includes step halving so the
# deviance is non-increasing, matching the simulation pipeline.

poisson_irls <- function(Y, X, beta_init = NULL, tol = 1e-7, max_iter = 50) {
  n <- nrow(X); p <- ncol(X)
  beta <- if (is.null(beta_init)) rep(0, p) else as.numeric(beta_init)
  dev <- function(b) {
    eta <- as.numeric(X %*% b)
    mu <- exp(pmin(pmax(eta, -30), 30))
    d <- ifelse(Y > 0, Y * log(pmax(Y, 1e-10) / mu), 0)
    2 * sum(d - (Y - mu))
  }
  cur <- dev(beta)
  for (it in seq_len(max_iter)) {
    eta <- as.numeric(X %*% beta)
    mu <- exp(pmin(pmax(eta, -30), 30))
    z <- eta + (Y - mu) / pmax(mu, 1e-10)
    z <- pmin(pmax(z, -50), 50)
    Wm <- mu * X
    XtWX <- crossprod(Wm, X)
    XtWz <- crossprod(Wm, z)
    scale <- max(1e-12, mean(diag(XtWX)) / p)
    beta_new <- safe_solve(XtWX, XtWz, scale = scale)
    dev_new <- dev(beta_new)
    halve <- dev_new > cur - 1e-12
    tries <- 0
    while (any(halve) && tries < 5) {
      beta_new[halve] <- 0.5 * (beta_new[halve] + beta[halve])
      dev_new <- dev(beta_new)
      halve <- dev_new > cur - 1e-12
      tries <- tries + 1
    }
    step <- max(abs(beta_new - beta))
    improved <- dev_new < cur - 1e-12
    if (improved) { beta <- beta_new; cur <- dev_new } else break
    if (step < tol) break
  }
  beta
}

# Solve with escalating ridge; least-squares fallback (mirrors _safe_solve).
safe_solve <- function(A, b, scale = 1) {
  for (ridge in c(0, 1e-10, 1e-6, 1e-3)) {
    Ap <- A + ridge * scale * diag(ncol(A))
    out <- tryCatch(solve(Ap, b), error = function(e) NULL)
    if (!is.null(out)) return(out)
  }
  qr.solve(A, b)
}

safe_invert <- function(A, scale = 1) {
  for (ridge in c(0, 1e-10, 1e-6, 1e-3)) {
    Ap <- A + ridge * scale * diag(ncol(A))
    out <- tryCatch(solve(Ap), error = function(e) NULL)
    if (!is.null(out)) return(out)
  }
  MASS_inv(A)
}

MASS_inv <- function(A) {
  s <- svd(A)
  keep <- s$d > max(s$d) * .Machine$double.eps
  s$v %*% (t(s$u[, keep, drop = FALSE]) / s$d[keep])
}

# Robust sandwich for a Poisson GLM fit.
poisson_sandwich <- function(Y, X, beta) {
  mu <- as.numeric(exp(pmin(pmax(X %*% beta, -30), 30)))
  H <- crossprod(mu * X, X)
  score <- X * (Y - mu)
  Bm <- crossprod(score)
  Ainv <- safe_invert(H, scale = mean(diag(H)))
  Ainv %*% Bm %*% Ainv
}
