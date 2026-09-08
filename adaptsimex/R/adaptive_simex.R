# Main driver: formula interface, CV selection, two-stage extrapolation,
# analytic sandwich (default) or pairs-bootstrap variance.

adaptive_simex <- function(formula, data, me_vars, sigma_error,
                           lambda_max_seq = seq(0.5, 2.5, by = 0.5),
                           poly_degree_seq = 1:3,
                           n_simex = 100, n_cv_folds = 5, n_cv_simex = 25,
                           se_method = c("sandwich", "bootstrap"),
                           n_boot = 200, seed = NULL, verbose = TRUE) {
  se_method <- match.arg(se_method)
  if (!is.null(seed)) set.seed(seed)
  mf <- model.frame(formula, data, na.action = na.omit)
  Y <- model.response(mf)
  if (!all(Y >= 0) || any(Y != floor(Y)))
    stop("response must be a non-negative count")
  X <- model.matrix(formula, mf)
  cn <- colnames(X)
  me_idx <- match(me_vars, cn)
  if (any(is.na(me_idx)))
    stop("me_vars not all found among predictors: ",
         paste(me_vars[is.na(me_idx)], collapse = ", "))
  sigma2_vec <- sigma_error^2
  if (length(sigma2_vec) != length(me_vars))
    stop("sigma_error must have one entry per me_var")
  n <- nrow(X); p <- ncol(X)

  if (verbose) cat("Cross-validating hyperparameters...\n")
  cv <- cv_select(Y, X, me_idx, sigma2_vec, folds = n_cv_folds,
                  B_cv = n_cv_simex, lam_max_seq = lambda_max_seq,
                  deg_seq = poly_degree_seq, seed = NULL)
  lm_star <- cv$lambda_max_1se; d_star <- cv$degree_1se
  if (verbose) cat("Selected lambda_max =", lm_star,
                   "degree =", d_star, "\n")

  grid <- seq(0, lm_star, by = 0.5)
  sc <- simex_curve(Y, X, me_idx, sigma2_vec, grid, n_simex, seed = NULL)
  d <- combined_weights(grid, d_star)
  beta <- as.numeric(d %*% sc$curve)
  sw <- simex_sandwich(Y, X, me_idx, sigma2_vec, grid, sc$curve,
                       sc$info, sc$beta_all, weights = d)
  if (se_method == "bootstrap") {
    se <- bootstrap_se(Y, X, me_idx, sigma2_vec, grid, d_star,
                       NB = n_boot, seed = seed)
    vcov <- diag(se^2)
  } else {
    vcov <- sw$vcov / n
    se <- sqrt(diag(vcov))
  }
  names(beta) <- names(se) <- cn
  colnames(vcov) <- rownames(vcov) <- cn
  out <- list(coefficients = beta, vcov = vcov, se = se,
              lambda_max = lm_star, poly_degree = d_star,
              lambda_max_argmin = cv$lambda_max_argmin,
              degree_argmin = cv$degree_argmin,
              cv_scores = cv$cv_scores, cv_se = cv$cv_se,
              simex_curve = sc$curve, lam_grid = grid,
              weights = d, n = n, call = match.call())
  class(out) <- "adaptive_simex"
  out
}
