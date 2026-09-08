# Cross-validation of (lambda_max, degree): held-out Poisson deviance on
# the observed scale, prefix property (one union-grid curve per fold,
# noise shared across lambda within a replicate), one-standard-error
# parsimony rule. Returns both the 1-SE and argmin selections.

cv_select <- function(Y, W, me_vars, sigma2_vec, folds = 5, B_cv = 25,
                      lam_max_seq = seq(0.5, 2.5, 0.5), deg_seq = 1:3,
                      seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(W)
  idx <- sample.int(n)
  fold_id <- split(idx, cut(seq_len(n), breaks = folds))
  union_grid <- seq(0, max(lam_max_seq), by = 0.5)
  fold_scores <- array(NA_real_,
    c(folds, length(lam_max_seq), length(deg_seq)))
  for (k in seq_len(folds)) {
    test_idx <- fold_id[[k]]
    train_idx <- setdiff(seq_len(n), test_idx)
    Ytr <- Y[train_idx]; Wtr <- W[train_idx, , drop = FALSE]
    Yte <- Y[test_idx]; Wte <- W[test_idx, , drop = FALSE]
    sc <- simex_curve(Ytr, Wtr, me_vars, sigma2_vec, union_grid, B_cv,
                      seed = NULL)
    for (a in seq_along(lam_max_seq)) {
      lm <- lam_max_seq[a]
      sub <- union_grid[union_grid <= lm + 1e-9]
      cp <- sc$curve[seq_along(sub), , drop = FALSE]
      for (di in seq_along(deg_seq)) {
        bex <- as.numeric(extrap_weights(sub, deg_seq[di]) %*% cp)
        mu <- as.numeric(exp(pmin(pmax(Wte %*% bex, -30), 30)))
        term <- ifelse(Yte > 0, Yte * log(pmax(Yte, 1e-10) / mu), 0)
        fold_scores[k, a, di] <- 2 * sum(term - (Yte - mu))
      }
    }
  }
  cv_scores <- colMeans(fold_scores)
  cv_se <- apply(fold_scores, c(2, 3), sd) / sqrt(folds)
  best <- arrayInd(which.min(cv_scores), dim(cv_scores))
  thresh <- cv_scores[best[1], best[2]] + cv_se[best[1], best[2]]
  ok <- which(cv_scores <= thresh, arr.ind = TRUE)
  ok <- ok[order(ok[, 2], ok[, 1]), , drop = FALSE]  # degree, then lambda_max
  list(lambda_max_1se = lam_max_seq[ok[1, 1]], degree_1se = deg_seq[ok[1, 2]],
       lambda_max_argmin = lam_max_seq[best[1]],
       degree_argmin = deg_seq[best[2]],
       cv_scores = cv_scores, cv_se = cv_se)
}
