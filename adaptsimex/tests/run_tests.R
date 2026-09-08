# Base-R test runner (works without testthat): mirrors the testthat suite.
source_all <- function(pkg) {
  for (f in list.files(file.path(pkg, "R"), full.names = TRUE))
    sys.source(f, envir = globalenv())
}
# prefer the installed package (as under R CMD check); fall back to
# sourcing the R/ directory for direct script use
if (requireNamespace("adaptsimex", quietly = TRUE)) {
  for (fn in c("adaptive_simex", "poisson_irls",
               "nakamura_corrected_score", "regression_calibration",
               "simex_curve", "combined_weights", "simex_sandwich",
               "bootstrap_se", "extrap_weights"))
    assign(fn, getFromNamespace(fn, asNamespace("adaptsimex")))
} else {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_dir <- if (length(fa))
    dirname(normalizePath(sub("^--file=", "", fa))) else "."
  pkg <- normalizePath(file.path(script_dir, ".."))
  source_all(pkg)
}
set.seed(20260905)
n <- 800; p <- 3
X <- matrix(rnorm(n * p), n, p)
beta0 <- c(1, -0.5, 0.8)
Y <- rpois(n, exp(X %*% beta0))
W <- X + matrix(rnorm(n * p), n, p) * sqrt(0.5)
df <- data.frame(Y = Y, W1 = W[,1], W2 = W[,2], W3 = W[,3])
fit <- adaptive_simex(Y ~ W1 + W2 + W3, data = df,
  me_vars = c("W1","W2","W3"), sigma_error = rep(sqrt(0.5), 3),
  n_simex = 50, n_cv_simex = 20, seed = 1, verbose = FALSE)
b <- fit$coefficients
bn1 <- poisson_irls(Y, cbind(1, W))[2]
stopifnot(abs(b[2] - beta0[1]) < abs(bn1 - beta0[1]))
stopifnot(abs(b[2] - beta0[1]) < 0.15)
ci <- confint(fit); stopifnot(dim(ci) == c(p + 1, 2))
pr <- summary(fit); stopifnot(nrow(pr$coefficients) == p + 1)
set.seed(1)
n <- 500; X <- matrix(rnorm(n*2), n, 2); beta0 <- c(0.8, -0.4)
Y <- rpois(n, exp(X %*% beta0))
W <- X + matrix(rnorm(n*2), n, 2) * sqrt(0.2)
cs <- nakamura_corrected_score(Y, W, c(0.2,0.2), 1:2)
stopifnot(all(is.finite(cs$beta)), max(abs(cs$beta - beta0)) < 0.15)
rc <- regression_calibration(Y, W, c(0.2,0.2), 1:2)
stopifnot(all(is.finite(diag(rc$vcov))))
set.seed(3)
n <- 400; X <- matrix(rnorm(n*2), n, 2)
Y <- rpois(n, exp(X %*% c(0.9, -0.6)))
W <- X + matrix(rnorm(n*2), n, 2) * sqrt(0.3)
sc <- simex_curve(Y, W, 1:2, c(0.3,0.3), seq(0,2,0.5), 50, seed = 5)
d <- combined_weights(seq(0,2,0.5), 2)
sw <- simex_sandwich(Y, W, 1:2, c(0.3,0.3), seq(0,2,0.5), sc$curve,
                     sc$info, sc$beta_all, weights = d)
stopifnot(all(diag(sw$vcov) > 0))
se_an <- sqrt(diag(sw$vcov) / n)
se_boot <- bootstrap_se(Y, W, 1:2, c(0.3,0.3), seq(0,2,0.5), 2, NB = 30, seed = 6)
stopifnot(all(se_an > 0.2 * se_boot) && all(se_an < 5 * se_boot))
cat("ALL BASE-R TESTS PASSED\n")
