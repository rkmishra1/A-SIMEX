test_that("A-SIMEX reduces attenuation and recovers coefficients", {
  set.seed(20260905)
  n <- 800; p <- 3
  X <- matrix(rnorm(n * p), n, p)
  beta0 <- c(1, -0.5, 0.8)
  Y <- rpois(n, exp(X %*% beta0))
  W <- X + matrix(rnorm(n * p), n, p) * sqrt(0.5)
  fit <- adaptive_simex(Y ~ W1 + W2 + W3, data = data.frame(Y, W1 = W[,1],
    W2 = W[,2], W3 = W[,3]), me_vars = c("W1","W2","W3"),
    sigma_error = rep(sqrt(0.5), 3), n_simex = 50, n_cv_simex = 20,
    seed = 1, verbose = FALSE)
  bn <- coef(lm(0))  # placeholder to avoid glm dependency: naive via irls
  b <- fit$coefficients
  # closer to truth than the naive fit for the strongly loaded coefficient
  bn1 <- adaptsimex:::poisson_irls(Y, cbind(1, W))[2]
  expect_true(abs(b[2] - beta0[1]) < abs(bn1 - beta0[1]))
  # within 0.15 of truth on the first coefficient at this error level
  expect_true(abs(b[2] - beta0[1]) < 0.15)
  ci <- confint(fit)
  expect_equal(dim(ci), c(p + 1, 2))
})
