test_that("corrected score and regression calibration run", {
  set.seed(1)
  n <- 500
  X <- matrix(rnorm(n * 2), n, 2)
  beta0 <- c(0.8, -0.4)
  Y <- rpois(n, exp(X %*% beta0))
  W <- X + matrix(rnorm(n * 2), n, 2) * sqrt(0.2)
  cs <- nakamura_corrected_score(Y, W, c(0.2, 0.2), 1:2)
  expect_true(all(is.finite(cs$beta)))
  expect_true(max(abs(cs$beta - beta0)) < 0.15)
  rc <- regression_calibration(Y, W, c(0.2, 0.2), 1:2)
  expect_true(all(is.finite(rc$beta)))
  expect_true(all(is.finite(diag(rc$vcov))))
})
