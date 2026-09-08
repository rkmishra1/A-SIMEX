test_that("sandwich variance is positive and near bootstrap scale", {
  set.seed(3)
  n <- 400
  X <- matrix(rnorm(n * 2), n, 2)
  Y <- rpois(n, exp(X %*% c(0.9, -0.6)))
  W <- X + matrix(rnorm(n * 2), n, 2) * sqrt(0.3)
  sc <- simex_curve(Y, W, 1:2, c(0.3, 0.3), seq(0, 2, 0.5), 50, seed = 5)
  d <- combined_weights(seq(0, 2, 0.5), 2)
  sw <- simex_sandwich(Y, W, 1:2, c(0.3, 0.3), seq(0, 2, 0.5), sc$curve,
                       sc$info, sc$beta_all, weights = d)
  expect_true(all(diag(sw$vcov) > 0))
  se_an <- sqrt(diag(sw$vcov) / n)
  se_boot <- bootstrap_se(Y, W, 1:2, c(0.3, 0.3), seq(0, 2, 0.5), 2,
                          NB = 30, seed = 6)
  expect_true(all(se_an > 0.2 * se_boot) && all(se_an < 5 * se_boot))
})
