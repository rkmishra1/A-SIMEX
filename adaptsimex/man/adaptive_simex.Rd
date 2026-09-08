\name{adaptive_simex}
\alias{adaptive_simex}
\alias{print.adaptive_simex}
\alias{summary.adaptive_simex}
\alias{confint.adaptive_simex}
\alias{plot.adaptive_simex}
\title{Adaptive SIMEX for Poisson Regression with Classical
  Measurement Error}
\description{
  Corrects covariate measurement error in Poisson regression by
  simulation extrapolation with cross-validated hyperparameters. The
  extrapolation region and polynomial degree are selected by held-out
  Poisson deviance with a one-standard-error parsimony rule, the
  extrapolation uses a two-stage (polynomial plus ridge) functional,
  and standard errors come from an analytic sandwich that decomposes
  into data and simulation components.
}
\usage{
adaptive_simex(formula, data, me_vars, sigma_error,
  lambda_max_seq = seq(0.5, 2.5, by = 0.5),
  poly_degree_seq = 1:3, n_simex = 100, n_cv_folds = 5,
  n_cv_simex = 25, se_method = c("sandwich", "bootstrap"),
  n_boot = 200, seed = NULL, verbose = TRUE)

\method{print}{adaptive_simex}(x, ...)
\method{summary}{adaptive_simex}(object, ...)
\method{confint}{adaptive_simex}(object, parm, level = 0.95, ...)
\method{plot}{adaptive_simex}(x, which = 1, ...)
}
\arguments{
  \item{formula}{an object of class \code{"formula"}: the outcome on
    the left, covariates (including the error-contaminated ones) on the
    right.}
  \item{data}{a data frame containing the variables in
    \code{formula}.}
  \item{me_vars}{character vector naming the error-contaminated
    predictors.}
  \item{sigma_error}{numeric vector of measurement-error standard
    deviations, one per entry of \code{me_vars}.}
  \item{lambda_max_seq}{candidate values of the extrapolation endpoint
    \eqn{\lambda_{\max}}; the grid within each candidate is
    \eqn{\{0, 0.5, \dots, \lambda_{\max}\}}.}
  \item{poly_degree_seq}{candidate polynomial degrees.}
  \item{n_simex}{number of SIMEX replicates \eqn{B} for the final fit.}
  \item{n_cv_folds}{number of cross-validation folds.}
  \item{n_cv_simex}{number of SIMEX replicates per fold curve during
    cross-validation.}
  \item{se_method}{\code{"sandwich"} (analytic, default) or
    \code{"bootstrap"} (pairs bootstrap over data and simulated
    errors; recommended at small samples).}
  \item{n_boot}{number of bootstrap replicates when
    \code{se_method = "bootstrap"}.}
  \item{seed}{optional integer seed for reproducibility.}
  \item{verbose}{logical; print progress.}
  \item{which}{which diagnostic plot to draw (1 = SIMEX curve).}
  \item{x, object, parm, level, ...}{standard S3 arguments.}
}
\value{
  An object of class \code{"adaptive_simex"}: a list with
  \code{coefficients}, \code{vcov}, \code{se}, the CV-selected
  \code{lambda_max} and \code{poly_degree} (both the one-standard-error
  and argmin choices), \code{cv_scores}, the fitted
  \code{simex_curve}, and the combined extrapolation
  \code{weights}.
}
\references{
  Garg, G. and Mishra, R. Adaptive SIMEX for Poisson Regression with
  Classical Measurement Error.
}
\examples{
set.seed(1)
n <- 500
X <- matrix(rnorm(n * 3), n, 3)
Y <- rpois(n, exp(X \%*\% c(1, -0.5, 0.8)))
W <- X + matrix(rnorm(n * 3), n, 3) * sqrt(0.5)
df <- data.frame(Y = Y, W1 = W[, 1], W2 = W[, 2], X3 = X[, 3])
fit <- adaptive_simex(Y ~ W1 + W2 + X3, data = df,
  me_vars = c("W1", "W2"), sigma_error = c(sqrt(0.5), sqrt(0.5)),
  seed = 1)
summary(fit)
confint(fit)
plot(fit)
}
