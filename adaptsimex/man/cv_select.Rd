\name{cv_select}
\alias{cv_select}
\title{Cross-Validation of Hyperparameters}
\description{
  Helper for the adaptive SIMEX estimator of Garg and Mishra: held-out Poisson deviance over the hyperparameter grid with the one-standard-error parsimony rule and the argmin diagnostic.
}
\usage{
cv_select(Y, W, me_vars, sigma2_vec, folds, B_cv, lam_max_seq, deg_seq, seed)
}
\arguments{
  \item{Y}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{W}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{me_vars}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{sigma2_vec}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{folds}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{B_cv}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{lam_max_seq}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{deg_seq}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{seed}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
}
\value{
  See \code{\link{adaptive_simex}} and the package vignette.
}
