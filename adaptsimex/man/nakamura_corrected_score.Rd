\name{nakamura_corrected_score}
\alias{nakamura_corrected_score}
\title{Nakamura Corrected-Score Estimator}
\description{
  Helper for the adaptive SIMEX estimator of Garg and Mishra: the Nakamura (1990) corrected-score estimator with a damped Newton solver and sandwich variance.
}
\usage{
nakamura_corrected_score(Y, W, sigma2_vec, me_vars, beta_init, tol, max_iter)
}
\arguments{
  \item{Y}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{W}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{sigma2_vec}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{me_vars}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{beta_init}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{tol}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{max_iter}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
}
\value{
  See \code{\link{adaptive_simex}} and the package vignette.
}
