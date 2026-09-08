\name{simex_curve}
\alias{simex_curve}
\title{Fitted SIMEX Curve and Replicate Information}
\description{
  Helper for the adaptive SIMEX estimator of Garg and Mishra: fits the SIMEX curve over a grid and returns the mean curve, mean replicate information matrices, and all replicate fits.
}
\usage{
simex_curve(Y, W, me_vars, sigma2_vec, lam_grid, B = 100,
  seed = NULL, beta_init = NULL)
}
\arguments{
  \item{Y}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{W}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{me_vars}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{sigma2_vec}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{lam_grid}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{B}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{seed}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{beta_init}{optional warm-start coefficient vector.}
}
\value{
  See \code{\link{adaptive_simex}} and the package vignette.
}
