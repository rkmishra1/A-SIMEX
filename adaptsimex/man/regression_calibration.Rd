\name{regression_calibration}
\alias{regression_calibration}
\title{Regression Calibration Estimator}
\description{
  Helper for the adaptive SIMEX estimator of Garg and Mishra: regression calibration with a robust sandwich variance.
}
\usage{
regression_calibration(Y, W, sigma2_vec, me_vars)
}
\arguments{
  \item{Y}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{W}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{sigma2_vec}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{me_vars}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
}
\value{
  See \code{\link{adaptive_simex}} and the package vignette.
}
