\name{combined_weights}
\alias{combined_weights}
\title{Combined Extrapolation Weights}
\description{
  Helper for the adaptive SIMEX estimator of Garg and Mishra: combined primary-plus-ridge extrapolation weights (summing to one).
}
\usage{
combined_weights(lam_grid, degree_primary, gamma, extra_degree)
}
\arguments{
  \item{lam_grid}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{degree_primary}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{gamma}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
  \item{extra_degree}{see \code{\link{adaptive_simex}}; documented in the package vignette.}
}
\value{
  See \code{\link{adaptive_simex}} and the package vignette.
}
