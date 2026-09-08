print.adaptive_simex <- function(x, ...) {
  cat("Adaptive SIMEX fit (Poisson regression, classical error)\n")
  cat("=========================================================\n")
  cat("n =", x$n,
      "| lambda_max =", x$lambda_max,
      "| degree =", x$poly_degree, "\n\n")
  tab <- data.frame(Estimate = x$coefficients, Std_Error = x$se,
                    z = x$coefficients / x$se,
                    p = 2 * pnorm(-abs(x$coefficients / x$se)))
  print(tab, digits = 4)
  invisible(x)
}

summary.adaptive_simex <- function(object, ...) {
  tab <- data.frame(Estimate = object$coefficients,
                    Std_Error = object$se,
                    z = object$coefficients / object$se,
                    p = 2 * pnorm(-abs(object$coefficients / object$se)))
  list(coefficients = tab, lambda_max = object$lambda_max,
       poly_degree = object$poly_degree,
       lambda_max_argmin = object$lambda_max_argmin,
       degree_argmin = object$degree_argmin,
       cv_scores = object$cv_scores)
}

confint.adaptive_simex <- function(object, parm, level = 0.95, ...) {
  cf <- object$coefficients
  se <- object$se
  z <- qnorm((1 + level) / 2)
  ci <- cbind(cf - z * se, cf + z * se)
  colnames(ci) <- paste0(c((1 - level) / 2, (1 + level) / 2) * 100, " %")
  if (!missing(parm)) ci <- ci[parm, , drop = FALSE]
  ci
}

plot.adaptive_simex <- function(x, which = 1, ...) {
  if (which == 1) {
    K <- nrow(x$simex_curve)
    plot(seq_len(K) - 1, x$simex_curve[, 1], type = "b", pch = 19,
         xlab = expression(lambda), ylab = expression(bar(beta)(lambda)),
         main = "SIMEX curve (first coefficient)", ...)
    abline(v = -1, lty = 2)
    points(-1, x$coefficients[1], col = 2, pch = 17)
  }
  invisible(x)
}
