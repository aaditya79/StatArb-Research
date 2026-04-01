"""
Ornstein-Uhlenbeck process estimation via AR(1) regression (Paper Appendix A).

The residual process X(t) is modeled as:
    dX(t) = kappa * (m - X(t)) * dt + sigma * dW(t)

We estimate this by fitting the discrete AR(1) model:
    X_{n+1} = a + b * X_n + epsilon

Then map to OU parameters:
    kappa = -ln(b) / dt
    m = a / (1 - b)
    sigma = sqrt(Var(epsilon) * 2 * kappa / (1 - b^2))
    sigma_eq = sigma / sqrt(2 * kappa) = sqrt(Var(epsilon) / (1 - b^2))
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    """Estimated Ornstein-Uhlenbeck parameters for a single stock."""
    kappa: float        # mean-reversion speed (annualized)
    m: float            # equilibrium level
    sigma: float        # OU diffusion coefficient
    sigma_eq: float     # equilibrium std dev = sigma / sqrt(2*kappa)
    half_life: float    # mean-reversion half-life in trading days
    a: float            # AR(1) intercept
    b: float            # AR(1) slope


def fit_ar1(series: np.ndarray) -> tuple[float, float, float] | None:
    """
    Fit AR(1) model: X_{n+1} = a + b * X_n + epsilon.

    Args:
        series: 1-D array of cumulative residuals X_1, ..., X_T.

    Returns:
        Tuple (a, b, var_eps) or None if fit fails.
    """
    if len(series) < 10:
        return None

    X = series[:-1]
    Y = series[1:]

    mask = np.isfinite(X) & np.isfinite(Y)
    if mask.sum() < 10:
        return None

    X = X[mask]
    Y = Y[mask]

    n = len(X)
    Sx = X.sum()
    Sy = Y.sum()
    Sxx = np.dot(X, X)
    Sxy = np.dot(X, Y)

    denom = n * Sxx - Sx * Sx
    if abs(denom) < 1e-12:
        return None

    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n

    # Clamp b for stationarity
    b = np.clip(b, -0.9999, 0.9999)

    residuals = Y - (a + b * X)
    var_eps = np.var(residuals, ddof=1) if n > 2 else np.var(residuals)

    return a, b, var_eps


def ar1_to_ou(
    a: float, b: float, var_eps: float, dt: float = 1.0 / 252.0
) -> OUParams | None:
    """
    Map AR(1) parameters to continuous-time OU parameters.

    Args:
        a: AR(1) intercept.
        b: AR(1) slope (must be in (0, 1) for mean-reversion).
        var_eps: Variance of AR(1) residuals.
        dt: Time step in years (default: 1/252).

    Returns:
        OUParams or None if parameters are invalid.
    """
    if b <= 0 or b >= 1:
        return None

    kappa = -np.log(b) / dt  # annualized
    m = a / (1.0 - b)
    sigma_eq = np.sqrt(var_eps / (1.0 - b * b))
    sigma = sigma_eq * np.sqrt(2.0 * kappa)
    half_life = np.log(2.0) / (kappa * dt)  # in trading days

    if kappa <= 0 or sigma_eq <= 0:
        return None

    return OUParams(
        kappa=kappa,
        m=m,
        sigma=sigma,
        sigma_eq=sigma_eq,
        half_life=half_life,
        a=a,
        b=b,
    )


def estimate_ou_params(
    residuals: pd.Series, window: int = 60, dt: float = 1.0 / 252.0
) -> OUParams | None:
    """
    Estimate OU parameters from a residual time series.

    Uses the last `window` observations to compute the cumulative
    residual process, then fits AR(1) and maps to OU.

    Args:
        residuals: Series of daily idiosyncratic returns.
        window: Number of trailing days for estimation (default: 60).
        dt: Time step in years.

    Returns:
        OUParams or None if estimation fails.
    """
    # Take last `window` observations
    tail = residuals.dropna().iloc[-window:]
    if len(tail) < 30:
        return None

    # Cumulative residual (discrete X process)
    X = tail.cumsum().values

    result = fit_ar1(X)
    if result is None:
        return None

    a, b, var_eps = result
    return ar1_to_ou(a, b, var_eps, dt)
