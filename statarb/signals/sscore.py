"""
S-score computation (Paper Section 4, Eq. 15, Appendix A Eq. A2).

The s-score measures how far a stock's residual is from equilibrium
in units of the equilibrium standard deviation:

    s_i = (X_i(t) - m_i_bar) / sigma_eq_i

With mean centering (Eq. 18):
    m_i_bar = m_i - mean(m_j) across all eligible stocks
"""
import numpy as np
import pandas as pd

from .ou_estimator import OUParams


def compute_sscores(
    residuals: pd.DataFrame,
    ou_params: dict[str, OUParams],
    mean_center: bool = True,
) -> pd.Series:
    """
    Compute s-scores for the last date in the residuals DataFrame.

    Following the paper (Appendix A), since the regression forces
    X_60 = 0 (the cumulative residual at the end of the window),
    we have:
        s = -m / sigma_eq  (before centering)

    With mean centering:
        s = (-m + mean(m)) / sigma_eq

    Args:
        residuals: DataFrame of daily idiosyncratic returns (dates x tickers).
            Only the last `window` rows are used per stock.
        ou_params: Dict mapping ticker -> OUParams (already estimated).
        mean_center: Whether to apply cross-sectional mean centering (Eq. 18).

    Returns:
        Series of s-scores indexed by ticker.
    """
    tickers = list(ou_params.keys())
    sscores = {}

    # Compute raw m values
    m_values = {t: ou_params[t].m for t in tickers}

    # Mean centering (Eq. 18)
    if mean_center and len(m_values) > 0:
        mean_m = np.mean(list(m_values.values()))
    else:
        mean_m = 0.0

    for ticker in tickers:
        params = ou_params[ticker]
        if params.sigma_eq <= 0:
            sscores[ticker] = 0.0
            continue

        m_centered = params.m - mean_m
        # s = -m_centered / sigma_eq (since X(t) = 0 at end of window)
        sscores[ticker] = -m_centered / params.sigma_eq

    return pd.Series(sscores)


def compute_sscores_timeseries(
    residuals: pd.DataFrame,
    ou_params_series: dict[str, dict[str, OUParams]],
    mean_center: bool = True,
) -> pd.DataFrame:
    """
    Compute s-scores for each date in a time series.

    Args:
        residuals: DataFrame of daily idiosyncratic returns (dates x tickers).
        ou_params_series: Dict mapping date_str -> {ticker -> OUParams}.
        mean_center: Whether to apply mean centering.

    Returns:
        DataFrame of s-scores (dates x tickers).
    """
    dates = sorted(ou_params_series.keys())
    all_tickers = residuals.columns.tolist()

    sscores_df = pd.DataFrame(
        np.nan, index=pd.to_datetime(dates), columns=all_tickers
    )

    for date_str in dates:
        params = ou_params_series[date_str]
        if not params:
            continue
        scores = compute_sscores(residuals, params, mean_center)
        for ticker in scores.index:
            if ticker in all_tickers:
                sscores_df.loc[pd.to_datetime(date_str), ticker] = scores[ticker]

    return sscores_df
