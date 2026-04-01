"""
Sector ETF Factor Model (Paper Section 2.3).

Each stock is regressed against its sector ETF to extract the
idiosyncratic residual. Uses rolling OLS for time-varying betas.
"""
import numpy as np
import pandas as pd

from .base import FactorModel, FactorResult


class ETFFactorModel(FactorModel):
    """
    ETF-based factor model following Avellaneda & Lee (2010).

    Each stock i is regressed on its assigned sector ETF:
        R_i = beta_i * R_ETF_i + residual_i

    Args:
        sector_mapping: Dict mapping ticker -> sector ETF symbol.
        rolling_window: Window size for rolling OLS beta estimation
            (paper default: 252).
    """

    def __init__(
        self,
        sector_mapping: dict[str, str],
        rolling_window: int = 252,
    ):
        self.sector_mapping = sector_mapping
        self.rolling_window = rolling_window

    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        """
        Fit ETF factor model.

        Args:
            returns: DataFrame of stock log returns (dates x tickers).
            **kwargs:
                etf_returns: DataFrame of ETF log returns (dates x ETFs).
                    Required.

        Returns:
            FactorResult with ETF-regression residuals.
        """
        etf_returns = kwargs.get("etf_returns")
        if etf_returns is None:
            raise ValueError("etf_returns is required for ETFFactorModel")

        tickers = returns.columns.tolist()
        dates = returns.index
        T = len(dates)

        residuals = pd.DataFrame(np.nan, index=dates, columns=tickers)
        betas_last = {}
        r_squared = {}

        for ticker in tickers:
            etf_ticker = self.sector_mapping.get(ticker)
            if etf_ticker is None or etf_ticker not in etf_returns.columns:
                residuals[ticker] = returns[ticker]
                betas_last[ticker] = 0.0
                continue

            stock_ret = returns[ticker].values
            etf_ret = etf_returns[etf_ticker].values

            # Rolling OLS: R_stock = beta * R_etf + residual (no intercept)
            for t in range(self.rolling_window, T):
                window_stock = stock_ret[t - self.rolling_window : t]
                window_etf = etf_ret[t - self.rolling_window : t]

                mask = np.isfinite(window_stock) & np.isfinite(window_etf)
                if mask.sum() < 30:
                    continue

                y = window_stock[mask]
                x = window_etf[mask]
                x_sq_sum = np.dot(x, x)
                if x_sq_sum < 1e-12:
                    continue

                beta = np.dot(x, y) / x_sq_sum
                residuals.iloc[t, residuals.columns.get_loc(ticker)] = stock_ret[t] - beta * etf_ret[t]

            # Store last beta
            betas_last[ticker] = beta if "beta" in dir() else 0.0

            # R-squared on last window
            if T >= self.rolling_window:
                last_y = stock_ret[-self.rolling_window:]
                last_x = etf_ret[-self.rolling_window:]
                mask = np.isfinite(last_y) & np.isfinite(last_x)
                if mask.sum() > 30:
                    y_m = last_y[mask]
                    x_m = last_x[mask]
                    beta_last = np.dot(x_m, y_m) / (np.dot(x_m, x_m) + 1e-12)
                    pred = beta_last * x_m
                    ss_res = np.sum((y_m - pred) ** 2)
                    ss_tot = np.sum((y_m - y_m.mean()) ** 2)
                    r_squared[ticker] = 1 - ss_res / (ss_tot + 1e-12)
                    betas_last[ticker] = beta_last

        # Construct factor returns and betas DataFrames
        etf_names = sorted(set(self.sector_mapping.values()))
        etf_names = [e for e in etf_names if e in etf_returns.columns]
        factor_returns = etf_returns[etf_names].copy()

        betas_df = pd.DataFrame(0.0, index=tickers, columns=etf_names)
        for ticker in tickers:
            etf = self.sector_mapping.get(ticker)
            if etf and etf in etf_names:
                betas_df.loc[ticker, etf] = betas_last.get(ticker, 0.0)

        metadata = {
            "sector_mapping": self.sector_mapping,
            "r_squared": r_squared,
            "rolling_window": self.rolling_window,
        }

        return FactorResult(
            residuals=residuals,
            factor_returns=factor_returns,
            betas=betas_df,
            metadata=metadata,
        )
