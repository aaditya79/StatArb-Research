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

        # Align ETF returns to the stock-returns index. Without this the two
        # frames can have different lengths (different fetch cutoffs, missing
        # ETF bars on specific dates) and positional indexing into .values
        # goes out of bounds.
        etf_returns = etf_returns.reindex(returns.index)

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

            stock_ret = returns[ticker].values.astype(float)
            etf_ret = etf_returns[etf_ticker].values.astype(float)

            # Rolling OLS (no intercept): R_stock = beta * R_etf + residual.
            # Vectorised: compute rolling sum(x^2) and rolling sum(x*y), then
            # beta_t = Sxy_t / Sxx_t over the trailing `rolling_window` bars.
            valid = np.isfinite(stock_ret) & np.isfinite(etf_ret)
            x = np.where(valid, etf_ret, 0.0)
            y = np.where(valid, stock_ret, 0.0)
            xy = x * y
            xx = x * x
            cnt = valid.astype(float)

            # Cumulative sums -> rolling sums via difference.
            c_xy = np.concatenate(([0.0], np.cumsum(xy)))
            c_xx = np.concatenate(([0.0], np.cumsum(xx)))
            c_cnt = np.concatenate(([0.0], np.cumsum(cnt)))

            W = self.rolling_window
            if T > W:
                roll_xy = c_xy[W:T + 1] - c_xy[: T + 1 - W]   # (T+1-W,)
                roll_xx = c_xx[W:T + 1] - c_xx[: T + 1 - W]
                roll_cnt = c_cnt[W:T + 1] - c_cnt[: T + 1 - W]
                # These cover t = W..T (inclusive), matching original loop
                # which set residuals at t = W, W+1, ..., T-1.
                # Slice the first (T - W) entries to align with those t's.
                roll_xy = roll_xy[: T - W]
                roll_xx = roll_xx[: T - W]
                roll_cnt = roll_cnt[: T - W]

                with np.errstate(divide="ignore", invalid="ignore"):
                    betas = np.where(
                        (roll_xx > 1e-12) & (roll_cnt >= 30),
                        roll_xy / np.where(roll_xx > 1e-12, roll_xx, 1.0),
                        np.nan,
                    )
                # Apply to days W..T-1
                stock_at_t = stock_ret[W:T]
                etf_at_t = etf_ret[W:T]
                day_resid = np.where(
                    np.isfinite(betas) & np.isfinite(stock_at_t) & np.isfinite(etf_at_t),
                    stock_at_t - betas * etf_at_t,
                    np.nan,
                )
                residuals.iloc[W:T, residuals.columns.get_loc(ticker)] = day_resid

                # Last valid beta for the betas_last dict.
                finite_betas = betas[np.isfinite(betas)]
                betas_last[ticker] = float(finite_betas[-1]) if len(finite_betas) else 0.0
            else:
                betas_last[ticker] = 0.0

            # R-squared on the final rolling window (diagnostic).
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
