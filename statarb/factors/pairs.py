import numpy as np
import pandas as pd
from .base import FactorModel, FactorResult
from statarb.signals.cointegration import test_cointegration


class PairsFactorModel(FactorModel):
    """
    Lookahead-clean pairs model.

    Pair selection runs on a *formation* slice consisting of the FIRST
    `lookback` rows of the price history. Trading happens after that
    slice, using rolling per-day hedge ratios that only see data
    strictly before the trading day. No information from the trade
    period leaks into the pair set or the hedge ratio.
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        max_pairs: int = 20,
        min_half_life: float = 1.0,
        max_half_life: float = 126.0,
        lookback: int = 252,
    ):
        self.pvalue_threshold = pvalue_threshold
        self.max_pairs = max_pairs
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.lookback = lookback
        self.coint_results = pd.DataFrame()

    @staticmethod
    def _rolling_beta_series(
        y: np.ndarray, x: np.ndarray, window: int, min_obs: int = 60
    ) -> np.ndarray:
        """
        Rolling OLS slope (with intercept) of y on x over the trailing
        `window` strictly preceding each index t. Returns array indexed
        by t with NaN where unavailable.
        """
        T = len(y)
        out = np.full(T, np.nan)
        for t in range(window, T):
            yi = y[t - window:t]
            xi = x[t - window:t]
            mask = np.isfinite(yi) & np.isfinite(xi)
            if mask.sum() < min_obs:
                continue
            ym, xm = yi[mask], xi[mask]
            xm_c = xm - xm.mean()
            ym_c = ym - ym.mean()
            denom = float((xm_c * xm_c).sum())
            if denom < 1e-12:
                continue
            out[t] = float((xm_c * ym_c).sum() / denom)
        return out

    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        prices = kwargs.get("prices")
        if prices is None:
            raise ValueError("PairsFactorModel requires 'prices' in kwargs.")

        common = returns.columns.intersection(prices.columns)
        prices = prices[common]
        returns = returns[common]

        if len(prices) <= self.lookback + 1:
            raise ValueError(
                f"Need more than lookback={self.lookback} rows to leave a "
                f"trading window after the formation period. Got {len(prices)}."
            )

        # ── Formation period: first `lookback` rows only ──
        # All pair selection / initial coint tests use this slice. The
        # trading-period engine sees no overlap with this window.
        formation = prices.iloc[: self.lookback]
        coint_df = test_cointegration(
            formation, self.pvalue_threshold, lookback=self.lookback
        )
        self.coint_results = coint_df

        if coint_df.empty:
            coint_df = test_cointegration(
                formation, pvalue_threshold=1.0, lookback=self.lookback
            )
            self.coint_results = coint_df
            if coint_df.empty:
                raise ValueError(
                    "No pairs could be tested in the formation window. "
                    "Ensure enough overlapping price history exists in the "
                    "first `lookback` rows."
                )

        valid = coint_df[
            coint_df["half_life"].between(self.min_half_life, self.max_half_life)
        ].head(self.max_pairs)
        if valid.empty:
            valid = coint_df.head(self.max_pairs)

        # ── Rolling-beta spread innovations ──
        # For every selected pair, refit β on a trailing `lookback` window
        # at each day. The spread innovation at day t uses β computed only
        # from prices strictly before t — no future leakage.
        log_prices = np.log(prices)
        T = len(log_prices)
        spread_returns: dict[str, pd.Series] = {}
        betas_data: dict[str, dict[str, float]] = {}

        for _, row in valid.iterrows():
            t1, t2 = row["ticker1"], row["ticker2"]
            pair_id = f"{t1}_{t2}"
            p1 = log_prices[t1].values.astype(float)
            p2 = log_prices[t2].values.astype(float)

            beta_series = self._rolling_beta_series(
                p1, p2, window=self.lookback
            )
            # Spread at day t uses β_t (which itself is fit on [t-W, t-1]).
            spread_t = p1 - beta_series * p2
            spread_series = pd.Series(spread_t, index=prices.index)
            # First-difference → spread innovation; engine treats this
            # as the pair's residual return time series.
            spread_returns[pair_id] = spread_series.diff()
            # Cosmetic: store the most recent finite β for the betas frame.
            finite_betas = beta_series[np.isfinite(beta_series)]
            beta_last = float(finite_betas[-1]) if len(finite_betas) else float(row["hedge_ratio"])
            betas_data[pair_id] = {t1: 1.0, t2: -beta_last}

        residuals = pd.DataFrame(spread_returns).dropna(how="all")
        betas = pd.DataFrame(betas_data).T.fillna(0.0)

        return FactorResult(
            residuals=residuals,
            factor_returns=pd.DataFrame(index=residuals.index),
            betas=betas,
            metadata={
                "coint_results": coint_df,
                "selected_pairs": valid,
                "n_pairs": len(valid),
                "model_type": "pairs",
                "formation_end": prices.index[self.lookback - 1],
                "trading_start": prices.index[self.lookback],
            },
        )
