import numpy as np
import pandas as pd
from .base import FactorModel, FactorResult
from statarb.signals.cointegration import test_cointegration


class PairsFactorModel(FactorModel):
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

    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        prices = kwargs.get("prices")
        if prices is None:
            raise ValueError("PairsFactorModel requires 'prices' in kwargs.")

        common = returns.columns.intersection(prices.columns)
        prices = prices[common]
        returns = returns[common]

        coint_df = test_cointegration(prices, self.pvalue_threshold, self.lookback)
        self.coint_results = coint_df

        if coint_df.empty:
            # Fallback: relax threshold and take best available pairs
            coint_df = test_cointegration(prices, pvalue_threshold=1.0, lookback=self.lookback)
            self.coint_results = coint_df
            if coint_df.empty:
                raise ValueError(
                    "No pairs could be tested. Ensure enough overlapping price history exists."
                )

        valid = coint_df[
            coint_df["half_life"].between(self.min_half_life, self.max_half_life)
        ].head(self.max_pairs)
        if valid.empty:
            valid = coint_df.head(self.max_pairs)

        log_prices = np.log(prices)
        spread_returns = {}
        betas_data = {}

        for _, row in valid.iterrows():
            t1, t2 = row["ticker1"], row["ticker2"]
            beta = row["hedge_ratio"]
            pair_id = f"{t1}_{t2}"
            spread = log_prices[t1] - beta * log_prices[t2]
            spread_returns[pair_id] = spread.diff()
            betas_data[pair_id] = {t1: 1.0, t2: -beta}

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
            },
        )
