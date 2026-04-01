"""
Combined Factor Model: SPY + Sector ETF + PCA (3-stage decomposition).

This implements the full Avellaneda-Lee decomposition:
    Stage 1: Remove market factor (SPY) via rolling beta
    Stage 2: Remove sector ETF factor from Stage 1 residuals
    Stage 3: PCA on Stage 2 residuals (eigenportfolios within sectors)
"""
import numpy as np
import pandas as pd

from .base import FactorModel, FactorResult
from .pca import PCAFactorModel


class CombinedFactorModel(FactorModel):
    """
    3-stage factor model combining market, sector, and PCA factors.

    Args:
        sector_mapping: Dict mapping ticker -> sector ETF symbol.
        rolling_window: Window for rolling OLS betas (default: 252).
        pca_n_components: Number of PCA components for Stage 3.
        pca_lookback: Lookback for PCA correlation matrix.
        use_ledoit_wolf: Ledoit-Wolf shrinkage for PCA.
    """

    def __init__(
        self,
        sector_mapping: dict[str, str],
        rolling_window: int = 252,
        pca_n_components: int | None = 15,
        pca_lookback: int = 252,
        use_ledoit_wolf: bool = True,
    ):
        self.sector_mapping = sector_mapping
        self.rolling_window = rolling_window
        self.pca_n_components = pca_n_components
        self.pca_lookback = pca_lookback
        self.use_ledoit_wolf = use_ledoit_wolf

    def _rolling_beta_residual(
        self,
        stock_returns: np.ndarray,
        factor_returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling OLS residual (no intercept)."""
        T = len(stock_returns)
        residuals = np.full(T, np.nan)
        for t in range(window, T):
            y = stock_returns[t - window : t]
            x = factor_returns[t - window : t]
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 30:
                continue
            x_m, y_m = x[mask], y[mask]
            denom = np.dot(x_m, x_m)
            if denom < 1e-12:
                continue
            beta = np.dot(x_m, y_m) / denom
            residuals[t] = stock_returns[t] - beta * factor_returns[t]
        return residuals

    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        """
        Fit 3-stage combined model.

        Args:
            returns: Stock log returns (dates x tickers).
            **kwargs:
                etf_returns: ETF log returns (dates x ETFs). Required.
                spy_returns: SPY log returns (dates x ['SPY']). Required.

        Returns:
            FactorResult with final residuals after all 3 stages.
        """
        etf_returns = kwargs.get("etf_returns")
        spy_returns = kwargs.get("spy_returns")
        if etf_returns is None or spy_returns is None:
            raise ValueError(
                "etf_returns and spy_returns required for CombinedFactorModel"
            )

        tickers = returns.columns.tolist()
        dates = returns.index
        T = len(dates)

        if isinstance(spy_returns, pd.DataFrame):
            spy_arr = spy_returns.iloc[:, 0].values
        else:
            spy_arr = spy_returns.values

        # Stage 1: Remove SPY market factor
        stage1_residuals = pd.DataFrame(np.nan, index=dates, columns=tickers)
        for ticker in tickers:
            stock_arr = returns[ticker].values
            stage1_residuals[ticker] = self._rolling_beta_residual(
                stock_arr, spy_arr, self.rolling_window
            )

        # Stage 2: Remove sector ETF factor from Stage 1 residuals
        stage2_residuals = pd.DataFrame(np.nan, index=dates, columns=tickers)
        for ticker in tickers:
            etf_ticker = self.sector_mapping.get(ticker)
            if etf_ticker is None or etf_ticker not in etf_returns.columns:
                stage2_residuals[ticker] = stage1_residuals[ticker]
                continue
            s1_arr = stage1_residuals[ticker].values
            etf_arr = etf_returns[etf_ticker].values
            stage2_residuals[ticker] = self._rolling_beta_residual(
                s1_arr, etf_arr, self.rolling_window
            )

        # Stage 3: PCA on Stage 2 residuals
        # Drop rows that are all NaN before PCA
        valid_mask = stage2_residuals.notna().all(axis=1)
        valid_residuals = stage2_residuals[valid_mask].copy()

        if len(valid_residuals) < self.pca_lookback:
            # Not enough data for PCA; return Stage 2 residuals
            all_factor_names = ["SPY"] + sorted(
                set(self.sector_mapping.values())
            )
            factor_ret = pd.DataFrame(
                index=dates,
                columns=all_factor_names,
            )
            factor_ret["SPY"] = spy_arr
            for etf in set(self.sector_mapping.values()):
                if etf in etf_returns.columns:
                    factor_ret[etf] = etf_returns[etf].values

            betas_df = pd.DataFrame(0.0, index=tickers, columns=all_factor_names)
            return FactorResult(
                residuals=stage2_residuals,
                factor_returns=factor_ret.astype(float),
                betas=betas_df,
                metadata={"stages_completed": 2},
            )

        pca_model = PCAFactorModel(
            n_components=self.pca_n_components,
            use_ledoit_wolf=self.use_ledoit_wolf,
            lookback=self.pca_lookback,
        )
        pca_result = pca_model.fit(valid_residuals)

        # Combine factor returns: SPY + sector ETFs + PCA components
        pca_factor_names = pca_result.factor_returns.columns.tolist()
        etf_names = sorted(set(self.sector_mapping.values()))
        all_factor_names = ["SPY"] + etf_names + pca_factor_names

        factor_ret = pd.DataFrame(index=dates, columns=all_factor_names)
        factor_ret["SPY"] = spy_arr
        for etf in etf_names:
            if etf in etf_returns.columns:
                factor_ret[etf] = etf_returns[etf].values
        for pc in pca_factor_names:
            if pc in pca_result.factor_returns.columns:
                factor_ret.loc[
                    pca_result.factor_returns.index, pc
                ] = pca_result.factor_returns[pc].values

        # Build betas DataFrame
        betas_df = pd.DataFrame(0.0, index=tickers, columns=all_factor_names)
        for ticker in tickers:
            if ticker in pca_result.betas.index:
                for pc in pca_factor_names:
                    betas_df.loc[ticker, pc] = pca_result.betas.loc[ticker, pc]

        # Final residuals from PCA stage, mapped back to full date range
        final_residuals = pd.DataFrame(np.nan, index=dates, columns=tickers)
        for ticker in tickers:
            if ticker in pca_result.residuals.columns:
                final_residuals.loc[
                    pca_result.residuals.index, ticker
                ] = pca_result.residuals[ticker].values

        metadata = {
            "stages_completed": 3,
            "pca_metadata": pca_result.metadata,
            "sector_mapping": self.sector_mapping,
        }

        return FactorResult(
            residuals=final_residuals,
            factor_returns=factor_ret.astype(float),
            betas=betas_df,
            metadata=metadata,
        )
