"""
PCA Factor Model (Paper Section 2.1).

Extracts eigenportfolios from the correlation matrix of stock returns
and uses them as risk factors. Residuals are the idiosyncratic component
after projecting out the top eigenportfolios.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .base import FactorModel, FactorResult


class PCAFactorModel(FactorModel):
    """
    PCA-based factor model following Avellaneda & Lee (2010).

    Args:
        n_components: Fixed number of eigenportfolios. If None, use
            explained_variance_threshold to select adaptively.
        explained_variance_threshold: Fraction of total variance to explain
            when n_components is None (paper default: 0.55).
        use_ledoit_wolf: Whether to use Ledoit-Wolf shrinkage for the
            covariance matrix (recommended for N > M).
        lookback: Number of days for the correlation matrix window
            (paper default: 252).
    """

    def __init__(
        self,
        n_components: int | None = 15,
        explained_variance_threshold: float = 0.55,
        use_ledoit_wolf: bool = True,
        lookback: int = 252,
    ):
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.use_ledoit_wolf = use_ledoit_wolf
        self.lookback = lookback

    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        """
        Fit PCA factor model.

        Steps (Paper Section 2.1):
            1. Use trailing `lookback` days of returns
            2. Compute correlation matrix (with optional Ledoit-Wolf shrinkage)
            3. Eigendecompose; select top-m eigenvalues
            4. Construct eigenportfolio returns (Eq. 9)
            5. Regress each stock on eigenportfolios to get betas
            6. Residual = actual - predicted

        Args:
            returns: DataFrame of log returns (dates x tickers).

        Returns:
            FactorResult with PCA residuals and eigenportfolio diagnostics.
        """
        returns_window = returns.iloc[-self.lookback:]

        # Drop tickers with more than 20% missing data in the window, then drop remaining NaN rows
        min_obs = int(len(returns_window) * 0.80)
        returns_window = returns_window.dropna(axis=1, thresh=min_obs).dropna(how="any")

        if len(returns_window) == 0:
            raise ValueError(
                "No complete rows in the PCA lookback window after dropping tickers with "
                "missing data. Shorten the lookback window or use a later start date."
            )

        tickers = returns_window.columns.tolist()
        N = len(tickers)
        means = returns_window.mean()
        stds = returns_window.std()
        stds = stds.replace(0, 1e-10)
        standardized = (returns_window - means) / stds

        # Correlation / covariance matrix
        if self.use_ledoit_wolf:
            lw = LedoitWolf().fit(standardized.values)
            cov_matrix = lw.covariance_
        else:
            cov_matrix = standardized.cov().values

        # Symmetrize and floor eigenvalues
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Floor small eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-6)

        # Select number of components
        if self.n_components is not None:
            m = min(self.n_components, N)
        else:
            total_var = eigenvalues.sum()
            cumulative = np.cumsum(eigenvalues) / total_var
            m = int(np.searchsorted(cumulative, self.explained_variance_threshold) + 1)
            m = min(m, N)

        # Stabilize eigenvector signs (flip so max-abs loading is positive)
        for j in range(m):
            max_idx = np.argmax(np.abs(eigenvectors[:, j]))
            if eigenvectors[max_idx, j] < 0:
                eigenvectors[:, j] *= -1

        # Top-m eigenvectors
        V = eigenvectors[:, :m]  # (N, m)
        top_eigenvalues = eigenvalues[:m]

        # Eigenportfolio returns: F_jk = sum_i (v_i^j / sigma_i) * R_ik  (Eq. 9)
        stds_arr = stds.values
        weights = V / stds_arr[:, np.newaxis]  # (N, m)

        factor_returns_arr = returns_window.values @ weights  # (T, m)
        factor_names = [f"PC{j+1}" for j in range(m)]
        factor_returns = pd.DataFrame(
            factor_returns_arr,
            index=returns_window.index,
            columns=factor_names,
        )

        # Betas: regress each stock's returns on factor returns
        # R_i = sum_j beta_ij * F_j + residual
        F = factor_returns_arr  # (T, m)
        FtF_inv = np.linalg.pinv(F.T @ F)
        betas_arr = FtF_inv @ F.T @ returns_window.values  # (m, N)
        betas_arr = betas_arr.T  # (N, m)

        betas = pd.DataFrame(
            betas_arr, index=tickers, columns=factor_names
        )

        # Residuals over the full returns DataFrame (restricted to surviving tickers)
        returns_full = returns[tickers]
        full_factor_returns = returns_full.values @ weights  # (T_full, m)
        predicted = full_factor_returns @ betas_arr.T  # (T_full, N)
        residuals_arr = returns_full.values - predicted

        residuals = pd.DataFrame(
            residuals_arr, index=returns.index, columns=tickers
        )

        # Full factor returns for all dates
        full_factor_df = pd.DataFrame(
            full_factor_returns,
            index=returns_full.index,
            columns=factor_names,
        )

        explained_var = top_eigenvalues.sum() / eigenvalues.sum()

        metadata = {
            "eigenvalues": top_eigenvalues,
            "all_eigenvalues": eigenvalues,
            "explained_variance_ratio": explained_var,
            "n_components": m,
            "eigenvectors": V,
            "lookback": self.lookback,
        }

        return FactorResult(
            residuals=residuals,
            factor_returns=full_factor_df,
            betas=betas,
            metadata=metadata,
        )
