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


def compute_pca_eigenportfolio_returns(
    returns_window: pd.DataFrame,
    n_components: int | None = 15,
    explained_variance_threshold: float = 0.55,
    use_ledoit_wolf: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA on a single 252-day window, return eigenportfolio returns
    over that window.

    This is the one-step primitive used by both the rolling factor model
    and the engine's per-day signal generation.

    Args:
        returns_window: DataFrame of log returns (T x N) for this window.
        n_components: Fixed number of eigenportfolios, or None for adaptive.
        explained_variance_threshold: Used when n_components is None.
        use_ledoit_wolf: Whether to apply Ledoit-Wolf shrinkage.

    Returns:
        eigenportfolio_returns: DataFrame (T x m), indexed by window's dates.
        eigenvalues: 1-D array of all eigenvalues (descending).
        V: eigenvector matrix (N x m) -- top m eigenvectors.
        stds: 1-D array of per-stock std devs (N,) used in standardization.
    """
    # Drop tickers with >20% NaN coverage. Remaining NaN days get a
    # backward-looking imputation: the trailing 20-day rolling mean of
    # the same column. This uses only data strictly before the NaN day
    # within the window (no in-window lookahead), and is closer to the
    # "expected return on a typical recent day" than a flat zero. Any
    # NaN that survives the rolling fill (e.g., at the very start of
    # the window) is set to the column's expanding mean and finally to
    # 0 as a last resort.
    min_obs = int(len(returns_window) * 0.80)
    w = returns_window.dropna(axis=1, thresh=min_obs)
    if w.empty or w.shape[1] < 2:
        raise ValueError("PCA window has insufficient data")

    rolling_mean = w.rolling(window=20, min_periods=5).mean().shift(1)
    expanding_mean = w.expanding(min_periods=5).mean().shift(1)
    w_filled = w.fillna(rolling_mean).fillna(expanding_mean).fillna(0.0)

    means = w_filled.mean()
    stds = w_filled.std().replace(0, 1e-10)
    standardized = (w_filled - means) / stds

    if use_ledoit_wolf:
        lw = LedoitWolf().fit(standardized.values)
        cov_matrix = lw.covariance_
    else:
        cov_matrix = standardized.cov().values

    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 1e-6)

    N = len(w.columns)
    if n_components is not None:
        m = min(n_components, N)
    else:
        total = eigenvalues.sum()
        cumulative = np.cumsum(eigenvalues) / total
        m = int(np.searchsorted(cumulative, explained_variance_threshold) + 1)
        m = min(m, N)

    # Stabilize signs so max-abs loading is positive
    for j in range(m):
        max_idx = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[max_idx, j] < 0:
            eigenvectors[:, j] *= -1

    V = eigenvectors[:, :m]
    stds_arr = stds.values

    # Eigenportfolio weights (Eq. 9): w_ij = v_ij / sigma_i
    weights = V / stds_arr[:, np.newaxis]  # (N, m)

    # Eigenportfolio returns over this window. Use the trailing-imputed
    # matrix (same backward-looking fill used for the cov matrix) so the
    # factor return on a missing-data day is the trailing-average of that
    # ticker's recent returns rather than an arbitrary zero.
    F = w_filled.values @ weights  # (T, m)
    factor_names = [f"PC{j + 1}" for j in range(m)]
    eigenport_returns = pd.DataFrame(F, index=w.index, columns=factor_names)

    return eigenport_returns, eigenvalues, V, stds_arr


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
        Rolling 252-day PCA (paper Section 2.1).

        For each trading day t >= lookback:
          - Fit PCA on returns[t-lookback:t] (no look-ahead).
          - Compute eigenportfolio weights W.
          - Project day t's returns onto W to get the m factor returns F_t.
          - Regress each stock's lookback-window returns on those
            eigenportfolio returns to get per-stock betas.
          - Residual[t] = return[t] - betas @ F_t.

        This produces paper-faithful diagnostics (ACF, residual distribution
        plots on the Factor Diagnostics page). The ENGINE'S signal path does
        its own fresh 60-day OLS per day using the same eigenportfolio
        returns, so the s-score formula s = -m/sigma_eq remains exact.

        Args:
            returns: DataFrame of log returns (dates x tickers).

        Returns:
            FactorResult with rolling PCA residuals / factor returns / betas.
        """
        T = len(returns)
        tickers_all = returns.columns.tolist()
        N = len(tickers_all)

        residuals = pd.DataFrame(np.nan, index=returns.index, columns=tickers_all)
        # We don't know m in advance if adaptive; pre-allocate a wide enough grid.
        m_cap = self.n_components if self.n_components is not None else N
        factor_names_cap = [f"PC{j + 1}" for j in range(m_cap)]
        factor_returns_out = pd.DataFrame(
            np.nan, index=returns.index, columns=factor_names_cap
        )

        last_V = None
        last_eigvals = None
        last_m = 0
        last_betas = None
        last_tickers = None

        for t in range(self.lookback, T):
            window = returns.iloc[t - self.lookback:t]
            try:
                eigenport_returns, eigvals, V, stds_arr = compute_pca_eigenportfolio_returns(
                    window,
                    n_components=self.n_components,
                    explained_variance_threshold=self.explained_variance_threshold,
                    use_ledoit_wolf=self.use_ledoit_wolf,
                )
            except Exception:
                continue

            surviving = eigenport_returns.columns.tolist()
            m_current = len(surviving)
            # Tickers surviving the PCA dropna step
            w_tickers = [
                t_ for t_ in window.dropna(axis=1, thresh=int(len(window) * 0.80))
                                  .dropna(how="any").columns.tolist()
            ]

            # Day t factor returns: project day t's standardized returns onto W
            r_t_full = returns.iloc[t]
            r_t_vals = r_t_full.reindex(w_tickers).values
            if np.any(~np.isfinite(r_t_vals)):
                continue
            weights = V / stds_arr[:, np.newaxis]
            F_t = r_t_vals @ weights  # (m,)

            factor_returns_out.iloc[t, :m_current] = F_t

            # Regress each stock's window returns on eigenportfolio returns (window)
            F_hist = eigenport_returns.values  # (T_w, m)
            Y_hist = window[w_tickers].values  # (T_w, N_surv)
            try:
                betas, *_ = np.linalg.lstsq(F_hist, Y_hist, rcond=None)
            except np.linalg.LinAlgError:
                continue
            # betas shape (m, N_surv)

            predicted_t = betas.T @ F_t  # (N_surv,)
            resid_t = r_t_vals - predicted_t

            for j, tk in enumerate(w_tickers):
                residuals.iloc[t, residuals.columns.get_loc(tk)] = resid_t[j]

            # Keep most recent fit for final metadata
            last_V = V
            last_eigvals = eigvals
            last_m = m_current
            last_betas = pd.DataFrame(betas.T, index=w_tickers, columns=surviving)
            last_tickers = w_tickers

        if last_V is None:
            raise ValueError(
                "Rolling PCA produced no fits. Ensure start_date allows >=252 "
                "days of data before any trading date."
            )

        betas_df = pd.DataFrame(0.0, index=tickers_all, columns=factor_names_cap)
        for tk in last_tickers:
            for col in last_betas.columns:
                betas_df.loc[tk, col] = last_betas.loc[tk, col]

        explained_var = last_eigvals[:last_m].sum() / last_eigvals.sum()

        metadata = {
            "eigenvalues": last_eigvals[:last_m],
            "all_eigenvalues": last_eigvals,
            "explained_variance_ratio": explained_var,
            "n_components": last_m,
            "eigenvectors": last_V,
            "lookback": self.lookback,
            "rolling": True,
        }

        return FactorResult(
            residuals=residuals,
            factor_returns=factor_returns_out,
            betas=betas_df,
            metadata=metadata,
        )
