"""Abstract base class and data structures for factor models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """
    Standardized output from any factor model.

    Attributes:
        residuals: DataFrame (dates x tickers) of idiosyncratic returns.
        factor_returns: DataFrame (dates x factor_names) of factor returns.
        betas: DataFrame (tickers x factor_names) of factor loadings.
        metadata: Dict with model-specific info (eigenvalues, R2, etc.).
    """
    residuals: pd.DataFrame
    factor_returns: pd.DataFrame
    betas: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class FactorModel(ABC):
    """
    Abstract interface for factor models.

    All factor models (PCA, ETF, Combined) implement this interface,
    ensuring they can be used interchangeably in the backtest engine.
    """

    @abstractmethod
    def fit(self, returns: pd.DataFrame, **kwargs) -> FactorResult:
        """
        Fit the factor model and extract residuals.

        Args:
            returns: DataFrame of log returns (dates x tickers).
            **kwargs: Model-specific arguments (e.g., etf_returns).

        Returns:
            FactorResult with residuals, factor returns, betas, metadata.
        """
        ...
