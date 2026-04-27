"""Abstract base class for data sources."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataSource(ABC):
    """
    Abstract interface for all data providers.

    To add a new data source, subclass this and implement fetch_prices()
    and fetch_volume(). The fetch_returns() method has a default
    implementation using log returns but can be overridden.
    """

    @abstractmethod
    def fetch_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices.

        Args:
            tickers: List of ticker symbols.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex and ticker columns.
        """
        ...

    @abstractmethod
    def fetch_volume(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Fetch daily trading volume.

        Args:
            tickers: List of ticker symbols.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with DatetimeIndex and ticker columns.
        """
        ...

    def fetch_returns(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Compute log returns from prices.

        Default implementation: ln(P_t / P_{t-1}).
        Override if your data source provides returns directly.
        """
        prices = self.fetch_prices(tickers, start, end)
        return np.log(prices / prices.shift(1)).dropna(how="all")

    def fetch_sector_mapping(self, tickers: list[str]) -> dict[str, str]:
        """
        Map each ticker to its sector ETF symbol.

        Default: returns empty dict. Subclasses override with their
        provider-specific sector lookup (yfinance `.info` metadata,
        CRSP SIC codes, etc.).

        Callers in universe.py layer a TICKER_TO_ETF_OVERRIDES dict
        and a fallback "XLY" on top of whatever this returns, so a
        partial or empty result from the source is fine.
        """
        return {}
