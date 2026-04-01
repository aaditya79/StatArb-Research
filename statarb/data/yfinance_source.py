"""Yahoo Finance data source implementation."""
import pandas as pd
import yfinance as yf

from .base import DataSource


class YFinanceSource(DataSource):
    """
    Data source using the yfinance library.

    Fetches adjusted close prices and daily volume from Yahoo Finance.
    No credentials required.
    """

    def fetch_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers
        prices = prices.ffill().bfill()
        return prices

    def fetch_volume(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
        if isinstance(data.columns, pd.MultiIndex):
            volume = data["Volume"]
        else:
            volume = data[["Volume"]]
            volume.columns = tickers
        volume = volume.ffill().bfill()
        return volume
