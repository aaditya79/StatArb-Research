"""Yahoo Finance data source implementation."""
import hashlib
import logging
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import SECTOR_TO_ETF_MAP
from .base import DataSource

logger = logging.getLogger(__name__)

# Cache directory: <project_root>/.cache/
_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


def _cache_path(prefix: str, key_str: str) -> Path:
    digest = hashlib.md5(key_str.encode()).hexdigest()
    return _CACHE_DIR / f"{prefix}_{digest}.pkl"


def _load(path: Path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _save(path: Path, obj) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class YFinanceSource(DataSource):
    """
    Data source using the yfinance library.

    Fetches adjusted close prices and daily volume from Yahoo Finance.
    No credentials required.
    """

    def _fetch_raw(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Download OHLCV from yfinance, caching the result to disk."""
        key_str = f"raw|{sorted(tickers)}|{start}|{end}"
        path = _cache_path("raw", key_str)
        cached = _load(path)
        if cached is not None:
            logger.debug(f"Cache hit: {path.name}")
            return cached
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        _save(path, data)
        return data

    def fetch_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = self._fetch_raw(tickers, start, end)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers
        # Only forward-fill short gaps (1-2 days, e.g. exchange closures).
        # Long gaps (delistings, pre-IPO periods) stay NaN so the OU fitter
        # drops those days instead of generating spurious zero returns.
        prices = prices.ffill(limit=2)
        return prices

    def fetch_volume(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        data = self._fetch_raw(tickers, start, end)
        if isinstance(data.columns, pd.MultiIndex):
            volume = data["Volume"]
        else:
            volume = data[["Volume"]]
            volume.columns = tickers
        # Leave NaN where data is genuinely missing (pre-IPO, post-delist,
        # provider gap). compute_volume_adjusted_returns handles NaN by
        # falling back to the unadjusted return for that day, which is the
        # right behavior. Filling with 0 used to bias the trailing-volume
        # average downward and inflate the volume-ratio adjustment on
        # adjacent real-trading days.
        volume = volume.astype(float)
        return volume

    def fetch_sector_mapping(self, tickers: list[str]) -> dict[str, str]:
        """
        Sector lookup via yfinance `.info` metadata. Only works for
        currently-listed tickers — delisted names typically return an
        empty or error-raising info dict. Unresolved tickers are left
        out of the returned dict so the caller can fall through to
        TICKER_TO_ETF_OVERRIDES or a final "XLY" default.
        """
        key_str = f"sector|{sorted(tickers)}"
        path = _cache_path("sector", key_str)
        cached = _load(path)
        if cached is not None:
            logger.debug(f"Cache hit: {path.name}")
            return cached

        mapping: dict[str, str] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "")
                if sector and sector in SECTOR_TO_ETF_MAP:
                    mapping[ticker] = SECTOR_TO_ETF_MAP[sector]
            except Exception:
                # Delisted / no profile — skip and let the caller handle.
                logger.debug(f"yfinance .info failed for {ticker}; skipping")

        _save(path, mapping)
        return mapping
