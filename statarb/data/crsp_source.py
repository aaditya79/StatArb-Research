"""CRSP data source via WRDS."""
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import SIC_TO_ETF, SIC3_TO_ETF_OVERRIDES
from .base import DataSource

load_dotenv()


class CRSPSource(DataSource):
    """
    Data source using CRSP daily stock data via the WRDS database.

    Requires WRDS_USERNAME and WRDS_PASSWORD in the .env file.
    Uses the wrds Python library to connect and query.
    """

    def __init__(self):
        self._conn = None

    def _connect(self):
        if self._conn is None:
            try:
                import wrds
            except ImportError:
                raise ImportError(
                    "wrds package required for CRSP data. "
                    "Install with: pip install wrds"
                )
            username = os.getenv("WRDS_USERNAME")
            if not username:
                raise ValueError(
                    "WRDS_USERNAME not found in environment. "
                    "Set it in your .env file."
                )
            self._conn = wrds.Connection(wrds_username=username)
        return self._conn

    @staticmethod
    def _validate_tickers(tickers: list[str]) -> None:
        """Reject tickers that could be used for SQL injection."""
        import re
        for t in tickers:
            if not re.fullmatch(r"[A-Z]{1,10}", t):
                raise ValueError(f"Invalid ticker symbol: {t!r}")

    def _ticker_to_permno(self, tickers: list[str]) -> pd.DataFrame:
        """Map ticker symbols to CRSP PERMNOs (most recent mapping per ticker)."""
        self._validate_tickers(tickers)
        conn = self._connect()
        ticker_str = "', '".join(tickers)
        query = f"""
            SELECT DISTINCT ticker, permno, nameenddt
            FROM crsp.stocknames
            WHERE ticker IN ('{ticker_str}')
        """
        mapping = conn.raw_sql(query)
        # A ticker can map to multiple PERMNOs over time; keep the most recent.
        # NULL nameenddt means the name is still active — treat as far future.
        mapping["nameenddt"] = mapping["nameenddt"].fillna(pd.Timestamp("2099-12-31"))
        mapping = (
            mapping
            .sort_values("nameenddt", ascending=False)
            .drop_duplicates(subset=["ticker"], keep="first")
            .drop(columns=["nameenddt"])
        )
        return mapping

    def fetch_prices(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        conn = self._connect()
        mapping = self._ticker_to_permno(tickers)
        if mapping.empty:
            raise ValueError(f"No PERMNOs found for tickers: {tickers}")

        permno_str = ", ".join(str(p) for p in mapping["permno"].unique())
        query = f"""
            SELECT date, permno, ABS(prc) AS price, cfacpr
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date, permno
        """
        raw = conn.raw_sql(query)
        raw["adj_price"] = raw["price"] / raw["cfacpr"]

        permno_to_ticker = dict(
            zip(mapping["permno"], mapping["ticker"])
        )
        raw["ticker"] = raw["permno"].map(permno_to_ticker)
        raw = raw.drop_duplicates(subset=["date", "ticker"], keep="last")

        prices = raw.pivot(index="date", columns="ticker", values="adj_price")
        prices.index = pd.to_datetime(prices.index)
        prices = prices[
            [t for t in tickers if t in prices.columns]
        ]
        # Only forward-fill short gaps; leave pre-IPO / post-delisting days NaN.
        prices = prices.ffill(limit=2).astype(float)
        return prices

    def fetch_volume(
        self, tickers: list[str], start: str, end: str
    ) -> pd.DataFrame:
        conn = self._connect()
        mapping = self._ticker_to_permno(tickers)
        if mapping.empty:
            raise ValueError(f"No PERMNOs found for tickers: {tickers}")

        permno_str = ", ".join(str(p) for p in mapping["permno"].unique())
        query = f"""
            SELECT date, permno, vol AS volume
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date, permno
        """
        raw = conn.raw_sql(query)

        permno_to_ticker = dict(
            zip(mapping["permno"], mapping["ticker"])
        )
        raw["ticker"] = raw["permno"].map(permno_to_ticker)
        raw = raw.drop_duplicates(subset=["date", "ticker"], keep="last")

        volume = raw.pivot(index="date", columns="ticker", values="volume")
        volume.index = pd.to_datetime(volume.index)
        volume = volume[
            [t for t in tickers if t in volume.columns]
        ]
        # Leave NaN for genuinely-missing days (pre-listing, post-delisting,
        # CRSP gaps). compute_volume_adjusted_returns falls back to the raw
        # return when the volume ratio is NaN — see yfinance_source for the
        # same rationale.
        volume = volume.astype(float)
        return volume

    def fetch_sector_mapping(self, tickers: list[str]) -> dict[str, str]:
        """
        Sector lookup via CRSP's `stocknames` SIC code (`hsiccd`), mapped
        through SIC_TO_ETF / SIC3_TO_ETF_OVERRIDES in config.py.

        Works for delisted names because stocknames carries the full
        listing history per PERMNO. Picks the most recent SIC code per
        ticker when a name changed industry codes across its life.
        """
        self._validate_tickers(tickers)
        conn = self._connect()
        ticker_str = "', '".join(tickers)
        # crsp.stocknames uses `siccd` as the column name (not `hsiccd` —
        # that's the column in crsp.msenames / dsenames header tables).
        query = f"""
            SELECT ticker, siccd, nameenddt
            FROM crsp.stocknames
            WHERE ticker IN ('{ticker_str}')
              AND siccd IS NOT NULL
        """
        raw = conn.raw_sql(query)
        if raw.empty:
            return {}

        # Most recent SIC per ticker (same pattern as _ticker_to_permno).
        raw["nameenddt"] = raw["nameenddt"].fillna(pd.Timestamp("2099-12-31"))
        raw = (
            raw
            .sort_values("nameenddt", ascending=False)
            .drop_duplicates(subset=["ticker"], keep="first")
        )

        mapping: dict[str, str] = {}
        for _, row in raw.iterrows():
            tk = row["ticker"]
            try:
                sic = int(row["siccd"])
            except (TypeError, ValueError):
                continue
            sic3 = sic // 10           # 3-digit prefix (e.g. 2834 → 283)
            sic2 = sic // 100          # 2-digit prefix (e.g. 2834 → 28)
            etf = SIC3_TO_ETF_OVERRIDES.get(sic3) or SIC_TO_ETF.get(sic2)
            if etf:
                mapping[tk] = etf
        return mapping