"""Ticker universe utilities: sector mapping and data source factory."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import SECTOR_TO_ETF_MAP

if TYPE_CHECKING:
    from .base import DataSource

logger = logging.getLogger(__name__)


def get_sector_mapping(tickers: list[str]) -> dict[str, str]:
    """
    Map each ticker to its sector ETF.

    Uses yfinance .info metadata to look up the sector, then maps
    to the corresponding ETF via SECTOR_TO_ETF_MAP. Falls back to
    XLK if sector is unknown.

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dict mapping ticker -> sector ETF symbol.
    """
    import yfinance as yf

    mapping = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "")
            etf = SECTOR_TO_ETF_MAP.get(sector, "XLK")
            mapping[ticker] = etf
        except Exception:
            logger.warning(
                f"Could not fetch sector for {ticker}, defaulting to XLK"
            )
            mapping[ticker] = "XLK"
    return mapping


def get_data_source(name: str) -> "DataSource":
    """
    Factory function to create data source by name.

    Args:
        name: One of "yfinance", "crsp".

    Returns:
        An instance of the corresponding DataSource subclass.

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "yfinance":
        from .yfinance_source import YFinanceSource
        return YFinanceSource()
    elif name == "crsp":
        from .crsp_source import CRSPSource
        return CRSPSource()
    else:
        raise ValueError(
            f"Unknown data source: '{name}'. "
            f"Available sources: yfinance, crsp"
        )
