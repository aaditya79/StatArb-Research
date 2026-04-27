"""Ticker universe utilities: sector mapping and data source factory."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import TICKER_TO_ETF_OVERRIDES

if TYPE_CHECKING:
    from .base import DataSource

logger = logging.getLogger(__name__)

# Final fallback when neither an override nor the data source resolves
# a ticker. XLY (broad consumer discretionary) is preferred over XLK
# because unknown mid/small-caps historically skew consumer/industrial.
_DEFAULT_FALLBACK_ETF = "XLY"


def get_sector_mapping(
    tickers: list[str],
    data_source: "DataSource | None" = None,
) -> dict[str, str]:
    """
    Map each ticker to its sector ETF. Three-stage lookup:

        1. TICKER_TO_ETF_OVERRIDES (config.py) — hardcoded fixes for
           delisted names and known-ambiguous symbol collisions.
        2. data_source.fetch_sector_mapping(tickers) — yfinance .info
           or CRSP SIC-code lookup, depending on the selected source.
        3. _DEFAULT_FALLBACK_ETF — for any ticker still unresolved.

    This replaces the old "yfinance-or-XLK" fallback which misrouted
    every delisted ticker to the tech ETF.

    Args:
        tickers: List of ticker symbols.
        data_source: Active DataSource (YFinanceSource or CRSPSource).
            If None, only overrides + fallback are used.

    Returns:
        Dict mapping ticker -> sector ETF symbol.
    """
    mapping: dict[str, str] = {}
    remaining: list[str] = []

    # Stage 1: hard-coded overrides (delisted names, known collisions).
    for ticker in tickers:
        if ticker in TICKER_TO_ETF_OVERRIDES:
            mapping[ticker] = TICKER_TO_ETF_OVERRIDES[ticker]
        else:
            remaining.append(ticker)

    # Stage 2: delegate to the data source (yfinance .info or CRSP SIC).
    if remaining and data_source is not None:
        try:
            source_map = data_source.fetch_sector_mapping(remaining)
            mapping.update(source_map)
        except Exception as e:
            logger.warning(f"data_source.fetch_sector_mapping failed: {e}")

    # Stage 3: final fallback for anything still unresolved.
    unresolved = [t for t in tickers if t not in mapping]
    if unresolved:
        logger.info(
            f"{len(unresolved)} tickers had no sector resolution; "
            f"defaulting to {_DEFAULT_FALLBACK_ETF}. Sample: {unresolved[:10]}"
        )
        for t in unresolved:
            mapping[t] = _DEFAULT_FALLBACK_ETF

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
