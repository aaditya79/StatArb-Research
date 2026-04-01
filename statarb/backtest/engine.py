"""
Backtest engine: day-by-day simulation (Paper Section 5).

Orchestrates the full pipeline:
    1. For each day, re-estimate OU parameters on trailing residuals
    2. Compute s-scores
    3. Generate entry/exit signals
    4. Execute trades via PortfolioManager
    5. Record equity, positions, and trades
"""
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from config import Config
from statarb.factors.base import FactorResult
from statarb.signals.ou_estimator import estimate_ou_params, OUParams
from statarb.signals.sscore import compute_sscores
from statarb.signals.filters import filter_eligible
from statarb.signals.volume_time import compute_volume_adjusted_returns
from .portfolio import PortfolioManager
from .metrics import compute_metrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete output from a backtest run."""
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_positions: pd.DataFrame
    daily_sscores: pd.DataFrame
    metrics: PerformanceMetrics
    factor_result: FactorResult
    daily_ou_params: dict = field(default_factory=dict)


def run_backtest(
    config: Config,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    factor_result: FactorResult,
) -> BacktestResult:
    """
    Run the full day-by-day backtest.

    Args:
        config: Master configuration object.
        prices: Adjusted close prices (dates x tickers).
        volume: Daily volume (dates x tickers).
        factor_result: Pre-computed factor model output with residuals.

    Returns:
        BacktestResult with equity curve, trades, and diagnostics.
    """
    residuals = factor_result.residuals
    tickers = [t for t in residuals.columns if t in prices.columns]
    residuals = residuals[tickers]
    prices = prices[tickers]

    # Volume adjustment if enabled
    if config.volume.enabled:
        vol_tickers = [t for t in tickers if t in volume.columns]
        if vol_tickers:
            adjusted = compute_volume_adjusted_returns(
                residuals[vol_tickers],
                volume[vol_tickers],
                trailing_window=config.volume.trailing_window,
            )
            residuals[vol_tickers] = adjusted

    dates = residuals.index
    ou_window = config.ou.estimation_window
    min_start = ou_window + 10  # need enough history

    # Initialize portfolio
    portfolio = PortfolioManager(
        initial_equity=config.backtest.initial_equity,
        leverage_long=config.backtest.leverage_long,
        leverage_short=config.backtest.leverage_short,
        tc_bps=config.backtest.tc_bps,
    )

    # Storage
    equity_values = []
    equity_dates = []
    trade_records = []
    position_records = []
    sscore_records = {}
    daily_ou_params = {}

    n_target = max(len(tickers) // 2, 10)  # expected positions for sizing

    for i in range(min_start, len(dates)):
        date = dates[i]
        date_ts = pd.Timestamp(date)

        # Current prices
        current_prices = prices.iloc[i]
        price_dict = current_prices.to_dict()

        # ── Step 1: Estimate OU parameters on trailing window ──
        trailing_residuals = residuals.iloc[max(0, i - ou_window) : i]
        ou_params = {}
        for ticker in tickers:
            if ticker not in trailing_residuals.columns:
                continue
            series = trailing_residuals[ticker]
            params = estimate_ou_params(
                series, window=ou_window, dt=config.backtest.dt
            )
            if params is not None:
                ou_params[ticker] = params

        daily_ou_params[str(date)[:10]] = ou_params

        # ── Step 2: Filter by kappa ──
        eligible = filter_eligible(ou_params, kappa_min=config.ou.kappa_min)

        # ── Step 3: Compute s-scores ──
        eligible_params = {t: ou_params[t] for t in eligible}
        if not eligible_params:
            # Mark to market and record
            portfolio.mark_to_market(price_dict)
            equity_values.append(portfolio.equity)
            equity_dates.append(date)
            continue

        sscores = compute_sscores(
            residuals.iloc[:i],
            eligible_params,
            mean_center=config.ou.mean_center,
        )
        sscore_records[date] = sscores

        # ── Step 4: Check exits ──
        tickers_to_close = []
        for ticker in list(portfolio.positions.keys()):
            if ticker not in sscores.index:
                # No longer eligible -> close
                tickers_to_close.append(ticker)
                continue

            s = sscores[ticker]
            pos = portfolio.positions[ticker]

            # Exit conditions (Paper Section 4, Eq. 16)
            should_close = False
            if pos.direction == 1:  # long
                if s >= -config.signal.s_sc:
                    should_close = True
            elif pos.direction == -1:  # short
                if s <= config.signal.s_bc:
                    should_close = True

            # Force exit on extreme s-score
            if abs(s) >= config.signal.s_limit:
                should_close = True

            if should_close:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if ticker in price_dict and np.isfinite(price_dict[ticker]):
                pos = portfolio.positions.get(ticker)
                if pos is None:
                    continue
                pnl = portfolio.close_position(ticker, price_dict[ticker], date_ts)
                trade_records.append({
                    "ticker": ticker,
                    "direction": pos.direction,
                    "entry_date": pos.entry_date,
                    "exit_date": date_ts,
                    "entry_price": pos.entry_price,
                    "exit_price": price_dict[ticker],
                    "pnl": pnl,
                    "notional": pos.notional,
                })

        # ── Step 5: Check entries ──
        notional_per_pos = portfolio.compute_notional_per_position(n_target)

        for ticker in eligible:
            if ticker in portfolio.positions:
                continue  # already have a position
            if ticker not in sscores.index:
                continue
            if ticker not in price_dict or not np.isfinite(price_dict[ticker]):
                continue

            s = sscores[ticker]
            direction = None

            if s <= -config.signal.s_bo:
                direction = 1   # buy to open (long)
            elif s >= config.signal.s_so:
                direction = -1  # sell to open (short)

            if direction is not None:
                portfolio.open_position(
                    ticker=ticker,
                    direction=direction,
                    price=price_dict[ticker],
                    date=date_ts,
                    notional=notional_per_pos,
                )

        # ── Step 6: Mark to market ──
        portfolio.mark_to_market(price_dict)
        equity_values.append(portfolio.equity)
        equity_dates.append(date)

        # Record positions snapshot
        for ticker, pos in portfolio.positions.items():
            position_records.append({
                "date": date,
                "ticker": ticker,
                "direction": pos.direction,
                "notional": pos.notional,
                "entry_price": pos.entry_price,
                "current_price": price_dict.get(ticker, np.nan),
            })

    # Build output DataFrames
    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))

    trades = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["ticker", "direction", "entry_date", "exit_date",
                 "entry_price", "exit_price", "pnl", "notional"]
    )

    daily_positions = pd.DataFrame(position_records) if position_records else pd.DataFrame(
        columns=["date", "ticker", "direction", "notional",
                 "entry_price", "current_price"]
    )

    # Build s-score DataFrame
    daily_sscores = pd.DataFrame(sscore_records).T
    daily_sscores.index = pd.DatetimeIndex(daily_sscores.index)

    # Compute metrics
    metrics = compute_metrics(
        equity_curve, trades,
        risk_free_rate=config.backtest.risk_free_rate,
        total_costs=portfolio.total_costs,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        daily_positions=daily_positions,
        daily_sscores=daily_sscores,
        metrics=metrics,
        factor_result=factor_result,
        daily_ou_params=daily_ou_params,
    )
