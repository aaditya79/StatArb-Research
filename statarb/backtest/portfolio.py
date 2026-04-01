"""
Portfolio management: position tracking, leverage, and beta-neutral hedging.

Implements the bang-bang position sizing from Paper Section 5:
equal-notional allocation with 2+2 leverage (2x long, 2x short).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import compute_transaction_cost


@dataclass
class Position:
    """A single stock position."""
    ticker: str
    direction: int              # +1 long, -1 short
    entry_date: pd.Timestamp
    entry_price: float
    notional: float             # dollar amount invested
    quantity: float             # shares (can be fractional)


class PortfolioManager:
    """
    Manages positions, equity, and leverage for the backtest.

    Args:
        initial_equity: Starting portfolio value.
        leverage_long: Maximum long leverage (default: 2.0).
        leverage_short: Maximum short leverage (default: 2.0).
        tc_bps: Transaction cost in basis points per side.
    """

    def __init__(
        self,
        initial_equity: float = 1_000_000.0,
        leverage_long: float = 2.0,
        leverage_short: float = 2.0,
        tc_bps: float = 5.0,
    ):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.leverage_long = leverage_long
        self.leverage_short = leverage_short
        self.tc_bps = tc_bps
        self.positions: dict[str, Position] = {}
        self.cash = initial_equity
        self.total_costs = 0.0

    @property
    def long_exposure(self) -> float:
        return sum(
            p.notional for p in self.positions.values() if p.direction == 1
        )

    @property
    def short_exposure(self) -> float:
        return sum(
            abs(p.notional) for p in self.positions.values() if p.direction == -1
        )

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    @property
    def net_exposure(self) -> float:
        return self.long_exposure - self.short_exposure

    def compute_notional_per_position(self, n_target_positions: int) -> float:
        """
        Compute equal-notional allocation per position.

        Paper: lambda_t = leverage / n_expected_positions
        The amount invested per stock is equity * lambda_t.
        """
        if n_target_positions <= 0:
            return 0.0
        # Average leverage across long and short
        avg_leverage = (self.leverage_long + self.leverage_short) / 2.0
        return self.equity * avg_leverage / n_target_positions

    def open_position(
        self,
        ticker: str,
        direction: int,
        price: float,
        date: pd.Timestamp,
        notional: float,
    ) -> Position | None:
        """
        Open a new position (bang-bang: full allocation at once).

        Returns the Position if opened, None if already exists.
        """
        if ticker in self.positions:
            return None

        if price <= 0 or notional <= 0:
            return None

        # Check leverage limits
        if direction == 1:
            if self.long_exposure + notional > self.equity * self.leverage_long:
                return None
        else:
            if self.short_exposure + notional > self.equity * self.leverage_short:
                return None

        quantity = notional / price
        cost = compute_transaction_cost(notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost

        pos = Position(
            ticker=ticker,
            direction=direction,
            entry_date=date,
            entry_price=price,
            notional=notional,
            quantity=quantity,
        )
        self.positions[ticker] = pos
        return pos

    def close_position(
        self, ticker: str, price: float, date: pd.Timestamp
    ) -> float:
        """
        Close an existing position. Returns realized PnL.
        """
        if ticker not in self.positions:
            return 0.0

        pos = self.positions[ticker]
        pnl = pos.direction * pos.quantity * (price - pos.entry_price)

        close_notional = abs(pos.quantity * price)
        cost = compute_transaction_cost(close_notional, self.tc_bps)
        self.cash -= cost
        self.total_costs += cost

        self.cash += pnl
        self.equity += pnl - cost

        del self.positions[ticker]
        return pnl

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """
        Update equity with unrealized PnL. Returns total daily PnL.
        """
        daily_pnl = 0.0
        for ticker, pos in self.positions.items():
            if ticker in prices and np.isfinite(prices[ticker]):
                current_price = prices[ticker]
                unrealized = pos.direction * pos.quantity * (
                    current_price - pos.entry_price
                )
                daily_pnl += unrealized

        self.equity = self.cash + sum(
            pos.direction * pos.quantity * prices.get(pos.ticker, pos.entry_price)
            - pos.direction * pos.quantity * pos.entry_price
            for pos in self.positions.values()
        ) + self.initial_equity - (self.initial_equity - self.cash + self.total_costs)

        # Simpler: equity = cash + unrealized PnL
        unrealized_total = sum(
            pos.direction * pos.quantity * (
                prices.get(pos.ticker, pos.entry_price) - pos.entry_price
            )
            for pos in self.positions.values()
        )
        self.equity = self.cash + unrealized_total

        return daily_pnl
