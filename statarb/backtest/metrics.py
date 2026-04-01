from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    trade_win_rate: float
    profit_factor: float
    avg_holding_period: float
    avg_daily_turnover: float
    num_trades: int
    total_costs: float
    turnover: float


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.02,
    total_costs: float = 0.0,
) -> PerformanceMetrics:
    if len(equity_curve) < 2:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_vol=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, win_rate=0, trade_win_rate=0,
            profit_factor=0, avg_holding_period=0, avg_daily_turnover=0,
            num_trades=0, total_costs=total_costs, turnover=0,
        )

    daily_returns = equity_curve.pct_change().dropna()
    n_days = len(daily_returns)
    n_years = n_days / 252.0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    annualized_return = (1 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0

    annualized_vol = daily_returns.std() * np.sqrt(252)

    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    sharpe = (
        excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252)
    )

    downside = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-10
    sortino = excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(252)

    dd_series = compute_drawdown_series(equity_curve)
    max_drawdown = dd_series.min()

    is_underwater = dd_series < 0
    if is_underwater.any():
        underwater_groups = (~is_underwater).cumsum()
        underwater_groups[~is_underwater] = np.nan
        durations = underwater_groups.groupby(underwater_groups).transform("count")
        max_dd_duration = int(durations.max()) if not durations.isna().all() else 0
    else:
        max_dd_duration = 0

    win_rate = (daily_returns > 0).mean()

    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-10)

    num_trades = len(trades) if trades is not None and not trades.empty else 0
    avg_holding = 0.0
    if num_trades > 0 and "entry_date" in trades.columns and "exit_date" in trades.columns:
        holding_days = (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        ).dt.days
        avg_holding = holding_days.mean()

    turnover = 0.0
    total_traded = 0.0
    avg_equity = equity_curve.mean()
    if num_trades > 0 and "notional" in trades.columns:
        total_traded = trades["notional"].sum() * 2
        turnover = total_traded / (avg_equity * n_years + 1e-10)

    trade_win_rate = 0.0
    if num_trades > 0 and "pnl" in trades.columns:
        trade_win_rate = float((trades["pnl"] > 0).mean())

    avg_daily_turnover = 0.0
    if num_trades > 0 and "notional" in trades.columns:
        avg_daily_turnover = total_traded / (avg_equity * n_days + 1e-10)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        win_rate=win_rate,
        trade_win_rate=trade_win_rate,
        profit_factor=profit_factor,
        avg_holding_period=avg_holding,
        avg_daily_turnover=avg_daily_turnover,
        num_trades=num_trades,
        total_costs=total_costs,
        turnover=turnover,
    )


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown
