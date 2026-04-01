"""Reusable KPI card row for the dashboard."""
import streamlit as st

from statarb.backtest.metrics import PerformanceMetrics


def render_kpi_cards(metrics: PerformanceMetrics):
    """Render a row of KPI metric cards."""
    cols = st.columns(6)

    cols[0].metric(
        "Total Return",
        f"{metrics.total_return:.1%}",
    )
    cols[1].metric(
        "Sharpe Ratio",
        f"{metrics.sharpe_ratio:.2f}",
    )
    cols[2].metric(
        "Max Drawdown",
        f"{metrics.max_drawdown:.1%}",
    )
    cols[3].metric(
        "Win Rate",
        f"{metrics.win_rate:.1%}",
    )
    cols[4].metric(
        "Num Trades",
        f"{metrics.num_trades:,}",
    )
    cols[5].metric(
        "Total Costs",
        f"${metrics.total_costs:,.0f}",
    )
