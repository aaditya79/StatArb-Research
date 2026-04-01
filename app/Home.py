import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from config import Config, SECTOR_ETFS, MARKET_ETF
from statarb.data.universe import get_data_source, get_sector_mapping
from statarb.factors.registry import build_factor_model
from statarb.backtest.engine import run_backtest
from app.state import (
    set_config, set_backtest_result, get_backtest_result, has_backtest_result,
    set_prices, set_volume,
)
from app.components.sidebar import build_sidebar
from app.components.kpi_cards import render_kpi_cards
from app.components.charts import (
    plot_equity_curve, plot_drawdown, plot_gross_exposure,
    plot_sscore_timeseries,
)


st.set_page_config(
    page_title="StatArb Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Statistical Arbitrage Dashboard")
st.caption("Avellaneda & Lee (2010) — PCA & ETF Mean-Reversion Strategies")

config = build_sidebar()
set_config(config)

if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    with st.spinner("Fetching data..."):
        data_source = get_data_source(config.data_source)

        all_tickers = config.tickers
        prices = data_source.fetch_prices(all_tickers, config.start_date, config.end_date)
        volume = data_source.fetch_volume(all_tickers, config.start_date, config.end_date)
        returns = data_source.fetch_returns(all_tickers, config.start_date, config.end_date)

        available = [t for t in all_tickers if t in prices.columns]
        prices = prices[available]
        volume = volume[[t for t in available if t in volume.columns]]
        returns = returns[[t for t in available if t in returns.columns]]

        set_prices(prices)
        set_volume(volume)

    with st.spinner("Computing sector mappings..."):
        sector_mapping = get_sector_mapping(available)

    with st.spinner("Fitting factor model..."):
        factor_model = build_factor_model(config.factor, sector_mapping, pairs_cfg=config.pairs)

        kwargs = {}
        if config.factor.model_type in ("etf", "combined"):
            etf_tickers = list(set(sector_mapping.values()))
            etf_prices = data_source.fetch_prices(
                etf_tickers, config.start_date, config.end_date
            )
            etf_returns = np.log(etf_prices / etf_prices.shift(1)).dropna(how="all")
            kwargs["etf_returns"] = etf_returns

        if config.factor.model_type == "combined":
            spy_prices = data_source.fetch_prices(
                [MARKET_ETF], config.start_date, config.end_date
            )
            spy_returns = np.log(spy_prices / spy_prices.shift(1)).dropna(how="all")
            kwargs["spy_returns"] = spy_returns

        if config.factor.model_type == "pairs":
            kwargs["prices"] = prices

        factor_result = factor_model.fit(returns, **kwargs)

    with st.spinner("Running backtest..."):
        if config.factor.model_type == "pairs":
            pair_prices = {}
            for col in factor_result.residuals.columns:
                cs = factor_result.residuals[col].cumsum()
                pair_prices[col] = 100 * np.exp(cs - cs.iloc[0])
            bt_prices = pd.DataFrame(pair_prices)
            bt_volume = pd.DataFrame(
                np.ones(bt_prices.shape), index=bt_prices.index, columns=bt_prices.columns
            )
        else:
            bt_prices = prices
            bt_volume = volume

        result = run_backtest(config, bt_prices, bt_volume, factor_result)
        set_backtest_result(result)

    st.success(
        f"Backtest complete: {result.metrics.num_trades} trades, "
        f"Sharpe = {result.metrics.sharpe_ratio:.2f}"
    )

if has_backtest_result():
    result = get_backtest_result()

    st.subheader("Performance Summary")
    render_kpi_cards(result.metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_equity_curve(result.equity_curve),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_drawdown(result.equity_curve),
            use_container_width=True,
        )

    st.plotly_chart(
        plot_gross_exposure(result.daily_positions, result.equity_curve),
        use_container_width=True,
    )

    st.subheader("Current S-Scores")
    if not result.daily_sscores.empty:
        last_sscores = result.daily_sscores.iloc[-1].dropna().sort_values()
        sscore_df = pd.DataFrame({
            "Ticker": last_sscores.index,
            "S-Score": last_sscores.values,
            "Signal": [
                "LONG" if s <= -config.signal.s_bo
                else "SHORT" if s >= config.signal.s_so
                else "NEUTRAL"
                for s in last_sscores.values
            ],
        })

        def color_signal(val):
            if val == "LONG":
                return "background-color: rgba(44, 160, 44, 0.3)"
            elif val == "SHORT":
                return "background-color: rgba(214, 39, 40, 0.3)"
            return ""

        st.dataframe(
            sscore_df.style.applymap(color_signal, subset=["Signal"]),
            use_container_width=True,
            height=400,
        )

    st.subheader("Per-Ticker Drill-Down")
    if not result.daily_sscores.empty:
        available_tickers = sorted(result.daily_sscores.columns.tolist())
        selected_ticker = st.selectbox("Select Ticker", available_tickers)

        if selected_ticker and selected_ticker in result.daily_sscores.columns:
            ticker_sscores = result.daily_sscores[selected_ticker].dropna()
            if not ticker_sscores.empty:
                st.plotly_chart(
                    plot_sscore_timeseries(
                        ticker_sscores, selected_ticker, config.signal
                    ),
                    use_container_width=True,
                )

            if not result.trades.empty:
                ticker_trades = result.trades[
                    result.trades["ticker"] == selected_ticker
                ]
                if not ticker_trades.empty:
                    st.write(f"**Trades for {selected_ticker}:**")
                    st.dataframe(ticker_trades, use_container_width=True)

    st.subheader("Annual Performance")
    eq = result.equity_curve
    yearly = eq.resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()
    if not yearly_ret.empty:
        ann_df = pd.DataFrame({
            "Year": yearly_ret.index.year,
            "Return": [f"{r:.1%}" for r in yearly_ret.values],
        })
        st.dataframe(ann_df, use_container_width=True)
else:
    st.info("Configure parameters in the sidebar and click **Run Backtest** to begin.")
