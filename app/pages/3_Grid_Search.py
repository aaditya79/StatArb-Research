import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import copy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.state import get_backtest_result, get_config, has_backtest_result, get_prices, get_volume
from statarb.backtest.engine import run_backtest


st.set_page_config(page_title="Grid Search", layout="wide")
st.title("Signal Threshold Grid Search")
st.caption("Re-runs the backtest engine across a grid of s_bo / s_so values using the cached factor model.")

if not has_backtest_result():
    st.warning("Run a backtest on the Home page first, then return here.")
    st.stop()

prices = get_prices()
volume = get_volume()

if prices is None or volume is None:
    st.warning("Prices and volume not found in session. Please re-run the backtest from Home.")
    st.stop()

result = get_backtest_result()
config = get_config()
factor_result = result.factor_result

st.sidebar.header("Grid Search Settings")

s_min = st.sidebar.number_input("s_bo / s_so min", 0.25, 2.0, 0.5, 0.25)
s_max = st.sidebar.number_input("s_bo / s_so max", 0.5, 4.0, 2.0, 0.25)
s_step = st.sidebar.number_input("Step size", 0.05, 0.5, 0.25, 0.05)

fix_sc_bc = st.sidebar.checkbox("Fix s_sc and s_bc", value=True)
if not fix_sc_bc:
    s_sc_val = st.sidebar.number_input("s_sc (exit long)", 0.1, 2.0, float(config.signal.s_sc), 0.05)
    s_bc_val = st.sidebar.number_input("s_bc (exit short)", 0.1, 2.0, float(config.signal.s_bc), 0.05)
else:
    s_sc_val = config.signal.s_sc
    s_bc_val = config.signal.s_bc

run_grid = st.button("Run Grid Search", type="primary")

if run_grid:
    s_values = list(np.arange(s_min, s_max + s_step / 2, s_step))
    s_values = [round(v, 4) for v in s_values]
    total_combos = len(s_values) ** 2

    st.info(f"Running {total_combos} backtest combinations ({len(s_values)} × {len(s_values)} grid)...")

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

    rows = []
    progress = st.progress(0)
    total = len(s_values) ** 2
    count = 0

    for sbo in s_values:
        for sso in s_values:
            cfg_copy = copy.deepcopy(config)
            cfg_copy.signal.s_bo = sbo
            cfg_copy.signal.s_so = sso
            cfg_copy.signal.s_sc = s_sc_val
            cfg_copy.signal.s_bc = s_bc_val

            try:
                r = run_backtest(cfg_copy, bt_prices, bt_volume, factor_result)
                rows.append({
                    "s_bo": sbo,
                    "s_so": sso,
                    "Sharpe": round(r.metrics.sharpe_ratio, 3),
                    "Max DD": round(r.metrics.max_drawdown, 4),
                    "Total Return": round(r.metrics.total_return, 4),
                    "Num Trades": r.metrics.num_trades,
                })
            except Exception:
                rows.append({
                    "s_bo": sbo,
                    "s_so": sso,
                    "Sharpe": np.nan,
                    "Max DD": np.nan,
                    "Total Return": np.nan,
                    "Num Trades": 0,
                })

            count += 1
            progress.progress(count / total)

    progress.empty()

    grid_df = pd.DataFrame(rows)
    st.session_state["grid_results"] = grid_df
    st.success(f"Grid search complete. Best Sharpe: {grid_df['Sharpe'].max():.3f}")

if "grid_results" in st.session_state:
    grid_df = st.session_state["grid_results"]

    best_idx = grid_df["Sharpe"].idxmax()
    best = grid_df.loc[best_idx]
    st.success(
        f"Best combo: s_bo={best['s_bo']}, s_so={best['s_so']} "
        f"→ Sharpe={best['Sharpe']:.3f}, Max DD={best['Max DD']:.2%}, "
        f"Return={best['Total Return']:.2%}, Trades={int(best['Num Trades'])}"
    )

    s_bo_vals = sorted(grid_df["s_bo"].unique())
    s_so_vals = sorted(grid_df["s_so"].unique())

    sharpe_matrix = grid_df.pivot(index="s_so", columns="s_bo", values="Sharpe")
    dd_matrix = grid_df.pivot(index="s_so", columns="s_bo", values="Max DD")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sharpe Ratio Heatmap")
        fig = go.Figure(go.Heatmap(
            z=sharpe_matrix.values,
            x=[str(v) for v in sharpe_matrix.columns],
            y=[str(v) for v in sharpe_matrix.index],
            colorscale="RdYlGn",
            colorbar=dict(title="Sharpe"),
        ))
        fig.add_trace(go.Scatter(
            x=[str(best["s_bo"])],
            y=[str(best["s_so"])],
            mode="markers",
            marker=dict(symbol="star", size=16, color="blue"),
            name="Best",
        ))
        fig.update_layout(
            xaxis_title="s_bo",
            yaxis_title="s_so",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Max Drawdown Heatmap")
        fig2 = go.Figure(go.Heatmap(
            z=dd_matrix.values,
            x=[str(v) for v in dd_matrix.columns],
            y=[str(v) for v in dd_matrix.index],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Max DD"),
        ))
        fig2.update_layout(
            xaxis_title="s_bo",
            yaxis_title="s_so",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full Results Table")
    display_df = grid_df.copy()
    display_df["Max DD"] = display_df["Max DD"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    display_df["Total Return"] = display_df["Total Return"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(display_df, use_container_width=True)
