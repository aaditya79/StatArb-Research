import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.state import get_backtest_result, get_config, has_backtest_result, get_prices
from statarb.signals.cointegration import test_cointegration, compute_pair_spread


st.set_page_config(page_title="Cointegration Analysis", layout="wide")
st.title("Cointegration Analysis")
st.caption("Engle-Granger cointegration tests across the ticker universe.")

prices = get_prices()

if prices is None:
    st.warning("No price data found. Please run a backtest from the Home page first.")
    st.stop()

# ── Sidebar Controls ──
st.sidebar.header("Cointegration Settings")
pvalue_threshold = st.sidebar.slider("p-value threshold", 0.01, 0.20, 0.05, 0.01)
lookback = st.sidebar.slider("Lookback window (days)", 60, 756, 252, 21)

run_test = st.sidebar.button("Run Cointegration Tests", type="primary")

if run_test or "coint_results" not in st.session_state:
    with st.spinner("Running Engle-Granger tests (this may take a moment)..."):
        try:
            coint_df = test_cointegration(prices, pvalue_threshold=pvalue_threshold, lookback=lookback)
            st.session_state["coint_results"] = coint_df
            st.session_state["coint_pvalue"] = pvalue_threshold
        except Exception as e:
            st.error(f"Cointegration test failed: {e}")
            st.stop()

if "coint_results" not in st.session_state:
    st.info("Click 'Run Cointegration Tests' to begin.")
    st.stop()

coint_df = st.session_state["coint_results"]

if coint_df.empty:
    st.warning(f"No cointegrated pairs found at p-value threshold {pvalue_threshold}. Try raising the threshold.")
    st.stop()

st.success(f"Found **{len(coint_df)}** cointegrated pair(s) at p ≤ {st.session_state.get('coint_pvalue', pvalue_threshold):.2f}")

# ── Pairs Table ──
st.subheader("Cointegrated Pairs")

display_df = coint_df.copy()
display_df["pvalue"] = display_df["pvalue"].map(lambda x: f"{x:.4f}")
display_df["score"] = display_df["score"].map(lambda x: f"{x:.3f}")
display_df["hedge_ratio"] = display_df["hedge_ratio"].map(lambda x: f"{x:.4f}")
display_df["spread_mean"] = display_df["spread_mean"].map(lambda x: f"{x:.5f}")
display_df["spread_std"] = display_df["spread_std"].map(lambda x: f"{x:.5f}")
display_df["half_life"] = display_df["half_life"].map(
    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
)

st.dataframe(display_df, use_container_width=True, height=300)

# ── Currently selected pairs (if pairs model was used) ──
if has_backtest_result():
    result = get_backtest_result()
    selected_pairs = result.factor_result.metadata.get("selected_pairs", None)
    if selected_pairs is not None and not selected_pairs.empty:
        st.info(
            f"The current backtest used **{len(selected_pairs)}** pair(s) from this universe "
            "(shown in the pairs model fit)."
        )
        with st.expander("Show selected pairs from backtest"):
            sp_display = selected_pairs.copy()
            for col in ["pvalue", "score", "hedge_ratio", "spread_mean", "spread_std"]:
                if col in sp_display.columns:
                    sp_display[col] = sp_display[col].map(lambda x: f"{x:.4f}")
            if "half_life" in sp_display.columns:
                sp_display["half_life"] = sp_display["half_life"].map(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
            st.dataframe(sp_display, use_container_width=True)

# ── Pair Deep Dive ──
st.subheader("Pair Analysis")

pair_labels = [f"{row['ticker1']} / {row['ticker2']}" for _, row in coint_df.iterrows()]
selected_label = st.selectbox("Select a pair to analyze", pair_labels)

if selected_label:
    idx = pair_labels.index(selected_label)
    row = coint_df.iloc[idx]
    t1, t2 = row["ticker1"], row["ticker2"]
    beta = row["hedge_ratio"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("p-value", f"{row['pvalue']:.4f}")
    col2.metric("Hedge Ratio (β)", f"{beta:.4f}")
    col3.metric("Half-life (days)", f"{row['half_life']:.1f}" if pd.notna(row["half_life"]) else "N/A")
    col4.metric("Spread Std", f"{row['spread_std']:.5f}")

    log_prices = np.log(prices[[t1, t2]].dropna())
    spread = compute_pair_spread(log_prices, t1, t2, beta)

    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / (spread_std + 1e-10)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Spread: {t1} - β·{t2}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spread.index, y=spread.values,
            mode="lines", name="Spread", line=dict(color="#1f77b4"),
        ))
        fig.add_hline(y=spread_mean, line_dash="dash", line_color="black", annotation_text="Mean")
        fig.add_hline(y=spread_mean + spread_std, line_dash="dot", line_color="orange", annotation_text="+1σ")
        fig.add_hline(y=spread_mean - spread_std, line_dash="dot", line_color="orange", annotation_text="-1σ")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Log Spread",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Z-Score (Trading Signal)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=z_score.index, y=z_score.values,
            mode="lines", name="Z-Score", line=dict(color="#ff7f0e"),
        ))
        fig2.add_hline(y=1.25, line_dash="dash", line_color="red", annotation_text="Short (+1.25)")
        fig2.add_hline(y=-1.25, line_dash="dash", line_color="green", annotation_text="Long (-1.25)")
        fig2.add_hline(y=0, line_dash="solid", line_color="gray")
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Z-Score",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── OU parameter estimates on the spread ──
    st.subheader("OU Parameter Estimates on Spread")
    try:
        lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        common_idx = lag.index.intersection(delta.index)
        if len(common_idx) >= 10:
            b, a = np.polyfit(lag[common_idx].values, delta[common_idx].values, 1)
            kappa_daily = -b
            kappa_ann = kappa_daily * 252
            half_life_est = -np.log(2) / b if b < 0 else np.nan
            sigma_res = delta[common_idx].std()
            sigma_eq = sigma_res / np.sqrt(2 * kappa_daily + 1e-10) if kappa_daily > 0 else np.nan

            ou_cols = st.columns(4)
            ou_cols[0].metric("κ (daily)", f"{kappa_daily:.4f}")
            ou_cols[1].metric("κ (annualized)", f"{kappa_ann:.2f}")
            ou_cols[2].metric("Half-life (days)", f"{half_life_est:.1f}" if pd.notna(half_life_est) else "N/A")
            ou_cols[3].metric("σ_eq", f"{sigma_eq:.5f}" if pd.notna(sigma_eq) else "N/A")
        else:
            st.info("Insufficient data to estimate OU parameters for this pair.")
    except Exception as e:
        st.warning(f"Could not estimate OU parameters: {e}")

    # ── Individual price series ──
    st.subheader(f"Price Series: {t1} vs {t2}")
    fig3 = go.Figure()
    px_norm = prices[[t1, t2]].dropna()
    fig3.add_trace(go.Scatter(
        x=px_norm.index,
        y=px_norm[t1] / px_norm[t1].iloc[0],
        mode="lines", name=t1,
    ))
    fig3.add_trace(go.Scatter(
        x=px_norm.index,
        y=px_norm[t2] / px_norm[t2].iloc[0],
        mode="lines", name=t2,
    ))
    fig3.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price (base=1)",
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)
