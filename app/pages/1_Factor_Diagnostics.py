import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from app.state import get_backtest_result, get_config, has_backtest_result
from app.components.charts import plot_correlation_heatmap, plot_eigenvalue_spectrum


st.set_page_config(page_title="Factor Diagnostics", layout="wide")
st.title("Factor Diagnostics")

if not has_backtest_result():
    st.warning("Run a backtest on the Home page first.")
    st.stop()

result = get_backtest_result()
config = get_config()
factor_result = result.factor_result

# ── Return Correlation Heatmap ──
st.subheader("Return Correlation Matrix")
if not factor_result.residuals.empty:
    returns_for_corr = factor_result.residuals.dropna(how="all")
    if len(returns_for_corr) > 30:
        corr = returns_for_corr.corr()
        st.plotly_chart(
            plot_correlation_heatmap(corr),
            use_container_width=True,
        )

# ── Eigenvalue Spectrum (PCA only) ──
if "all_eigenvalues" in factor_result.metadata:
    st.subheader("Eigenvalue Spectrum")
    eigenvalues = factor_result.metadata["all_eigenvalues"]
    st.plotly_chart(
        plot_eigenvalue_spectrum(eigenvalues),
        use_container_width=True,
    )

    n_components = factor_result.metadata.get("n_components", 0)
    explained = factor_result.metadata.get("explained_variance_ratio", 0)
    st.info(
        f"Using **{n_components}** components explaining "
        f"**{explained:.1%}** of total variance."
    )

# ── Beta Loadings Heatmap ──
st.subheader("Factor Loadings (Betas)")
if not factor_result.betas.empty:
    betas = factor_result.betas
    if betas.shape[1] > 10:
        betas = betas.iloc[:, :10]

    fig = go.Figure(data=go.Heatmap(
        z=betas.values,
        x=betas.columns.tolist(),
        y=betas.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
    ))
    fig.update_layout(
        title="Factor Loadings Heatmap",
        xaxis_title="Factor",
        yaxis_title="Ticker",
        height=max(400, len(betas) * 15),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Eigenportfolio Cumulative Returns (Paper Fig. 2) ──
st.subheader("Eigenportfolio Cumulative Returns (Paper Fig. 2)")
if not factor_result.factor_returns.empty:
    fr = factor_result.factor_returns.dropna(how="all")
    factor_cols = fr.columns.tolist()
    default_cols = factor_cols[:min(3, len(factor_cols))]

    selected = st.multiselect(
        "Select factors to plot",
        factor_cols,
        default=default_cols,
    )

    if selected:
        cum_returns = fr[selected].cumsum()
        fig = go.Figure()
        for col in selected:
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode="lines",
                name=col,
            ))
        fig.update_layout(
            title="Cumulative Eigenportfolio Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── R-squared per Stock ──
r_squared = factor_result.metadata.get("r_squared", {})
if r_squared:
    st.subheader("R-squared per Stock")
    r2_df = pd.Series(r_squared).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=r2_df.index.tolist(),
        y=r2_df.values,
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title="Variance Explained by Factor Model (per Stock)",
        xaxis_title="Ticker",
        yaxis_title="R-squared",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Residual Return Statistics ──
st.subheader("Residual Return Statistics")
if not factor_result.residuals.empty:
    r = factor_result.residuals.dropna(how="all")
    stats_df = pd.DataFrame({
        "Mean (ann %)": (r.mean() * 252 * 100).round(2),
        "Vol (ann %)": (r.std() * np.sqrt(252) * 100).round(2),
        "Skewness": r.skew().round(3),
        "Kurtosis": r.kurt().round(3),
        "Min": r.min().round(4),
        "Max": r.max().round(4),
    })
    st.dataframe(stats_df, use_container_width=True)

# ── OU Parameters (Last Estimation) ──
st.subheader("OU Parameters (Last Estimation)")
daily_ou = result.daily_ou_params
if daily_ou:
    last_date = sorted(daily_ou.keys())[-1]
    ou_data = daily_ou[last_date]
    if ou_data:
        rows = []
        for ticker, p in ou_data.items():
            rows.append({
                "Ticker": ticker,
                "κ (annualized)": round(p.kappa, 2),
                "Half-life (days)": round(p.half_life, 1),
                "σ_eq": round(p.sigma_eq, 5),
                "m (equilibrium)": round(p.m, 5),
            })
        ou_df = pd.DataFrame(rows).sort_values("Half-life (days)")
        st.dataframe(ou_df, use_container_width=True)

# ── Residual Analysis ──
st.subheader("Residual Analysis")
if not factor_result.residuals.empty:
    resid = factor_result.residuals.dropna(how="all")
    ticker_opts = resid.columns.tolist()
    sel = st.multiselect("Select tickers", ticker_opts, default=ticker_opts[:5])
    if sel:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for t in sel:
                fig.add_trace(go.Scatter(x=resid.index, y=resid[t].cumsum(), mode="lines", name=t))
            fig.update_layout(
                title="Cumulative Residuals",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            for t in sel:
                fig2.add_trace(go.Histogram(x=resid[t].dropna(), name=t, opacity=0.6, nbinsx=50))
            fig2.update_layout(
                title="Residual Distribution",
                barmode="overlay",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Residual Autocorrelation (first 20 lags)")
        from statsmodels.tsa.stattools import acf
        acf_fig = go.Figure()
        for t in sel[:3]:
            ac = acf(resid[t].dropna(), nlags=20, fft=True)
            acf_fig.add_trace(go.Bar(x=list(range(21)), y=ac, name=t, opacity=0.7))
        acf_fig.add_hline(
            y=1.96 / np.sqrt(len(resid)),
            line_dash="dash", line_color="red", annotation_text="95% CI",
        )
        acf_fig.add_hline(
            y=-1.96 / np.sqrt(len(resid)),
            line_dash="dash", line_color="red",
        )
        acf_fig.update_layout(
            barmode="group",
            template="plotly_white",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
        )
        st.plotly_chart(acf_fig, use_container_width=True)
