import datetime

import streamlit as st

from config import (
    Config, FactorConfig, OUConfig, SignalConfig, VolumeConfig,
    BacktestConfig, PairsConfig, DEFAULT_TICKERS, DATA_SOURCES,
)


def build_sidebar() -> Config:
    st.sidebar.header("Data Settings")

    data_source = st.sidebar.selectbox(
        "Data Source", DATA_SOURCES, index=0
    )

    tickers_input = st.sidebar.text_area(
        "Tickers (comma-separated)",
        value=", ".join(DEFAULT_TICKERS),
        height=100,
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start Date",
        value=datetime.date(1997, 1, 1),
        min_value=datetime.date(1960, 1, 1),
    )
    end_date = col2.date_input(
        "End Date",
        value=datetime.date(2007, 12, 31),
        min_value=datetime.date(1960, 1, 1),
    )

    st.sidebar.header("Factor Model")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["pca", "etf", "combined", "pairs"],
        format_func=lambda x: {
            "pca": "PCA (Eigenportfolios)",
            "etf": "Sector ETF Regression",
            "combined": "Combined (SPY + ETF + PCA)",
            "pairs": "Pairs Trading (Cointegration)",
        }[x],
    )

    pca_lookback = st.sidebar.slider(
        "PCA Correlation Window (days)", 120, 504, 252
    )

    use_fixed_components = st.sidebar.checkbox("Fixed # of PCA Components", value=True)
    if use_fixed_components:
        pca_n_components = st.sidebar.slider("PCA Components", 1, 30, 15)
    else:
        pca_n_components = None

    explained_var = st.sidebar.slider(
        "Explained Variance Threshold", 0.30, 0.90, 0.55, 0.05
    )

    use_ledoit_wolf = st.sidebar.checkbox("Ledoit-Wolf Shrinkage", value=True)

    beta_window = st.sidebar.slider(
        "Rolling Beta Window (days)", 40, 504, 252
    )

    pairs_pvalue = 0.05
    pairs_max = 20
    pairs_min_hl = 1.0
    pairs_max_hl = 126.0

    if model_type == "pairs":
        st.sidebar.header("Pairs Config")
        pairs_pvalue = st.sidebar.slider("Cointegration p-value threshold", 0.01, 0.20, 0.05, 0.01)
        pairs_max = st.sidebar.slider("Max pairs", 5, 50, 20)
        pairs_min_hl = st.sidebar.number_input("Min half-life (days)", 1.0, 30.0, 1.0)
        pairs_max_hl = st.sidebar.number_input("Max half-life (days)", 10.0, 252.0, 126.0)

    st.sidebar.header("OU Parameters")
    ou_window = st.sidebar.slider(
        "OU Estimation Window (days)", 40, 120, 60
    )
    kappa_min = st.sidebar.number_input(
        "Min Kappa (mean-reversion speed)", 0.0, 50.0, 8.4, 0.1
    )
    mean_center = st.sidebar.checkbox(
        "Mean Centering",
        value=True,
        help="Cross-sectional mean centering of s-scores (Eq. 18). Subtracts the mean equilibrium level across eligible stocks, making signals relative to peers rather than absolute.",
    )

    st.sidebar.header("Signal Thresholds")
    col1, col2 = st.sidebar.columns(2)
    s_bo = col1.number_input("Entry (s_bo)", 0.5, 3.0, 1.25, 0.05)
    s_so = col2.number_input("Entry (s_so)", 0.5, 3.0, 1.25, 0.05)
    col1, col2 = st.sidebar.columns(2)
    s_sc = col1.number_input("Exit Long (s_sc)", 0.1, 2.0, 0.50, 0.05)
    s_bc = col2.number_input("Exit Short (s_bc)", 0.1, 2.0, 0.75, 0.05)
    s_limit = st.sidebar.number_input("Force Exit (s_limit)", 2.0, 10.0, 4.0, 0.5)

    st.sidebar.header("Volume Adjustment")
    vol_enabled = st.sidebar.checkbox(
        "Enable Trading Time",
        value=False,
        help="Volume-adjusted trading time (Section 6, Eq. 20). Rescales returns by inverse volume ratio: low-volume days get higher weight (accentuating signals), high-volume days get lower weight.",
    )
    vol_window = st.sidebar.slider("Volume Trailing Window", 5, 30, 10)

    st.sidebar.header("Backtest Settings")
    initial_equity = st.sidebar.number_input(
        "Initial Equity ($)", 10_000, 10_000_000, 1_000_000, 100_000
    )
    col1, col2 = st.sidebar.columns(2)
    leverage_long = col1.number_input("Long Leverage", 0.0, 5.0, 2.0, 0.5)
    leverage_short = col2.number_input("Short Leverage", 0.0, 5.0, 2.0, 0.5)
    tc_bps = st.sidebar.number_input(
        "Transaction Cost (bps/side)", 0.0, 50.0, 5.0, 1.0
    )
    hedge = st.sidebar.selectbox(
        "Hedge Instrument",
        ["SPY", "sector_etf", "none"],
        help="SPY: market beta hedge. sector_etf: removes market + sector beta (uses each stock's sector ETF). none: unhedged.",
    )

    return Config(
        factor=FactorConfig(
            model_type=model_type,
            pca_lookback=pca_lookback,
            pca_n_components=pca_n_components,
            explained_variance_threshold=explained_var,
            use_ledoit_wolf=use_ledoit_wolf,
            beta_rolling_window=beta_window,
        ),
        ou=OUConfig(
            estimation_window=ou_window,
            kappa_min=kappa_min,
            mean_center=mean_center,
        ),
        signal=SignalConfig(
            s_bo=s_bo, s_so=s_so, s_sc=s_sc, s_bc=s_bc, s_limit=s_limit,
        ),
        volume=VolumeConfig(enabled=vol_enabled, trailing_window=vol_window),
        backtest=BacktestConfig(
            initial_equity=float(initial_equity),
            leverage_long=leverage_long,
            leverage_short=leverage_short,
            tc_bps=tc_bps,
            hedge_instrument=hedge,
        ),
        pairs=PairsConfig(
            pvalue_threshold=pairs_pvalue,
            max_pairs=pairs_max,
            min_half_life=pairs_min_hl,
            max_half_life=pairs_max_hl,
        ),
        data_source=data_source,
        start_date=str(start_date),
        end_date=str(end_date),
        tickers=tickers,
    )
