import streamlit as st

from config import Config
from statarb.backtest.engine import BacktestResult


def get_config() -> Config | None:
    return st.session_state.get("config")


def set_config(config: Config):
    st.session_state["config"] = config


def get_backtest_result() -> BacktestResult | None:
    return st.session_state.get("backtest_result")


def set_backtest_result(result: BacktestResult):
    st.session_state["backtest_result"] = result


def has_backtest_result() -> bool:
    return "backtest_result" in st.session_state


def get_prices() -> "pd.DataFrame | None":
    return st.session_state.get("prices")


def set_prices(prices):
    st.session_state["prices"] = prices


def get_volume() -> "pd.DataFrame | None":
    return st.session_state.get("volume")


def set_volume(volume):
    st.session_state["volume"] = volume
