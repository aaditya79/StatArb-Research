"""
Backtest engine: day-by-day simulation (Paper Sections 4-5).

Orchestrates the full paper-faithful pipeline:
    For each trading day t (paper Appendix A):
      1. Build the factor history for day t.
         - ETF model: the stock's sector ETF return series.
         - PCA model: rolling 252-day PCA -> eigenportfolio returns.
      2. For each stock, run a FRESH 60-day OLS with intercept of stock
         returns on the factor returns. By OLS identity cumsum(epsilon)
         terminates at 0, which is the condition that makes the paper's
         shortcut s = -m/sigma_eq exact.
      3. Fit AR(1) on the 60-day cumulative residual process to get OU
         parameters (kappa, m, sigma_eq).
      4. Filter by kappa >= kappa_min (paper 8.4).
      5. Compute mean-centered s-scores (Eq. 18).
      6. Apply entry/exit rules (Eq. 16).
      7. Open offsetting hedge position for every new stock trade
         (paper Section 5.1 / 5.3 beta-neutrality).
      8. Mark to market.
"""
from dataclasses import dataclass, field
import hashlib
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


def _backtest_cache_key(config: "Config") -> str:
    payload = {
        "tickers": sorted(config.tickers),
        "start": config.start_date,
        "end": config.end_date,
        "data_source": config.data_source,
        "trading_mode": config.trading_mode,
        "factor": {
            "model_type": config.factor.model_type,
            "pca_lookback": config.factor.pca_lookback,
            "pca_n_components": config.factor.pca_n_components,
            "explained_variance_threshold": config.factor.explained_variance_threshold,
            "use_ledoit_wolf": config.factor.use_ledoit_wolf,
            "beta_rolling_window": config.factor.beta_rolling_window,
        },
        "ou": {
            "estimation_window": config.ou.estimation_window,
            "kappa_min": config.ou.kappa_min,
            "mean_center": config.ou.mean_center,
        },
        "signal": {
            "s_bo": config.signal.s_bo,
            "s_so": config.signal.s_so,
            "s_sc": config.signal.s_sc,
            "s_bc": config.signal.s_bc,
            "s_limit": config.signal.s_limit,
        },
        "volume": {
            "enabled": config.volume.enabled,
            "trailing_window": config.volume.trailing_window,
        },
        "backtest": {
            "initial_equity": config.backtest.initial_equity,
            "leverage_long": config.backtest.leverage_long,
            "leverage_short": config.backtest.leverage_short,
            "tc_bps": config.backtest.tc_bps,
            "hedge_instrument": config.backtest.hedge_instrument,
            "risk_free_rate": config.backtest.risk_free_rate,
            "dt": config.backtest.dt,
        },
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest


def _load_backtest_cache(digest: str) -> "BacktestResult | None":
    path = _CACHE_DIR / f"backtest_{digest}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _save_backtest_cache(digest: str, result: "BacktestResult") -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"backtest_{digest}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
from statarb.factors.base import FactorResult
from statarb.factors.pca import compute_pca_eigenportfolio_returns
from statarb.signals.ou_estimator import (
    estimate_ou_params,
    estimate_ou_params_window,
    OUParams,
)
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


def _rolling_beta(
    y: np.ndarray, x: np.ndarray, end: int, window: int
) -> float:
    """Trailing no-intercept OLS beta of y on x over [end-window, end)."""
    start = max(0, end - window)
    y_w = y[start:end]
    x_w = x[start:end]
    mask = np.isfinite(y_w) & np.isfinite(x_w)
    if mask.sum() < 30:
        return 0.0
    y_m = y_w[mask]
    x_m = x_w[mask]
    denom = float(np.dot(x_m, x_m))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_m, y_m) / denom)


def run_backtest(
    config: Config,
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    factor_result: FactorResult,
    returns: pd.DataFrame | None = None,
    etf_returns: pd.DataFrame | None = None,
    spy_returns: pd.DataFrame | pd.Series | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> BacktestResult:
    """
    Paper-faithful backtest.

    Args:
        config: Master configuration object.
        prices: Adjusted close prices (dates x tickers).
        volume: Daily volume (dates x tickers).
        factor_result: Pre-computed FactorResult (used for diagnostics and
            for extracting sector_mapping if not passed explicitly).
        returns: Stock log returns (dates x tickers). Computed from prices
            if None.
        etf_returns: ETF log returns (dates x ETFs). Required for
            model_type = 'etf' and for sector-ETF hedge.
        spy_returns: SPY log returns (dates x ['SPY']) or Series. Required
            for model_type = 'combined' and for SPY hedge.
        sector_mapping: Ticker -> sector ETF. Required for ETF model and
            for sector-ETF hedge.

    Returns:
        BacktestResult.
    """
    cache_digest = _backtest_cache_key(config)
    cached = _load_backtest_cache(cache_digest)
    if cached is not None:
        logger.info(f"[run_backtest] cache hit ({cache_digest[:8]}…); skipping simulation.")
        print(f"[run_backtest] cache hit ({cache_digest[:8]}…); returning cached result.")
        return cached

    model_type = config.factor.model_type

    # ── Canonicalize inputs ──
    if returns is None:
        returns = np.log(prices / prices.shift(1))

    # Align every auxiliary frame to the stock index so positional slicing
    # into .values stays consistent.
    returns = returns.reindex(prices.index)
    if etf_returns is not None:
        etf_returns = etf_returns.reindex(prices.index)
    if spy_returns is not None:
        if isinstance(spy_returns, pd.DataFrame):
            spy_series = spy_returns.iloc[:, 0]
        else:
            spy_series = pd.Series(spy_returns)
        spy_series = spy_series.reindex(prices.index)
    else:
        spy_series = None

    if sector_mapping is None:
        sector_mapping = factor_result.metadata.get("sector_mapping", {}) or {}

    tickers = [t for t in returns.columns if t in prices.columns]
    returns = returns[tickers]
    prices = prices[tickers]

    # ── Hedge setup ──
    hedge_instr = config.backtest.hedge_instrument
    hedge_enabled = hedge_instr not in (None, "none", "")
    hedge_prices_frame = None
    stock_to_hedge: dict[str, str] = {}

    if hedge_enabled:
        if hedge_instr == "SPY":
            if spy_series is None or spy_series.isna().all():
                print("[hedge] SPY hedge selected but SPY returns missing; disabling hedge.")
                hedge_enabled = False
            else:
                spy_prices_series = np.exp(spy_series.cumsum()) * 100.0
                hedge_prices_frame = pd.DataFrame(
                    {"SPY": spy_prices_series}, index=prices.index
                )
                stock_to_hedge = {t: "SPY" for t in tickers}
        elif hedge_instr == "sector_etf":
            if etf_returns is None or not sector_mapping:
                print(
                    "[hedge] sector_etf hedge selected but etf_returns/sector_mapping "
                    "missing; disabling hedge."
                )
                hedge_enabled = False
            else:
                etf_prices_from_ret = np.exp(etf_returns.cumsum()) * 100.0
                hedge_prices_frame = etf_prices_from_ret
                stock_to_hedge = {
                    t: sector_mapping[t]
                    for t in tickers
                    if sector_mapping.get(t) in etf_returns.columns
                }
        else:
            print(f"[hedge] Unknown hedge_instrument='{hedge_instr}'; disabling hedge.")
            hedge_enabled = False

    # ── Diagnostics (stdout) ──
    print("=" * 60)
    print(f"[run_backtest] model_type={model_type}  "
          f"hedge={hedge_instr if hedge_enabled else 'none (disabled)'}")
    print(f"[run_backtest] universe size requested: {len(tickers)}")
    coverage = returns.notna().sum()
    nonempty = int((coverage > 0).sum())
    med = coverage.median() if len(coverage) else 0
    median_cov = int(med) if pd.notna(med) else 0
    print(f"[run_backtest] tickers with any return data: {nonempty} / {len(tickers)}, "
          f"median days of coverage: {median_cov}")
    if model_type == "etf" or model_type == "combined":
        if etf_returns is None:
            print("[run_backtest] WARNING: ETF model selected but etf_returns is None.")
        else:
            etf_cov = etf_returns.notna().sum()
            missing_etfs = etf_cov[etf_cov == 0].index.tolist()
            if missing_etfs:
                print(f"[run_backtest] ETFs with NO data at all: {missing_etfs}")
            thin = etf_cov[(etf_cov > 0) & (etf_cov < len(etf_returns) * 0.2)].index.tolist()
            if thin:
                print(f"[run_backtest] ETFs with <20% coverage: {thin}")
    if model_type == "etf" and sector_mapping:
        orphaned = [t for t in tickers if sector_mapping.get(t) not in (etf_returns.columns if etf_returns is not None else [])]
        if orphaned:
            print(f"[run_backtest] {len(orphaned)} tickers mapped to an ETF with no data "
                  f"(they will not trade): {orphaned[:10]}{'...' if len(orphaned) > 10 else ''}")

    # ── Volume-time adjustment (Section 6, Eq. 20) ──
    # Applied to raw returns BEFORE the factor regression, consistent with paper:
    # R_tilde = R * avg_V / V_t.
    if config.volume.enabled:
        adj_returns = returns.copy()
        vol_tickers = [t for t in tickers if t in volume.columns]
        if vol_tickers:
            adj_sub = compute_volume_adjusted_returns(
                returns[vol_tickers], volume[vol_tickers],
                trailing_window=config.volume.trailing_window,
            )
            adj_returns[vol_tickers] = adj_sub
        signal_returns = adj_returns
        print(f"[run_backtest] trading-time adjustment enabled "
              f"(trailing window {config.volume.trailing_window}).")
    else:
        signal_returns = returns

    dates = prices.index
    ou_window = config.ou.estimation_window
    pca_lookback = config.factor.pca_lookback if model_type == "pca" else 0
    # Need PCA lookback before first OU window; for ETF model we just need ou_window.
    if model_type == "pca":
        min_start = pca_lookback + 10
    else:
        min_start = ou_window + 10
    print(f"[run_backtest] first eligible trading index: {min_start} "
          f"(~ {str(dates[min_start])[:10] if min_start < len(dates) else 'beyond sample'})")

    # Warn when PCA is over-parameterized for the universe size. The paper
    # used N=1,417 with 15 components; at small N the higher eigenvectors
    # fit correlation-matrix noise and the 60-day OLS residuals become
    # fitting noise rather than true idiosyncratic returns.
    if model_type in ("pca", "combined"):
        m_cfg = config.factor.pca_n_components
        if m_cfg is not None and nonempty > 0 and m_cfg > max(1, nonempty // 10):
            print(
                f"[run_backtest] WARNING: pca_n_components={m_cfg} is large "
                f"relative to N={nonempty}. With N < ~10*m the 60-day OLS "
                f"over-fits (R² → 1) and residuals become noise — signal "
                f"will be unreliable. Try pca_n_components in {{1, 2, 3}}."
            )

    # ── Portfolio ──
    portfolio = PortfolioManager(
        initial_equity=config.backtest.initial_equity,
        leverage_long=config.backtest.leverage_long,
        leverage_short=config.backtest.leverage_short,
        tc_bps=config.backtest.tc_bps,
    )

    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    trade_records: list[dict] = []
    position_records: list[dict] = []
    sscore_records: dict = {}
    daily_ou_params: dict = {}

    # n_target: expected concurrently-open positions. Paper (§5.1) targets
    # ~2% of equity per position → ~100 concurrent positions at 2+2 leverage.
    # Scale with universe size so per-position notional stays near 2%; floor
    # at 10 so tiny test universes still function, cap at 200 so very large
    # universes don't produce unreasonably small per-name trades.
    n_target = max(min(len(tickers) // 7, 200), 10)
    print(f"[run_backtest] n_target (position-sizing denominator) = {n_target}")

    # Pre-cache numpy arrays for speed.
    signal_arr = signal_returns.values  # (T, N)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    etf_arr = etf_returns.values if etf_returns is not None else None
    etf_idx = {c: i for i, c in enumerate(etf_returns.columns)} if etf_returns is not None else {}
    spy_arr = spy_series.values if spy_series is not None else None

    # Paper §5.1 investable-universe filter:
    # "Our basic investable universe is formed by stocks that have a
    #  continuous trading history of at least 500 days preceding the
    #  trading date."
    # Precompute per-ticker cumulative count of finite returns so the
    # per-day filter is an O(1) lookup.
    min_history_days = 500
    valid_counts = np.isfinite(signal_arr).cumsum(axis=0)  # (T, N)
    history_skipped_logged = False

    first_eligible_logged = False

    for i in range(min_start, len(dates)):
        date = dates[i]
        date_ts = pd.Timestamp(date)

        current_prices = prices.iloc[i]
        price_dict = current_prices.to_dict()
        if hedge_enabled and hedge_prices_frame is not None:
            hp = hedge_prices_frame.iloc[i].to_dict()
            for k, v in hp.items():
                if np.isfinite(v):
                    price_dict[k] = v

        # ── Step 1: build factor history for this day ──
        factor_matrix_60 = None     # (ou_window, k) numpy array, or None
        pca_factor_matrix_60 = None  # for PCA model, shared across tickers
        pca_dates_60 = None          # DatetimeIndex for alignment

        if model_type == "pca":
            window_slice = signal_returns.iloc[i - pca_lookback:i]
            try:
                eigenport_returns, _eigvals, _V, _stds = compute_pca_eigenportfolio_returns(
                    window_slice,
                    n_components=config.factor.pca_n_components,
                    explained_variance_threshold=config.factor.explained_variance_threshold,
                    use_ledoit_wolf=config.factor.use_ledoit_wolf,
                )
                # The PCA helper drops rows with any NaN, so eigenport_returns
                # is indexed on SURVIVING dates — not on consecutive calendar
                # days. We carry the date index through so the stock leg
                # (below) can be indexed on the same rows.
                tail = eigenport_returns.iloc[-ou_window:]
                if len(tail) < 30:
                    portfolio.mark_to_market(price_dict)
                    equity_values.append(portfolio.equity)
                    equity_dates.append(date)
                    continue
                pca_factor_matrix_60 = tail.values
                pca_dates_60 = tail.index
            except Exception as e:
                # PCA failed this day -- mark to market only, skip trading.
                portfolio.mark_to_market(price_dict)
                equity_values.append(portfolio.equity)
                equity_dates.append(date)
                continue

        # ── Step 2: per-ticker fresh 60-day OLS ──
        ou_params: dict[str, OUParams] = {}
        # Precompute positional indices for PCA-aligned dates once per day.
        pca_date_positions = None
        if pca_dates_60 is not None:
            pca_date_positions = signal_returns.index.get_indexer(pca_dates_60)

        # Stats for the 500-day history gate (first-day diagnostic only).
        filtered_by_history = 0

        for ticker in tickers:
            # Paper §5.1 500-day history requirement. `valid_counts[i-1, idx]`
            # is the running number of finite return observations strictly
            # before day i (i.e. available when deciding trades at the close
            # of day i-1 for execution on day i).
            if valid_counts[i - 1, ticker_idx[ticker]] < min_history_days:
                filtered_by_history += 1
                continue

            if model_type == "etf":
                stock_60 = signal_arr[i - ou_window:i, ticker_idx[ticker]]
                etf_sym = sector_mapping.get(ticker)
                if etf_sym is None or etf_arr is None or etf_sym not in etf_idx:
                    continue
                f_60 = etf_arr[i - ou_window:i, etf_idx[etf_sym]]
                params = estimate_ou_params_window(
                    stock_60, f_60, dt=config.backtest.dt
                )
            elif model_type == "pca":
                if pca_factor_matrix_60 is None or pca_date_positions is None:
                    continue
                # Align stock leg to the SURVIVING PCA dates (not the raw
                # 60-day calendar slice). Both sides now refer to the same
                # trading days, so the 60-day OLS residuals are meaningful.
                stock_60 = signal_arr[pca_date_positions, ticker_idx[ticker]]
                params = estimate_ou_params_window(
                    stock_60, pca_factor_matrix_60, dt=config.backtest.dt
                )
            elif model_type == "combined":
                # [SPY, sector_ETF, PCA_factors] stacked as multi-factor.
                # For the combined model we align everything to the PCA dates
                # when PCA is present; otherwise use the calendar slice.
                if pca_date_positions is not None:
                    pos = pca_date_positions
                    stock_60 = signal_arr[pos, ticker_idx[ticker]]
                    parts = []
                    if spy_arr is not None:
                        parts.append(spy_arr[pos].reshape(-1, 1))
                    etf_sym = sector_mapping.get(ticker)
                    if etf_sym is not None and etf_arr is not None and etf_sym in etf_idx:
                        parts.append(etf_arr[pos, etf_idx[etf_sym]].reshape(-1, 1))
                    parts.append(pca_factor_matrix_60)
                else:
                    stock_60 = signal_arr[i - ou_window:i, ticker_idx[ticker]]
                    parts = []
                    if spy_arr is not None:
                        parts.append(spy_arr[i - ou_window:i].reshape(-1, 1))
                    etf_sym = sector_mapping.get(ticker)
                    if etf_sym is not None and etf_arr is not None and etf_sym in etf_idx:
                        parts.append(
                            etf_arr[i - ou_window:i, etf_idx[etf_sym]].reshape(-1, 1)
                        )
                if not parts:
                    continue
                f_60 = np.hstack(parts)
                params = estimate_ou_params_window(
                    stock_60, f_60, dt=config.backtest.dt
                )
            elif model_type == "pairs":
                # Pairs path uses pre-computed residuals (legacy).
                series = factor_result.residuals.get(ticker)
                if series is None:
                    continue
                stock_60 = signal_arr[i - ou_window:i, ticker_idx[ticker]]  # unused, kept for parity
                params = estimate_ou_params(
                    series.iloc[max(0, i - ou_window):i],
                    window=ou_window, dt=config.backtest.dt,
                )
            else:
                continue
            if params is not None:
                ou_params[ticker] = params

        daily_ou_params[str(date)[:10]] = ou_params

        # ── Step 3: kappa filter ──
        eligible = filter_eligible(ou_params, kappa_min=config.ou.kappa_min)

        if not first_eligible_logged and eligible:
            print(f"[run_backtest] first day with eligible stocks: "
                  f"{str(date)[:10]}  ({len(eligible)} / {len(ou_params)} passed kappa>={config.ou.kappa_min})")
            first_eligible_logged = True

        if not history_skipped_logged and filtered_by_history > 0:
            print(f"[run_backtest] paper §5.1 history filter (≥{min_history_days} days): "
                  f"{filtered_by_history}/{len(tickers)} tickers skipped on {str(date)[:10]}")
            history_skipped_logged = True

        # ── Step 4: s-scores ──
        eligible_params = {t: ou_params[t] for t in eligible}
        if not eligible_params:
            portfolio.mark_to_market(price_dict)
            equity_values.append(portfolio.equity)
            equity_dates.append(date)
            continue

        sscores = compute_sscores(
            signal_returns.iloc[:i],
            eligible_params,
            mean_center=config.ou.mean_center,
        )
        sscore_records[date] = sscores

        # ── Step 5: exits ──
        tickers_to_close: list[str] = []
        for ticker in list(portfolio.positions.keys()):
            if ticker not in sscores.index:
                tickers_to_close.append(ticker)
                continue
            s = sscores[ticker]
            pos = portfolio.positions[ticker]
            should_close = False
            if pos.direction == 1 and s >= -config.signal.s_sc:
                should_close = True
            elif pos.direction == -1 and s <= config.signal.s_bc:
                should_close = True
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
                if hedge_enabled:
                    htk = stock_to_hedge.get(ticker)
                    if htk is not None:
                        hprice = price_dict.get(htk)
                        if hprice is not None and np.isfinite(hprice):
                            portfolio.close_hedge_slice(ticker, htk, hprice, date_ts)
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

        # ── Step 6: entries ──
        notional_per_pos = portfolio.compute_notional_per_position(n_target)
        for ticker in eligible:
            if ticker in portfolio.positions:
                continue
            if ticker not in sscores.index:
                continue
            if ticker not in price_dict or not np.isfinite(price_dict[ticker]):
                continue

            s = sscores[ticker]
            direction = None
            if s <= -config.signal.s_bo:
                direction = 1
            elif s >= config.signal.s_so:
                direction = -1
            if direction is None:
                continue

            opened = portfolio.open_position(
                ticker=ticker,
                direction=direction,
                price=price_dict[ticker],
                date=date_ts,
                notional=notional_per_pos,
            )
            if opened is None or not hedge_enabled:
                continue

            htk = stock_to_hedge.get(ticker)
            if htk is None:
                continue
            hprice = price_dict.get(htk)
            if hprice is None or not np.isfinite(hprice):
                continue

            # Hedge beta:
            #   ETF model -> use the beta_1 produced by the same 60-day OLS
            #                that generated the signal (paper-consistent).
            #   PCA / combined -> use stock vs SPY rolling beta (paper §5.3
            #                     hedges PCA book with SPY).
            if model_type == "etf":
                beta = ou_params[ticker].factor_beta
            else:
                # Use a trailing 252-day beta of stock on SPY.
                if spy_arr is None:
                    continue
                beta = _rolling_beta(
                    signal_arr[:, ticker_idx[ticker]],
                    spy_arr,
                    end=i,
                    window=config.factor.beta_rolling_window,
                )
            portfolio.open_hedge_slice(
                stock_ticker=ticker,
                hedge_ticker=htk,
                stock_direction=direction,
                stock_notional=notional_per_pos,
                beta=beta,
                hedge_price=hprice,
                date=date_ts,
            )

        # ── Step 7: mark to market (updates stale_days counters) ──
        portfolio.mark_to_market(price_dict)

        # Force-close positions whose underlying has been dark for 10+ days
        # so delisted names don't occupy leverage budget forever. Uses each
        # position's last-known finite price to realize the close.
        portfolio.purge_stale_positions(date_ts, stale_threshold=10)

        equity_values.append(portfolio.equity)
        equity_dates.append(date)

        for ticker, pos in portfolio.positions.items():
            position_records.append({
                "date": date,
                "ticker": ticker,
                "direction": pos.direction,
                "notional": pos.notional,
                "entry_price": pos.entry_price,
                "current_price": price_dict.get(ticker, np.nan),
            })

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))

    trades = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["ticker", "direction", "entry_date", "exit_date",
                 "entry_price", "exit_price", "pnl", "notional"]
    )
    daily_positions = pd.DataFrame(position_records) if position_records else pd.DataFrame(
        columns=["date", "ticker", "direction", "notional",
                 "entry_price", "current_price"]
    )

    daily_sscores = pd.DataFrame(sscore_records).T
    if not daily_sscores.empty:
        daily_sscores.index = pd.DatetimeIndex(daily_sscores.index)

    metrics = compute_metrics(
        equity_curve, trades,
        risk_free_rate=config.backtest.risk_free_rate,
        total_costs=portfolio.total_costs,
    )

    print(f"[run_backtest] DONE. trades={metrics.num_trades}, "
          f"sharpe={metrics.sharpe_ratio:.3f}, total_return={metrics.total_return:.2%}, "
          f"max_dd={metrics.max_drawdown:.2%}, total_costs=${portfolio.total_costs:,.0f}")
    print("=" * 60)

    result = BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        daily_positions=daily_positions,
        daily_sscores=daily_sscores,
        metrics=metrics,
        factor_result=factor_result,
        daily_ou_params=daily_ou_params,
    )
    _save_backtest_cache(cache_digest, result)
    logger.info(f"[run_backtest] result cached ({cache_digest[:8]}…).")
    return result
