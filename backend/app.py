"""
FastAPI backend exposing the existing statarb pipeline as JSON endpoints.

This file ONLY orchestrates calls to the unchanged statarb.* and config.*
modules — no trading / signal / model logic lives here.

Cache hierarchy (all disk-based, survives restarts):
  Level 1 – result cache   : keyed on every request field → instant replay
  Level 2 – factor cache   : keyed on data + factor params → skip factor fit
  Level 3 – data cache     : keyed on tickers + dates + source → skip download
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import pickle
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Make the project root importable so `import config`, `import statarb` work
# regardless of where uvicorn is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (  # noqa: E402
    Config, FactorConfig, OUConfig, SignalConfig, VolumeConfig,
    BacktestConfig, PairsConfig, DEFAULT_TICKERS, DATA_SOURCES, MARKET_ETF,
    PAPER_TICKERS, MODERN_TICKERS,
)
from statarb.data.universe import get_data_source, get_sector_mapping  # noqa: E402
from statarb.factors.registry import build_factor_model  # noqa: E402
from statarb.backtest.engine import run_backtest  # noqa: E402

app = FastAPI(title="StatArb API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / ".backtest_cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key_data(req: BacktestRequest) -> str:
    """Key depends only on universe + date window + data source."""
    payload = (sorted(req.tickers), req.start_date, req.end_date, req.data_source)
    return "d_" + hashlib.sha256(str(payload).encode()).hexdigest()[:20]


def _cache_key_factor(req: BacktestRequest, data_key: str) -> str:
    """Key depends on data key + every factor-model parameter."""
    payload = (
        data_key,
        req.model_type, req.pca_lookback, req.pca_n_components,
        req.explained_variance_threshold, req.use_ledoit_wolf,
        req.beta_rolling_window, req.hedge_instrument,
        req.pairs_pvalue, req.pairs_max, req.pairs_min_hl, req.pairs_max_hl,
    )
    return "f_" + hashlib.sha256(str(payload).encode()).hexdigest()[:20]


def _cache_key_result(req: BacktestRequest) -> str:
    """Key depends on every request field (tickers sorted for order-independence)."""
    d = req.model_dump()
    d["tickers"] = sorted(d["tickers"])
    return "r_" + hashlib.sha256(str(sorted(d.items())).encode()).hexdigest()[:20]


def _cache_load(key: str):
    path = _CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        _logger.warning("Corrupted cache entry %s — removing: %s", key, exc)
        path.unlink(missing_ok=True)
        return None


def _cache_save(key: str, value) -> None:
    path = _CACHE_DIR / f"{key}.pkl"
    try:
        with path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        _logger.warning("Failed to write cache entry %s: %s", key, exc)
        path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    data_source: str = "yfinance"
    tickers: List[str]
    start_date: str
    end_date: str

    model_type: str = "pca"
    pca_lookback: int = 252
    pca_n_components: Optional[int] = 15
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252

    ou_window: int = 60
    kappa_min: float = 8.4
    mean_center: bool = True

    s_bo: float = 1.25
    s_so: float = 1.25
    s_sc: float = 0.50
    s_bc: float = 0.75
    s_limit: float = 4.0

    vol_enabled: bool = False
    vol_window: int = 10

    initial_equity: float = 1_000_000
    leverage_long: float = 2.0
    leverage_short: float = 2.0
    tc_bps: float = 1.0
    hedge_instrument: str = "SPY"

    pairs_pvalue: float = 0.05
    pairs_max: int = 20
    pairs_min_hl: float = 1.0
    pairs_max_hl: float = 126.0

    # Extra fields from newer frontend — accepted but not yet wired to engine
    hmm_enabled: bool = False
    hmm_n_states: int = 2
    hmm_training_window: int = 252
    hmm_feature_window: int = 20
    hmm_entry_threshold: float = 0.5
    hmm_favorable_high_vol: bool = True
    hmm_soft_gate: bool = True
    hmm_soft_gate_floor: float = 0.2
    vol_target_enabled: bool = False
    vol_target_floor: float = 0.2
    vol_target_cap: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config/defaults")
def get_defaults():
    return {
        "default_tickers": DEFAULT_TICKERS,
        "paper_tickers_count": len(PAPER_TICKERS),
        "modern_tickers_count": len(MODERN_TICKERS),
        "data_sources": DATA_SOURCES,
        "model_types": [
            {"value": "pca", "label": "PCA (Eigenportfolios)"},
            {"value": "etf", "label": "Sector ETF Regression"},
            {"value": "combined", "label": "Combined (SPY + ETF + PCA)"},
            {"value": "pairs", "label": "Pairs Trading (Cointegration)"},
        ],
        "hedge_instruments": ["SPY", "sector_etf", "none"],
        "ticker_presets": {
            "default": DEFAULT_TICKERS,
            "paper": PAPER_TICKERS,
            "modern": MODERN_TICKERS,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cache management endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/cache")
def cache_info():
    _KIND = {"d_": "data", "f_": "factor", "r_": "result"}
    entries = []
    total = 0
    for p in sorted(_CACHE_DIR.glob("*.pkl")):
        size = p.stat().st_size
        total += size
        kind = _KIND.get(p.stem[:2], "unknown")
        entries.append({"file": p.name, "type": kind, "size_mb": round(size / 1e6, 2)})
    return {"count": len(entries), "total_mb": round(total / 1e6, 2), "entries": entries}


@app.delete("/api/cache")
def cache_clear():
    cleared = 0
    for p in _CACHE_DIR.glob("*.pkl"):
        p.unlink()
        cleared += 1
    return {"cleared": cleared}


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_config(req: BacktestRequest) -> Config:
    return Config(
        factor=FactorConfig(
            model_type=req.model_type,
            pca_lookback=req.pca_lookback,
            pca_n_components=req.pca_n_components,
            explained_variance_threshold=req.explained_variance_threshold,
            use_ledoit_wolf=req.use_ledoit_wolf,
            beta_rolling_window=req.beta_rolling_window,
        ),
        ou=OUConfig(
            estimation_window=req.ou_window,
            kappa_min=req.kappa_min,
            mean_center=req.mean_center,
        ),
        signal=SignalConfig(
            s_bo=req.s_bo, s_so=req.s_so, s_sc=req.s_sc,
            s_bc=req.s_bc, s_limit=req.s_limit,
        ),
        volume=VolumeConfig(
            enabled=req.vol_enabled, trailing_window=req.vol_window
        ),
        backtest=BacktestConfig(
            initial_equity=float(req.initial_equity),
            leverage_long=req.leverage_long,
            leverage_short=req.leverage_short,
            tc_bps=req.tc_bps,
            hedge_instrument=req.hedge_instrument,
        ),
        pairs=PairsConfig(
            pvalue_threshold=req.pairs_pvalue,
            max_pairs=req.pairs_max,
            min_half_life=req.pairs_min_hl,
            max_half_life=req.pairs_max_hl,
        ),
        data_source=req.data_source,
        start_date=req.start_date,
        end_date=req.end_date,
        tickers=req.tickers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers used by both backtest and grid-search endpoints
# ─────────────────────────────────────────────────────────────────────────────
def _load_or_fetch_data(req: BacktestRequest, config, data_source, p: Callable):
    """
    Return (prices, volume, returns, available, sector_mapping, dkey).
    Checks the data cache first; falls through to a live fetch on miss.
    """
    dkey = _cache_key_data(req)
    cached = _cache_load(dkey)
    if cached is not None:
        prices, volume, returns, available, sector_mapping = cached
        p("universe",
          f"Market data loaded from cache ({len(available)} tickers).", 0.22)
        return prices, volume, returns, available, sector_mapping, dkey

    p("fetch_data", "Fetching prices, volume and returns…", 0.08)
    all_tickers = config.tickers
    prices  = data_source.fetch_prices(all_tickers, config.start_date, config.end_date)
    volume  = data_source.fetch_volume(all_tickers, config.start_date, config.end_date)
    returns = data_source.fetch_returns(all_tickers, config.start_date, config.end_date)

    available = [t for t in all_tickers if t in prices.columns]
    prices  = prices[available]
    volume  = volume[[t for t in available if t in volume.columns]]
    returns = returns[[t for t in available if t in returns.columns]]
    p("universe",
      f"Universe ready: {len(available)}/{len(all_tickers)} tickers.", 0.18)

    p("sector_map", "Resolving sector mapping…", 0.22)
    sector_mapping = get_sector_mapping(available, data_source=data_source)

    _cache_save(dkey, (prices, volume, returns, available, sector_mapping))
    return prices, volume, returns, available, sector_mapping, dkey


def _load_or_fit_factor(req: BacktestRequest, config, data_source,
                        prices, volume, returns, available, sector_mapping,
                        dkey: str, p: Callable):
    """
    Return (factor_result, bt_prices, bt_volume, bt_returns,
            etf_returns_df, spy_returns_df, fkey).
    Checks the factor cache first; falls through to a live fit on miss.
    """
    fkey = _cache_key_factor(req, dkey)
    cached = _cache_load(fkey)
    if cached is not None:
        factor_result, bt_prices, bt_volume, bt_returns, etf_df, spy_df = cached
        p("fit_factor_model", "Factor model loaded from cache.", 0.55)
        return factor_result, bt_prices, bt_volume, bt_returns, etf_df, spy_df, fkey

    p("build_factor_model",
      f"Building {config.factor.model_type.upper()} factor model…", 0.28)
    factor_model = build_factor_model(
        config.factor, sector_mapping, pairs_cfg=config.pairs
    )

    kwargs: dict = {}
    etf_df = spy_df = None
    needs_etf = (
        config.factor.model_type in ("etf", "combined")
        or config.backtest.hedge_instrument == "sector_etf"
    )
    needs_spy = (
        config.factor.model_type in ("combined", "pca")
        or config.backtest.hedge_instrument == "SPY"
    )

    if needs_etf or needs_spy:
        p("fetch_factors", "Fetching ETF / SPY reference series…", 0.34)
    if needs_etf:
        etf_tickers = list(set(sector_mapping.values()))
        etf_prices = data_source.fetch_prices(
            etf_tickers, config.start_date, config.end_date
        )
        etf_df = np.log(etf_prices / etf_prices.shift(1)).dropna(how="all")
        kwargs["etf_returns"] = etf_df
    if needs_spy:
        spy_prices = data_source.fetch_prices(
            [MARKET_ETF], config.start_date, config.end_date
        )
        spy_df = np.log(spy_prices / spy_prices.shift(1)).dropna(how="all")
        kwargs["spy_returns"] = spy_df
    if config.factor.model_type == "pairs":
        kwargs["prices"] = prices

    p("fit_factor_model",
      "Fitting factor model (rolling PCA / OLS / cointegration)…", 0.42)
    factor_result = factor_model.fit(returns, **kwargs)

    if config.factor.model_type == "pairs":
        pair_prices = {}
        for col in factor_result.residuals.columns:
            cs = factor_result.residuals[col].cumsum()
            first_finite = cs.dropna()
            if first_finite.empty:
                continue
            pair_prices[col] = 100 * np.exp(cs - first_finite.iloc[0])
        bt_prices = pd.DataFrame(pair_prices)
        bt_volume = pd.DataFrame(
            np.ones(bt_prices.shape),
            index=bt_prices.index,
            columns=bt_prices.columns,
        )
        bt_returns = None
    else:
        bt_prices  = prices
        bt_volume  = volume
        bt_returns = returns

    _cache_save(fkey, (factor_result, bt_prices, bt_volume, bt_returns, etf_df, spy_df))
    return factor_result, bt_prices, bt_volume, bt_returns, etf_df, spy_df, fkey


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest runner (three-level cache)
# ─────────────────────────────────────────────────────────────────────────────
def _execute_backtest(
    req: BacktestRequest,
    progress: Callable[[str, str, float], None] | None = None,
) -> dict:
    """Run the full backtest pipeline with three-level disk caching.

    Cache levels (checked in order):
      1. Result cache  — identical request → return immediately
      2. Factor cache  — same data + factor params → skip factor fit
      3. Data cache    — same universe + dates → skip data download
    """
    p = progress or (lambda *_: None)

    # ── Level 1: full result cache ────────────────────────────────────────────
    rkey = _cache_key_result(req)
    cached_result = _cache_load(rkey)
    if cached_result is not None:
        p("done", "Result loaded from cache — instant replay.", 1.0)
        return cached_result

    p("config", "Building configuration…", 0.02)
    config      = _build_config(req)
    data_source = get_data_source(config.data_source)
    all_tickers = config.tickers

    dkey = _cache_key_data(req)
    fkey = _cache_key_factor(req, dkey)

    # ── Level 2: factor cache (skips data download + factor fitting) ──────────
    cached_factor = _cache_load(fkey)
    if cached_factor is not None:
        (factor_result, bt_prices, bt_volume,
         bt_returns, etf_df, spy_df) = cached_factor
        # Recover available from data cache (always present when factor is)
        cached_data = _cache_load(dkey)
        available = cached_data[3] if cached_data is not None else []
        sector_mapping = cached_data[4] if cached_data is not None else {}
        p("fit_factor_model", "Factor model loaded from cache.", 0.55)
    else:
        # ── Level 3: data cache (skips only the download) ─────────────────────
        prices, volume, returns, available, sector_mapping, dkey = \
            _load_or_fetch_data(req, config, data_source, p)

        (factor_result, bt_prices, bt_volume,
         bt_returns, etf_df, spy_df, fkey) = \
            _load_or_fit_factor(req, config, data_source,
                                prices, volume, returns,
                                available, sector_mapping, dkey, p)

    # ── Backtest engine ───────────────────────────────────────────────────────
    p("run_backtest", "Running backtest engine…", 0.55)
    result = run_backtest(
        config, bt_prices, bt_volume, factor_result,
        returns=bt_returns,
        etf_returns=etf_df,
        spy_returns=spy_df,
        sector_mapping=sector_mapping,
    )

    # ── Metrics & serialisation ───────────────────────────────────────────────
    p("metrics", "Computing metrics, drawdowns and trade book…", 0.92)
    eq          = result.equity_curve
    running_max = eq.cummax()
    drawdown    = eq / running_max - 1.0

    yearly     = eq.resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()

    last_sscores: list = []
    if not result.daily_sscores.empty:
        last = result.daily_sscores.iloc[-1].dropna()
        for tk, val in last.items():
            if val <= -config.signal.s_bo:
                sig = "LONG"
            elif val >= config.signal.s_so:
                sig = "SHORT"
            else:
                sig = "NEUTRAL"
            last_sscores.append(
                {"ticker": str(tk), "sscore": float(val), "signal": sig}
            )
        last_sscores.sort(key=lambda r: r["sscore"])

    exposure_curve: list = []
    if not result.daily_positions.empty:
        dp = result.daily_positions.copy()
        dp["long"]  = np.where(dp["direction"] == 1,  dp["notional"], 0.0)
        dp["short"] = np.where(dp["direction"] == -1, dp["notional"], 0.0)
        agg = dp.groupby("date").agg(
            long=("long", "sum"), short=("short", "sum")
        ).reset_index()
        for r in agg.itertuples():
            exposure_curve.append({
                "date": str(r.date)[:10],
                "long": float(r.long),
                "short": float(r.short),
            })

    trades_list: list = []
    if not result.trades.empty:
        for r in result.trades.tail(5000).itertuples():
            trades_list.append({
                "ticker":      str(r.ticker),
                "direction":   int(r.direction),
                "entry_date":  str(r.entry_date)[:10],
                "exit_date":   str(r.exit_date)[:10],
                "entry_price": float(r.entry_price),
                "exit_price":  float(r.exit_price),
                "pnl":         float(r.pnl),
                "notional":    float(r.notional),
            })

    meta           = factor_result.metadata or {}
    eigenvalues    = meta.get("eigenvalues")
    all_eigenvals  = meta.get("all_eigenvalues")
    diag = {
        "model_type": config.factor.model_type,
        "eigenvalues": (
            [float(x) for x in eigenvalues[:20]] if eigenvalues is not None else []
        ),
        "all_eigenvalues_top": (
            [float(x) for x in all_eigenvals[:50]] if all_eigenvals is not None else []
        ),
        "explained_variance_ratio": float(meta.get("explained_variance_ratio") or 0.0),
        "n_components": int(meta.get("n_components") or 0),
        "r_squared": [
            {"ticker": k, "r2": float(v)}
            for k, v in (meta.get("r_squared") or {}).items()
        ],
    }

    ou_rows: list = []
    if result.daily_ou_params:
        last_ou_key = sorted(result.daily_ou_params.keys())[-1]
        for tk, ou in result.daily_ou_params[last_ou_key].items():
            ou_rows.append({
                "ticker":      str(tk),
                "kappa":       float(ou.kappa),
                "m":           float(ou.m),
                "sigma_eq":    float(ou.sigma_eq),
                "half_life":   float(ou.half_life),
                "factor_beta": float(ou.factor_beta),
            })

    m = result.metrics
    response = {
        "metrics": {
            "total_return":       float(m.total_return),
            "annualized_return":  float(m.annualized_return),
            "annualized_vol":     float(m.annualized_vol),
            "sharpe_ratio":       float(m.sharpe_ratio),
            "sortino_ratio":      float(m.sortino_ratio),
            "max_drawdown":       float(m.max_drawdown),
            "win_rate":           float(m.win_rate),
            "trade_win_rate":     float(m.trade_win_rate),
            "profit_factor":      float(m.profit_factor),
            "num_trades":         int(m.num_trades),
            "total_costs":        float(m.total_costs),
            "avg_holding_period": float(m.avg_holding_period),
        },
        "equity_curve": [
            {"date": str(d)[:10], "equity": float(v)} for d, v in eq.items()
        ],
        "drawdown_curve": [
            {"date": str(d)[:10], "drawdown": float(v)}
            for d, v in drawdown.items()
        ],
        "exposure_curve": exposure_curve,
        "annual_returns": [
            {"year": int(y.year), "return": float(v)}
            for y, v in yearly_ret.items()
        ],
        "last_sscores": last_sscores,
        "data_summary": {
            "n_requested": len(all_tickers),
            "n_returned":  len(available),
            "n_dropped":   len(all_tickers) - len(available),
        },
        "trades":      trades_list,
        "diagnostics": diag,
        "ou_last":     ou_rows,
        "regime_curve": [],
    }

    _cache_save(rkey, response)
    p("done", "Backtest complete.", 1.0)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Backtest endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        return _execute_backtest(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/stream")
def run_backtest_stream(req: BacktestRequest):
    q: queue.Queue = queue.Queue()

    def progress(stage: str, message: str, pct: float) -> None:
        q.put({"event": "progress", "stage": stage,
               "message": message, "progress": float(pct)})

    def worker() -> None:
        try:
            data = _execute_backtest(req, progress)
            q.put({"event": "result", "data": data})
        except Exception as exc:
            _logger.exception("Backtest worker error")
            q.put({"event": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cointegration endpoint
# ─────────────────────────────────────────────────────────────────────────────
class CointRequest(BaseModel):
    data_source: str = "yfinance"
    tickers: List[str]
    start_date: str
    end_date: str
    pvalue_threshold: float = 0.05
    lookback: int = 252
    pair_t1: Optional[str] = None
    pair_t2: Optional[str] = None


@app.post("/api/cointegration")
def cointegration(req: CointRequest):
    try:
        from statarb.signals.cointegration import test_cointegration

        ds     = get_data_source(req.data_source)
        prices = ds.fetch_prices(req.tickers, req.start_date, req.end_date)
        prices = prices.dropna(axis=1, how="all")

        coint_df = test_cointegration(prices, req.pvalue_threshold, lookback=req.lookback)
        rows = []
        for _, r in coint_df.iterrows():
            rows.append({
                "ticker1":     r["ticker1"],
                "ticker2":     r["ticker2"],
                "pvalue":      float(r["pvalue"]),
                "score":       float(r["score"]),
                "hedge_ratio": float(r["hedge_ratio"]),
                "spread_mean": float(r["spread_mean"]),
                "spread_std":  float(r["spread_std"]),
                "half_life": (
                    float(r["half_life"]) if pd.notna(r["half_life"]) else None
                ),
            })

        spread = []
        if (req.pair_t1 and req.pair_t2
                and req.pair_t1 in prices.columns
                and req.pair_t2 in prices.columns):
            log_p = np.log(prices[[req.pair_t1, req.pair_t2]]).dropna()
            beta  = float(
                np.polyfit(log_p[req.pair_t2].values, log_p[req.pair_t1].values, 1)[0]
            )
            s    = log_p[req.pair_t1] - beta * log_p[req.pair_t2]
            mean = float(s.mean())
            std  = float(s.std())
            for d, v in s.items():
                spread.append({
                    "date":   str(d)[:10],
                    "spread": float(v),
                    "z": (float(v) - mean) / std if std > 0 else 0.0,
                })

        return {"pairs": rows, "spread": spread}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Grid search endpoint — reuses data + factor cache
# ─────────────────────────────────────────────────────────────────────────────
class GridRequest(BacktestRequest):
    s_bo_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]
    s_so_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]


@app.post("/api/grid-search")
def grid_search(req: GridRequest):
    try:
        config      = _build_config(req)
        data_source = get_data_source(config.data_source)
        all_tickers = config.tickers
        noop        = lambda *_: None

        dkey = _cache_key_data(req)
        fkey = _cache_key_factor(req, dkey)

        # Try factor cache first (skips both download and factor fit)
        cached_factor = _cache_load(fkey)
        if cached_factor is not None:
            (factor_result, bt_prices, bt_volume,
             bt_returns, etf_df, spy_df) = cached_factor
            cached_data    = _cache_load(dkey)
            available      = cached_data[3] if cached_data is not None else []
            sector_mapping = cached_data[4] if cached_data is not None else {}
        else:
            prices, volume, returns, available, sector_mapping, dkey = \
                _load_or_fetch_data(req, config, data_source, noop)
            (factor_result, bt_prices, bt_volume,
             bt_returns, etf_df, spy_df, fkey) = \
                _load_or_fit_factor(req, config, data_source,
                                    prices, volume, returns,
                                    available, sector_mapping, dkey, noop)

        cells = []
        for s_bo in req.s_bo_values:
            for s_so in req.s_so_values:
                cfg = copy.deepcopy(config)
                cfg.signal.s_bo = float(s_bo)
                cfg.signal.s_so = float(s_so)
                res = run_backtest(
                    cfg, bt_prices, bt_volume, factor_result,
                    returns=bt_returns,
                    etf_returns=etf_df,
                    spy_returns=spy_df,
                    sector_mapping=sector_mapping,
                )
                cells.append({
                    "s_bo":         float(s_bo),
                    "s_so":         float(s_so),
                    "sharpe":       float(res.metrics.sharpe_ratio),
                    "total_return": float(res.metrics.total_return),
                    "max_drawdown": float(res.metrics.max_drawdown),
                    "num_trades":   int(res.metrics.num_trades),
                })

        best = max(cells, key=lambda c: c["sharpe"]) if cells else None
        return {
            "s_bo_values": req.s_bo_values,
            "s_so_values": req.s_so_values,
            "cells":       cells,
            "best":        best,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
