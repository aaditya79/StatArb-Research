"""
FastAPI backend exposing the existing statarb pipeline as JSON endpoints.

This file ONLY orchestrates calls to the unchanged statarb.* and config.*
modules — no trading / signal / model logic lives here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

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
    BacktestConfig, PairsConfig, VolTargetConfig, HMMConfig,
    DEFAULT_TICKERS, DATA_SOURCES, MARKET_ETF,
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

logger = logging.getLogger(__name__)

CACHE_VERSION = "api-cache-v1"
CACHE_DIR = Path(os.getenv("STATARB_CACHE_DIR", Path(ROOT) / ".cache" / "api"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        {"version": CACHE_VERSION, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"


def _cache_load(key: str) -> Any | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Dropping unreadable cache entry %s: %s", path.name, exc)
        path.unlink(missing_ok=True)
        return None


def _cache_save(key: str, value: Any) -> None:
    path = _cache_path(key)
    tmp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("Failed to write cache entry %s: %s", path.name, exc)
        tmp_path.unlink(missing_ok=True)


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

    # ── Extensions ──
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


def _request_dict(req: BaseModel) -> dict[str, Any]:
    return req.model_dump() if hasattr(req, "model_dump") else req.dict()


def _cache_key_data(req: BacktestRequest) -> str:
    return "data_" + _stable_hash({
        "data_source": req.data_source,
        "tickers": list(req.tickers),
        "start_date": req.start_date,
        "end_date": req.end_date,
    })


def _cache_key_factor(req: BacktestRequest, data_key: str) -> str:
    return "factor_" + _stable_hash({
        "data_key": data_key,
        "model_type": req.model_type,
        "pca_lookback": req.pca_lookback,
        "pca_n_components": req.pca_n_components,
        "explained_variance_threshold": req.explained_variance_threshold,
        "use_ledoit_wolf": req.use_ledoit_wolf,
        "beta_rolling_window": req.beta_rolling_window,
        "hedge_instrument": req.hedge_instrument,
        "pairs_pvalue": req.pairs_pvalue,
        "pairs_max": req.pairs_max,
        "pairs_min_hl": req.pairs_min_hl,
        "pairs_max_hl": req.pairs_max_hl,
    })


def _cache_key_result(req: BacktestRequest) -> str:
    return "result_" + _stable_hash(_request_dict(req))


def _cache_key_grid(req: "GridRequest") -> str:
    return "grid_" + _stable_hash(_request_dict(req))


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


@app.get("/api/cache")
def cache_info():
    entries = []
    total_bytes = 0
    for path in sorted(CACHE_DIR.glob("*.pkl")):
        stat = path.stat()
        total_bytes += stat.st_size
        cache_type = path.stem.split("_", 1)[0]
        entries.append({
            "file": path.name,
            "type": cache_type,
            "size_mb": round(stat.st_size / 1_000_000, 3),
            "modified": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
        })
    return {
        "directory": str(CACHE_DIR),
        "count": len(entries),
        "total_mb": round(total_bytes / 1_000_000, 3),
        "entries": entries,
    }


@app.delete("/api/cache")
def cache_clear():
    cleared = 0
    for path in CACHE_DIR.glob("*.pkl"):
        path.unlink(missing_ok=True)
        cleared += 1
    return {"cleared": cleared}


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
        vol_target=VolTargetConfig(
            enabled=req.vol_target_enabled,
            floor_multiplier=req.vol_target_floor,
            cap_multiplier=req.vol_target_cap,
        ),
        hmm=HMMConfig(
            enabled=req.hmm_enabled,
            n_states=req.hmm_n_states,
            training_window=req.hmm_training_window,
            feature_window=req.hmm_feature_window,
            entry_threshold=req.hmm_entry_threshold,
            favorable_high_vol=req.hmm_favorable_high_vol,
            soft_gate=req.hmm_soft_gate,
            soft_gate_floor=req.hmm_soft_gate_floor,
        ),
        data_source=req.data_source,
        start_date=req.start_date,
        end_date=req.end_date,
        tickers=req.tickers,
    )


def _select_available_frames(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]
    volume = volume[[t for t in available if t in volume.columns]]
    returns = returns[[t for t in available if t in returns.columns]]
    return prices, volume, returns, available


def _load_or_fetch_data(
    req: BacktestRequest,
    config: Config,
    data_source,
    progress: Callable[[str, str, float], None],
) -> dict[str, Any]:
    data_key = _cache_key_data(req)
    cached = _cache_load(data_key)
    if isinstance(cached, dict) and {"prices", "volume", "returns"} <= set(cached):
        prices, volume, returns, available = _select_available_frames(
            cached["prices"],
            cached["volume"],
            cached["returns"],
            config.tickers,
        )
        sector_mapping = dict(cached.get("sector_mapping") or {})
        missing = [t for t in available if t not in sector_mapping]
        if missing:
            sector_mapping.update(get_sector_mapping(missing, data_source=data_source))
        progress(
            "data_cache",
            f"Market data loaded from cache ({len(available)} tickers).",
            0.18,
        )
        return {
            "data_key": data_key,
            "prices": prices,
            "volume": volume,
            "returns": returns,
            "available": available,
            "sector_mapping": sector_mapping,
        }

    progress("fetch_data", "Fetching prices and volume…", 0.08)
    prices = data_source.fetch_prices(
        config.tickers, config.start_date, config.end_date
    )
    volume = data_source.fetch_volume(
        config.tickers, config.start_date, config.end_date
    )
    returns = np.log(prices / prices.shift(1)).dropna(how="all")

    prices, volume, returns, available = _select_available_frames(
        prices, volume, returns, config.tickers
    )
    progress(
        "universe",
        f"Universe ready: {len(available)}/{len(config.tickers)} tickers loaded.",
        0.18,
    )

    progress("sector_map", "Resolving sector mapping…", 0.22)
    sector_mapping = get_sector_mapping(available, data_source=data_source)

    payload = {
        "prices": prices,
        "volume": volume,
        "returns": returns,
        "sector_mapping": sector_mapping,
    }
    _cache_save(data_key, payload)

    return {
        "data_key": data_key,
        **payload,
        "available": available,
    }


def _fit_and_cache_factor(
    req: BacktestRequest,
    config: Config,
    data_source,
    data: dict[str, Any],
    progress: Callable[[str, str, float], None],
) -> dict[str, Any]:
    factor_key = _cache_key_factor(req, data["data_key"])
    prices = data["prices"]
    volume = data["volume"]
    returns = data["returns"]
    sector_mapping = data["sector_mapping"]

    progress(
        "build_factor_model",
        f"Building {config.factor.model_type.upper()} factor model…",
        0.28,
    )
    factor_model = build_factor_model(
        config.factor, sector_mapping, pairs_cfg=config.pairs
    )

    kwargs: dict[str, Any] = {}
    etf_returns_df = None
    spy_returns_df = None
    needs_etf = (
        config.factor.model_type in ("etf", "combined")
        or config.backtest.hedge_instrument == "sector_etf"
    )
    needs_spy = (
        config.factor.model_type in ("combined", "pca")
        or config.backtest.hedge_instrument == "SPY"
    )

    if needs_etf or needs_spy:
        progress("fetch_factors", "Fetching ETF / SPY reference series…", 0.34)
    if needs_etf:
        etf_tickers = sorted(set(sector_mapping.values()))
        etf_prices = data_source.fetch_prices(
            etf_tickers, config.start_date, config.end_date
        )
        etf_returns_df = np.log(etf_prices / etf_prices.shift(1)).dropna(how="all")
        kwargs["etf_returns"] = etf_returns_df
    if needs_spy:
        spy_prices = data_source.fetch_prices(
            [MARKET_ETF], config.start_date, config.end_date
        )
        spy_returns_df = np.log(spy_prices / spy_prices.shift(1)).dropna(how="all")
        kwargs["spy_returns"] = spy_returns_df
    if config.factor.model_type == "pairs":
        kwargs["prices"] = prices

    progress(
        "fit_factor_model",
        "Fitting factor model (rolling PCA / OLS / cointegration)…",
        0.42,
    )
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
        bt_prices = prices
        bt_volume = volume
        bt_returns = returns

    payload = {
        "factor_result": factor_result,
        "bt_prices": bt_prices,
        "bt_volume": bt_volume,
        "bt_returns": bt_returns,
        "etf_returns_df": etf_returns_df,
        "spy_returns_df": spy_returns_df,
        "available": data["available"],
        "sector_mapping": sector_mapping,
    }
    _cache_save(factor_key, payload)
    return {"factor_key": factor_key, **payload}


def _prepare_backtest_inputs(
    req: BacktestRequest,
    config: Config,
    data_source,
    progress: Callable[[str, str, float], None],
) -> dict[str, Any]:
    data_key = _cache_key_data(req)
    factor_key = _cache_key_factor(req, data_key)
    cached_factor = _cache_load(factor_key)
    required = {
        "factor_result", "bt_prices", "bt_volume", "bt_returns",
        "etf_returns_df", "spy_returns_df", "available", "sector_mapping",
    }
    if isinstance(cached_factor, dict) and required <= set(cached_factor):
        progress(
            "factor_cache",
            "Data and factor model loaded from cache.",
            0.55,
        )
        return {
            "data_key": data_key,
            "factor_key": factor_key,
            **cached_factor,
        }

    data = _load_or_fetch_data(req, config, data_source, progress)
    return {
        "data_key": data["data_key"],
        **_fit_and_cache_factor(req, config, data_source, data, progress),
    }


def _execute_backtest(
    req: BacktestRequest,
    progress: Callable[[str, str, float], None] | None = None,
) -> dict:
    """Run the full backtest pipeline, calling `progress(stage, message, pct)`
    at each milestone. Returns the JSON-serialisable response payload."""
    p = progress or (lambda *_: None)

    result_key = _cache_key_result(req)
    cached_result = _cache_load(result_key)
    if isinstance(cached_result, dict):
        p("result_cache", "Backtest result loaded from cache.", 1.0)
        return cached_result

    p("config", "Building configuration…", 0.02)
    config = _build_config(req)
    data_source = get_data_source(config.data_source)
    all_tickers = config.tickers
    prepared = _prepare_backtest_inputs(req, config, data_source, p)

    factor_result = prepared["factor_result"]
    bt_prices = prepared["bt_prices"]
    bt_volume = prepared["bt_volume"]
    bt_returns = prepared["bt_returns"]
    etf_returns_df = prepared["etf_returns_df"]
    spy_returns_df = prepared["spy_returns_df"]
    available = prepared["available"]
    sector_mapping = prepared["sector_mapping"]

    p(
        "run_backtest",
        "Running backtest engine — OU fit, signals"
        + (", HMM regime gating" if config.hmm.enabled else "")
        + (", vol-targeted sizing" if config.vol_target.enabled else "")
        + "…",
        0.55,
    )
    result = run_backtest(
        config, bt_prices, bt_volume, factor_result,
        returns=bt_returns,
        etf_returns=etf_returns_df,
        spy_returns=spy_returns_df,
        sector_mapping=sector_mapping,
    )

    p("metrics", "Computing metrics, drawdowns and trade book…", 0.92)
    eq = result.equity_curve
    running_max = eq.cummax()
    drawdown = (eq / running_max - 1.0)

    yearly = eq.resample("YE").last()
    yearly_ret = yearly.pct_change().dropna()

    last_sscores = []
    if not result.daily_sscores.empty:
        last = result.daily_sscores.iloc[-1].dropna()
        for tk, val in last.items():
            if val <= -config.signal.s_bo:
                sig = "LONG"
            elif val >= config.signal.s_so:
                sig = "SHORT"
            else:
                sig = "NEUTRAL"
            last_sscores.append({
                "ticker": str(tk), "sscore": float(val), "signal": sig,
            })
        last_sscores.sort(key=lambda r: r["sscore"])

    exposure_curve = []
    if not result.daily_positions.empty:
        dp = result.daily_positions.copy()
        dp["long"] = np.where(dp["direction"] == 1, dp["notional"], 0.0)
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

    trades_list = []
    if not result.trades.empty:
        tr = result.trades.tail(5000)
        for r in tr.itertuples():
            trades_list.append({
                "ticker": str(r.ticker),
                "direction": int(r.direction),
                "entry_date": str(r.entry_date)[:10],
                "exit_date": str(r.exit_date)[:10],
                "entry_price": float(r.entry_price),
                "exit_price": float(r.exit_price),
                "pnl": float(r.pnl),
                "notional": float(r.notional),
            })

    meta = factor_result.metadata or {}
    eigenvalues = meta.get("eigenvalues")
    all_eigenvalues = meta.get("all_eigenvalues")
    diag = {
        "model_type": config.factor.model_type,
        "eigenvalues": (
            [float(x) for x in eigenvalues[:20]]
            if eigenvalues is not None else []
        ),
        "all_eigenvalues_top": (
            [float(x) for x in all_eigenvalues[:50]]
            if all_eigenvalues is not None else []
        ),
        "explained_variance_ratio": float(
            meta.get("explained_variance_ratio") or 0.0
        ),
        "n_components": int(meta.get("n_components") or 0),
        "r_squared": [
            {"ticker": k, "r2": float(v)}
            for k, v in (meta.get("r_squared") or {}).items()
        ],
    }

    ou_rows = []
    if result.daily_ou_params:
        last_key = sorted(result.daily_ou_params.keys())[-1]
        for tk, ou in result.daily_ou_params[last_key].items():
            ou_rows.append({
                "ticker": str(tk),
                "kappa": float(ou.kappa),
                "m": float(ou.m),
                "sigma_eq": float(ou.sigma_eq),
                "half_life": float(ou.half_life),
                "factor_beta": float(ou.factor_beta),
            })

    m = result.metrics
    response = {
        "metrics": {
            "total_return": float(m.total_return),
            "annualized_return": float(m.annualized_return),
            "annualized_vol": float(m.annualized_vol),
            "sharpe_ratio": float(m.sharpe_ratio),
            "sortino_ratio": float(m.sortino_ratio),
            "max_drawdown": float(m.max_drawdown),
            "win_rate": float(m.win_rate),
            "trade_win_rate": float(m.trade_win_rate),
            "profit_factor": float(m.profit_factor),
            "num_trades": int(m.num_trades),
            "total_costs": float(m.total_costs),
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
            "n_returned": len(available),
            "n_dropped": len(all_tickers) - len(available),
        },
        "trades": trades_list,
        "diagnostics": diag,
        "ou_last": ou_rows,
        "regime_curve": (
            [{"date": str(d)[:10], "p_favorable": float(v)}
             for d, v in result.regime_proba.items()]
            if result.regime_proba is not None else []
        ),
    }
    _cache_save(result_key, response)
    p("done", "Backtest complete.", 1.0)
    return response


@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        return _execute_backtest(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/stream")
def run_backtest_stream(req: BacktestRequest):
    """Server-sent-events stream of progress + final result.

    Each event is a single JSON object on a `data:` line:
      {"event":"progress","stage":"...","message":"...","progress":0.42}
      {"event":"result","data":{...}}
      {"event":"error","message":"..."}
    """
    q: queue.Queue = queue.Queue()

    def progress(stage: str, message: str, pct: float) -> None:
        q.put({"event": "progress", "stage": stage,
               "message": message, "progress": float(pct)})

    def worker() -> None:
        try:
            data = _execute_backtest(req, progress)
            q.put({"event": "result", "data": data})
        except Exception as exc:
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

        ds = get_data_source(req.data_source)
        prices = ds.fetch_prices(req.tickers, req.start_date, req.end_date)
        prices = prices.dropna(axis=1, how="all")

        coint_df = test_cointegration(
            prices, req.pvalue_threshold, lookback=req.lookback
        )
        rows = []
        for _, r in coint_df.iterrows():
            rows.append({
                "ticker1": r["ticker1"],
                "ticker2": r["ticker2"],
                "pvalue": float(r["pvalue"]),
                "score": float(r["score"]),
                "hedge_ratio": float(r["hedge_ratio"]),
                "spread_mean": float(r["spread_mean"]),
                "spread_std": float(r["spread_std"]),
                "half_life": (
                    float(r["half_life"]) if pd.notna(r["half_life"]) else None
                ),
            })

        spread = []
        if (req.pair_t1 and req.pair_t2
                and req.pair_t1 in prices.columns
                and req.pair_t2 in prices.columns):
            log_p = np.log(prices[[req.pair_t1, req.pair_t2]]).dropna()
            beta = float(np.polyfit(
                log_p[req.pair_t2].values, log_p[req.pair_t1].values, 1
            )[0])
            s = log_p[req.pair_t1] - beta * log_p[req.pair_t2]
            mean = float(s.mean())
            std = float(s.std())
            for d, v in s.items():
                spread.append({
                    "date": str(d)[:10],
                    "spread": float(v),
                    "z": (float(v) - mean) / std if std > 0 else 0.0,
                })

        return {"pairs": rows, "spread": spread}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Grid search endpoint (sweeps s_bo / s_so)
# ─────────────────────────────────────────────────────────────────────────────
class GridRequest(BacktestRequest):
    s_bo_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]
    s_so_values: List[float] = [1.0, 1.25, 1.5, 1.75, 2.0]


@app.post("/api/grid-search")
def grid_search(req: GridRequest):
    try:
        grid_key = _cache_key_grid(req)
        cached_grid = _cache_load(grid_key)
        if isinstance(cached_grid, dict):
            return cached_grid

        config = _build_config(req)
        ds = get_data_source(config.data_source)
        prepared = _prepare_backtest_inputs(req, config, ds, lambda *_: None)

        factor_result = prepared["factor_result"]
        bt_prices = prepared["bt_prices"]
        bt_volume = prepared["bt_volume"]
        bt_returns = prepared["bt_returns"]
        etf_returns_df = prepared["etf_returns_df"]
        spy_returns_df = prepared["spy_returns_df"]
        sector_mapping = prepared["sector_mapping"]

        cells = []
        import copy as _copy
        for s_bo in req.s_bo_values:
            for s_so in req.s_so_values:
                cfg = _copy.deepcopy(config)
                cfg.signal.s_bo = float(s_bo)
                cfg.signal.s_so = float(s_so)
                res = run_backtest(
                    cfg, bt_prices, bt_volume, factor_result,
                    returns=bt_returns,
                    etf_returns=etf_returns_df,
                    spy_returns=spy_returns_df,
                    sector_mapping=sector_mapping,
                )
                cells.append({
                    "s_bo": float(s_bo),
                    "s_so": float(s_so),
                    "sharpe": float(res.metrics.sharpe_ratio),
                    "total_return": float(res.metrics.total_return),
                    "max_drawdown": float(res.metrics.max_drawdown),
                    "num_trades": int(res.metrics.num_trades),
                })

        # Best by Sharpe.
        best = max(cells, key=lambda c: c["sharpe"]) if cells else None
        response = {
            "s_bo_values": req.s_bo_values,
            "s_so_values": req.s_so_values,
            "cells": cells,
            "best": best,
        }
        _cache_save(grid_key, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
