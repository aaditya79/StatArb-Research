from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass
class CointResult:
    ticker1: str
    ticker2: str
    score: float
    pvalue: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float


def _estimate_half_life(spread: pd.Series) -> float:
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    idx = lag.index.intersection(delta.index)
    if len(idx) < 10:
        return np.nan
    b = np.polyfit(lag[idx].values, delta[idx].values, 1)[0]
    if b >= 0 or b <= -2:
        return np.nan
    return -np.log(2) / b


def test_cointegration(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    lookback: int = 252,
) -> pd.DataFrame:
    # Use per-pair dropna so tickers with partial history don't exclude all others
    log_px = np.log(prices.iloc[-lookback:])
    tickers = log_px.columns.tolist()
    rows = []
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            pair = log_px[[t1, t2]].dropna(how="any")
            if len(pair) < 60:
                continue
            try:
                score, pvalue, _ = coint(pair[t1].values, pair[t2].values)
            except Exception:
                continue
            if pvalue > pvalue_threshold:
                continue
            beta = float(np.polyfit(pair[t2].values, pair[t1].values, 1)[0])
            spread = pd.Series(
                pair[t1].values - beta * pair[t2].values, index=pair.index
            )
            hl = _estimate_half_life(spread)
            rows.append({
                "ticker1": t1, "ticker2": t2,
                "score": score, "pvalue": pvalue,
                "hedge_ratio": beta,
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std()),
                "half_life": hl,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ticker1", "ticker2", "score", "pvalue",
            "hedge_ratio", "spread_mean", "spread_std", "half_life"
        ])
    return pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)


def compute_pair_spread(log_prices: pd.DataFrame, t1: str, t2: str, beta: float) -> pd.Series:
    return log_prices[t1] - beta * log_prices[t2]
