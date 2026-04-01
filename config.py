from dataclasses import dataclass, field
from typing import List, Optional


SECTOR_ETFS: List[str] = [
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY",
    "IYR", "IYT", "OIH", "SMH", "RTH", "RKH", "UTH",
    "XLB",
    "XLU",
    "GDX",
    "IBB",
    "KRE",
    "XOP",
    "XHB",
    "XRT",
    "IYZ",
    "XME",
]

MARKET_ETF: str = "SPY"

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS", "WFC", "C",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "UNH", "ABT", "MRK",
    "HD", "PG", "KO", "PEP", "WMT",
    "NEE", "DUK", "SO", "D", "AEP",
    "UNP", "UPS", "FDX", "CSX", "NSC",
    "NVDA", "INTC", "AVGO", "TXN", "QCOM",
]

SECTOR_TO_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Real Estate": "IYR",
    "Utilities": "UTH",
    "Basic Materials": "XLI",
    "Communication Services": "XLK",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Materials": "XLB",
    "Semiconductors": "SMH",
    "Telecom": "IYZ",
    "Telecommunications": "IYZ",
}

DATA_SOURCES = ["yfinance", "crsp"]


@dataclass
class FactorConfig:
    model_type: str = "pca"
    pca_lookback: int = 252
    pca_n_components: Optional[int] = 15
    explained_variance_threshold: float = 0.55
    use_ledoit_wolf: bool = True
    beta_rolling_window: int = 252


@dataclass
class OUConfig:
    estimation_window: int = 60
    kappa_min: float = 8.4
    mean_center: bool = True


@dataclass
class SignalConfig:
    s_bo: float = 1.25
    s_so: float = 1.25
    s_sc: float = 0.50
    s_bc: float = 0.75
    s_limit: float = 4.0


@dataclass
class VolumeConfig:
    enabled: bool = False
    trailing_window: int = 10


@dataclass
class PairsConfig:
    pvalue_threshold: float = 0.10
    max_pairs: int = 20
    min_half_life: float = 1.0
    max_half_life: float = 126.0
    lookback_window: int = 252
    auto_select: bool = True


@dataclass
class BacktestConfig:
    initial_equity: float = 1_000_000.0
    leverage_long: float = 2.0
    leverage_short: float = 2.0
    tc_bps: float = 5.0
    hedge_instrument: str = "SPY"
    risk_free_rate: float = 0.02
    dt: float = 1.0 / 252.0


@dataclass
class Config:
    factor: FactorConfig = field(default_factory=FactorConfig)
    ou: OUConfig = field(default_factory=OUConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    trading_mode: str = "statarb"
    data_source: str = "yfinance"
    start_date: str = "1997-01-01"
    end_date: str = "2007-12-31"
    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
