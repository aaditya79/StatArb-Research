"""
HMM Regime Detection for Statistical Arbitrage.

Planned Enhancement:
    Use a Hidden Markov Model to detect market regimes (e.g., trending vs.
    mean-reverting) and dynamically adjust s-score thresholds or disable
    trading during unfavorable regimes.

Interface (planned):
    class HMMRegimeDetector:
        def __init__(self, n_regimes: int = 2, lookback: int = 252): ...
        def fit(self, returns: pd.DataFrame) -> 'HMMRegimeDetector': ...
        def predict_regime(self, returns: pd.DataFrame) -> pd.Series: ...
        def get_regime_params(self) -> dict: ...

Integration Points:
    - Called before signal generation in backtest/engine.py
    - Output regime label fed to signals/generator.py to gate entries
    - Regime probabilities available for frontend visualization

References:
    - Hamilton (1989) "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle"
    - Avellaneda & Lee (2010) Section 7 discussion on market cycles
"""
