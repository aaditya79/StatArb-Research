"""
Volatility-Targeted Position Sizing.

Planned Enhancement:
    Replace equal-notional allocation with volatility-parity sizing so each
    position contributes approximately equal risk to the portfolio. High-
    volatility residuals get smaller notional, low-volatility get larger.

Interface (planned):
    class VolTargetedSizer:
        def __init__(self, target_vol: float = 0.10,
                     lookback: int = 60): ...
        def compute_weights(self, residuals: pd.DataFrame,
                            ou_params: dict) -> pd.Series: ...
        def scale_notional(self, base_notional: float,
                           sigma_eq: float, target_sigma: float) -> float: ...

Integration Points:
    - Called in backtest/portfolio.py when sizing new positions
    - Uses sigma_eq from signals/ou_estimator.py
    - Weight visualization on frontend

References:
    - Risk parity / volatility targeting literature
    - Presentation slide: "Uniform Sizing" weakness discussion
"""
