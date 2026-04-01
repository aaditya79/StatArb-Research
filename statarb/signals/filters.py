"""
Eligibility filters for trading signals.

The paper requires stocks to have sufficiently fast mean-reversion
(kappa > 8.4, corresponding to mean-reversion time < ~30 trading days)
before they can be traded.
"""
from .ou_estimator import OUParams


def filter_eligible(
    ou_params: dict[str, OUParams],
    kappa_min: float = 8.4,
) -> list[str]:
    """
    Filter tickers by mean-reversion speed.

    Args:
        ou_params: Dict mapping ticker -> OUParams.
        kappa_min: Minimum annualized kappa (paper default: 8.4,
            corresponding to mean-reversion time < T1/2 where T1 = 60/252).

    Returns:
        List of eligible ticker symbols.
    """
    eligible = []
    for ticker, params in ou_params.items():
        if params.kappa >= kappa_min:
            eligible.append(ticker)
    return eligible
