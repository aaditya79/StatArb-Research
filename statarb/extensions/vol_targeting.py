"""
Volatility-Targeted Position Sizing.

Replaces equal-notional allocation with vol-parity sizing so each position
contributes roughly equal risk. The risk of position i is:

    risk_i = notional_i * sigma_eq_i

Setting risk_i = constant and using cross-sectional median sigma_eq as the
target gives:

    notional_i = base_notional * (target_sigma / sigma_eq_i)

Stocks with high equilibrium vol receive smaller notional; stable stocks
receive more. A floor and cap prevent degenerate positions.
"""
import numpy as np

from statarb.signals.ou_estimator import OUParams


class VolTargetedSizer:
    """
    Scale position notionals inversely with residual equilibrium volatility.

    Args:
        floor_multiplier: Minimum scale factor relative to equal-notional
            (prevents very small positions for high-vol stocks).
        cap_multiplier: Maximum scale factor relative to equal-notional
            (prevents very large positions for unusually calm stocks).
    """

    def __init__(
        self,
        floor_multiplier: float = 0.2,
        cap_multiplier: float = 5.0,
    ):
        self.floor_multiplier = floor_multiplier
        self.cap_multiplier = cap_multiplier

    def compute_scales(
        self, ou_params: dict[str, OUParams]
    ) -> dict[str, float]:
        """
        Compute per-ticker risk-parity scale factors, clipped and then
        renormalised so the mean scale across the eligible set is exactly 1.0.

        Renormalisation is the critical step — it guarantees that

            sum_i (base_notional * scale_i) == base_notional * n

        so the total deployed notional matches the equal-notional baseline
        regardless of the cross-sectional sigma distribution or the asymmetric
        floor/cap clip. Without this, scales = target / sigma with target =
        median (or any fixed target) drift the gross exposure upward whenever
        the harmonic mean of sigma is below the target, silently increasing
        leverage.

        Args:
            ou_params: Dict mapping ticker -> OUParams for eligible stocks.

        Returns:
            Dict mapping ticker -> scale factor (≥ 0, mean = 1.0). Tickers
            whose sigma_eq is non-positive are mapped to 1.0.
        """
        if not ou_params:
            return {}

        # Inverse-sigma weights (risk parity).
        raw: dict[str, float] = {}
        for tk, p in ou_params.items():
            raw[tk] = 1.0 / p.sigma_eq if p.sigma_eq > 0 else float("nan")

        finite = np.array([v for v in raw.values() if np.isfinite(v)])
        if finite.size == 0:
            return {tk: 1.0 for tk in ou_params}

        # Pre-normalise: mean(weight) = 1.0 before clipping.
        mean_w = float(np.mean(finite))
        if mean_w <= 0:
            return {tk: 1.0 for tk in ou_params}

        scales: dict[str, float] = {}
        for tk, w in raw.items():
            if not np.isfinite(w):
                scales[tk] = 1.0  # fallback for missing sigma
                continue
            s = w / mean_w
            scales[tk] = float(
                np.clip(s, self.floor_multiplier, self.cap_multiplier)
            )

        # Re-normalise after clipping so the budget is exactly preserved.
        finite_scales = np.array(list(scales.values()))
        mean_s = float(np.mean(finite_scales))
        if mean_s > 0:
            for tk in scales:
                scales[tk] = scales[tk] / mean_s
        return scales
