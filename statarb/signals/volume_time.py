"""
Volume-adjusted trading time returns (Paper Section 6, Eq. 20).

Measuring mean-reversion in trading time rescales stock returns:

    R_tilde_t = R_t * (avg_V / V_t)

where avg_V is the trailing average daily volume. This accentuates
signals on low volume and mitigates signals on high volume.
"""
import numpy as np
import pandas as pd


def compute_volume_adjusted_returns(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    trailing_window: int = 10,
) -> pd.DataFrame:
    """
    Rescale returns by inverse volume ratio.

    Args:
        returns: DataFrame of log returns (dates x tickers).
        volume: DataFrame of daily volume (dates x tickers).
        trailing_window: Number of days for trailing average volume
            (paper default: 10).

    Returns:
        DataFrame of volume-adjusted returns, same shape as input.
    """
    # Trailing average volume (uses the current and prior `trailing_window-1`
    # days — paper Section 6, Eq. 20).
    avg_volume = volume.rolling(window=trailing_window, min_periods=1).mean()

    # Current-day volume. A 0 print is treated as missing (rather than
    # divide-by-zero) and a NaN is genuinely missing data. Both are
    # imputed with the trailing rolling average shifted by one day — i.e.,
    # only past data is used to fill, so there's no lookahead. When the
    # imputation lands on NaN itself (very start of the series), fall
    # back to the per-day cross-sectional non-imputation route via NaN.
    raw = volume.replace(0, np.nan)
    trailing_lagged = avg_volume.shift(1)
    current_volume = raw.fillna(trailing_lagged)

    # Volume ratio: avg_V / V_t. With the trailing fill, a missing-print
    # day yields ratio ~= 1 (no adjustment), which is the right default.
    volume_ratio = avg_volume / current_volume
    volume_ratio = volume_ratio.clip(upper=10.0)

    adjusted = returns * volume_ratio
    # Final safety net: any day where neither raw nor trailing fill was
    # available (early sample) keeps the unadjusted return.
    adjusted = adjusted.fillna(returns)
    return adjusted
