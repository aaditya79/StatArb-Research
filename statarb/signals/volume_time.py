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
    # Trailing average volume
    avg_volume = volume.rolling(window=trailing_window, min_periods=1).mean()

    # Current period volume (avoid division by zero)
    current_volume = volume.replace(0, np.nan)

    # Volume ratio: avg_V / V_t
    volume_ratio = avg_volume / current_volume
    volume_ratio = volume_ratio.clip(upper=10.0)  # cap extreme ratios

    # Adjusted returns
    adjusted = returns * volume_ratio

    # Fill NaN with original returns (if volume data missing)
    adjusted = adjusted.fillna(returns)

    return adjusted
