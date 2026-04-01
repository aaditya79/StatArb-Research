"""
Transaction cost model.

Paper assumes 5 basis points per side (10 bps round-trip) to incorporate
estimated price slippage and other costs as a single friction coefficient.
"""


def compute_transaction_cost(notional: float, tc_bps: float = 5.0) -> float:
    """
    Compute one-way transaction cost.

    Args:
        notional: Dollar amount traded.
        tc_bps: Cost in basis points (default: 5.0 = 0.05%).

    Returns:
        Dollar cost of the transaction.
    """
    return abs(notional) * tc_bps / 10_000.0
