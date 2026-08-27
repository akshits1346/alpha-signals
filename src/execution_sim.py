"""
Simple execution simulation: instead of assuming a pairs-trading signal
fills instantly and completely at the decision price, slice the order
across time (TWAP: equal-sized slices at equal time intervals; VWAP:
slices proportional to expected volume) and measure the REALIZED
average execution price against the price that existed when the
decision was made.

IMPLEMENTATION SHORTFALL: (average_execution_price - arrival_price) *
quantity, signed so that a positive number means the execution cost
MORE than if you'd been filled instantly at the arrival price (for a
buy) -- this is the standard measure of execution quality, separate
from whether the underlying signal itself was profitable.

INTEGRATION NOTE: this module takes a plain price series (and, for
VWAP, a volume profile) as input -- it doesn't require the order book
engine from the orderbook-engine project to run or be tested. But the
intended real use is to feed it a price/liquidity series extracted from
replaying real order book data (e.g. via that project's OrderBook and
LOBSTER replay), rather than an assumed/synthetic price path, which is
what actually connects a trading signal to a realistic cost of
execution. See run_pairs_with_execution.py for how a pairs-strategy
entry would be routed through this.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionResult:
    avg_execution_price: float
    arrival_price: float
    total_quantity: float
    implementation_shortfall: float  # positive = execution cost more than arrival price (for a buy)


def twap_execute(prices: np.ndarray, total_quantity: float, n_slices: int, side: str = "buy") -> ExecutionResult:
    """
    Splits total_quantity into n_slices EQUAL parts, executing one slice
    at each of n_slices evenly-spaced points across the given price
    series (first point, last point, and evenly spaced between).
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < n_slices:
        raise ValueError(f"need at least {n_slices} price points to slice into {n_slices} pieces, got {len(prices)}")
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    indices = np.linspace(0, len(prices) - 1, n_slices).round().astype(int)
    slice_prices = prices[indices]
    slice_qty = total_quantity / n_slices

    avg_price = float(np.mean(slice_prices))
    arrival_price = float(prices[0])
    sign = 1 if side == "buy" else -1
    shortfall = sign * (avg_price - arrival_price) * total_quantity

    return ExecutionResult(
        avg_execution_price=avg_price,
        arrival_price=arrival_price,
        total_quantity=total_quantity,
        implementation_shortfall=shortfall,
    )


def vwap_execute(prices: np.ndarray, volumes: np.ndarray, total_quantity: float, side: str = "buy") -> ExecutionResult:
    """
    Allocates total_quantity across the given price points PROPORTIONAL
    to the given volume profile (more quantity executed where more
    volume is expected to trade, which is the standard VWAP participation
    logic -- trade in proportion to the market, not evenly over time).
    """
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if len(prices) != len(volumes):
        raise ValueError("prices and volumes must be the same length")
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    total_volume = volumes.sum()
    if total_volume <= 0:
        raise ValueError("volume profile must have positive total volume")

    weights = volumes / total_volume
    slice_qty = total_quantity * weights
    notional = float(np.sum(slice_qty * prices))
    avg_price = notional / total_quantity

    arrival_price = float(prices[0])
    sign = 1 if side == "buy" else -1
    shortfall = sign * (avg_price - arrival_price) * total_quantity

    return ExecutionResult(
        avg_execution_price=avg_price,
        arrival_price=arrival_price,
        total_quantity=total_quantity,
        implementation_shortfall=shortfall,
    )
