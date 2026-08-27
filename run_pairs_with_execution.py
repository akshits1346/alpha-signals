"""
Ties the pairs strategy to the execution layer: instead of assuming
every entry/exit fills instantly at the exact spread price the signal
fired at, route each entry through TWAP execution over a short window
of subsequent prices, and report the resulting implementation
shortfall alongside the strategy's raw signal count.

This uses SYNTHETIC price data (see generate_synthetic_pair below) --
same honesty rule as everywhere else in this project: this demonstrates
the signal-to-execution pipeline works end-to-end, it is not a claim
about real trading costs. Swapping in real price/liquidity data (ideally
from replaying real order book data, as in the orderbook-engine project)
is the actual next step before treating any number here as a finding.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.cointegration import test_cointegration, compute_spread
from src.pairs_strategy import generate_signals, PairsStrategyConfig, Position
from src.execution_sim import twap_execute


def generate_synthetic_pair(n=400, seed=42):
    np.random.seed(seed)
    B = np.cumsum(np.random.normal(0, 1, n))
    A = 2.0 * B + 5.0 + np.random.normal(0, 1, n)
    return A, B


def main():
    A, B = generate_synthetic_pair()

    coint_result = test_cointegration(A, B)
    print(f"Cointegration test: is_cointegrated={coint_result.is_cointegrated}, "
          f"p={coint_result.p_value:.5f}, hedge_ratio={coint_result.hedge_ratio:.3f}\n")

    spread = compute_spread(A, B, coint_result.hedge_ratio, coint_result.intercept)
    cfg = PairsStrategyConfig(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    positions = generate_signals(spread, cfg)

    # Find each point where the position CHANGES from FLAT into a trade --
    # that's an entry, and it's the moment we'd actually need to execute.
    entries = []
    for i in range(1, len(positions)):
        if positions[i - 1] == Position.FLAT and positions[i] != Position.FLAT:
            entries.append((i, positions[i]))

    print(f"Found {len(entries)} entry signals\n")

    total_shortfall = 0.0
    execution_window = 5  # TWAP slice the entry over the next 5 spread observations

    for idx, side_position in entries:
        if idx + execution_window > len(spread):
            continue  # not enough data left to slice over -- skip near the end of the series

        window_prices = spread[idx: idx + execution_window]
        side = "buy" if side_position == Position.LONG_SPREAD else "sell"

        result = twap_execute(window_prices, total_quantity=100, n_slices=execution_window, side=side)
        total_shortfall += result.implementation_shortfall

        print(f"  entry at t={idx} ({side_position.name}): "
              f"arrival={result.arrival_price:.3f}, avg_fill={result.avg_execution_price:.3f}, "
              f"shortfall={result.implementation_shortfall:.2f}")

    print(f"\nTotal implementation shortfall across all entries: {total_shortfall:.2f}")
    print("(Positive = executions cost more than instant-fill-at-signal-price would have.)")


if __name__ == "__main__":
    main()
