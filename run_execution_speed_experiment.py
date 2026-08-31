"""
Runs the SAME pairs-strategy entries used in run_pairs_with_execution.py
through TWAP execution at several different speeds (window/slice counts
from 1 -- near-instant -- up to 20), holding the entry set and quantity
fixed across every speed. Directly tests the question left open in that
script's finding #3: if slicing a mean-reversion entry into the
reversion is really what's driving the positive shortfall found there,
does faster (fewer-slice) execution reduce or eliminate it?

Same honesty rule as everywhere else in this project: this is synthetic
data, so it demonstrates the mechanism and pipeline, not a claim about a
real trading cost.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cointegration import test_cointegration, compute_spread
from src.pairs_strategy import generate_signals, PairsStrategyConfig, Position
from src.execution_speed_experiment import speed_sweep
from run_pairs_with_execution import generate_synthetic_pair


def main():
    A, B = generate_synthetic_pair()

    coint_result = test_cointegration(A, B)
    spread = compute_spread(A, B, coint_result.hedge_ratio, coint_result.intercept)
    cfg = PairsStrategyConfig(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    positions = generate_signals(spread, cfg)

    entries = []
    for i in range(1, len(positions)):
        if positions[i - 1] == Position.FLAT and positions[i] != Position.FLAT:
            entries.append((i, positions[i]))

    windows = [1, 2, 3, 5, 10, 20]
    max_window = max(windows)

    paths, sides = [], []
    for idx, side_position in entries:
        if idx + max_window > len(spread):
            continue  # not enough data left for the largest window -- excluded from the WHOLE sweep
        paths.append(spread[idx: idx + max_window])
        sides.append("buy" if side_position == Position.LONG_SPREAD else "sell")

    print(f"Found {len(entries)} entry signals, {len(paths)} usable across every window size in the sweep\n")

    results = speed_sweep(paths, sides, quantity=100, windows=windows)

    print(f"{'window (slices)':>16} {'total shortfall':>18} {'mean shortfall':>16} {'n entries':>10}")
    print("-" * 62)
    for point in results:
        print(f"{point.window:>16} {point.total_shortfall:>18.2f} {point.mean_shortfall:>16.2f} {point.n_entries:>10}")

    print("\n(Positive = executions cost more than instant-fill-at-signal-price would have.")
    print(" If finding #3's mechanism is what's really driving the cost, shortfall should")
    print(" shrink toward zero as window size shrinks toward 1.)")


if __name__ == "__main__":
    main()
