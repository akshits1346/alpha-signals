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
from src.execution_speed_experiment import speed_sweep, speed_sweep_with_impact
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

    # --- now add market impact: does "always execute in 1 slice" stop
    # being optimal once your own trading footprint has a cost too?
    # NOTE on these parameters: avg_volume=50 with quantity=100 means a
    # single-slice fill is 2x "average volume" -- a deliberately
    # illiquid/oversized scenario, not a typical liquid blue-chip stock.
    # That's not an accident: at gentler, more realistic participation
    # rates (tried first -- see the README), drift cost dominated
    # impact by 1-2 orders of magnitude on this dataset's actual spread
    # scale, and window=1 stayed optimal outright. It takes a fairly
    # illiquid instrument for the impact effect to actually flip the
    # practical answer here, which is itself worth reporting honestly
    # rather than picking parameters after the fact to force a more
    # dramatic-looking result. ---
    print("\n" + "=" * 70)
    print("Same entries, WITH market impact priced in (square-root law,")
    print("avg_volume=50, impact_coefficient=2.5 -- a deliberately illiquid")
    print("scenario; see the comment in this file for why):")
    print("=" * 70 + "\n")

    impact_results = speed_sweep_with_impact(paths, sides, quantity=100, windows=windows,
                                              avg_volume=50, impact_coefficient=2.5)
    print(f"{'window (slices)':>16} {'total shortfall':>18} {'mean shortfall':>16} {'n entries':>10}")
    print("-" * 62)
    for point in impact_results:
        print(f"{point.window:>16} {point.total_shortfall:>18.2f} {point.mean_shortfall:>16.2f} {point.n_entries:>10}")

    totals = [p.total_shortfall for p in impact_results]
    min_idx = totals.index(min(totals))
    optimal_window = impact_results[min_idx].window
    if min_idx == 0:
        print(f"\nOptimal window is still window=1 (instant) -- at these parameters, impact isn't")
        print("large enough to outweigh the drift-avoidance benefit of executing instantly.")
    elif min_idx == len(impact_results) - 1:
        print(f"\nOptimal window is the slowest one tested (window={optimal_window}) -- impact")
        print("dominates so much here that the sweep didn't go slow enough to find the true optimum.")
    else:
        print(f"\nOptimal window is INTERIOR: window={optimal_window}, total shortfall "
              f"{totals[min_idx]:.2f} -- beats both window=1 ({totals[0]:.2f}) and "
              f"window={windows[-1]} ({totals[-1]:.2f}). Neither 'always instant' nor 'always slow' "
              f"is optimal once both effects are priced in.")


if __name__ == "__main__":
    main()
