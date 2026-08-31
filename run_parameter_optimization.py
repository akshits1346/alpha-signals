"""
Runs walk_forward_optimize() on a realistic synthetic pair's spread
(generate_realistic_synthetic_pair() -- see synthetic_data.py) to
honestly answer the question left open in this project's earlier
"what I'd build next": does optimizing the z-score strategy's
parameters via the walk-forward framework actually beat the untuned
conventional defaults (lookback=20, entry_z=2.0, exit_z=0.5), or is
that just as arbitrary a choice as the defaults were?

Same honesty rule as everywhere else in this project: synthetic data,
demonstrates the mechanism and pipeline, not a claim about a real
trading result.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cointegration import test_cointegration, compute_spread
from src.parameter_optimization import walk_forward_optimize
from src.pairs_strategy import PairsStrategyConfig
from src.synthetic_data import generate_realistic_synthetic_pair


def main(seed=42, n=800, window_size=100, step_size=50):
    A, B = generate_realistic_synthetic_pair(n=n, seed=seed)
    coint = test_cointegration(A, B)
    print(f"Cointegration test: is_cointegrated={coint.is_cointegrated}, p={coint.p_value:.5f}, "
          f"hedge_ratio={coint.hedge_ratio:.3f}\n")

    spread = compute_spread(A, B, coint.hedge_ratio, coint.intercept)

    grid = {"lookback": [10, 20, 30], "entry_z": [1.0, 1.5, 2.0, 2.5], "exit_z": [0.3, 0.5, 0.8]}
    baseline = PairsStrategyConfig(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)

    results = walk_forward_optimize(spread, window_size=window_size, step_size=step_size,
                                    param_grid=grid, baseline_config=baseline)
    print(f"Grid: {grid}")
    print(f"Baseline (untuned): {baseline}\n")

    print(f"{'window start':>12} {'chosen lookback':>16} {'chosen entry_z':>15} {'chosen exit_z':>14} "
          f"{'out-of-sample optimized':>24} {'out-of-sample baseline':>22}")
    print("-" * 108)
    wins = 0
    for r in results:
        wins += 1 if r.out_of_sample_pnl_optimized > r.out_of_sample_pnl_baseline else 0
        c = r.chosen_config
        print(f"{r.start:>12} {c.lookback:>16} {c.entry_z:>15} {c.exit_z:>14} "
              f"{r.out_of_sample_pnl_optimized:>24.2f} {r.out_of_sample_pnl_baseline:>22.2f}")

    total_optimized = sum(r.out_of_sample_pnl_optimized for r in results)
    total_baseline = sum(r.out_of_sample_pnl_baseline for r in results)
    n_windows = len(results)

    print(f"\nTotals: optimized={total_optimized:.2f}, baseline={total_baseline:.2f}")
    print(f"Optimized beats untuned baseline in {wins}/{n_windows} windows")


if __name__ == "__main__":
    main()
