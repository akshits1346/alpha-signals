"""
Tests for parameter_optimization.py.

Three checks, each isolating a different piece:

  1. pnl_from_positions(): a HAND-TRACED exact P&L calculation, kept
     separate from position generation (same reason pairs_strategy.py
     splits rolling_zscore() from _positions_from_zscore()).

  2. walk_forward_optimize()'s grid search itself: verify the chosen
     config for one window is ACTUALLY the argmax over the grid, by
     independently recomputing every combo's in-sample P&L via
     strategy_pnl() directly and comparing. This is the part that could
     have a real bug (e.g. picking the argmin, or the first combo
     regardless of P&L) independent of whether strategy_pnl() itself is
     correct.

  3. A qualitative, ground-truth case: a clean, strongly mean-reverting
     synthetic spread (sinusoid + small noise) where the baseline
     defaults (lookback=20, entry_z=2.0) are deliberately a poor match
     to the oscillation period, so walk-forward-optimized parameters
     SHOULD beat the untuned baseline out-of-sample -- and do, giving a
     legitimate (not cherry-picked) case to check the whole pipeline
     produces a sensible qualitative result, before trusting it on
     noisier, more realistic data (see run_parameter_optimization.py
     for the honest answer on that -- NOT assumed to be the same
     result, since ground-truth-clean and noisy-realistic are
     genuinely different regimes).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itertools import product

import numpy as np

from src.pairs_strategy import Position, PairsStrategyConfig
from src.parameter_optimization import pnl_from_positions, strategy_pnl, walk_forward_optimize

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- 1. hand-traced P&L ---
    # spread = [10, 12, 11, 9, 9] -> returns = [2, -1, -2, 0]
    # positions = [FLAT, LONG, LONG, SHORT, FLAT] -> values[:-1] = [0, 1, 1, -1]
    # pnl = 0*2 + 1*(-1) + 1*(-2) + (-1)*0 = -3.0
    spread = np.array([10.0, 12.0, 11.0, 9.0, 9.0])
    positions = [Position.FLAT, Position.LONG_SPREAD, Position.LONG_SPREAD, Position.SHORT_SPREAD, Position.FLAT]
    pnl = pnl_from_positions(spread, positions)
    check(abs(pnl - (-3.0)) < 1e-9, f"hand-traced P&L == -3.0, got {pnl}")

    try:
        pnl_from_positions(spread, positions[:-1])
        check(False, "should reject mismatched positions/spread lengths")
    except ValueError:
        check(True, "correctly rejects mismatched positions/spread lengths")

    # --- 2. grid search correctness: window 0's chosen config really is
    # the argmax over the grid, verified independently ---
    np.random.seed(5)
    t = np.arange(300)
    clean_spread = 5 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 0.5, 300)

    grid = {"lookback": [10, 20, 30], "entry_z": [1.0, 1.5, 2.0], "exit_z": [0.3, 0.5]}
    baseline = PairsStrategyConfig(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)

    results = walk_forward_optimize(clean_spread, window_size=60, step_size=30,
                                    param_grid=grid, baseline_config=baseline)
    check(len(results) > 0, f"at least one window was optimized, got {len(results)}")

    first_window_in_sample = clean_spread[0:60]
    combos = [PairsStrategyConfig(lookback=lb, entry_z=ez, exit_z=xz, stop_z=baseline.stop_z)
              for lb, ez, xz in product(grid["lookback"], grid["entry_z"], grid["exit_z"])]
    independently_computed = [(cfg, strategy_pnl(first_window_in_sample, cfg)) for cfg in combos]
    independent_best_cfg, independent_best_pnl = max(independently_computed, key=lambda pair: pair[1])

    chosen = results[0].chosen_config
    check(chosen.lookback == independent_best_cfg.lookback and chosen.entry_z == independent_best_cfg.entry_z
          and chosen.exit_z == independent_best_cfg.exit_z,
          f"window 0's chosen config ({chosen}) matches the independently-verified grid argmax "
          f"({independent_best_cfg})")
    check(abs(results[0].in_sample_pnl - independent_best_pnl) < 1e-9,
          f"window 0's reported in-sample P&L matches the independently-verified best, "
          f"{results[0].in_sample_pnl} vs {independent_best_pnl}")

    # --- 3. qualitative ground-truth case: on this deliberately clean,
    # strongly-periodic spread (baseline's lookback=20/entry_z=2.0 don't
    # match the period-40 oscillation well), walk-forward-optimized
    # out-of-sample P&L should beat the untouched baseline ---
    total_optimized = sum(r.out_of_sample_pnl_optimized for r in results)
    total_baseline = sum(r.out_of_sample_pnl_baseline for r in results)
    check(total_optimized > total_baseline,
          f"on a clean, strongly mean-reverting spread, walk-forward-optimized out-of-sample P&L "
          f"({total_optimized:.2f}) beats the untuned baseline ({total_baseline:.2f})")

    # --- edge case: data too short for even one window -> empty list, not an error ---
    short_results = walk_forward_optimize(clean_spread[:50], window_size=60, step_size=30,
                                          param_grid=grid, baseline_config=baseline)
    check(short_results == [], f"too-short data returns an empty list, not an error, got {short_results}")

    print()
    if failures == 0:
        print("All parameter optimization checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
