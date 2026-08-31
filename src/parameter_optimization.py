"""
Walk-forward optimization of the z-score strategy's parameters
(lookback, entry_z, exit_z), rather than the conventional defaults used
everywhere else in this project (lookback=20, entry_z=2.0, exit_z=0.5,
explicitly called out as un-tuned in the README).

Uses walk_forward.py's exact rolling in-sample/out-of-sample window
structure, but instead of re-testing cointegration, grid-searches over
candidate (lookback, entry_z, exit_z) combinations on the IN-SAMPLE
window, picks the one with the highest in-sample total P&L, and applies
THAT chosen configuration -- UNTOUCHED, not re-fit -- to the immediately
following OUT-OF-SAMPLE window. This is the only honest way to ask "does
optimizing help": picking parameters that fit the out-of-sample window
best would be look-ahead bias, not a real trading result. The
out-of-sample P&L of the untouched baseline config is computed on the
SAME out-of-sample window for a direct, paired comparison.

P&L convention: LONG_SPREAD earns money when the spread RISES
(spread[t+1] - spread[t]), SHORT_SPREAD earns when it falls, matching
pairs_strategy.py's own definition (long the spread when it's
abnormally low, betting on reversion upward) -- and conveniently,
Position's own enum values (LONG_SPREAD=1, SHORT_SPREAD=-1, FLAT=0) are
already exactly the right position-size multiplier, so no separate
mapping is needed. One unit of spread notional per period, no
transaction costs -- same simplifying assumptions as the rest of this
project's strategy layer.
"""
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

import numpy as np

from src.pairs_strategy import generate_signals, PairsStrategyConfig, Position


def pnl_from_positions(spread: np.ndarray, positions: Sequence[Position]) -> float:
    """
    Pure P&L computation given ALREADY-COMPUTED positions, independent
    of how they were generated -- kept separate from strategy_pnl() for
    the same reason pairs_strategy.py splits rolling_zscore() from
    _positions_from_zscore(): each piece can be tested against a
    hand-constructed input without also depending on the other piece
    being correct.

    position[t] is the position HELD DURING the period from t to t+1,
    so it earns spread[t+1] - spread[t] -- the last position in the
    list has no following return to earn, and is excluded.
    """
    spread = np.asarray(spread, dtype=float)
    if len(positions) != len(spread):
        raise ValueError("positions and spread must be the same length")

    position_values = np.array([p.value for p in positions], dtype=float)
    returns = np.diff(spread)
    return float(np.sum(position_values[:-1] * returns))


def strategy_pnl(spread: np.ndarray, config: PairsStrategyConfig) -> float:
    """Convenience wrapper: generate_signals(spread, config), then
    pnl_from_positions on the result."""
    positions = generate_signals(spread, config)
    return pnl_from_positions(spread, positions)


@dataclass
class WindowOptimizationResult:
    start: int
    chosen_config: PairsStrategyConfig
    in_sample_pnl: float
    out_of_sample_pnl_optimized: float
    out_of_sample_pnl_baseline: float


def walk_forward_optimize(spread: np.ndarray, window_size: int, step_size: int,
                           param_grid: Dict[str, Sequence[float]],
                           baseline_config: PairsStrategyConfig) -> List[WindowOptimizationResult]:
    """
    param_grid: {"lookback": [...], "entry_z": [...], "exit_z": [...]}
    -- every combination in the Cartesian product is tried in-sample.
    stop_z is held fixed at baseline_config.stop_z for every combo (not
    part of the grid -- it's a risk control, not a signal parameter).
    """
    spread = np.asarray(spread, dtype=float)
    n = len(spread)

    combos = [
        PairsStrategyConfig(lookback=lb, entry_z=ez, exit_z=xz, stop_z=baseline_config.stop_z)
        for lb, ez, xz in product(param_grid["lookback"], param_grid["entry_z"], param_grid["exit_z"])
    ]

    results = []
    start = 0
    while start + window_size + window_size <= n:  # need room for both in-sample AND out-of-sample windows
        in_sample = spread[start: start + window_size]
        out_sample = spread[start + window_size: start + 2 * window_size]

        in_sample_pnls = [(cfg, strategy_pnl(in_sample, cfg)) for cfg in combos]
        best_cfg, best_in_sample_pnl = max(in_sample_pnls, key=lambda pair: pair[1])

        out_optimized_pnl = strategy_pnl(out_sample, best_cfg)
        out_baseline_pnl = strategy_pnl(out_sample, baseline_config)

        results.append(WindowOptimizationResult(
            start=start, chosen_config=best_cfg, in_sample_pnl=best_in_sample_pnl,
            out_of_sample_pnl_optimized=out_optimized_pnl, out_of_sample_pnl_baseline=out_baseline_pnl,
        ))
        start += step_size

    return results
