"""
Hand-traced tests for pairs_strategy.py, split to match the module's
own split: rolling_zscore() checked against a hand-computed value,
_positions_from_zscore() checked against a fully hand-traced sequence
of state transitions.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.pairs_strategy import rolling_zscore, _positions_from_zscore, generate_signals, Position, PairsStrategyConfig

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- rolling_zscore: hand-computed single value ---
    # spread=[0,0,0,10], lookback=3: window at i=3 is [0,0,10],
    # mean=3.3333, population std=4.7140, z = (10-3.3333)/4.7140 = 1.4142
    spread = np.array([0.0, 0.0, 0.0, 10.0])
    z = rolling_zscore(spread, lookback=3)
    check(np.isnan(z[0]) and np.isnan(z[1]), "first lookback-1 entries are NaN (not enough history)")
    check(abs(z[3] - 1.4142) < 0.001, f"z[3] matches hand calculation, got {z[3]:.4f}")

    # --- _positions_from_zscore: fully hand-traced state sequence ---
    # config: entry_z=2.0, exit_z=0.5, stop_z=3.5
    #
    # idx  z      state before -> action                  -> state after
    # 0    nan    FLAT          (no data)                     FLAT
    # 1    0.0    FLAT          not > 2 or < -2               FLAT
    # 2    2.5    FLAT          2.5 > entry_z(2.0)             SHORT_SPREAD
    # 3    1.0    SHORT         |1.0|<=3.5, 1.0 not < 0.5      SHORT
    # 4    0.3    SHORT         |0.3|<=3.5, 0.3 < exit_z(0.5)  FLAT (profit exit)
    # 5   -0.2    FLAT          not > 2 or < -2                FLAT
    # 6   -2.5    FLAT          -2.5 < -entry_z(-2.0)          LONG_SPREAD
    # 7   -4.0    LONG          |-4.0| > stop_z(3.5)           FLAT (stopped out)
    # 8    0.0    FLAT          not > 2 or < -2                FLAT
    z_sequence = np.array([np.nan, 0.0, 2.5, 1.0, 0.3, -0.2, -2.5, -4.0, 0.0])
    cfg = PairsStrategyConfig(lookback=1, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    positions = _positions_from_zscore(z_sequence, cfg)

    expected = [
        Position.FLAT, Position.FLAT, Position.SHORT_SPREAD, Position.SHORT_SPREAD,
        Position.FLAT, Position.FLAT, Position.LONG_SPREAD, Position.FLAT, Position.FLAT,
    ]
    check(positions == expected, f"state sequence matches hand trace exactly, got {[p.name for p in positions]}")

    # --- integration: generate_signals runs end-to-end without error on a longer series ---
    np.random.seed(1)
    long_spread = np.random.normal(0, 1, 200)
    full_cfg = PairsStrategyConfig(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    full_positions = generate_signals(long_spread, full_cfg)
    check(len(full_positions) == 200, "generate_signals returns one position per input point")
    check(any(p != Position.FLAT for p in full_positions),
          "strategy actually takes at least one position over 200 random points (sanity check, not a performance claim)")

    print()
    if failures == 0:
        print("All pairs strategy checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
