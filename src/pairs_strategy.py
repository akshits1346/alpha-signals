"""
Z-score mean-reversion strategy for a cointegrated pair's spread.

Given a spread that's supposed to mean-revert (from cointegration.py),
this computes a rolling z-score and generates entry/exit signals:
  - enter SHORT the spread when z > entry_z (spread abnormally high,
    bet it falls back toward its mean)
  - enter LONG the spread when z < -entry_z (spread abnormally low,
    bet it rises back toward its mean)
  - exit when z crosses back within exit_z of zero
  - hard stop-loss if z blows past stop_z, since a cointegrating
    relationship can break down -- see the walk-forward validation in
    walk_forward.py, which exists specifically to check how often that
    actually happens rather than assuming the relationship is stable
    forever just because it tested significant once.

"Long the spread" means: long A, short hedge_ratio units of B.
"Short the spread" means: short A, long hedge_ratio units of B.

Deliberately split into two pieces: rolling_zscore() (pure numeric
computation) and _positions_from_zscore() (the state machine). This
lets the state machine be tested against a hand-constructed z-score
array directly, without needing to also verify the rolling-window math
in the same test -- each piece is independently checkable.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Position(Enum):
    FLAT = 0
    LONG_SPREAD = 1
    SHORT_SPREAD = -1


@dataclass
class PairsStrategyConfig:
    lookback: int = 20       # rolling window for mean/std used in the z-score
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5


def rolling_zscore(spread: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling z-score of the spread. The first `lookback - 1` entries are
    NaN (not enough history yet to compute a rolling window) -- callers
    must handle that, not treat NaN as a signal of any kind."""
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    z = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        window = spread[i - lookback + 1: i + 1]
        mean = window.mean()
        std = window.std()
        if std > 0:
            z[i] = (spread[i] - mean) / std
    return z


def _positions_from_zscore(z: np.ndarray, config: PairsStrategyConfig) -> list:
    """The state machine itself, operating on an already-computed
    z-score array. Kept separate from rolling_zscore() specifically so
    this logic can be unit tested against a hand-constructed z array,
    independent of the rolling-window computation."""
    positions = []
    current = Position.FLAT

    for z_i in z:
        if np.isnan(z_i):
            positions.append(Position.FLAT)
            continue

        if current == Position.FLAT:
            if z_i > config.entry_z:
                current = Position.SHORT_SPREAD
            elif z_i < -config.entry_z:
                current = Position.LONG_SPREAD

        elif current == Position.SHORT_SPREAD:
            if abs(z_i) > config.stop_z:
                current = Position.FLAT  # stopped out -- relationship may be breaking down
            elif z_i < config.exit_z:
                current = Position.FLAT  # reverted back toward mean, take profit

        elif current == Position.LONG_SPREAD:
            if abs(z_i) > config.stop_z:
                current = Position.FLAT
            elif z_i > -config.exit_z:
                current = Position.FLAT

        positions.append(current)

    return positions


def generate_signals(spread: np.ndarray, config: PairsStrategyConfig) -> list:
    """Full pipeline: compute the rolling z-score, then run the state
    machine over it. Returns one Position per input time step."""
    z = rolling_zscore(spread, config.lookback)
    return _positions_from_zscore(z, config)
