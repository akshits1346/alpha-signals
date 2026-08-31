"""
Direct test of the causal claim behind this project's finding #3 (see
run_pairs_with_execution.py / README): TWAP-slicing a mean-reversion
entry works against the same reversion the strategy is betting on,
because the entry fires exactly when the spread is at a statistical
extreme and expected to start moving back. If that mechanism is really
what's driving the positive shortfall found there, then faster
execution (fewer slices, more concentrated near the arrival price)
should shrink shortfall toward zero, and a single-slice ("instant")
fill should give EXACTLY zero shortfall by construction (it's filled
entirely at the arrival price, by definition).

speed_sweep() re-runs TWAP execution for the SAME set of entries and
the SAME quantity at several different window/slice counts, so window
size is the only thing that varies between points in the sweep -- the
question is whether shortfall changes with execution speed, not
whether a faster sweep happened to land on an easier sample of entries.
To keep that comparison clean, only entries with enough future price
data for the LARGEST window in the sweep are used at all, for every
window size in the sweep (not just the large ones).
"""
from dataclasses import dataclass
from typing import List, Sequence

from src.execution_sim import twap_execute


@dataclass
class SpeedSweepPoint:
    window: int
    total_shortfall: float
    mean_shortfall: float
    n_entries: int


def speed_sweep(entry_price_paths: Sequence, sides: Sequence[str], quantity: float,
                 windows: Sequence[int]) -> List[SpeedSweepPoint]:
    if len(entry_price_paths) != len(sides):
        raise ValueError("entry_price_paths and sides must be the same length")

    max_window = max(windows)
    usable = [(path, side) for path, side in zip(entry_price_paths, sides) if len(path) >= max_window]

    results = []
    for window in windows:
        shortfalls = [
            twap_execute(path[:window], quantity, n_slices=window, side=side).implementation_shortfall
            for path, side in usable
        ]
        total = sum(shortfalls)
        mean = total / len(shortfalls) if shortfalls else 0.0
        results.append(SpeedSweepPoint(window=window, total_shortfall=total, mean_shortfall=mean,
                                        n_entries=len(shortfalls)))
    return results
