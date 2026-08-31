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

from src.execution_sim import twap_execute, twap_execute_with_impact


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


def speed_sweep_with_impact(entry_price_paths: Sequence, sides: Sequence[str], quantity: float,
                             windows: Sequence[int], avg_volume: float,
                             impact_coefficient: float = 0.001, impact_exponent: float = 0.5
                             ) -> List[SpeedSweepPoint]:
    """
    Same comparison as speed_sweep() (same entries, same quantity, only
    window size varies), but each slice's fill is ALSO charged
    market_impact_fraction() from execution_sim.py. This is the point of
    this function: speed_sweep() alone only ever shows shortfall
    strictly increasing with window size (slicing into a mean-reversion
    entry has a cost, faster is always better there) -- with market
    impact ALSO in the picture, fewer/bigger slices cost MORE impact per
    slice, which pulls the total the other way. Combined, the total
    shortfall vs window curve can have a genuine INTERIOR minimum
    instead of being monotonic in either direction -- see
    run_execution_speed_experiment.py for where that minimum actually
    falls on this project's synthetic entries.
    """
    if len(entry_price_paths) != len(sides):
        raise ValueError("entry_price_paths and sides must be the same length")

    max_window = max(windows)
    usable = [(path, side) for path, side in zip(entry_price_paths, sides) if len(path) >= max_window]

    results = []
    for window in windows:
        shortfalls = [
            twap_execute_with_impact(path[:window], quantity, n_slices=window, avg_volume=avg_volume,
                                     side=side, impact_coefficient=impact_coefficient,
                                     impact_exponent=impact_exponent).implementation_shortfall
            for path, side in usable
        ]
        total = sum(shortfalls)
        mean = total / len(shortfalls) if shortfalls else 0.0
        results.append(SpeedSweepPoint(window=window, total_shortfall=total, mean_shortfall=mean,
                                        n_entries=len(shortfalls)))
    return results
