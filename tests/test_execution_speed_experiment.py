"""
Tests for execution_speed_experiment.py.

VALIDATION STRATEGY: an EXACT closed form, not just an observed trend.
Build price paths that decline/rise LINEARLY and deterministically
(price[i] = P0 -+ step*i) -- a clean stand-in for "the spread reverting
right after the entry fires," the exact mechanism finding #3 claims is
driving positive shortfall. twap_execute's TWAP indices for a path of
length `window` sliced into `window` slices land on EXACTLY indices
0..window-1 (linspace(0, window-1, window) is already integer-valued),
so the average execution price over the first `window` points of a
linear path has a closed form: avg = P0 -+ step*(window-1)/2. That
gives an exact expected shortfall per window size, not just "smaller
windows should give smaller shortfall" -- this test checks the actual
predicted number.

If this test passes, it's a mathematical guarantee (for this
idealized linear-reversion path) that faster execution (smaller
window) STRICTLY reduces shortfall, all the way to exactly zero at
window=1 -- which is exactly the claim finding #3's "what I'd build
next" wanted tested.

A SECOND section below tests speed_sweep_with_impact(): once market
impact is priced in, "always execute in 1 slice" stops being free, and
the total-shortfall-vs-window curve should have a genuine INTERIOR
minimum rather than being monotonic -- verified empirically (found the
minimum by directly running the sweep, not asserted from theory) rather
than with a closed form, since impact + drift together don't have as
clean an exact formula as drift alone.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.execution_speed_experiment import speed_sweep, speed_sweep_with_impact

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    step = 0.5
    qty = 100.0
    max_window = 11

    # A "sell into a declining price" entry (mirrors: spread was high,
    # strategy shorts it, spread reverts DOWN right after -- selling
    # into a falling price is the unfavorable case) ...
    decline_path = np.array([100.0 - step * i for i in range(max_window)])
    # ... and a "buy into a rising price" entry (spread was low, strategy
    # goes long, spread reverts UP right after).
    rise_path = np.array([100.0 + step * i for i in range(max_window)])
    # A too-short entry that must be excluded from the WHOLE sweep, not
    # just from window sizes larger than it.
    short_path = np.array([100.0, 100.5, 101.0])

    paths = [decline_path, rise_path, short_path]
    sides = ["sell", "buy", "sell"]
    windows = [1, 3, 5, 11]

    results = speed_sweep(paths, sides, qty, windows)

    for point in results:
        check(point.n_entries == 2,
              f"window={point.window}: short entry excluded from the whole sweep, got n_entries={point.n_entries}")
        expected_per_entry = step * (point.window - 1) / 2 * qty
        expected_total = 2 * expected_per_entry
        check(abs(point.total_shortfall - expected_total) < 1e-9,
              f"window={point.window}: total_shortfall == {expected_total}, got {point.total_shortfall}")
        check(abs(point.mean_shortfall - expected_per_entry) < 1e-9,
              f"window={point.window}: mean_shortfall == {expected_per_entry}, got {point.mean_shortfall}")

    # window=1 is the "instant fill" degenerate case: filled entirely at
    # the arrival price, by definition -- shortfall must be EXACTLY zero.
    window1 = next(p for p in results if p.window == 1)
    check(window1.total_shortfall == 0.0,
          f"window=1 (instant fill) gives EXACTLY zero shortfall, got {window1.total_shortfall}")

    # Shortfall must be STRICTLY monotonically increasing as window grows
    # (i.e. strictly decreasing toward zero as execution gets faster) --
    # this is the direct test of finding #3's causal claim.
    totals = [p.total_shortfall for p in results]
    check(all(totals[i] < totals[i + 1] for i in range(len(totals) - 1)),
          f"total shortfall strictly increases with window size (slower execution costs more): {totals}")

    # --- with market impact ALSO in the picture, the monotonic
    # "faster is always better" result above stops holding: fewer/bigger
    # slices cost more impact, which pulls the total the other way,
    # creating a genuine INTERIOR minimum instead of a boundary one.
    # Small, slow drift (so impact dominates for small windows) + a
    # square-root impact law that's meaningful at these clip sizes. ---
    small_step = 0.01
    impact_qty = 10000.0
    impact_max_window = 20
    slow_decline_path = np.array([100.0 - small_step * i for i in range(impact_max_window + 1)])
    impact_windows = list(range(1, impact_max_window + 1))

    impact_results = speed_sweep_with_impact([slow_decline_path], ["sell"], impact_qty, impact_windows,
                                              avg_volume=50000, impact_coefficient=0.01, impact_exponent=0.5)
    totals_with_impact = [p.total_shortfall for p in impact_results]
    min_idx = int(np.argmin(totals_with_impact))
    min_window = impact_results[min_idx].window

    check(0 < min_idx < len(impact_results) - 1,
          f"total shortfall (drift + impact) is minimized at an INTERIOR window ({min_window}), "
          f"not at either boundary (window=1 or window={impact_max_window}) -- confirming market "
          f"impact really does pull the optimum away from 'always execute instantly'")
    check(totals_with_impact[min_idx] < totals_with_impact[0],
          f"the interior minimum ({totals_with_impact[min_idx]:.2f}) beats window=1's "
          f"({totals_with_impact[0]:.2f}) -- pure drift-minimization (window=1) is no longer optimal")
    check(totals_with_impact[min_idx] < totals_with_impact[-1],
          f"the interior minimum ({totals_with_impact[min_idx]:.2f}) also beats the slowest window's "
          f"({totals_with_impact[-1]:.2f}) -- pure impact-minimization (slowest) isn't optimal either")

    print()
    if failures == 0:
        print("All execution speed experiment checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
