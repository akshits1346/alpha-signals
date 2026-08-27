"""
Tests walk_forward_validate() against two constructed scenarios where
the ground truth is known:

  - A pair that's cointegrated with a STABLE relationship for its
    entire length -- survival rate should be high (most windows that
    find cointegration in-sample should still find it out-of-sample,
    since the true relationship never changes).

  - A pair that's cointegrated for the first half, then the underlying
    relationship BREAKS (second half becomes an independent random
    walk unrelated to B) -- survival rate should be meaningfully lower,
    since windows straddling the breakpoint will find in-sample
    cointegration (still partly built on the stable first half) but
    fail out-of-sample once the window has moved past the break.

This mirrors the honest-reversal finding pattern elsewhere in this
project: the test doesn't assert a specific number for the breaking
pair (that would overfit to one random seed's exact behavior), it
asserts the qualitatively correct relationship -- stable pair survives
more often than breaking pair.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.walk_forward import walk_forward_validate

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    n = 400
    window_size = 60
    step_size = 30

    # --- stable pair: cointegrated throughout ---
    np.random.seed(10)
    B_stable = np.cumsum(np.random.normal(0, 1, n))
    A_stable = 2.0 * B_stable + 5.0 + np.random.normal(0, 1, n)
    result_stable = walk_forward_validate(A_stable, B_stable, window_size, step_size)

    check(result_stable.n_windows > 0, "stable pair: at least one window tested")
    check(result_stable.n_in_sample_cointegrated > 0,
          "stable pair: cointegration found in-sample in at least one window")
    check(result_stable.survival_rate > 0.7,
          f"stable pair: high survival rate, got {result_stable.survival_rate:.3f}")

    # --- breaking pair: cointegrated first half, independent random walk second half ---
    np.random.seed(20)
    half = n // 2
    B_break = np.cumsum(np.random.normal(0, 1, n))
    first_half_a = 2.0 * B_break[:half] + 5.0 + np.random.normal(0, 1, half)
    # second half continues from wherever the first half ended, but as an
    # INDEPENDENT random walk from that point on -- no relationship to B_break[half:]
    continuation_start = first_half_a[-1]
    second_half_a = continuation_start + np.cumsum(np.random.normal(0, 1, n - half))
    A_break = np.concatenate([first_half_a, second_half_a])

    result_break = walk_forward_validate(A_break, B_break, window_size, step_size)

    check(result_break.n_windows > 0, "breaking pair: at least one window tested")
    check(result_break.survival_rate < result_stable.survival_rate,
          f"breaking pair survival rate ({result_break.survival_rate:.3f}) is lower than "
          f"stable pair's ({result_stable.survival_rate:.3f})")

    # --- edge case: no in-sample cointegration found at all -> survival_rate is 0.0, not NaN/error ---
    np.random.seed(1)
    np.random.seed(2)
    X = np.cumsum(np.random.normal(0, 1, n))
    np.random.seed(3)
    Y = np.cumsum(np.random.normal(0, 1, n))
    result_none = walk_forward_validate(X, Y, window_size, step_size)
    check(result_none.survival_rate == 0.0 or result_none.n_in_sample_cointegrated > 0,
          "survival_rate is well-defined (0.0, not NaN) even if no window ever found cointegration")

    print()
    if failures == 0:
        print("All walk-forward validation checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
