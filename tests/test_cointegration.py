"""
Tests the cointegration test itself against synthetic data where the
ground truth is known by construction:

  - A cointegrated pair: B is a random walk, A = true_hedge_ratio*B +
    true_intercept + stationary noise. By construction, A - hedge*B -
    intercept is stationary (it IS the noise term), so this pair MUST
    be detected as cointegrated, and the fitted hedge_ratio/intercept
    should closely recover the true values used to build it.

  - Two independent random walks: no relationship between them, so
    they should NOT be flagged as cointegrated (a spurious low p-value
    here would mean the test itself is broken, not that pairs trading
    "works" -- a well-known trap with naive correlation-based pair
    selection is exactly this: two unrelated trending series can look
    related by pure chance, which is why cointegration testing, not
    correlation, is the standard tool here).

Fixed seeds are used so this test is deterministic and doesn't
occasionally fail due to random chance.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.cointegration import test_cointegration, compute_spread

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    n = 500

    # --- constructed cointegrated pair ---
    np.random.seed(42)
    B = np.cumsum(np.random.normal(0, 1, n))
    true_hedge_ratio = 2.0
    true_intercept = 5.0
    A = true_hedge_ratio * B + true_intercept + np.random.normal(0, 1, n)

    result = test_cointegration(A, B)
    check(result.is_cointegrated, f"constructed cointegrated pair IS detected as cointegrated (p={result.p_value:.5f})")
    check(result.p_value < 0.01, f"p-value is strongly significant, got {result.p_value:.5f}")
    check(abs(result.hedge_ratio - true_hedge_ratio) < 0.1,
          f"fitted hedge_ratio ({result.hedge_ratio:.3f}) close to true value ({true_hedge_ratio})")
    check(abs(result.intercept - true_intercept) < 0.5,
          f"fitted intercept ({result.intercept:.3f}) close to true value ({true_intercept})")

    spread = compute_spread(A, B, result.hedge_ratio, result.intercept)
    check(abs(np.mean(spread)) < 1.0, f"resulting spread has near-zero mean, got {np.mean(spread):.4f}")
    check(np.std(spread) < 2.0, f"resulting spread has bounded (not exploding) variance, std={np.std(spread):.4f}")

    # --- independent random walks: should NOT be cointegrated ---
    np.random.seed(7)
    X = np.cumsum(np.random.normal(0, 1, n))
    np.random.seed(99)
    Y = np.cumsum(np.random.normal(0, 1, n))

    result2 = test_cointegration(X, Y)
    check(not result2.is_cointegrated,
          f"independent random walks correctly NOT flagged as cointegrated (p={result2.p_value:.5f})")

    # --- input validation ---
    try:
        test_cointegration(np.array([1, 2, 3]), np.array([1, 2, 3]))
        check(False, "should reject series shorter than 20 observations")
    except ValueError:
        check(True, "correctly rejects series shorter than 20 observations")

    try:
        test_cointegration(np.arange(50), np.arange(30))
        check(False, "should reject mismatched series lengths")
    except ValueError:
        check(True, "correctly rejects mismatched series lengths")

    print()
    if failures == 0:
        print("All cointegration checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
