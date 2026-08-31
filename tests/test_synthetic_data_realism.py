"""
Statistical validation that generate_realistic_synthetic_pair() (in
src/synthetic_data.py) actually reproduces two well-known "stylized
facts" of real price data that generate_synthetic_pair()'s plain
Gaussian random walk does NOT have by construction (i.i.d. Gaussian
increments have zero excess kurtosis and zero autocorrelation, by the
definition of i.i.d. Gaussian):

  - FAT TAILS: real return distributions have positive excess kurtosis
    (more extreme moves than a Gaussian of the same variance predicts).
  - VOLATILITY CLUSTERING: real |returns| are positively autocorrelated
    (big moves tend to follow big moves, calm periods follow calm
    periods) -- volatility clusters, it isn't i.i.d. noise.

VALIDATION STRATEGY, in two different ways for the two properties (this
mirrors orderbook-engine's tests/test_synthetic_data_realism.py exactly
-- same two mechanisms, same reasoning for why each check is built the
way it is):

  - Fat tails: run BOTH generators on the SAME seed and length, and
    check the realistic one has measurably higher excess kurtosis. This
    comparison is legitimate here (unlike a degenerate fixed-step walk
    would be) because the plain generator's increments really are i.i.d.
    Gaussian, whose theoretical excess kurtosis is exactly 0 -- a clean,
    non-degenerate baseline.

  - Volatility clustering: comparing autocorrelation directly against
    the plain generator would still work here (i.i.d. Gaussian has zero
    autocorrelation in expectation, not a spurious near-1.0 like
    orderbook-engine's fixed-step case), but the more rigorous test is
    the same one used there regardless: compare the realistic series'
    OWN time-ordered autocorrelation against a random permutation of
    itself. Shuffling preserves the exact marginal distribution while
    destroying temporal order -- if the original is measurably higher
    than its own shuffled control, that difference can only come from
    genuine temporal clustering, not the distribution's shape.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.synthetic_data import generate_realistic_synthetic_pair
from run_pairs_with_execution import generate_synthetic_pair

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def excess_kurtosis(x):
    x = x - x.mean()
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    return m4 / (m2 ** 2) - 3.0  # 0 for a true Gaussian; > 0 means fatter tails


def autocorr_abs_returns(x, lag=1):
    a = np.abs(x)
    a = a - a.mean()
    denom = np.sum(a ** 2)
    if denom == 0:
        return 0.0
    return np.sum(a[:-lag] * a[lag:]) / denom


def main():
    n = 3000
    seed = 7

    A_realistic, B_realistic = generate_realistic_synthetic_pair(n=n, seed=seed)
    A_plain, B_plain = generate_synthetic_pair(n=n, seed=seed)

    realistic_returns = np.diff(B_realistic)
    plain_returns = np.diff(B_plain)

    # --- basic sanity: still a genuinely cointegrated pair by
    # construction (A - hedge_ratio*B - intercept should be exactly the
    # idiosyncratic noise, nothing else) ---
    residual = A_realistic - 2.0 * B_realistic - 5.0
    check(np.std(residual) < 10 * np.std(np.diff(B_realistic)),
          "A remains linearly tied to B (idiosyncratic residual isn't dominating the relationship)")

    # --- fat tails ---
    realistic_kurt = excess_kurtosis(realistic_returns)
    plain_kurt = excess_kurtosis(plain_returns)
    check(realistic_kurt > plain_kurt + 1.0,
          f"realistic generator has measurably fatter tails: excess kurtosis {realistic_kurt:.2f} "
          f"vs plain generator's {plain_kurt:.2f} (theoretical value for true i.i.d. Gaussian: 0)")

    # --- volatility clustering: original time-order vs a shuffled
    # control with the IDENTICAL marginal distribution ---
    realistic_autocorr = autocorr_abs_returns(realistic_returns)

    shuffle_rng = np.random.RandomState(123)
    shuffled_autocorrs = []
    for _ in range(30):
        shuffled = realistic_returns.copy()
        shuffle_rng.shuffle(shuffled)
        shuffled_autocorrs.append(autocorr_abs_returns(shuffled))
    mean_shuffled_autocorr = float(np.mean(shuffled_autocorrs))
    std_shuffled_autocorr = float(np.std(shuffled_autocorrs))

    check(abs(mean_shuffled_autocorr) < 0.05,
          f"shuffled control has ~zero autocorrelation, as expected (mean over 30 shuffles: "
          f"{mean_shuffled_autocorr:.4f}) -- confirms the marginal distribution alone doesn't "
          f"produce clustering, only temporal order can")
    check(realistic_autocorr > mean_shuffled_autocorr + 5 * std_shuffled_autocorr,
          f"realistic generator's TIME-ORDERED |return| autocorrelation ({realistic_autocorr:.3f}) is "
          f"far above its own shuffled-control distribution (mean {mean_shuffled_autocorr:.4f}, "
          f"std {std_shuffled_autocorr:.4f}) -- clustering comes from temporal order, not just the "
          f"distribution's shape")
    check(realistic_autocorr > 0.1,
          f"realistic generator's volatility clustering is positive in absolute terms too, got {realistic_autocorr:.3f}")

    # --- determinism: same seed reproduces the exact same series ---
    A_again, B_again = generate_realistic_synthetic_pair(n=n, seed=seed)
    check(np.array_equal(B_realistic, B_again) and np.array_equal(A_realistic, A_again),
          "same seed reproduces an identical pair (deterministic, reproducible)")

    print()
    if failures == 0:
        print("All synthetic data realism checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
