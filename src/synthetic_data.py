"""
A more statistically realistic synthetic pair generator than
run_pairs_with_execution.py's generate_synthetic_pair() (a plain
Gaussian random walk for B, i.i.d. Gaussian noise for A's idiosyncratic
residual). Real price series have two well-documented properties that
plain Gaussian innovations don't produce: FAT TAILS (positive excess
kurtosis in returns -- more extreme moves than a Gaussian of the same
variance predicts) and VOLATILITY CLUSTERING (positively autocorrelated
|returns| -- calm periods and turbulent periods cluster in time, rather
than every step independently being the same typical size).

Uses the same two mechanisms as the orderbook-engine project's
generate_realistic_lobster_data() (see that project for the full
derivation and the statistical tests that validate each mechanism in
isolation): Student-t innovations for fat tails, and a textbook
GARCH(1,1) variance recursion (variance_t = omega + alpha*z_{t-1}^2 +
beta*variance_{t-1}, z = the standardized shock) for volatility
clustering. Applied independently to BOTH B's random-walk increments
and A's idiosyncratic residual, so the resulting pair still has a
genuinely stationary, cointegrated spread (A - hedge_ratio*B is still
just intercept + idiosyncratic noise, exactly as in the plain
generator) but with realistic marginal statistics on both legs.

See tests/test_synthetic_data_realism.py for the statistical
validation (excess kurtosis vs the plain generator directly; |return|
autocorrelation via a shuffled-control comparison against itself,
which is the correct way to isolate genuine temporal clustering from
just the marginal distribution's shape -- see that test's docstring).
"""
import numpy as np


def _garch_fat_tailed_increments(rng: np.random.RandomState, n: int, df: int = 4,
                                  garch_alpha: float = 0.15, garch_beta: float = 0.80,
                                  long_run_variance: float = 1.0) -> np.ndarray:
    """
    n increments, Student-t distributed (df degrees of freedom -- low df
    means fat tails), scaled by a GARCH(1,1) variance state that
    mean-reverts to long_run_variance but is pushed up by the PREVIOUS
    step's realized standardized shock z^2 -- that persistence (decaying
    geometrically at rate garch_beta rather than resetting each step) is
    what creates measurable autocorrelation in |increments|.
    """
    omega = (1.0 - garch_alpha - garch_beta) * long_run_variance
    variance_state = long_run_variance
    increments = np.empty(n)
    for i in range(n):
        z = rng.standard_t(df)
        increments[i] = z * np.sqrt(variance_state)
        variance_state = omega + garch_alpha * (z ** 2) + garch_beta * variance_state
    return increments


def generate_realistic_synthetic_pair(n: int = 400, seed: int = 42,
                                       hedge_ratio: float = 2.0, intercept: float = 5.0,
                                       fat_tail_df: int = 4,
                                       garch_alpha: float = 0.15, garch_beta: float = 0.80):
    """
    Returns (A, B), a cointegrated pair with hedge ratio `hedge_ratio`
    and intercept `intercept` EXACTLY as generate_synthetic_pair()
    constructs them (A = hedge_ratio*B + intercept + idiosyncratic
    noise), but with both B's increments and A's idiosyncratic noise
    drawn from the fat-tailed, volatility-clustered process above
    instead of i.i.d. Gaussian.
    """
    rng = np.random.RandomState(seed)

    b_increments = _garch_fat_tailed_increments(rng, n, df=fat_tail_df,
                                                 garch_alpha=garch_alpha, garch_beta=garch_beta)
    B = np.cumsum(b_increments)

    idiosyncratic = _garch_fat_tailed_increments(rng, n, df=fat_tail_df,
                                                  garch_alpha=garch_alpha, garch_beta=garch_beta)
    A = hedge_ratio * B + intercept + idiosyncratic

    return A, B
