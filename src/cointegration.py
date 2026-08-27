"""
Engle-Granger cointegration testing for pairs trading.

Two price series are cointegrated if some linear combination of them is
stationary (mean-reverting), even though each series individually is
not (typically a random walk / unit-root process). That's the entire
premise pairs trading rests on: if A and B are cointegrated, the spread
`A - hedge_ratio * B` should mean-revert, which is what a z-score
entry/exit strategy tries to exploit.

Engle-Granger is a two-step test:
  1. Regress A on B (OLS): A = alpha + beta*B + residual.
     `beta` here is the hedge ratio.
  2. Test the regression's residuals for a unit root (ADF test). If the
     residuals reject the unit-root null (i.e. ARE stationary), A and B
     are cointegrated with that hedge ratio.

This module uses statsmodels' `coint()`, which implements exactly this
two-step procedure with the correct (MacKinnon) critical values for the
Engle-Granger test specifically -- these differ from a plain ADF test's
critical values, which is a common mistake if you reimplement this by
hand using a generic ADF function on the residuals without adjusting
for the fact that the residuals came from an estimated regression, not
observed data. Using the library function sidesteps that entire
correctness trap.
"""
from dataclasses import dataclass

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint


@dataclass
class CointegrationResult:
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    hedge_ratio: float
    intercept: float


def test_cointegration(series_a: np.ndarray, series_b: np.ndarray, significance: float = 0.05) -> CointegrationResult:
    """
    Tests whether series_a and series_b are cointegrated, and if so,
    what hedge ratio defines the stationary spread.

    Returns is_cointegrated=True only if the Engle-Granger p-value is
    below `significance` -- the conventional 0.05 threshold by default,
    but callable with a stricter one for a more conservative pair
    selection filter.
    """
    series_a = np.asarray(series_a, dtype=float)
    series_b = np.asarray(series_b, dtype=float)

    if len(series_a) != len(series_b):
        raise ValueError("series_a and series_b must be the same length")
    if len(series_a) < 20:
        raise ValueError("need at least 20 observations for a meaningful cointegration test")

    test_stat, p_value, _crit_values = coint(series_a, series_b)

    # Separately fit the hedge ratio via OLS -- coint() runs its own
    # internal regression for the test statistic, but doesn't expose the
    # fitted beta directly, so we fit it again here to get hedge_ratio
    # for actually constructing the spread later.
    X = add_constant(series_b)
    ols_result = OLS(series_a, X).fit()
    intercept, hedge_ratio = ols_result.params

    return CointegrationResult(
        is_cointegrated=bool(p_value < significance),
        p_value=float(p_value),
        test_statistic=float(test_stat),
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
    )


def compute_spread(series_a: np.ndarray, series_b: np.ndarray, hedge_ratio: float, intercept: float = 0.0) -> np.ndarray:
    """The stationary linear combination the cointegration test found:
    spread = A - hedge_ratio*B - intercept. This is what should
    mean-revert if the pair really is cointegrated."""
    return np.asarray(series_a, dtype=float) - hedge_ratio * np.asarray(series_b, dtype=float) - intercept
