"""
Walk-forward validation for cointegration.

A single Engle-Granger test on the full sample tells you the pair LOOKED
cointegrated over that whole window. It says nothing about whether that
relationship was stable, or whether it happened to hold only because of
one particular stretch of the data. Walk-forward validation checks this
directly: roll a window across the data, test cointegration in each
window, and see how often a pair that tested cointegrated in one window
ALSO tests cointegrated in the immediately following window.

This is the same honest-reversal pattern used elsewhere in this
project's research: a relationship that only holds in-sample and falls
apart out-of-sample is a real, reportable finding, not a bug to hide.
"""
from dataclasses import dataclass

import numpy as np

from src.cointegration import test_cointegration


@dataclass
class WalkForwardResult:
    n_windows: int
    n_in_sample_cointegrated: int
    n_survived_out_of_sample: int

    @property
    def survival_rate(self) -> float:
        """Of the windows where cointegration was found in-sample, what
        fraction ALSO held in the following out-of-sample window?
        Returns 0.0 if cointegration was never found in-sample at all
        (rather than raising or returning NaN) -- "the relationship was
        never even found" is a valid, reportable outcome."""
        if self.n_in_sample_cointegrated == 0:
            return 0.0
        return self.n_survived_out_of_sample / self.n_in_sample_cointegrated


def walk_forward_validate(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window_size: int,
    step_size: int,
    significance: float = 0.05,
) -> WalkForwardResult:
    """
    Slides a window of `window_size` across the data in steps of
    `step_size`. For each position, tests cointegration in that window
    (in-sample), then tests cointegration in the NEXT window of the same
    size immediately following it (out-of-sample). Counts how often
    in-sample-cointegrated pairs remain cointegrated out-of-sample.

    Windows that don't have a full following window (i.e. near the end
    of the data) are simply not tested -- there's nothing to validate
    against past the end of the series.
    """
    series_a = np.asarray(series_a, dtype=float)
    series_b = np.asarray(series_b, dtype=float)
    n = len(series_a)

    n_windows = 0
    n_in_sample_cointegrated = 0
    n_survived = 0

    start = 0
    while start + window_size + window_size <= n:  # need room for both in-sample AND out-of-sample windows
        in_sample_a = series_a[start: start + window_size]
        in_sample_b = series_b[start: start + window_size]
        out_sample_a = series_a[start + window_size: start + 2 * window_size]
        out_sample_b = series_b[start + window_size: start + 2 * window_size]

        n_windows += 1
        in_result = test_cointegration(in_sample_a, in_sample_b, significance=significance)

        if in_result.is_cointegrated:
            n_in_sample_cointegrated += 1
            out_result = test_cointegration(out_sample_a, out_sample_b, significance=significance)
            if out_result.is_cointegrated:
                n_survived += 1

        start += step_size

    return WalkForwardResult(
        n_windows=n_windows,
        n_in_sample_cointegrated=n_in_sample_cointegrated,
        n_survived_out_of_sample=n_survived,
    )
