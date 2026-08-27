# alpha-signals

Statistical arbitrage research: Engle-Granger cointegration testing for
pair selection, z-score mean-reversion trading, walk-forward validation
of pair stability, and an execution layer measuring the real cost of
actually entering a signal rather than assuming instant fills.

## Architecture

```
cointegration.py    -- Engle-Granger test (statsmodels' coint()) + hedge
                        ratio fit via OLS. Answers: are these two series
                        cointegrated, and what linear combination of them
                        is stationary?

pairs_strategy.py   -- rolling z-score of the spread + a state machine
                        (entry_z / exit_z / stop_z) generating FLAT /
                        LONG_SPREAD / SHORT_SPREAD positions over time.

walk_forward.py     -- rolls a window across the data, tests
                        cointegration in-sample, then re-tests in the
                        immediately following out-of-sample window.
                        Reports what fraction of "found cointegrated"
                        windows actually held up one window later.

execution_sim.py    -- TWAP (equal slices, equal time) and VWAP
                        (slices proportional to volume) execution
                        simulation, with implementation shortfall
                        (realized avg fill price vs. arrival price).

run_pairs_with_execution.py
                     -- ties the above together: generates entry
                        signals from the pairs strategy, routes each
                        one through TWAP execution, reports realized
                        shortfall per entry.
```

## Build & test

```bash
pip3 install statsmodels numpy

python3 tests/test_cointegration.py     # 9 checks
python3 tests/test_pairs_strategy.py    # 5 checks
python3 tests/test_walk_forward.py      # 6 checks
python3 tests/test_execution_sim.py     # 8 checks

python3 run_pairs_with_execution.py     # end-to-end demonstration
```

## Findings

**1. The cointegration test and hedge ratio fit are correct**, verified
against constructed synthetic pairs with known ground truth: a pair
built as `A = 2.0*B + 5.0 + noise` is correctly detected as cointegrated
(p < 0.00001) with a fitted hedge ratio of 2.006 (true: 2.0) and
intercept 5.041 (true: 5.0). Two independent random walks are correctly
NOT flagged as cointegrated (p = 0.72) -- this matters because a naive
correlation-based pair selection would be vulnerable to exactly this
kind of spurious relationship between unrelated trending series.

**2. Cointegration is not necessarily stable.** Walk-forward validation
on a pair engineered to be cointegrated for its first half and then
break down (become an independent random walk) in its second half shows
survival rate dropping from 0.889 (stable pair, cointegrated throughout)
to 0.600 (breaking pair) -- a real, structural difference, not noise.
The practical implication: a single significant cointegration test on
historical data is not sufficient justification to trade a pair
indefinitely; the relationship needs to be re-validated on a rolling
basis.

**3. Every simulated entry shows positive implementation shortfall --
consistently, not occasionally.** Routing each pairs-strategy entry
through 5-slice TWAP execution instead of assuming an instant fill, the
total shortfall across 16 entries was +2845.58 (all positive, no
exceptions). The mechanism is structural, not incidental: entries are
triggered exactly when the spread has hit a statistical extreme -- the
same moment it's expected to start reverting. Slicing the entry slowly
means executing directly into that reversion: a short entry sells into
a falling price, a long entry buys into a rising price, both against
the trader. **TWAP-slicing a mean-reversion entry works against the
same reversion the strategy is betting on** -- a real, explainable
result, not an artifact of the synthetic data it was demonstrated on.

## Honest limitations

- **All of the above is on synthetic data** (constructed pairs and
  spreads with known statistical properties), used specifically to
  verify the math and mechanisms are implemented correctly. None of it
  is a claim about real equity pairs. Real price data (e.g. via a
  financial data API) should replace this before treating any specific
  number as an actual trading result.
- **The execution tie-in uses the spread series itself as the "price
  path,"** not a real order book's liquidity/depth. A more realistic
  version would replay actual bid/ask depth (e.g. from the
  `orderbook-engine` project's LOBSTER-based replay) to get a genuine
  volume profile for VWAP and realistic slippage for TWAP, rather than
  assuming the spread itself is directly tradable at whatever price it
  shows.
- **The z-score strategy's parameters (lookback=20, entry_z=2.0,
  exit_z=0.5, stop_z=3.5) are conventional defaults, not optimized or
  validated** against any specific data.

## What I'd build next

- Swap in real price data (equities or crypto) for an actual pair
  selection and trading result
- Replace the execution tie-in's spread-as-price-path assumption with
  a real depth/volume profile from order book data
- Test whether faster execution (fewer slices, more aggressive) reduces
  or eliminates the structural shortfall found in finding #3 above --
  if the mechanism described is really what's driving it, a near-instant
  fill should show shortfall close to zero
- Optimize/validate the z-score strategy's parameters via the
  walk-forward framework already built, rather than using un-tuned defaults
