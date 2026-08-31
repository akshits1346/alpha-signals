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
                        twap_execute_with_impact adds a square-root-law
                        market impact cost on top of TWAP -- see
                        finding #5 below.

run_pairs_with_execution.py
                     -- ties the above together: generates entry
                        signals from the pairs strategy, routes each
                        one through TWAP execution, reports realized
                        shortfall per entry.

execution_speed_experiment.py / run_execution_speed_experiment.py
                     -- direct test of WHY shortfall was positive above:
                        re-runs the SAME entries through TWAP at several
                        different speeds (1 to 20 slices), holding the
                        entry set and quantity fixed, to test whether
                        faster execution actually reduces the shortfall
                        the mechanism predicts it should.

synthetic_data.py    -- a more statistically realistic synthetic pair
                        generator (fat-tailed, volatility-clustered
                        innovations via Student-t + GARCH(1,1), same
                        mechanism as the orderbook-engine project's
                        generator) alongside run_pairs_with_execution.py's
                        original plain-Gaussian-random-walk one.
```

## Build & test

```bash
pip3 install statsmodels numpy

python3 tests/test_cointegration.py     # 9 checks
python3 tests/test_pairs_strategy.py    # 5 checks
python3 tests/test_walk_forward.py      # 6 checks
python3 tests/test_execution_sim.py     # 15 checks (incl. market impact)
python3 tests/test_execution_speed_experiment.py  # 17 checks (incl. impact-vs-drift optimum)
python3 tests/test_synthetic_data_realism.py      # 6 checks: fat tails + volatility clustering

python3 run_pairs_with_execution.py         # end-to-end demonstration
python3 run_execution_speed_experiment.py   # execution-speed-vs-shortfall experiment
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

**4. Testing that mechanism directly: does faster execution actually
reduce the shortfall?** `run_execution_speed_experiment.py` re-runs the
same 15 entries (of the 16 above, one excluded for lacking enough
trailing data at the largest window size) through TWAP at slice counts
from 1 (near-instant) to 20 (the original setting), holding the entry
set and order size fixed across every speed so window size is the only
thing that varies:

| window (slices) | total shortfall | mean shortfall |
|---|---|---|
| 1 | 0.00 | 0.00 |
| 2 | 1438.63 | 95.91 |
| 3 | 2180.05 | 145.34 |
| 5 | 2719.50 | 181.30 |
| 10 | 3093.80 | 206.25 |
| 20 | 3159.86 | 210.66 |

Shortfall increases monotonically with window size and hits exactly
zero at window=1 (a single slice fills entirely at the arrival price,
by definition -- not a finding, a check that the experiment is wired
up correctly). This directly confirms finding #3's causal claim rather
than just restating it: it isn't merely that this particular 5-slice
TWAP setting happened to cost money, it's that execution speed itself
is a lever on the cost, in the direction the reversion mechanism
predicts, monotonically, at every window size tested. The exact version
of this claim (a linear, deterministic reversion path where the
relationship has a closed form) is proven exactly, not just observed,
in `tests/test_execution_speed_experiment.py`.

**5. But finding #4 makes "always execute in 1 slice" look strictly
free -- it isn't, once market impact is priced in.** `execution_sim.py`
now has `twap_execute_with_impact`, a square-root-law temporary impact
model (`impact = impact_coefficient * (slice_qty/avg_volume)^0.5`, the
standard empirical form for temporary price impact): bigger clips (fewer
slices) cost MORE impact per slice, the opposite direction from finding
#4's drift effect. Re-running the same 15 entries with impact priced in
(`avg_volume=50, impact_coefficient=2.5` -- a deliberately illiquid
scenario, see the comment in `run_execution_speed_experiment.py` for
why: at gentler, more realistic participation rates tried first, drift
cost dominated impact by 1-2 orders of magnitude on this dataset's
actual spread scale, and window=1 stayed optimal outright):

| window (slices) | total shortfall (drift + impact) |
|---|---|
| 1 | 3073.66 |
| 2 | **2790.03** |
| 3 | 2944.30 |
| 5 | 3175.44 |
| 10 | 3321.11 |
| 20 | 3283.42 |

An INTERIOR minimum at window=2, beating both window=1 and window=20 --
neither "always execute instantly" nor "always execute slowly" is
optimal once both effects are priced in. The exact version of this (a
deterministic case with a provable interior minimum, found empirically
and verified, not asserted from a closed form since drift+impact
together don't have as clean a formula as drift alone) is in
`tests/test_execution_speed_experiment.py`. Reporting honestly that it
took an illiquid-instrument parameter regime to actually flip the
window=1-is-optimal conclusion on THIS dataset matters as much as the
interior-optimum result itself -- the mechanism is real and provably
correct, but it doesn't automatically change the practical
recommendation at every parameter setting, and pretending otherwise
would be exactly the kind of overclaiming this project tries to avoid.

## Honest limitations

- **All of the above is on synthetic data** (constructed pairs and
  spreads with known statistical properties -- `synthetic_data.py`'s
  generator is statistically validated to reproduce fat tails and
  volatility clustering, see `tests/test_synthetic_data_realism.py`,
  which is a materially better stand-in than a plain Gaussian random
  walk, but is still not real order flow), used specifically to verify
  the math and mechanisms are implemented correctly. None of it is a
  claim about real equity pairs. This was a deliberate fallback, not an
  oversight: this project was built in a sandboxed environment with
  outbound network access restricted to an allowlist (package
  registries only) -- every free financial data source tried (e.g.
  stooq.com) was confirmed unreachable (`EGRESS_BLOCKED`) before
  falling back to a better synthetic generator instead. Real price data
  (e.g. via a financial data API, the moment one is reachable) should
  replace this before treating any specific number as an actual
  trading result -- `test_cointegration`, `generate_signals`, and
  `walk_forward_validate` all take plain numpy arrays, so no code
  changes are needed to point them at real prices instead of
  `synthetic_data.py`'s output.
- **The execution tie-in uses the spread series itself as the "price
  path,"** not a real order book's liquidity/depth. A more realistic
  version would replay actual bid/ask depth (e.g. from the
  `orderbook-engine` project's LOBSTER-based replay) to get a genuine
  volume profile for VWAP and realistic slippage for TWAP, rather than
  assuming the spread itself is directly tradable at whatever price it
  shows.
- **The market impact model is TEMPORARY-only**: each slice's impact is
  computed independently and doesn't persist to the next slice or
  permanently shift the price path. A large enough real trade also has
  a PERMANENT component (the market re-prices based on the information
  your trading reveals), which this model doesn't capture -- it likely
  understates the true cost of trading a genuinely large position.
  `avg_volume` and `impact_coefficient` are also illustrative, not
  calibrated to any specific real instrument's actual liquidity.
- **The z-score strategy's parameters (lookback=20, entry_z=2.0,
  exit_z=0.5, stop_z=3.5) are conventional defaults, not optimized or
  validated** against any specific data.

## What I'd build next

- Real price data (equities or crypto) for an actual pair selection and
  trading result, the moment it's reachable from wherever this runs
  next -- see the honest-limitations note on why it isn't here now,
  and that no code changes are needed once it is
- Replace the execution tie-in's spread-as-price-path assumption with
  a real depth/volume profile from order book data
- Optimize/validate the z-score strategy's parameters via the
  walk-forward framework already built, rather than using un-tuned defaults
- Find the actual optimal window/impact-coefficient tradeoff curve
  (sweep impact_coefficient itself, not just window) rather than the
  single illustrative parameter setting used in finding #5 above
- Permanent, not just temporary, impact (a large enough trade can move
  the price for the REST of the session, not just its own fill) --
  the current model resets to zero impact influence on every fresh
  slice, which understates the true cost of trading a large position
