"""
Hand-traced tests for execution_sim.py.

TWAP case: prices=[100,101,102,103], n_slices=4, qty=400
  -> executes 100 shares at each price (equal time slices)
  -> avg_price = (100+101+102+103)/4 = 101.5
  -> arrival_price = 100 (first price)
  -> shortfall (buy) = (101.5 - 100) * 400 = 600

VWAP case: prices=[100,101,102], volumes=[100,200,100], qty=400
  -> weights = [0.25, 0.5, 0.25] -> slice_qty = [100, 200, 100]
  -> notional = 100*100 + 101*200 + 102*100 = 10000+20200+10200 = 40400
  -> avg_price = 40400/400 = 101.0
  -> shortfall (buy) = (101.0 - 100) * 400 = 400

Sell-side sign check: a sell that executes ABOVE arrival price should
show NEGATIVE shortfall (that's a GOOD outcome for a sell -- you got a
better price than expected), which is why the sign flips by side.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.execution_sim import twap_execute, vwap_execute

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    # --- TWAP ---
    prices = np.array([100.0, 101.0, 102.0, 103.0])
    result = twap_execute(prices, total_quantity=400, n_slices=4, side="buy")
    check(abs(result.avg_execution_price - 101.5) < 1e-9, f"TWAP avg_execution_price == 101.5, got {result.avg_execution_price}")
    check(result.arrival_price == 100.0, f"TWAP arrival_price == 100.0, got {result.arrival_price}")
    check(abs(result.implementation_shortfall - 600.0) < 1e-9,
          f"TWAP shortfall == 600.0, got {result.implementation_shortfall}")

    # --- VWAP ---
    prices2 = np.array([100.0, 101.0, 102.0])
    volumes2 = np.array([100.0, 200.0, 100.0])
    result2 = vwap_execute(prices2, volumes2, total_quantity=400, side="buy")
    check(abs(result2.avg_execution_price - 101.0) < 1e-9, f"VWAP avg_execution_price == 101.0, got {result2.avg_execution_price}")
    check(abs(result2.implementation_shortfall - 400.0) < 1e-9,
          f"VWAP shortfall == 400.0, got {result2.implementation_shortfall}")

    # --- sign check: same TWAP scenario but as a SELL should flip shortfall's sign ---
    result_sell = twap_execute(prices, total_quantity=400, n_slices=4, side="sell")
    check(abs(result_sell.implementation_shortfall - (-600.0)) < 1e-9,
          f"selling into a rising price gives NEGATIVE shortfall (favorable), got {result_sell.implementation_shortfall}")

    # --- input validation ---
    try:
        twap_execute(np.array([100.0, 101.0]), total_quantity=100, n_slices=4)
        check(False, "should reject fewer price points than n_slices")
    except ValueError:
        check(True, "correctly rejects fewer price points than n_slices")

    try:
        vwap_execute(np.array([100.0, 101.0]), np.array([50.0]), total_quantity=100)
        check(False, "should reject mismatched prices/volumes lengths")
    except ValueError:
        check(True, "correctly rejects mismatched prices/volumes lengths")

    print()
    if failures == 0:
        print("All execution simulation checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
