# scripts/export_trades.py
# Export trade list (entries / exits / P&L) to CSV.

import os
import sys
import pandas as pd

# --- Make package importable ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


def main():
    # 1) Load price data
    df = load_prices()
    price = df["price"]

    # 2) Build positions (SMA 20/100 example)
    positions = sma_crossover_positions(price, short=20, long=100)

    # 3) Run backtest (FIXED: cost_bps replaces trans_cost_bps)
    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,          # <--- FIXED HERE
        initial_capital=1.0,
    )

    # 4) Extract trades into DataFrame
    trades = result.trades.copy()
    trades.reset_index(drop=True, inplace=True)

    # 5) Save CSV
    out_csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "trades_export.csv"
    )
    trades.to_csv(out_csv, index=False)

    print(f"\n[OK] Trades exported → {out_csv}\n")
    print(trades.head())


if __name__ == "__main__":
    main()


