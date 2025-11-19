# scripts/run_breakout.py
# Run a trend-following breakout strategy and display results.

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

# Make the project importable when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.engine import run_backtest
from backtesting.trend_breakout import breakout_positions


def plot_breakout_equity(df, positions, result):
    """
    Plot:
    - price with position background
    - equity curve (rebased to 1.0)
    """
    price = df["price"]
    eq = result.equity / result.equity.iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- 1) Price + position colouring ---
    ax1.plot(price.index, price, color="black", lw=1.2, label="Price")
    ax1.set_title("Breakout strategy — price & positions")

    # Shade background by position
    # +1 long = light green, -1 short = light red, 0 flat = grey
    long_mask = positions > 0
    short_mask = positions < 0
    flat_mask = positions == 0

    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=long_mask,
        color="green",
        alpha=0.08,
        label="Long",
    )
    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=short_mask,
        color="red",
        alpha=0.08,
        label="Short",
    )
    ax1.fill_between(
        price.index,
        price.min(),
        price.max(),
        where=flat_mask,
        color="grey",
        alpha=0.04,
        label="Flat",
    )

    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- 2) Equity curve ---
    ax2.plot(eq.index, eq, lw=1.5, label="Breakout equity")
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_title("Equity curve (rebased to 1.0)")
    ax2.set_ylabel("Index")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "breakout_equity.png")
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Breakout equity figure saved → {out_png}")
    plt.show()


def main():
    # 1) Load prices (CSV or synthetic)
    df = load_prices()
    price = df["price"]

    # 2) Build breakout positions
    positions = breakout_positions(
        price,
        lookback=80,   # you can play with this
        hold_bars=5,   # minimum holding period
    )

    # 3) Run backtest using your generic engine
    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # 4) Print metrics
    print("\n===== Breakout strategy results =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {result.equity.iloc[-1]:.4f}")

    # 5) Plot
    plot_breakout_equity(df, positions, result)


if __name__ == "__main__":
    main()
