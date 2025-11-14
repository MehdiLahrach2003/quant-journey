# scripts/run_backtest.py
# Run a single SMA-crossover backtest and plot the results.

import os
import sys
import matplotlib.pyplot as plt

# Allow "backtesting" and "utils" imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


def plot_backtest(df, positions, result):
    """Two panels: price+signals and equity curve."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

    # 1) Price + signals
    ax1.plot(df.index, df["price"], label="Price", color="black", lw=1.2)
    buys = positions[(positions.shift(1) <= 0) & (positions > 0)].index
    sells = positions[(positions.shift(1) >= 0) & (positions < 0)].index
    ax1.scatter(buys,  df.loc[buys,  "price"], color="green", marker="^", s=70, label="Buy")
    ax1.scatter(sells, df.loc[sells, "price"], color="red",   marker="v", s=70, label="Sell")
    ax1.set_title("SMA crossover — trading signals")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 2) Equity curve (+ drawdown shadow)
    eq = result.equity
    ax2.plot(eq.index, eq, lw=1.6, label="Equity")
    ax2.fill_between(eq.index, eq, eq.cummax(), color="red", alpha=0.12, label="Drawdown")
    ax2.set_title("Equity curve")
    ax2.set_ylabel("Capital")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backtest_plot.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"\n[OK] Plot saved to: {out_png}\n")
    plt.show()


def main():
    # Load prices (utils/data_loader.py will synthesize data if file is missing)
    df = load_prices()  # DataFrame with 'price' column

    # Build SMA crossover positions (+1 / 0 / -1), shifted by 1 bar inside the function
    positions = sma_crossover_positions(df["price"], short=20, long=100)

    # >>> IMPORTANT: engine.run_backtest uses 'cost_bps' <<<
    result = run_backtest(
        df["price"],
        positions,
        cost_bps=1.0,         # 1 basis point per unit of |Δposition|
        # initial_capital=1.0  # uncomment if you want to change starting capital
    )

    print("\n===== Backtest results =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")

    print("\nFinal equity:", round(result.equity.iloc[-1], 4))

    # Plot
    plot_backtest(df, positions, result)


if __name__ == "__main__":
    main()


