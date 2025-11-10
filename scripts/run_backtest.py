import sys
import os
import matplotlib.pyplot as plt

# === Make package imports work when running this file directly ===
# (adds the repository root to sys.path)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ---------- Plotting helpers ----------
def plot_backtest(df, positions, result):
    """
    Two-panel plot:
      1) Price with buy/sell markers from the SMA crossover signals
      2) Equity curve with drawdown shading
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # 1) Price + signals
    ax1.plot(df.index, df["price"], label="Price", color="black", lw=1.2)

    # Buy when position goes from <= 0 to > 0; sell when it goes from >= 0 to < 0
    buy_idx = positions[(positions.shift(1) <= 0) & (positions > 0)].index
    sell_idx = positions[(positions.shift(1) >= 0) & (positions < 0)].index
    ax1.scatter(buy_idx, df.loc[buy_idx, "price"], color="green", marker="^", s=80, label="Buy")
    ax1.scatter(sell_idx, df.loc[sell_idx, "price"], color="red", marker="v", s=80, label="Sell")

    ax1.set_title("SMA Crossover — Trading Signals", fontsize=13)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Equity curve + drawdown (shade vs running max)
    equity = result.equity
    ax2.plot(equity.index, equity, color="tab:blue", lw=1.5, label="Equity Curve")
    ax2.fill_between(equity.index, equity, equity.cummax(), color="tab:red", alpha=0.12, label="Drawdown")
    ax2.set_title("Equity Curve", fontsize=13)
    ax2.set_ylabel("Equity")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure next to /data
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backtest_plot.png")
    try:
        plt.savefig(save_path, dpi=150)
        print(f"\n📊 Plot saved to: {save_path}\n")
    except Exception as e:
        print(f"[WARN] Could not save plot: {e}")

    plt.show()


# ---------- Main script ----------
def main():
    # Load prices (utils/data_loader.py will fallback to synthetic if file not found)
    df = load_prices()  # dataframe with a 'price' column and a datetime-like index

    # Build SMA crossover positions (+1 / 0 / -1)
    positions = sma_crossover_positions(
        df["price"],
        short=20,
        long=100,
    )

    # Run backtest (IMPORTANT: use total_bps to match engine.py)
    result = run_backtest(
        df,
        positions,
        total_bps=1.0,       # transaction cost in basis points per turnover
        # initial_capital=1.0 # uncomment if your engine supports/needs it
    )

    # Print metrics
    print("\n===== Backtest Results =====")
    for k, v in result.metrics.items():
        try:
            print(f"{k:25s}: {v:.4f}")
        except Exception:
            print(f"{k:25s}: {v}")

    # Final equity (also visible in result.equity.iloc[-1])
    print("\n📈 Final equity:", round(result.equity.iloc[-1], 4))

    # Plot
    plot_backtest(df, positions, result)


# Entry point
if __name__ == "__main__":
    main()

