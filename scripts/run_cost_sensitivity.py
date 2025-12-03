# scripts/run_cost_sensitivity.py
"""
Cost-sensitivity analysis for the SMA 20/100 strategy.

We:
- load prices (AAPL or synthetic if data/prices.csv is missing),
- build a SMA 20/100 signal,
- run the backtest for a range of transaction costs (in bps),
- collect risk/return metrics,
- save a CSV + plot showing how performance degrades with higher costs.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ---------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------
def evaluate_cost_grid(
    df_prices: pd.DataFrame,
    cost_grid_bps: np.ndarray,
    short_window: int = 20,
    long_window: int = 100,
) -> pd.DataFrame:
    """
    Run the SMA strategy for a grid of transaction costs (in bps).

    Parameters
    ----------
    df_prices : pd.DataFrame
        Must have a column 'price'.
    cost_grid_bps : array-like
        List/array of cost levels in basis points.
    short_window : int
        Short SMA length.
    long_window : int
        Long SMA length.

    Returns
    -------
    pd.DataFrame
        Index = cost_bps, columns = metrics (Sharpe, cumret, etc.).
    """
    price = df_prices["price"].astype(float)

    # Build positions once; only costs change across runs
    positions = sma_crossover_positions(price, short=short_window, long=long_window)

    rows = []

    for c_bps in cost_grid_bps:
        res = run_backtest(
            price_like=price,
            positions_like=positions,
            cost_bps=float(c_bps),
            initial_capital=1.0,
        )

        m = res.metrics
        rows.append(
            {
                "cost_bps": float(c_bps),
                "Cumulative Return": m.get("Cumulative Return", np.nan),
                "Annualized Volatility": m.get("Annualized Volatility", np.nan),
                "Sharpe Ratio": m.get("Sharpe Ratio", np.nan),
                "Max Drawdown": m.get("Max Drawdown", np.nan),
                "Total Costs": m.get("Total Costs", np.nan),
                "Final Equity": float(res.equity.iloc[-1]),
            }
        )

    df_res = pd.DataFrame(rows).set_index("cost_bps").sort_index()
    return df_res


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def plot_cost_sensitivity(df_res: pd.DataFrame, out_png: str | None = None) -> None:
    """
    Make a 2x2 plot: Sharpe vs cost, cumulative return vs cost,
    max drawdown vs cost, and final equity vs cost.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = df_res.index.values

    # Sharpe vs cost
    ax1.plot(x, df_res["Sharpe Ratio"], marker="o")
    ax1.set_title("Sharpe vs transaction costs")
    ax1.set_xlabel("Cost (bps)")
    ax1.set_ylabel("Sharpe")
    ax1.grid(alpha=0.3)

    # Cumulative return vs cost
    ax2.plot(x, df_res["Cumulative Return"], marker="o")
    ax2.set_title("Cumulative return vs costs")
    ax2.set_xlabel("Cost (bps)")
    ax2.set_ylabel("Cumulative return")
    ax2.grid(alpha=0.3)

    # Max drawdown vs cost
    ax3.plot(x, df_res["Max Drawdown"], marker="o")
    ax3.set_title("Max drawdown vs costs")
    ax3.set_xlabel("Cost (bps)")
    ax3.set_ylabel("Max drawdown")
    ax3.grid(alpha=0.3)

    # Final equity vs cost
    ax4.plot(x, df_res["Final Equity"], marker="o")
    ax4.set_title("Final equity vs costs")
    ax4.set_xlabel("Cost (bps)")
    ax4.set_ylabel("Final equity")
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Cost-sensitivity plot saved → {out_png}")

    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    # 1) Load prices (AAPL or synthetic)
    df = load_prices()  # default = data/prices.csv or synthetic GBM

    # 2) Define a grid of transaction costs in basis points
    #    e.g. from 0 bps to 50 bps
    cost_grid_bps = np.array([0, 1, 2, 5, 10, 20, 30, 40, 50], dtype=float)

    # 3) Run the cost-sensitivity analysis
    df_res = evaluate_cost_grid(df, cost_grid_bps, short_window=20, long_window=100)

    # 4) Save results to CSV
    base_path = os.path.dirname(os.path.dirname(__file__))
    out_csv = os.path.join(base_path, "data", "cost_sensitivity_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv)
    print(f"[OK] Cost-sensitivity metrics saved → {out_csv}")

    # 5) Plot summary
    out_png = os.path.join(base_path, "data", "cost_sensitivity_plot.png")
    plot_cost_sensitivity(df_res, out_png=out_png)

    # 6) Print small table in terminal
    print("\n===== Cost-sensitivity summary (SMA 20/100) =====")
    print(df_res.round(4))


if __name__ == "__main__":
    main()
