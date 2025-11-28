# scripts/run_portfolio_grid.py
# Grid-search on portfolio weights between SMA and Breakout strategies.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make project root importable (so "backtesting.*" and "utils.*" work)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions
from backtesting.engine import run_backtest, combine_backtests


def run_individual_strategies(df, cost_bps: float = 1.0):
    """
    Run the two base strategies (SMA 20/100 and Breakout 50d)
    and return a dict of BacktestResult objects.
    """
    price = df["price"]

    # --- SMA 20/100 ---
    sma_pos = sma_crossover_positions(price, short=20, long=100)
    sma_res = run_backtest(
        df,
        sma_pos,
        cost_bps=cost_bps,
        initial_capital=1.0,
    )

    # --- Breakout 50d ---
    brk_pos = breakout_positions(price, lookback=50)
    brk_res = run_backtest(
        df,
        brk_pos,
        cost_bps=cost_bps,
        initial_capital=1.0,
    )

    results = {
        "sma_20_100": sma_res,
        "breakout_50": brk_res,
    }
    return results


def portfolio_grid(results, n_steps: int = 11):
    """
    Build a grid of portfolios between SMA and Breakout.

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Must contain keys "sma_20_100" and "breakout_50".
    n_steps : int
        Number of points between 0 and 1 (inclusive) for the SMA weight.

    Returns
    -------
    pd.DataFrame
        One row per portfolio with weights and performance metrics.
    """
    sma_key = "sma_20_100"
    brk_key = "breakout_50"

    weights_sma = np.linspace(0.0, 1.0, n_steps)
    rows = []

    for w_sma in weights_sma:
        w_brk = 1.0 - w_sma

        weights = {
            sma_key: w_sma,
            brk_key: w_brk,
        }

        # Combine the two BacktestResult objects into a portfolio
        port_res = combine_backtests(results, weights)

        # Copy metrics and augment with weights + final equity
        m = dict(port_res.metrics)  # shallow copy
        m["w_sma_20_100"] = w_sma
        m["w_breakout_50"] = w_brk
        m["Final equity"] = float(port_res.equity.iloc[-1])

        rows.append(m)

    df_grid = pd.DataFrame(rows)

    # Order columns a bit more nicely if they exist
    cols = [
        "w_sma_20_100",
        "w_breakout_50",
        "Cumulative Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Total Costs",
        "Final equity",
    ]
    df_grid = df_grid[[c for c in cols if c in df_grid.columns]]
    return df_grid


def plot_risk_return(df_grid: pd.DataFrame, out_png: str | None = None):
    """
    Plot Cumulative Return vs Annualized Volatility for the portfolio grid.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = df_grid["Annualized Volatility"]
    y = df_grid["Cumulative Return"]

    ax.scatter(x, y, alpha=0.8)

    # Annotate a few key points: pure SMA, 50/50, pure Breakout
    for _, row in df_grid.iterrows():
        w = row["w_sma_20_100"]
        label = None
        if abs(w - 0.0) < 1e-8:
            label = "0% SMA / 100% Breakout"
        elif abs(w - 0.5) < 1e-8:
            label = "50% / 50%"
        elif abs(w - 1.0) < 1e-8:
            label = "100% SMA / 0% Breakout"

        if label is not None:
            ax.annotate(
                label,
                (row["Annualized Volatility"], row["Cumulative Return"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Risk/return grid – SMA 20/100 vs Breakout 50d portfolio")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Frontier plot saved → {out_png}")

    plt.show()


def main():
    # 1) Load price data
    df = load_prices()

    # 2) Run base strategies once
    results = run_individual_strategies(df, cost_bps=1.0)

    # 3) Build portfolio grid
    df_grid = portfolio_grid(results, n_steps=11)

    # 4) Save metrics to CSV
    root = os.path.dirname(os.path.dirname(__file__))
    out_csv = os.path.join(root, "data", "portfolio_grid_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_grid.to_csv(out_csv, index=False)
    print(f"[OK] Portfolio grid results saved → {out_csv}")

    # 5) Plot risk/return “frontier”
    out_png = os.path.join(root, "data", "portfolio_frontier.png")
    plot_risk_return(df_grid, out_png=out_png)


if __name__ == "__main__":
    main()
