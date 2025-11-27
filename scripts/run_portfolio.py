# scripts/run_portfolio.py
"""
Example: combine two strategies (SMA crossover and breakout)
into a single portfolio using backtesting.portfolio.combine_backtests.
"""

import os
import sys

import matplotlib.pyplot as plt

# Make project root importable (so 'backtesting.*' and 'utils.*' work)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions
from backtesting.engine import run_backtest
from backtesting.portfolio import combine_backtests


def plot_equity_comparison(results_dict):
    """
    Plot equity curves for several BacktestResult objects.
    """
    plt.figure(figsize=(10, 6))

    for name, res in results_dict.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity curves – individual strategies vs portfolio")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio_equity.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Portfolio equity plot saved → {out_png}")

    plt.show()


def print_metrics(title, res):
    """
    Pretty-print metrics for a single BacktestResult.
    """
    print(f"\n===== {title} =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")


def main():
    # 1) Load prices (CSV if present, otherwise synthetic GBM)
    df = load_prices()
    price = df["price"]

    # 2) Build positions for each strategy
    sma_pos = sma_crossover_positions(price, short=20, long=100)
    brk_pos = breakout_positions(price, lookback=50)

    # 3) Run backtests for each strategy
    sma_res = run_backtest(df, sma_pos, cost_bps=1.0, initial_capital=1.0)
    brk_res = run_backtest(df, brk_pos, cost_bps=1.0, initial_capital=1.0)

    # 4) Combine them into a 50/50 portfolio
    results = {
        "SMA 20/100": sma_res,
        "Breakout 50d": brk_res,
    }
    weights = {
        "SMA 20/100": 0.5,
        "Breakout 50d": 0.5,
    }

    portfolio_res = combine_backtests(results, weights=weights)
    results["Portfolio 50/50"] = portfolio_res

    # 5) Print metrics
    print_metrics("SMA 20/100", sma_res)
    print_metrics("Breakout 50d", brk_res)
    print_metrics("Portfolio 50/50", portfolio_res)

    # 6) Plot equity comparison
    plot_equity_comparison(results)


if __name__ == "__main__":
    main()
