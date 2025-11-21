# scripts/run_risk_mc.py
# Monte Carlo risk analysis for the SMA 20/100 strategy.

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# --- Make 'quant-journey' package importable ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from utils.risk_mc import monte_carlo_from_returns


def plot_mc_results(stats: pd.DataFrame) -> None:
    """
    Plot histograms of:
    - final equity distribution
    - max drawdown distribution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Final equity
    ax1.hist(stats["final_equity"], bins=40, alpha=0.8)
    ax1.set_title("Final equity distribution")
    ax1.set_xlabel("Final equity (1.0 = starting capital)")
    ax1.set_ylabel("Frequency")
    ax1.grid(alpha=0.3)

    # Max drawdown (should be <= 0)
    ax2.hist(stats["max_drawdown"], bins=40, alpha=0.8)
    ax2.set_title("Max drawdown distribution")
    ax2.set_xlabel("Max drawdown (negative)")
    ax2.set_ylabel("Frequency")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "risk_mc_histograms.png",
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Monte Carlo histograms saved -> {out_png}")

    plt.show()


def main() -> None:
    # 1) Run the baseline SMA 20/100 backtest
    df = load_prices()
    price = df["price"]

    positions = sma_crossover_positions(price, short=20, long=100)

    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,       # keep it consistent with your other scripts
        initial_capital=1.0,
    )

    # 2) Build daily returns from the equity curve
    equity = result.equity
    returns = equity.pct_change().dropna()

    # 3) Run Monte Carlo risk analysis
    stats = monte_carlo_from_returns(
        returns,
        n_paths=2000,    # number of scenarios
        horizon=252,     # 1 trading year
        seed=42,
    )

    # 4) Print summary statistics
    print("\n===== Monte Carlo risk (1-year horizon) =====")
    print(f"Mean final equity      : {stats['final_equity'].mean():.4f}")
    print(f"Median final equity    : {stats['final_equity'].median():.4f}")
    print(f"5% worst-case equity   : {stats['final_equity'].quantile(0.05):.4f}")
    print(f"1% worst-case equity   : {stats['final_equity'].quantile(0.01):.4f}")

    # Drawdown is negative, so 'worst' = most negative (low quantile)
    print(f"Median max drawdown    : {stats['max_drawdown'].median():.4f}")
    print(f"95% worst max drawdown : {stats['max_drawdown'].quantile(0.95):.4f}")

    # 5) Plot histograms
    plot_mc_results(stats)


if __name__ == "__main__":
    main()
