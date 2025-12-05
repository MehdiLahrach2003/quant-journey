# scripts/run_var_cornish.py
"""
Compare Gaussian VaR vs Cornish–Fisher VaR for the SMA 20/100 strategy.
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.var_cornish import compute_cornish_report, cornish_fisher_var


def main():
    # Load data
    df = load_prices()
    price = df["price"]

    # Build strategy
    pos = sma_crossover_positions(price, 20, 100)
    res = run_backtest(price, pos, cost_bps=1.0)

    # Compute risk measures
    report = compute_cornish_report(res.returns, alpha=0.05)

    print("\n===== Gaussian vs Cornish–Fisher VaR (5%) =====")
    for k, v in report.items():
        print(f"{k:20s}: {v:.5f}")

    # Plot histogram
    r = res.returns.dropna()
    plt.figure(figsize=(8,5))
    plt.hist(r, bins=50, alpha=0.6, label="Returns")

    var_g = report["VaR_gaussian"]
    var_cf = report["VaR_cornish"]

    plt.axvline(-var_g, color="blue", linestyle="--", label=f"Gaussian VaR = {var_g:.4f}")
    plt.axvline(-var_cf, color="red", linestyle="--", label=f"Cornish–Fisher VaR = {var_cf:.4f}")

    plt.title("Gaussian vs Cornish–Fisher VaR (SMA 20/100)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "var_cornish_comparison.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
