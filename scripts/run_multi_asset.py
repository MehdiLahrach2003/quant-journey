# scripts/run_multi_asset.py
"""
Multi-asset backtest:
- Load multiple assets (AAPL, BTCUSD, MSFT, etc.)
- Apply SMA 20/100 on each asset
- Run backtest engine for each one
- Build an equal-weight portfolio
- Plot portfolio equity vs individual components
"""

from __future__ import annotations
import os
import sys
import matplotlib.pyplot as plt

# Allow project imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_multi import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest, combine_backtests


# -----------------------------------------
# Plot equity comparison
# -----------------------------------------
def plot_equities(res_dict):
    plt.figure(figsize=(12, 6))

    for ticker, res in res_dict.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=ticker)

    plt.axhline(1.0, color="black", ls="--", lw=0.8)
    plt.title("Multi-asset SMA 20/100 strategies vs portfolio")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "multi_asset_equity.png")
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Equity figure saved → {out_png}")

    plt.show()


# -----------------------------------------
# Pretty-print metrics
# -----------------------------------------
def print_metrics(name, res):
    print(f"\n===== {name} =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")


# -----------------------------------------
# Main
# -----------------------------------------
def main():
    # 1) Load all assets
    assets = load_multi_assets()

    print("\nLoaded assets:")
    for t in assets:
        print(" -", t)

    # 2) Build positions + backtest each asset separately
    results = {}
    for ticker, price in assets.items():
        pos = sma_crossover_positions(price, short=20, long=100)
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)
        results[ticker] = res

    # 3) Equal-weight portfolio
    weights = {t: 1.0 / len(results) for t in results}
    portfolio_res = combine_backtests(results, weights=weights)
    results["Portfolio EW"] = portfolio_res

    # 4) Print metrics
    for name, res in results.items():
        print_metrics(name, res)

    # 5) Plot all curves
    plot_equities(results)


if __name__ == "__main__":
    main()
