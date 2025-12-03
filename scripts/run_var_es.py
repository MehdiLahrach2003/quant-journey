# scripts/run_var_es.py
"""
Compute Value-at-Risk (VaR) and Expected Shortfall (CVaR)
for the SMA 20/100 strategy.
"""

import os
import sys
import matplotlib.pyplot as plt

# project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.risk_measures import compute_risk_measures


def main():
    # 1) Load prices (AAPL or synthetic)
    df = load_prices()
    price = df["price"]

    # 2) Build positions for SMA 20/100
    pos = sma_crossover_positions(price, short=20, long=100)

    # 3) Run backtest
    res = run_backtest(price, pos, cost_bps=1.0)

    # 4) Compute VaR & CVaR
    metrics = compute_risk_measures(res.returns, alpha=0.05)

    print("\n===== Risk Measures (5% tail) =====")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.5f}")

    print(f"\nFinal equity: {res.equity.iloc[-1]:.4f}")

    # 5) Optional plot of returns histogram + VaR threshold
    plt.figure(figsize=(8,5))
    r = res.returns.dropna()
    plt.hist(r, bins=50, alpha=0.6, label="Returns")

    var_hist = metrics["VaR_hist"]
    plt.axvline(-var_hist, color="red", linestyle="--", label=f"VaR 5% = {var_hist:.4f}")

    plt.title("Strategy returns distribution with VaR threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "var_es_histogram.png")
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Histogram saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
