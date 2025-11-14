# scripts/run_vol_target.py
# Run SMA crossover with a volatility-targeting overlay and plot results.

import os
import sys
import matplotlib.pyplot as plt

# Allow "import backtesting / utils"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.risk import vol_target_positions
from backtesting.engine import run_backtest


def plot_equity(result, title="Vol-targeted equity"):
    plt.figure(figsize=(11, 5))
    eq = result.equity
    plt.plot(eq.index, eq.values, label="Equity", lw=1.6)
    plt.fill_between(eq.index, eq.values, eq.cummax(), alpha=0.12, color="red")
    plt.title(title)
    plt.ylabel("Capital")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vol_target_equity.png")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"[OK] Equity plot saved -> {out}")
    plt.show()


def main():
    # 1) Prices (your loader will synth fallback if no CSV)
    df = load_prices()                 # DataFrame with a 'price' column
    price = df["price"]
    returns = price.pct_change().fillna(0.0)

    # 2) Raw SMA crossover positions {-1,0,+1} (already shifted inside)
    pos_raw = sma_crossover_positions(price, short=20, long=100)

    # 3) Volatility targeting overlay (fractional exposure with leverage cap)
    pos_vt = vol_target_positions(
        returns=returns,
        positions=pos_raw,
        target_vol_annual=0.10,    # 10% vol target
        lookback=20,
        max_leverage=3.0,
    )

    # 4) Backtest (same API as d’habitude, param de coûts = cost_bps)
    res = run_backtest(df, pos_vt, cost_bps=1.0)

    print("\n===== Vol Target Backtest (SMA 20/100) =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"\nFinal equity: {res.equity.iloc[-1]:.2f}")

    # 5) Plot
    plot_equity(res, title="SMA 20/100 with Vol Target (10%)")


if __name__ == "__main__":
    main()
