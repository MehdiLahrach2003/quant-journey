# scripts/run_regime_filter.py
# Compare SMA crossover with and without a long-term regime filter.

import os
import sys
import matplotlib.pyplot as plt

# --- Make project package importable ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.regime import RegimeConfig, long_only_regime


def plot_equity_comparison(base_res, filtered_res):
    """
    Plot equity curves for:
    - baseline SMA 20/100
    - SMA 20/100 with long-only regime filter
    """
    eq_base = base_res.equity / base_res.equity.iloc[0]
    eq_filt = filtered_res.equity / filtered_res.equity.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(eq_base.index, eq_base, label="SMA 20/100 (no filter)", lw=1.4)
    plt.plot(eq_filt.index, eq_filt, label="SMA 20/100 + regime filter", lw=1.4)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity: SMA crossover with vs without regime filter")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "regime_equity.png",
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Equity comparison saved → {out_png}")

    plt.show()


def print_metrics(title: str, result):
    """Pretty-print the metrics dictionary from BacktestResult."""
    print(f"\n===== {title} =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {result.equity.iloc[-1]:.4f}")


def main():
    # 1) Load price data
    df = load_prices()
    price = df["price"]

    # 2) Baseline SMA 20/100 positions
    base_pos = sma_crossover_positions(price, short=20, long=100)

    # 3) Long-only regime based on SMA 200
    cfg = RegimeConfig(long_window=200)
    regime = long_only_regime(price, cfg)

    # Filtered positions: only trade when regime == 1
    filt_pos = base_pos * regime
    filt_pos.name = "position_filtered"

    # 4) Backtest both versions
    base_res = run_backtest(
        df,
        base_pos,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    filt_res = run_backtest(
        df,
        filt_pos,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # 5) Print metrics
    print_metrics("SMA 20/100 (no filter)", base_res)
    print_metrics("SMA 20/100 + regime filter", filt_res)

    # 6) Plot equity comparison
    plot_equity_comparison(base_res, filt_res)


if __name__ == "__main__":
    main()
