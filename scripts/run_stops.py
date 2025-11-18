# scripts/run_stops.py
# Compare SMA crossover with and without stop-loss / take-profit.

import os
import sys
import matplotlib.pyplot as plt

# --- Make 'quant-journey' package importable ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.rules import StopConfig, apply_stop_loss_take_profit


def plot_equity_comparison(base_res, stopped_res):
    """Plot equity curves for baseline SMA and SMA + stops."""
    eq_base = base_res.equity / base_res.equity.iloc[0]
    eq_stop = stopped_res.equity / stopped_res.equity.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(eq_base.index, eq_base, label="SMA 20/100 (no stops)", lw=1.4)
    plt.plot(eq_stop.index, eq_stop, label="SMA 20/100 + stops", lw=1.4)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Equity curve: SMA crossover with vs without stops")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stops_equity.png")
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Equity comparison saved → {out_png}")
    plt.show()


def print_metrics(title: str, result):
    """Pretty-print result metrics."""
    print(f"\n===== {title} =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {result.equity.iloc[-1]:.4f}")


def main():
    # 1) Load prices
    df = load_prices()
    price = df["price"]

    # 2) Build baseline positions
    base_positions = sma_crossover_positions(price, short=20, long=100)

    # 3) Backtest baseline (NO NAMED ARGUMENT 'price=' !!!)
    base_res = run_backtest(
        price,
        base_positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # 4) Stop parameters
    cfg = StopConfig(
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )

    # 5) Apply stop-loss / take-profit
    stopped_positions = apply_stop_loss_take_profit(
        prices =price,
        base_positions=base_positions,
        config=cfg,
    )

    # 6) Backtest with stops
    stopped_res = run_backtest(
        price,
        stopped_positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # 7) Print metrics
    print_metrics("SMA 20/100 (no stops)", base_res)
    print_metrics("SMA 20/100 + stops", stopped_res)

    # 8) Plot result
    plot_equity_comparison(base_res, stopped_res)


if __name__ == "__main__":
    main()