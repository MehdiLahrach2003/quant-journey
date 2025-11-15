# scripts/run_stops.py
# Demo: SMA crossover -> optional vol-target -> apply stops -> backtest

from __future__ import annotations
import os
import sys
import matplotlib.pyplot as plt

# Allow "import backtesting / utils"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.risk import vol_target_positions
from backtesting.rules import apply_stops
from backtesting.engine import run_backtest


def plot_equity(result, title="Equity"):
    eq = result.equity
    plt.figure(figsize=(11, 5))
    plt.plot(eq.index, eq.values, label="Equity", lw=1.6)
    plt.fill_between(eq.index, eq.values, eq.cummax(), alpha=0.12, color="red")
    plt.title(title)
    plt.ylabel("Capital")
    plt.grid(alpha=0.3)
    plt.legend()
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stops_equity.png")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"[OK] Equity plot saved -> {out}")
    plt.show()


def main():
    # 1) Load prices (synthetic fallback if no CSV)
    df = load_prices()
    price = df["price"]
    returns = price.pct_change().fillna(0.0)

    # 2) Base signal (SMA crossover)
    pos_raw = sma_crossover_positions(price, short=20, long=100)

    # 3) (Optional) Vol targeting on top of SMA (comment out if you want raw)
    pos_vt = vol_target_positions(
        returns=returns,
        positions=pos_raw,
        target_vol_annual=0.10,
        lookback=20,
        max_leverage=3.0,
    )

    # 4) Apply stops on the (possibly vol-targeted) exposure
    pos_stopped = apply_stops(
        prices=price,
        positions=pos_vt,
        stop_loss_pct=0.05,      # 5% SL
        take_profit_pct=0.10,    # 10% TP
        trailing=True,           # trailing behavior
    )

    # 5) Backtest (same engine API as d’habitude)
    res = run_backtest(df, pos_stopped, cost_bps=1.0)

    print("\n===== Backtest with Stops (SMA 20/100, vol-target 10%, SL 5%, TP 10%) =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"\nFinal equity: {res.equity.iloc[-1]:.2f}")

    # 6) Plot
    plot_equity(res, title="SMA 20/100 + VolTarget + Stops")


if __name__ == "__main__":
    main()
