# scripts/make_report.py
"""
Run an SMA(20/100) backtest and generate a simple performance report:
- Print summary metrics from BacktestResult.metrics
- Plot a tear sheet (equity vs buy & hold, rolling Sharpe, drawdown)
- Save the tear sheet as PNG in data/tear_sheet.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Make the project root importable (quant-journey) ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ---------------------------------------------------------------------
# Helper functions for risk / performance
# ---------------------------------------------------------------------
def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from an equity curve.

    Drawdown_t = equity_t / max_{s <= t}(equity_s) - 1
    """
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    dd.name = "drawdown"
    return dd


def rolling_sharpe(returns: pd.Series, window: int = 126, ann_factor: int = 252) -> pd.Series:
    """
    Compute rolling Sharpe ratio on a rolling window.

    We assume daily simple returns and use:
        Sharpe = mean(ret) / std(ret) * sqrt(ann_factor)
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)
    sharpe = roll_mean / roll_std * np.sqrt(ann_factor)
    sharpe.name = f"rolling_sharpe_{window}"
    return sharpe


# ---------------------------------------------------------------------
# Tear sheet plotting
# ---------------------------------------------------------------------
def plot_tear_sheet(
    price: pd.Series,
    equity: pd.Series,
    strategy_returns: pd.Series,
    out_png: str | None = None,
) -> None:
    """
    Plot a simple tear sheet with:
    - Strategy equity vs buy & hold (both rebased to 1.0)
    - Rolling Sharpe (126 days)
    - Drawdown
    """
    # Rebase strategy equity to 1.0
    eq = equity.astype(float)
    eq_rebased = eq / eq.iloc[0]

    # Buy & hold benchmark (also rebased)
    bh = price.astype(float) / price.iloc[0]
    bh.name = "buy_and_hold"

    # Drawdown and rolling Sharpe
    dd = compute_drawdown(eq)
    roll_sh = rolling_sharpe(strategy_returns, window=126)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # 1) Equity vs benchmark
    ax1.plot(eq_rebased.index, eq_rebased, label="Strategy", lw=1.5)
    ax1.plot(bh.index, bh, label="Buy & Hold", lw=1.2)
    ax1.set_ylabel("Index (× start)")
    ax1.set_title("Equity (rebased) vs Benchmark")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Rolling Sharpe
    ax2.plot(roll_sh.index, roll_sh, lw=1.2)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Rolling Sharpe (126d)")
    ax2.grid(alpha=0.3)

    # 3) Drawdown
    ax3.fill_between(dd.index, dd, 0.0, color="steelblue", alpha=0.4)
    ax3.set_ylabel("Drawdown")
    ax3.set_title("Drawdown")
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Tear sheet saved → {out_png}")

    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    # 1) Load prices (CSV or synthetic if file is missing)
    df = load_prices()
    price = df["price"]

    # 2) Build SMA crossover positions (+1 / 0 / -1)
    positions = sma_crossover_positions(price, short=20, long=100)

    # 3) Run backtest with your engine (note: cost_bps, not trans_cost_bps)
    result = run_backtest(
        df,                # can pass the full DataFrame (engine will use "price")
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # 4) Print summary metrics
    print("\n===== Summary metrics =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v: .4f}")

    print(f"{'Final equity':25s}: {result.equity.iloc[-1]: .4f}")

    # 5) Plot tear sheet and save to PNG
    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tear_sheet.png")
    plot_tear_sheet(
        price=price,
        equity=result.equity,
        strategy_returns=result.returns,
        out_png=out_png,
    )


if __name__ == "__main__":
    main()
