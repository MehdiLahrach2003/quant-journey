# scripts/make_report.py
# Build a tear sheet for the plain SMA 20/100 crossover strategy.
#
# You run this file (Run ▶️) to:
# - load prices (or generate synthetic ones),
# - build SMA crossover positions,
# - run the backtest,
# - compute stats and a tear sheet figure,
# - save the figure + metrics to the /data folder.

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ---- Make sure we can import from the project root ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.report import (
    compute_rolling_sharpe,
    compute_drawdown,
    summarize_equity,
    plot_tear_sheet,
)


def main() -> None:
    # 1) Load prices (utils/data_loader.py will create synthetic data if file is missing)
    df = load_prices()  # dataframe with at least a "price" column
    price = df["price"].astype(float)

    # 2) Base SMA crossover signal (no vol targeting, no stops here)
    pos_raw = sma_crossover_positions(price, short=20, long=100)

    # For this report we just use the raw SMA signal
    pos_final = pos_raw

    # 3) Run the backtest
    result = run_backtest(
        df,          # full dataframe, engine will grab df["price"]
        pos_final,   # positions in {-1, 0, +1}
        cost_bps=1.0,
        # initial_capital=1.0,  # uncomment if your engine supports this
    )

    equity = result.equity.astype(float)
    rets = result.returns.astype(float)

    # 4) Rebase equity and benchmark to 1.0 for nicer comparison
    equity_rebased = equity / equity.iloc[0]
    benchmark_rebased = price / price.iloc[0]

    # 5) Rolling Sharpe + drawdown
    rolling_sharpe = compute_rolling_sharpe(rets, window=126, periods_per_year=252)
    drawdown = compute_drawdown(equity)

    # 6) Print a small summary in the terminal
    stats = summarize_equity(equity, rets)
    print("\n===== Summary =====")
    for k, v in stats.items():
        if "dd" in k:
            print(f"{k:22s}: {v: .2%}")
        else:
            print(f"{k:22s}: {v: .4f}")

    print("\nFinal equity:", round(equity.iloc[-1], 2))

    # 7) Save summary metrics to CSV (nice for your README or further analysis)
    metrics_csv = os.path.join(PROJECT_ROOT, "data", "sma_report_metrics.csv")
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    pd.Series(stats).to_csv(metrics_csv)
    print(f"[OK] Metrics saved -> {metrics_csv}")

    # 8) Plot tear sheet and save the figure
    tear_sheet_path = os.path.join(PROJECT_ROOT, "data", "tear_sheet.png")
    plot_tear_sheet(
        equity_rebased=equity_rebased,
        benchmark_rebased=benchmark_rebased,
        rolling_sharpe=rolling_sharpe,
        drawdown=drawdown,
        title="Equity (rebased) vs Benchmark",
        save_path=tear_sheet_path,
    )


if __name__ == "__main__":
    main()
