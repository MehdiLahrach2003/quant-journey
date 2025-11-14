# utils/param_search.py
# Grid-search utilities for SMA crossover parameters.

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Tuple

from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions


def evaluate_sma_grid(
    df: pd.DataFrame,
    shorts: Iterable[int],
    longs: Iterable[int],
    cost_bps: float = 1.0,
    criterion: str = "sharpe",         # "sharpe" | "cumret"
    initial_capital: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Brute-force grid over (short, long) SMA windows.

    Returns
    -------
    results_df : long-form table with metrics for each (short, long)
    pivot      : matrix (rows=long, cols=short) of the selected criterion
    """
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price'.")

    price = df["price"].astype(float)

    rows = []
    for s in shorts:
        for l in longs:
            if s >= l:
                continue

            pos = sma_crossover_positions(price, short=s, long=l)
            res = run_backtest(price, pos, cost_bps=cost_bps, initial_capital=initial_capital)

            rows.append(
                {
                    "short": s,
                    "long": l,
                    "sharpe": res.metrics.get("Sharpe Ratio", np.nan),
                    "cumret": res.metrics.get("Cumulative Return", np.nan),
                    "max_dd": res.metrics.get("Max Drawdown", np.nan),
                    "total_costs": res.metrics.get("Total Costs", np.nan),
                }
            )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df, pd.DataFrame()

    # pivot for quick inspection / heatmap
    if criterion not in {"sharpe", "cumret"}:
        criterion = "sharpe"
    pivot = results_df.pivot_table(index="long", columns="short", values=criterion, aggfunc="mean")

    return results_df, pivot
