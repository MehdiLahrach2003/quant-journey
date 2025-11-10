# utils/param_search.py
# Grid evaluation + robust selection of the best hyperparameters.

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions


def evaluate_sma_grid(
    df: pd.DataFrame,
    shorts: list[int],
    longs: list[int],
    total_bps: float = 10.0,
    initial_capital: float = 1.0,
    criterion: str = "sharpe",  # 'sharpe' | 'cumret' | 'maxdd' (only used for the heatmap pivot)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all (short, long) SMA combinations with long > short.

    Returns
    -------
    results_df : pd.DataFrame
        One row per (short, long) with key metrics.
    pivot : pd.DataFrame
        Pivot table for heatmap visualization on the selected `criterion`.
    """
    rows = []
    price = df["price"]

    for s, l in itertools.product(shorts, longs):
        if l <= s:
            continue
        pos = sma_crossover_positions(price, s, l)
        res = run_backtest(df, pos, total_bps=total_bps, initial_capital=initial_capital)

        rows.append({
            "short": s,
            "long": l,
            "cumret": res.metrics["cumret"],
            "sharpe": res.metrics["sharpe"],
            "ann_vol": res.metrics["ann_vol"],
            "maxdd": res.metrics["maxdd"],
            "total_costs": res.metrics["total_costs"],
        })

    results_df = pd.DataFrame(rows).sort_values(["long", "short"]).reset_index(drop=True)

    # Choose the value column for the heatmap
    value_col = {"sharpe": "sharpe", "cumret": "cumret", "maxdd": "maxdd"}.get(criterion, "sharpe")
    pivot = results_df.pivot(index="long", columns="short", values=value_col)

    return results_df, pivot


def pick_best_params(results_df: pd.DataFrame, by: str = "sharpe") -> pd.Series:
    """
    Selects the best parameter combination from a backtest results DataFrame.

    - 'sharpe' and 'cumret' → maximize
    - 'maxdd' → minimize (most negative is worst)

    NaN and ±inf are ignored; raises a helpful error if all values are invalid.
    """
    if by not in {"sharpe", "cumret", "maxdd"}:
        raise ValueError("`by` must be one of {'sharpe', 'cumret', 'maxdd'}")

    if by not in results_df.columns:
        raise KeyError(f"Column '{by}' not found in results_df.")

    s = results_df[by].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise ValueError(
            f"No valid values for criterion '{by}'. "
            "This can happen if volatility ≈ 0 (Sharpe NaN/inf) or returns are empty. "
            "Try 'cumret', generate longer data, or widen your (short,long) grid."
        )

    idx = s.idxmax() if by in {"sharpe", "cumret"} else s.idxmin()
    return results_df.loc[idx]