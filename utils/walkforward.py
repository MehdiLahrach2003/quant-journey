# utils/walkforward.py
# Walk-forward SMA: tune on train slice, evaluate on next test slice.

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import math
import numpy as np
import pandas as pd

# Make project importable if this file is run/imported in isolation
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


@dataclass
class WFWindowResult:
    start: pd.Timestamp
    end: pd.Timestamp
    best_short: int
    best_long: int
    sharpe_in_sample: float
    equity: pd.Series
    metrics: Dict[str, float]


@dataclass
class WalkForwardResult:
    equity_oos: pd.Series
    windows: List[WFWindowResult]
    params_table: pd.DataFrame


def _annualize_sharpe(daily_returns: pd.Series, freq_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * math.sqrt(freq_per_year)


def _tune_sma_on_train(
    price_train: pd.Series,
    short_grid: Iterable[int],
    long_grid: Iterable[int],
    cost_bps: float,
) -> Tuple[int, int, float]:
    best_short, best_long, best_sharpe = None, None, -np.inf
    for s in short_grid:
        for l in long_grid:
            if s >= l:
                continue
            pos = sma_crossover_positions(price_train, short=s, long=l)
            res = run_backtest(price_train, pos, cost_bps)
            sharpe = res.metrics.get("Sharpe Ratio", _annualize_sharpe(res.returns))
            if sharpe > best_sharpe:
                best_short, best_long, best_sharpe = s, l, sharpe
    return int(best_short), int(best_long), float(best_sharpe)


def walkforward_sma(
    df: pd.DataFrame,
    short_grid: Iterable[int] = (5, 10, 20, 30),
    long_grid: Iterable[int] = (50, 100, 150, 200),
    train_months: int = 24,
    test_months: int = 3,
    cost_bps: float = 1.0,
) -> WalkForwardResult:
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain a 'price' column.")

    df = df.sort_index()
    price_all = df["price"].astype(float)

    def add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
        return (dt + pd.DateOffset(months=months)).normalize()

    start, end = price_all.index.min(), price_all.index.max()
    cursor = start

    windows: List[WFWindowResult] = []
    stitched_parts: List[pd.Series] = []
    level = 1.0

    while True:
        train_start = cursor
        train_end = add_months(train_start, train_months) - pd.Timedelta(days=1)
        test_start = add_months(train_end, 1)
        test_end = add_months(test_start, test_months) - pd.Timedelta(days=1)

        if test_start > end or train_end > end:
            break

        train_slice = price_all.loc[train_start:train_end]
        test_slice = price_all.loc[test_start:test_end]

        if len(train_slice) < 30 or len(test_slice) < 5:
            cursor = add_months(cursor, test_months)
            continue

        # Tune on train (Series everywhere)
        best_s, best_l, sharpe_is = _tune_sma_on_train(train_slice, short_grid, long_grid, cost_bps)

        # Evaluate on test
        pos_test = sma_crossover_positions(test_slice, short=best_s, long=best_l)
        res_test = run_backtest(test_slice, pos_test, cost_bps)

        eq = res_test.equity.copy()
        # Stitch (normalize each segment to continue from previous level)
        eq = eq / float(eq.iloc[0]) * level
        level = float(eq.iloc[-1])
        stitched_parts.append(eq)

        windows.append(
            WFWindowResult(
                start=test_start,
                end=min(test_end, eq.index.max()),
                best_short=best_s,
                best_long=best_l,
                sharpe_in_sample=sharpe_is,
                equity=eq,
                metrics=res_test.metrics,
            )
        )

        cursor = add_months(cursor, test_months)

        if add_months(cursor, train_months + test_months) > end + pd.DateOffset(months=1):
            break

    # Final stitched equity
    if stitched_parts:
        equity_oos = pd.concat(stitched_parts).groupby(level=0).last()
    else:
        equity_oos = pd.Series(dtype=float)

    params_table = pd.DataFrame(
        [
            {
                "start": w.start,
                "end": w.end,
                "best_short": w.best_short,
                "best_long": w.best_long,
                "train_sharpe": w.sharpe_in_sample,
                "oos_sharpe": w.metrics.get("Sharpe Ratio", np.nan),
                "oos_cum_return": w.metrics.get("Cumulative Return", np.nan),
            }
            for w in windows
        ]
    ).set_index("start") if windows else pd.DataFrame()

    return WalkForwardResult(
        equity_oos=equity_oos,
        windows=windows,
        params_table=params_table,
    )
