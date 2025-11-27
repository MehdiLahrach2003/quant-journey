# backtesting/portfolio.py
"""
Simple tools to combine several single-strategy backtests
into one portfolio backtest.

We work on the output of backtesting.engine.run_backtest
(BacktestResult objects) and build a weighted portfolio:
    r_portfolio(t) = sum_i w_i * r_i(t)

Assumption: all strategies are run on the same capital and
rebalanced to target weights every day.
"""

from __future__ import annotations

from typing import Dict, Mapping
import math

import numpy as np
import pandas as pd

from .engine import BacktestResult, _annualize_sharpe


def combine_backtests(
    results: Mapping[str, BacktestResult],
    weights: Mapping[str, float] | None = None,
) -> BacktestResult:
    """
    Combine several BacktestResult objects into a single portfolio.

    Parameters
    ----------
    results : mapping name -> BacktestResult
        Output of run_backtest for each individual strategy.
    weights : mapping name -> float, optional
        Portfolio weights for each strategy. If None, equal weights
        are used across all strategies present in `results`.

    Returns
    -------
    BacktestResult
        New BacktestResult for the combined portfolio. Returns,
        equity, costs etc. are all portfolio quantities.
    """
    if len(results) == 0:
        raise ValueError("`results` must contain at least one strategy.")

    # Put daily strategy returns into one DataFrame
    ret_df = pd.DataFrame({name: res.returns for name, res in results.items()})
    ret_df = ret_df.fillna(0.0)

    # Default = equal weights
    if weights is None:
        n = ret_df.shape[1]
        w = pd.Series(1.0 / n, index=ret_df.columns, dtype=float)
    else:
        w = pd.Series(weights, dtype=float)
        # Align on columns, missing weights -> 0
        w = w.reindex(ret_df.columns).fillna(0.0)
        if w.sum() == 0.0:
            raise ValueError("All portfolio weights are zero.")
        # Normalise so that sum(weights) = 1
        w = w / w.sum()

    # --- Portfolio returns ---
    port_ret = (ret_df * w).sum(axis=1)
    port_ret.name = "portfolio_returns"

    # --- Portfolio equity curve ---
    equity = (1.0 + port_ret).cumprod()
    equity.name = "equity"

    # --- Combine positions, costs and trades (linear in weights) ---
    pos_df = pd.DataFrame({name: res.positions for name, res in results.items()}).fillna(0.0)
    portfolio_pos = (pos_df * w).sum(axis=1)
    portfolio_pos.name = "position"

    costs_df = pd.DataFrame({name: res.costs for name, res in results.items()}).fillna(0.0)
    portfolio_costs = (costs_df * w.abs()).sum(axis=1)
    portfolio_costs.name = "costs"

    trades_df = pd.DataFrame({name: res.trades for name, res in results.items()}).fillna(0.0)
    portfolio_trades = (trades_df * w.abs()).sum(axis=1)
    portfolio_trades.name = "trades"

    # --- Metrics (same style as engine.run_backtest) ---
    if len(equity) > 1:
        cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    else:
        cum_ret = 0.0

    if len(port_ret) > 1:
        ann_vol = float(port_ret.std(ddof=1) * math.sqrt(252))
    else:
        ann_vol = 0.0

    sharpe = _annualize_sharpe(port_ret)

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    total_costs = float(portfolio_costs.sum())

    metrics = {
        "Cumulative Return": cum_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Costs": total_costs,
    }

    return BacktestResult(
        equity=equity,
        returns=port_ret,
        costs=portfolio_costs,
        positions=portfolio_pos,
        trades=portfolio_trades,
        metrics=metrics,
    )
