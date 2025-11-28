# backtesting/engine.py
# Simple long/short backtest engine for daily data.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, Mapping

import math
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Container for a single backtest run.

    All series are indexed by date and aligned on the same index.
    """
    equity: pd.Series          # equity curve
    returns: pd.Series         # strategy daily returns (net, after costs)
    costs: pd.Series           # daily transaction costs
    positions: pd.Series       # position (or size) time series
    trades: pd.Series          # abs(position change), proxy for turnover
    metrics: Dict[str, float]  # summary metrics (Sharpe, max DD, etc.)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _to_price_series(x: Union[pd.Series, pd.DataFrame], col: str = "price") -> pd.Series:
    """
    Accept either a Series (already price) or a DataFrame with a 'price' column.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if col not in x.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
        s = x[col].copy()
    else:
        s = pd.Series(x)

    s = s.astype(float).sort_index()
    s.name = "price"
    return s


def _annualize_sharpe(daily: pd.Series, freq_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio from daily returns.
    """
    r = daily.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * math.sqrt(freq_per_year)


def _compute_metrics(
    strat_net: pd.Series,
    equity: pd.Series,
    costs: pd.Series,
) -> Dict[str, float]:
    """
    Compute standard performance metrics from returns / equity / costs.
    """
    if len(equity) > 1:
        cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    else:
        cum_ret = 0.0

    if len(strat_net) > 1:
        ann_vol = float(strat_net.std() * math.sqrt(252))
    else:
        ann_vol = 0.0

    sharpe = _annualize_sharpe(strat_net)

    # Max drawdown on equity
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    total_costs = float(costs.sum())

    return {
        "Cumulative Return": cum_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Costs": total_costs,
    }


# -------------------------------------------------------------------------
# Single-strategy backtest
# -------------------------------------------------------------------------


def run_backtest(
    price_like: Union[pd.Series, pd.DataFrame],
    positions_like: Union[pd.Series, pd.DataFrame],
    cost_bps: float = 1.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Vectorized backtest.

    Parameters
    ----------
    price_like : pd.Series or pd.DataFrame
        Underlying price series, or a DataFrame containing a 'price' column.
    positions_like : pd.Series or pd.DataFrame
        Strategy positions, typically in {-1, 0, +1} (already shifted to
        avoid lookahead). If a DataFrame is provided, uses the 'position'
        column if present, otherwise the first column.
    cost_bps : float
        Transaction costs in basis points per unit of |Δposition|.
    initial_capital : float
        Starting equity for the strategy.

    Returns
    -------
    BacktestResult
    """
    price = _to_price_series(price_like)

    # Positions → Series aligned on price index
    if isinstance(positions_like, pd.DataFrame):
        if "position" in positions_like.columns:
            pos = positions_like["position"]
        else:
            pos = positions_like.iloc[:, 0]
    else:
        pos = positions_like

    pos = pd.Series(pos, index=price.index).reindex(price.index).fillna(0.0).astype(float)
    pos.name = "position"

    # Underlying returns
    ret_under = price.pct_change().fillna(0.0)

    # Strategy gross P&L
    strat_gross = pos.shift(1).fillna(0.0) * ret_under
    strat_gross.name = "strat_gross"

    # Trades & costs (|Δposition| * cost_bps)
    trades = pos.diff().abs().fillna(0.0)
    trades.name = "trades"
    costs = trades * (cost_bps / 10_000.0)
    costs.name = "costs"

    # Net returns & equity
    strat_net = strat_gross - costs
    strat_net.name = "strat_net"
    equity = (1.0 + strat_net).cumprod() * initial_capital
    equity.name = "equity"

    metrics = _compute_metrics(strat_net, equity, costs)

    return BacktestResult(
        equity=equity,
        returns=strat_net,
        costs=costs,
        positions=pos,
        trades=trades,
        metrics=metrics,
    )


# -------------------------------------------------------------------------
# Portfolio combination of several BacktestResult objects
# -------------------------------------------------------------------------


def combine_backtests(
    results: Mapping[str, BacktestResult],
    weights: Mapping[str, float],
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Linearly combine several BacktestResult objects into a portfolio.

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Mapping from strategy name to BacktestResult.
    weights : dict[str, float]
        Mapping from strategy name to portfolio weight (must match keys
        in `results`). The weights do not need to sum exactly to 1 but
        they will be normalized internally.
    initial_capital : float
        Starting equity of the combined portfolio.

    Returns
    -------
    BacktestResult
        Portfolio result with aggregated returns / equity / metrics.
    """
    if not results:
        raise ValueError("`results` is empty in combine_backtests.")
    if not weights:
        raise ValueError("`weights` is empty in combine_backtests.")

    # Normalize weights to sum to 1
    w_series = pd.Series(weights, dtype=float)
    if w_series.sum() == 0:
        raise ValueError("All portfolio weights are zero.")
    w_series = w_series / w_series.sum()

    # Use the index of the first strategy as master calendar
    first_key = next(iter(results))
    master_index = results[first_key].returns.index

    # Aggregate series
    port_ret = pd.Series(0.0, index=master_index)
    port_costs = pd.Series(0.0, index=master_index)
    port_pos = pd.Series(0.0, index=master_index)
    port_trades = pd.Series(0.0, index=master_index)

    for name, res in results.items():
        if name not in w_series.index:
            continue  # weight = 0 implicitly
        w = float(w_series[name])

        # Reindex each series on the master index just in case
        r = res.returns.reindex(master_index).fillna(0.0)
        c = res.costs.reindex(master_index).fillna(0.0)
        p = res.positions.reindex(master_index).fillna(0.0)
        t = res.trades.reindex(master_index).fillna(0.0)

        port_ret += w * r
        port_costs += w * c
        port_pos += w * p
        port_trades += w * t

    port_ret.name = "portfolio_returns"
    port_costs.name = "portfolio_costs"
    port_pos.name = "portfolio_position"
    port_trades.name = "portfolio_trades"

    # Rebuild equity and metrics
    equity = (1.0 + port_ret).cumprod() * initial_capital
    equity.name = "equity"
    metrics = _compute_metrics(port_ret, equity, port_costs)

    return BacktestResult(
        equity=equity,
        returns=port_ret,
        costs=port_costs,
        positions=port_pos,
        trades=port_trades,
        metrics=metrics,
    )
