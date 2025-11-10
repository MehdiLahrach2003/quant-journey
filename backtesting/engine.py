# backtesting/engine.py
# Core backtesting engine: computes strategy returns, costs, equity curve, and metrics.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252
_EPS = 1e-12  # Small epsilon to avoid division by ~0 when computing volatility or Sharpe ratio


@dataclass
class BacktestResult:
    equity: pd.Series
    returns_ret_under: pd.Series
    strategy_ret: pd.Series
    costs: pd.Series
    positions: pd.Series
    trades: pd.Series
    metrics: Dict[str, float]


def _safe_series(x: pd.Series, name: str) -> pd.Series:
    """Ensures a clean pandas Series with a proper name."""
    s = pd.Series(x, index=x.index if isinstance(x, pd.Series) else None, copy=False)
    s.name = name
    return s


def _max_drawdown(equity: pd.Series) -> float:
    """Computes the maximum drawdown (returns the lowest relative drawdown value)."""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def run_backtest(
    df: pd.DataFrame,
    positions: pd.Series,
    total_bps: float = 10.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Vectorized long/short backtest engine.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'price' column (indexed by dates).
    positions : pd.Series
        Strategy positions per date (aligned with df.index).
        Expected range: {-1, 0, +1} or continuous weights.
    total_bps : float, default=10.0
        Total transaction cost (in basis points), applied to absolute position changes.
    initial_capital : float, default=1.0
        Starting portfolio value.

    Returns
    -------
    BacktestResult
        Dataclass containing:
        - equity curve
        - underlying and strategy returns
        - transaction costs and trades
        - key performance metrics (Sharpe, Volatility, etc.)

    Notes
    -----
    - Returns are computed daily based on shifted positions.
    - Sharpe ratio is protected with a small epsilon (_EPS) to avoid infinite values.
    """

    if "price" not in df.columns:
        raise KeyError("DataFrame must contain a 'price' column.")

    prices = df["price"].astype(float).copy()
    prices.name = "price"

    # Underlying returns (simple percentage returns)
    ret_under = prices.pct_change().fillna(0.0)
    ret_under = _safe_series(ret_under, "underlying_ret")

    # Clean and align positions
    positions = positions.reindex(prices.index).fillna(0.0).astype(float)
    positions.name = "position"

    # Trades = absolute position change
    trades = positions.diff().abs().fillna(0.0)
    trades.name = "trades"

    # Transaction cost rate (bps → percentage)
    cost_rate = total_bps / 10_000.0
    costs = (trades * cost_rate).rename("costs")

    # Gross strategy return (position_{t-1} × return_t)
    strat_gross = positions.shift(1).fillna(0.0) * ret_under
    strat_gross.name = "strategy_gross"

    # Net return after transaction costs
    strat_net = strat_gross - costs
    strat_net.name = "strategy_net"

    # Equity curve
    equity = (1.0 + strat_net).cumprod() * float(initial_capital)
    equity = _safe_series(equity, "equity")

    # Metrics calculation
    mean_daily = float(strat_net.mean())
    std_daily = float(strat_net.std(ddof=0))
    std_ann = (std_daily * np.sqrt(TRADING_DAYS)) if std_daily > 0 else 0.0
    sharpe = (mean_daily * TRADING_DAYS) / max(std_ann, _EPS)

    cumret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    maxdd = _max_drawdown(equity)
    total_costs = float(costs.sum())

    metrics = {
        "cumret": cumret,
        "ann_vol": std_ann,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "total_costs": total_costs,
    }

    return BacktestResult(
        equity=equity,
        returns_ret_under=ret_under.rename("underlying_ret"),
        strategy_ret=strat_net.rename("strategy_ret"),
        costs=costs,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )

