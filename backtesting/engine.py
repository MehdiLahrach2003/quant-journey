# backtesting/engine.py
# Simple long/short backtest engine for daily data.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import math
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series          # equity curve
    returns: pd.Series         # strategy daily returns (net)
    costs: pd.Series           # daily costs series
    positions: pd.Series       # aligned positions
    trades: pd.Series          # abs(position change)
    metrics: Dict[str, float]  # summary metrics


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
    r = daily.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * math.sqrt(freq_per_year)


def run_backtest(
    price_like: Union[pd.Series, pd.DataFrame],
    positions_like: Union[pd.Series, pd.DataFrame],
    cost_bps: float = 1.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Vectorized backtest:
      - price_like: Series (price) or DataFrame with 'price'
      - positions_like: Series in {-1,0,+1} (shifted by 1 bar, pas de lookahead)
      - cost_bps: transaction costs in basis points per trade unit (|Δposition|)
    """
    price = _to_price_series(price_like)

    # Positions → Series alignée sur l'index prix
    if isinstance(positions_like, pd.DataFrame):
        if "position" in positions_like.columns:
            pos = positions_like["position"]
        else:
            # take first column
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

    # Metrics
    cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    ann_vol = float(strat_net.std() * math.sqrt(252)) if len(strat_net) > 1 else 0.0
    sharpe = _annualize_sharpe(strat_net)
    # Max drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0
    total_costs = float(costs.sum())

    metrics = {
        "Cumulative Return": cum_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Costs": total_costs,
    }

    return BacktestResult(
        equity=equity,
        returns=strat_net,
        costs=costs,
        positions=pos,
        trades=trades,
        metrics=metrics,
    )


