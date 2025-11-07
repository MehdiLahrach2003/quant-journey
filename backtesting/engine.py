import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    strategy_ret: pd.Series
    costs: pd.Series
    positions: pd.Series
    trades: pd.Series
    metrics: dict


def compute_metrics(strategy_ret, equity, costs):
    cumret = (1 + strategy_ret).prod() - 1
    vol = strategy_ret.std() * np.sqrt(252)
    sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
    drawdown = (equity / equity.cummax() - 1).min()
    return {
        "Cumulative Return": cumret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown,
        "Total Costs": costs.sum(),
    }


def run_backtest(prices: pd.Series, positions: pd.Series, trans_cost_bps=1.0, initial_capital=1.0):
    ret_under = prices.pct_change().fillna(0.0)
    trades = positions.diff().abs().fillna(0.0)
    total_bps = trans_cost_bps / 1e4

    strat_gross = positions.shift(1).fillna(0.0) * ret_under
    costs = (trades * total_bps).rename("costs")
    strat_net = strat_gross - costs

    equity = initial_capital * (1 + strat_net).cumprod()

    metrics = compute_metrics(strat_net, equity, costs)

    return BacktestResult(
        equity=equity,
        returns=ret_under,
        strategy_ret=strat_net,
        costs=costs,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )
